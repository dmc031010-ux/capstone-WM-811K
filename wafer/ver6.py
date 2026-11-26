"""ver6 Phase 1: Latent Multi-Masking Ensemble Consistency (디코더 없음, 1차원 잠재 벡터 연산만).

원본 x → 타깃 인코더(EMA, gradient 멈춤) → z (원본 특징 벡터)
마스킹 4장 → 온라인 인코더 → 선형 Predictor → 예측 벡터 4개 → 평균 = 앙상블 복원 벡터 p
loss = 1 - cos(p, z)
Phase 2 는 ver3 와 같은 labeled 미세조정 (selfsup.finetune) 을 그대로 사용.
"""
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import augment as A
from .engine import BalancedSampler, set_seed
from .models import FiLMResNet, to_input
from .selfsup import V3, _encoder, batch_hard_triplet

V6 = dict(n_masks=4, mask_area=0.2, ema=0.996)
# ver6.1: 바둑판 50% 마스킹 + 원본/마스킹본에 서로 다른 무작위 회전 + 2층 MLP Predictor
V61 = dict(n_masks=4, cell=12, ema=0.996)


class InterViewSelfAttention(nn.Module):
    """ver9.3A: 마스킹된 뷰 예측 벡터들 간 Self-Attention 후 뷰별 가중치를 학습해 결합."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_scale = nn.Parameter(torch.zeros(()))
        self.score = nn.Linear(dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, p):
        normed = self.norm(p)
        contextual, attn_weights = self.attn(normed, normed, normed, need_weights=True,
                                             average_attn_weights=False)
        refined = p + self.attn_scale * contextual
        weights = self.score(self.norm(refined)).squeeze(-1).softmax(dim=1)
        pooled = (refined * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights, attn_weights.detach()


class TargetConditionedViewGate(nn.Module):
    """ver9.3C: 뷰 간 어텐션 + EMA 타깃 벡터 z 와의 관계를 함께 보고 뷰별 가중치를 산출."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.view_norm = nn.LayerNorm(dim)
        self.target_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_scale = nn.Parameter(torch.zeros(()))
        self.score = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, p, z):
        normed = self.view_norm(p)
        contextual, attn_weights = self.attn(normed, normed, normed, need_weights=True,
                                              average_attn_weights=False)
        refined = p + self.attn_scale * contextual
        target = self.target_norm(z).unsqueeze(1).expand_as(refined)
        view = self.view_norm(refined)
        gate_input = torch.cat((view, target, (view - target).abs(), view * target), dim=-1)
        weights = self.score(gate_input).squeeze(-1).softmax(dim=1)
        pooled = (refined * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights, attn_weights.detach()


class MaskedLatentModel(nn.Module):
    """Encoder(ResNet-18, 512) + Predictor. 타깃 인코더는 온라인 인코더의 EMA 복사본.

    mlp=False: 선형 Predictor(512→512, ver6) / mlp=True: 512→512→512 (BN+ReLU, BYOL 구조, ver6.1)
    """

    def __init__(self, arch, mlp=False, view_attention=False, view_gate_mode=None, size=None):
        super().__init__()
        self.encoder, dim = _encoder(arch)
        if size is not None:   # ver8.2: 인코더 (온라인 / 타깃 모두) 를 크기 조건 FiLM ResNet 으로
            self.encoder = FiLMResNet(self.encoder, size)
        self.predictor = (nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
                          if mlp else nn.Linear(dim, dim))
        self.view_gate_mode = view_gate_mode
        self.view_agg = (InterViewSelfAttention(dim) if view_attention else
                         TargetConditionedViewGate(dim) if view_gate_mode == "target_conditioned" else None)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, m):
        for t, o in zip(self.target.state_dict().values(), self.encoder.state_dict().values()):
            if t.dtype.is_floating_point:
                t.mul_(m).add_(o.detach(), alpha=1 - m)
            else:
                t.copy_(o)


def _predict(model, xm, bn_groups=None, idx=None):
    """마스킹본 (n,B,...) → 예측 (n,B,d). bn_groups: BN 통계를 같이 쓰는 묶음 (치팅 검증용)

    None: n*B 장 전체 한 묶음 (기존) / "split": 묶음 = 같은 순번 마스킹본 B 장 (형제 뷰와 통계 공유 없음)
    "together": 묶음 = 웨이퍼 B/n 장의 마스킹본 n 장씩 (형제 뷰와 통계 공유, 묶음 크기는 split 과 같음)
    idx: (B,) 웨이퍼 번호 — 크기 조건 인코더 (ver8.2) 일 때만
    """
    n, b = xm.shape[:2]
    if idx is not None:
        assert bn_groups is None
        return model.predictor(model.encoder(to_input(xm.flatten(0, 1)), idx.repeat(n))).view(n, b, -1)
    f = lambda v: model.predictor(model.encoder(to_input(v)))
    if bn_groups == "split":
        return torch.stack([f(xm[i]) for i in range(n)])
    if bn_groups == "together":
        k = b // n
        return torch.cat([f(xm[:, j * k:(j + 1) * k].flatten(0, 1)).view(n, k, -1) for j in range(n)], 1)
    return f(xm.flatten(0, 1)).view(n, b, -1)


def consistency_loss(model, x, xm, per_view=False, bn_groups=None, idx=None, view_gate_tau=0.2, view_gate_lambda=0.05):
    """x: 원본 (B,1,H,W), xm: 마스킹본 (n,B,1,H,W). 반환: (loss, 원본 벡터 z).

    per_view=False: 1 - cos(mean(p_i), z) (ver6~ver7) / True: mean_i (1 - cos(p_i, z)) (ver7.2)
    idx: (B,) 웨이퍼 번호 — 크기 조건 인코더 (ver8.2) 일 때만
    """
    with torch.no_grad():
        model.target.train()  # BYOL 처럼 배치 통계로 BN (EMA 된 running 통계는 쓰지 않음)
        z = (model.target(to_input(x)) if idx is None else model.target(to_input(x), idx)).float()
    p = _predict(model, xm, bn_groups, idx).float()
    if per_view:                                       # 마스킹본 각각을 z 와 비교한 뒤 손실 평균
        return 1 - F.cosine_similarity(p, z.unsqueeze(0), dim=2).mean(), z
    p_views = p.transpose(0, 1)                        # (B, views, D)
    gate_loss = z.new_zeros(())
    if model.view_gate_mode == "target_similarity":
        w = (F.cosine_similarity(p_views, z.unsqueeze(1), dim=-1).detach() / view_gate_tau).softmax(dim=1)
        pooled = (p_views * w.unsqueeze(-1)).sum(dim=1)
    elif model.view_gate_mode == "target_conditioned":
        pooled, w, _ = model.view_agg(p_views, z)
        quality = (F.cosine_similarity(p_views, z.unsqueeze(1), dim=-1).detach() / view_gate_tau).softmax(dim=1)
        gate_loss = F.kl_div(w.clamp_min(1e-8).log(), quality, reduction="batchmean")
    elif model.view_agg is not None:
        pooled, _, _ = model.view_agg(p_views)
    else:
        pooled = p.mean(0)                             # ver6~ver7: 4개 예측 벡터의 앙상블 평균
    loss = 1 - F.cosine_similarity(pooled, z, dim=1).mean()
    if model.view_gate_mode == "target_conditioned":
        loss = loss + view_gate_lambda * gate_loss
    return loss, z


def _views(x, v61, cell=V61["cell"], n_masks=V61["n_masks"], ratio=0.5, dilate=0, stripes=False, crop=False):
    """(원본 입력, 마스킹본 (n,B,1,H,W)). ver6.1~ 은 원본과 마스킹본이 각자 다른 회전을 받은 뒤 마스킹.

    dilate: ver7.3 — 회전 → 팽창 채널 추가 (B,2,H,W) → 두 채널에 같은 바둑판 마스킹
    stripes: ver7.7 — 마지막 2장을 바둑판 대신 세로줄 / 가로줄 (같은 칸 크기, 회전 후 마스킹)
             "all" — 모두 줄무늬 (세로, 가로 번갈아; ver7.X1 / 7.95 의 줄무늬 단계)
    crop: ver7.X — 마스킹본만 회전 → 무작위 크기 자르기 (ver3 의 crop, 면적 50~100%) → 마스킹, 원본은 그대로
    """
    if not v61:
        return x, A.multi_mask(x, V6["n_masks"], V6["mask_area"])
    rot = (lambda v: A.add_dilated(A.rotate(v), dilate)) if dilate else A.rotate
    masks = [lambda v: A.checker_mask(v, cell, ratio)] * n_masks
    sv = [lambda v: A.stripe_mask(v, cell, "v"), lambda v: A.stripe_mask(v, cell, "h")]
    if stripes == "all":
        masks = sv * (n_masks // 2)
    elif stripes:
        masks[-2:] = sv
    xm = torch.stack([m(A.crop(rot(x)) if crop else rot(x)) for m in masks])
    return rot(x), xm


def pretrain(ds, sub, args, device, out_path, v61=False, cell=V61["cell"], n_masks=V61["n_masks"], ratio=0.5, mlp=None,
             per_view=False, dilate=0, stripes=False, curriculum=None, lab_idx=None, triplet_w=0.0, crop=False,
             view_attention=False, view_gate_mode=None, bn_groups=None, size=None):
    """Phase 1. 온라인 인코더(state_dict, fc 제외)를 저장 — selfsup.finetune 이 그대로 읽음.

    cell: 바둑판 칸 크기 (ver6.1 = 12, ver6.2~ = 6), n_masks: 마스킹본 수 (ver6.3 = 3), ratio: 가리는 비율 (ver6.4 = 0.3)
    mlp: Predictor 구조 (None 이면 v61 을 따름, ver6.6 = False 로 선형)
    per_view: 손실을 마스킹본별로 계산 후 평균 (ver7.2), dilate: 2채널 입력의 팽창 커널 (ver7.3 = 3)
    stripes: 마스킹본 2장을 세로줄 / 가로줄로 (ver7.7)
    curriculum: 에폭 절반씩 마스킹 단계 전환 — "stripe_first" 줄무늬 → 바둑판 (ver7.X1), "checker_first" 반대 (ver7.95)
    lab_idx, triplet_w: 매 스텝 라벨 데이터 128장 (클래스 균형, 회전만) 의 인코더 특징에 batch-hard Triplet 추가 (ver7.X2)
    crop: 마스킹본만 무작위 크기 자르기 (ver7.X)
    view_attention, view_gate_mode: ver9.3A/B/C 멀티뷰 어텐션/게이트 결합
    bn_groups: 마스킹본의 BN 묶음 (_predict 참고, 치팅 검증용)
    size: 웨이퍼별 표준화 log(die 수) 표 — 주면 인코더가 FiLM 으로 크기 조건을 받음 (ver8.2)
    """
    set_seed(args.seed)
    cfg = V61 if v61 else V6
    assert size is None or not (triplet_w or bn_groups)
    model = MaskedLatentModel(args.arch, mlp=v61 if mlp is None else mlp,
                              view_attention=view_attention, view_gate_mode=view_gate_mode, size=size).to(device)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    if model.view_agg is not None:
        params += list(model.view_agg.parameters())
    opt = torch.optim.AdamW(params, lr=V3["ssl_lr"], weight_decay=1e-4)
    B = args.batch_size
    steps = len(sub) // B
    total = steps * args.ssl_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / total)))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(args.seed)
    if triplet_w:
        y_lab = ds.labels[lab_idx]
        lsampler = BalancedSampler(y_lab, args.seed)
    # 에폭마다 중간 저장 → 중간에 죽으면 다시 실행 시 이어서 (끝나면 삭제)
    ckpt_path, start = out_path + ".ckpt", 1
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"].cpu()); torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        if triplet_w:
            lsampler = ck["lsampler"]
        start = ck["ep"] + 1
        print(f"--- resume from {ckpt_path} (epoch {ck['ep']} done)", flush=True)
    for ep in range(start, args.ssl_epochs + 1):
        model.encoder.train(); model.predictor.train()
        if model.view_agg is not None:
            model.view_agg.train()
        t, tot, zstd, tl = time.time(), 0.0, 0.0, 0.0
        if curriculum:
            first = ep <= args.ssl_epochs // 2
            stripes = "all" if first == (curriculum == "stripe_first") else False
        perm = rng.permutation(sub)
        for s in range(steps):
            with torch.no_grad():
                bi = np.sort(perm[s * B:(s + 1) * B])
                x, xm = _views(ds.fetch(bi, device), v61, cell, n_masks, ratio, dilate, stripes, crop)
                idx = torch.as_tensor(bi, device=device) if size is not None else None
                if triplet_w:
                    j = lsampler(128)
                    xl = A.rotate(ds.fetch(lab_idx[j], device))
                    yl = torch.from_numpy(y_lab[j]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, z = consistency_loss(model, x, xm, per_view, bn_groups, idx)
                if triplet_w:
                    Lt = batch_hard_triplet(model.encoder(to_input(xl)), yl)
                    loss = loss + triplet_w * Lt
                    tl += Lt.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            model.update_target(cfg["ema"])
            tot += loss.item()
            # 붕괴 감시: 정규화 벡터의 차원별 표준편차 평균 (건강 ≈ 1/sqrt(512) ≈ 0.044, 붕괴 → 0)
            zstd += F.normalize(z, dim=1).std(0).mean().item()
        print(f"ssl epoch={ep} loss={tot / steps:.4f} z_std={zstd / steps:.4f} sec={time.time() - t:.1f}"
              + (f" mask={stripes or 'checker'}" if curriculum else "") + (f" Lt={tl / steps:.4f}" if triplet_w else ""), flush=True)
        torch.save(dict(ep=ep, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        lsampler=lsampler if triplet_w else None), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    torch.save(model.encoder.state_dict(), out_path)
    os.remove(ckpt_path)
