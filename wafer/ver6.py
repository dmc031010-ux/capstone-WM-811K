"""ver6 Phase 1: Latent Multi-Masking Ensemble Consistency (디코더 없음, 1차원 잠재 벡터 연산만).

원본 x → 타깃 인코더(EMA, gradient 멈춤) → z (원본 특징 벡터)
마스킹 4장 → 온라인 인코더 → 선형 Predictor → 예측 벡터 4개 → 평균 = 앙상블 복원 벡터 p
loss = 1 - cos(p, z)
Phase 2 는 ver3 와 같은 labeled 미세조정 (selfsup.finetune) 을 그대로 사용.
"""
import copy
import json
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
    """Contextualize masked-view predictions, then learn a weight per view."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_scale = nn.Parameter(torch.zeros(()))
        self.score = nn.Linear(dim, 1)
        # Start as the ver7 arithmetic mean; learn view weighting during pretraining.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, p):
        # p: (B, views, dim). Views are exchangeable, so there is no view-index embedding.
        normed = self.norm(p)
        contextual, attn_weights = self.attn(normed, normed, normed, need_weights=True,
                                             average_attn_weights=False)
        context_delta = self.attn_scale * contextual
        refined = p + context_delta
        weights = self.score(self.norm(refined)).squeeze(-1).softmax(dim=1)
        pooled = (refined * weights.unsqueeze(-1)).sum(dim=1)
        diag = {
            "mha_scale": self.attn_scale.detach(),
            "mha_delta_ratio": (context_delta.norm(dim=-1) / p.norm(dim=-1).clamp_min(1e-12)).detach(),
            "mha_weights": attn_weights.detach(),
        }
        return pooled, weights, diag


class TargetConditionedViewGate(nn.Module):
    """Score each view using its relation to the full-view EMA target."""

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
        # p: (B, views, D); z: (B, D). The gate is permutation-equivariant over views.
        normed = self.view_norm(p)
        contextual, attn_weights = self.attn(normed, normed, normed, need_weights=True,
                                              average_attn_weights=False)
        context_delta = self.attn_scale * contextual
        refined = p + context_delta
        target = self.target_norm(z).unsqueeze(1).expand_as(refined)
        view = self.view_norm(refined)
        gate_input = torch.cat((view, target, (view - target).abs(), view * target), dim=-1)
        weights = self.score(gate_input).squeeze(-1).softmax(dim=1)
        pooled = (refined * weights.unsqueeze(-1)).sum(dim=1)
        diag = {
            "mha_scale": self.attn_scale.detach(),
            "mha_delta_ratio": (context_delta.norm(dim=-1) / p.norm(dim=-1).clamp_min(1e-12)).detach(),
            "mha_weights": attn_weights.detach(),
        }
        return pooled, weights, diag


class MaskedLatentModel(nn.Module):
    """Encoder(ResNet-18, 512) + Predictor. 타깃 인코더는 온라인 인코더의 EMA 복사본.

    mlp=False: 선형 Predictor(512→512, ver6) / mlp=True: 512→512→512 (BN+ReLU, BYOL 구조, ver6.1)
    """

    def __init__(self, arch, mlp=False, view_attention=False, view_gate_mode=None, use_projector=False, size=None):
        super().__init__()
        self.encoder, dim = _encoder(arch)
        if size is not None:   # ver8.2: 인코더 (온라인 / 타깃 모두) 를 크기 조건 FiLM ResNet 으로
            self.encoder = FiLMResNet(self.encoder, size)
        if use_projector:
            if use_projector == "wide":
                proj_hid, proj_out, pred_hid = 512, 256, 256
            else:
                proj_hid, proj_out, pred_hid = 256, 128, 256
            self.projector = nn.Sequential(
                nn.Linear(dim, proj_hid), nn.BatchNorm1d(proj_hid), nn.ReLU(inplace=True), nn.Linear(proj_hid, proj_out)
            )
            self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)
            self.predictor = nn.Sequential(
                nn.Linear(proj_out, pred_hid), nn.BatchNorm1d(pred_hid), nn.ReLU(inplace=True), nn.Linear(pred_hid, proj_out)
            )
            head_dim = proj_out
        else:
            self.projector = None
            self.target_projector = None
            self.predictor = (nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
                              if mlp else nn.Linear(dim, dim))
            head_dim = dim
        self.view_gate_mode = view_gate_mode
        self.view_agg = (InterViewSelfAttention(head_dim) if view_attention else
                         TargetConditionedViewGate(head_dim) if view_gate_mode == "target_conditioned" else None)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, m):
        pairs = [(self.target, self.encoder)]
        if self.target_projector is not None:
            pairs.append((self.target_projector, self.projector))
        for tgt, src in pairs:
            for t, o in zip(tgt.state_dict().values(), src.state_dict().values()):
                if t.dtype.is_floating_point:
                    t.mul_(m).add_(o.detach(), alpha=1 - m)
                else:
                    t.copy_(o)


def consistency_loss(model, x, xm, per_view=False, view_gate_tau=0.1, view_gate_lambda=0.1, loss_mode=None, bn_groups=None, idx=None):
    """x: 원본 (B,1,H,W), xm: 마스킹본 (n,B,1,H,W). 반환: (loss, 원본 벡터 z).

    per_view=False: 1 - cos(mean(p_i), z) (ver6~ver7) / True: mean_i (1 - cos(p_i, z)) (ver7.2)
    loss_mode: None (ver7 기본) | "infonce" (L1) | "uniformity" (L2) | "vicreg" (L3) | "debiased_infonce" (L4)
    """
    n, b = xm.shape[:2]
    with torch.no_grad():
        model.target.train()  # BYOL 처럼 배치 통계로 BN (EMA 된 running 통계는 쓰지 않음)
        z = model.target(to_input(x)) if idx is None else model.target(to_input(x), idx)
        if model.target_projector is not None:
            model.target_projector.train()
            z = model.target_projector(z)
        z = z.float()
    if idx is not None:
        h = model.encoder(to_input(xm.flatten(0, 1)), idx.repeat(n))
    else:
        h = model.encoder(to_input(xm.flatten(0, 1)))
    if model.projector is not None:
        h = model.projector(h)
    p = model.predictor(h).float()
    p = p.view(n, b, -1)
    if per_view:                                       # 마스킹본 각각을 z 와 비교한 뒤 손실 평균
        return 1 - F.cosine_similarity(p, z.unsqueeze(0), dim=2).mean(), z, None, None, z.new_zeros(()), None
    p = p.transpose(0, 1)                              # (B, views, D)
    alignment = F.cosine_similarity(p, z.unsqueeze(1), dim=-1)
    quality = (alignment.detach() / view_gate_tau).softmax(dim=1)
    gate_loss = z.new_zeros(())
    view_diag = None
    if model.view_gate_mode == "target_similarity":
        view_weights = quality
        pooled = (p * view_weights.unsqueeze(-1)).sum(dim=1)
    elif model.view_gate_mode == "target_conditioned":
        pooled, view_weights, view_diag = model.view_agg(p, z)
        gate_loss = F.kl_div(view_weights.clamp_min(1e-8).log(), quality, reduction="batchmean")
    elif model.view_agg is not None:
        pooled, view_weights, view_diag = model.view_agg(p)  # ver9.2: inter-view self-attention
    else:
        view_weights = None
        pooled = p.mean(1)                             # ver6~ver7: 4개 예측 벡터의 산술 평균
    if loss_mode == "infonce":
        p_norm = F.normalize(pooled, dim=1)
        z_norm = F.normalize(z, dim=1)
        logits = (p_norm @ z_norm.T) / V3["ssl_temp"]
        loss = F.cross_entropy(logits, torch.arange(b, device=z.device))
    elif loss_mode == "uniformity":
        p_norm = F.normalize(pooled, dim=1)
        align = 1 - F.cosine_similarity(pooled, z, dim=1).mean()
        sq_pdist = (2.0 - 2.0 * (p_norm @ p_norm.T)).clamp_min(0.0)
        off_mask = ~torch.eye(b, dtype=torch.bool, device=z.device)
        unif = torch.log(torch.exp(-2.0 * sq_pdist[off_mask]).mean().clamp_min(1e-12))
        loss = align + 0.1 * unif
    elif loss_mode == "vicreg":
        align = 1 - F.cosine_similarity(pooled, z, dim=1).mean()
        u = F.normalize(pooled, dim=1) * math.sqrt(pooled.shape[1])
        std = torch.sqrt(u.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        u_c = u - u.mean(dim=0, keepdim=True)
        cov = (u_c.T @ u_c) / max(b - 1, 1)
        off_cov = cov[~torch.eye(u.shape[1], dtype=torch.bool, device=z.device)]
        cov_loss = (off_cov ** 2).mean()
        loss = align + 0.1 * var_loss + 0.1 * cov_loss
    elif loss_mode == "debiased_infonce":
        p_norm = F.normalize(pooled, dim=1)
        z_norm = F.normalize(z, dim=1)
        z_sim = (z_norm @ z_norm.T).detach()
        neg_w = (1.0 - z_sim.clamp_min(0.0)).clamp(min=0.05)
        log_w = torch.log(neg_w)
        log_w.fill_diagonal_(0.0)
        logits = (p_norm @ z_norm.T) / V3["ssl_temp"] + log_w
        loss = F.cross_entropy(logits, torch.arange(b, device=z.device))
    else:
        loss = 1 - F.cosine_similarity(pooled, z, dim=1).mean()
    if model.view_gate_mode == "target_conditioned":
        loss = loss + view_gate_lambda * gate_loss
    return loss, z, view_weights, alignment.detach(), gate_loss, view_diag


def _views(x, v61, cell=V61["cell"], n_masks=V61["n_masks"], ratio=0.5, dilate=0, stripes=False, crop=False, flip=False):
    """(원본 입력, 마스킹본 (n,B,1,H,W)). ver6.1~ 은 원본과 마스킹본이 각자 다른 회전을 받은 뒤 마스킹.

    dilate: ver7.3 — 회전 → 팽창 채널 추가 (B,2,H,W) → 두 채널에 같은 바둑판 마스킹
    stripes: ver7.7 — 마지막 2장을 바둑판 대신 세로줄 / 가로줄 (같은 칸 크기, 회전 후 마스킹)
             "all" — 모두 줄무늬 (세로, 가로 번갈아; ver7.X1 / 7.95 의 줄무늬 단계)
    crop: ver7.X — 마스킹본만 회전 → 무작위 크기 자르기 (ver3 의 crop, 면적 50~100%) → 마스킹, 원본은 그대로
    flip: ver999 — 원본·마스킹본 모두 회전 후 무작위 좌우반전(hflip, p=0.5) 추가 (완전 O(2) 대칭)
    """
    if not v61:
        return x, A.multi_mask(x, V6["n_masks"], V6["mask_area"])
    base_geo = (lambda v: A.hflip(A.rotate(v))) if flip else A.rotate
    rot = (lambda v: A.add_dilated(base_geo(v), dilate)) if dilate else base_geo
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
             view_attention=False, view_gate_mode=None, flip=False, loss_mode=None, use_projector=False,
             bn_groups=None, size=None):
    """Phase 1. 온라인 인코더(state_dict, fc 제외)를 저장 — selfsup.finetune 이 그대로 읽음.

    cell: 바둑판 칸 크기 (ver6.1 = 12, ver6.2~ = 6), n_masks: 마스킹본 수 (ver6.3 = 3), ratio: 가리는 비율 (ver6.4 = 0.3)
    mlp: Predictor 구조 (None 이면 v61 을 따름, ver6.6 = False 로 선형)
    per_view: 손실을 마스킹본별로 계산 후 평균 (ver7.2), dilate: 2채널 입력의 팽창 커널 (ver7.3 = 3)
    stripes: 마스킹본 2장을 세로줄 / 가로줄로 (ver7.7)
    curriculum: 에폭 절반씩 마스킹 단계 전환 — "stripe_first" 줄무늬 → 바둑판 (ver7.X1), "checker_first" 반대 (ver7.95)
    lab_idx, triplet_w: 매 스텝 라벨 데이터 128장 (클래스 균형, 회전만) 의 인코더 특징에 batch-hard Triplet 추가 (ver7.X2)
    crop: 마스킹본만 무작위 크기 자르기 (ver7.X)
    loss_mode: None | "infonce" (L1) | "uniformity" (L2) | "vicreg" (L3) | "debiased_infonce" (L4)
    use_projector: True 이면 경량 Projector(512→256→128) + Predictor(128→256→128) 분리 (A1)
    size: 웨이퍼별 표준화 log(die 수) 표 — 주면 인코더가 FiLM 으로 크기 조건을 받음 (ver8.2)
    """
    set_seed(args.seed)
    cfg = V61 if v61 else V6
    assert size is None or not (triplet_w or bn_groups)
    model = MaskedLatentModel(args.arch, mlp=v61 if mlp is None else mlp, view_attention=view_attention,
                              view_gate_mode=view_gate_mode, use_projector=use_projector, size=size).to(device)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    if model.projector is not None:
        params += list(model.projector.parameters())
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
    view_weight_history = {}
    view_weight_path = os.path.splitext(out_path)[0] + "_view_weights.json"
    if (view_attention or view_gate_mode is not None) and os.path.exists(view_weight_path):
        with open(view_weight_path, encoding="utf-8") as f:
            previous_rows = json.load(f)["epochs"]
            # Keep the richer per-epoch records when resuming; older ver9.2 files
            # already contain dict rows with just epoch and mean_weights.
            view_weight_history = {str(row["epoch"]): row for row in previous_rows}
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
        if model.projector is not None:
            model.projector.train()
        t, tot, zstd, tl = time.time(), 0.0, 0.0, 0.0
        view_weight_sum = np.zeros(n_masks, dtype=np.float64)
        view_weight_batches = 0
        view_weight_samples, view_alignment_samples, view_sample_ids = [], [], []
        mha_scales, mha_delta_ratios, mha_attention_means, gate_losses = [], [], [], []
        if curriculum:
            first = ep <= args.ssl_epochs // 2
            stripes = "all" if first == (curriculum == "stripe_first") else False
        perm = rng.permutation(sub)
        for s in range(steps):
            with torch.no_grad():
                batch_ids = np.sort(perm[s * B:(s + 1) * B])
                x, xm = _views(ds.fetch(batch_ids, device), v61, cell, n_masks, ratio, dilate, stripes, crop, flip)
                idx = torch.as_tensor(batch_ids, device=device) if size is not None else None
                if triplet_w:
                    j = lsampler(128)
                    xl = A.rotate(ds.fetch(lab_idx[j], device))
                    yl = torch.from_numpy(y_lab[j]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, z, view_weights, view_alignment, gate_loss, view_diag = consistency_loss(
                    model, x, xm, per_view, args.view_gate_tau, args.view_gate_lambda,
                    loss_mode=loss_mode, bn_groups=bn_groups, idx=idx)
                if triplet_w:
                    Lt = batch_hard_triplet(model.encoder(to_input(xl)), yl)
                    loss = loss + triplet_w * Lt
                    tl += Lt.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            model.update_target(cfg["ema"])
            tot += loss.item()
            if view_weights is not None:
                weights_np = view_weights.detach().float().cpu().numpy()
                view_weight_samples.append(weights_np)
                view_alignment_samples.append(view_alignment.float().cpu().numpy())
                if ep == args.ssl_epochs:
                    view_sample_ids.append(batch_ids.copy())
                view_weight_sum += weights_np.mean(0)
                view_weight_batches += 1
                gate_losses.append(float(gate_loss.detach().float().cpu()))
                if view_diag is not None:
                    mha_scales.append(float(view_diag["mha_scale"].float().cpu()))
                    mha_delta_ratios.append(view_diag["mha_delta_ratio"].float().cpu().numpy())
                    mha_attention_means.append(view_diag["mha_weights"].float().mean(0).cpu().numpy())
            # 붕괴 감시: 정규화 벡터의 차원별 표준편차 평균 (건강 ≈ 1/sqrt(512) ≈ 0.044, 붕괴 → 0)
            zstd += F.normalize(z, dim=1).std(0).mean().item()
        mean_view_weights = None
        if view_weight_batches:
            weights_all = np.concatenate(view_weight_samples, axis=0)
            alignment_all = np.concatenate(view_alignment_samples, axis=0)
            mean_view_weights = weights_all.mean(0).round(6).tolist()
            entropy = -(weights_all * np.log(np.clip(weights_all, 1e-12, 1.0))).sum(1) / math.log(n_masks)
            max_weights = weights_all.max(1)
            correlation = 0.0
            if weights_all.std() > 1e-12 and alignment_all.std() > 1e-12:
                correlation = float(np.corrcoef(weights_all.reshape(-1), alignment_all.reshape(-1))[0, 1])
            row = dict(
                epoch=ep,
                mean_weights=mean_view_weights,
                std_weights=weights_all.std(0).round(6).tolist(),
                max_weight_quantiles=np.quantile(max_weights, [0.5, 0.9, 0.99]).round(6).tolist(),
                normalized_entropy_quantiles=np.quantile(entropy, [0.1, 0.5, 0.9]).round(6).tolist(),
                max_weight_histogram=np.histogram(max_weights, bins=[0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.000001])[0].tolist(),
                mean_teacher_similarity=alignment_all.mean(0).round(6).tolist(),
                weight_similarity_correlation=correlation,
                gate_kl=float(np.mean(gate_losses)),
            )
            if mha_scales:
                delta_all = np.concatenate(mha_delta_ratios, axis=0).reshape(-1)
                row.update(
                    mha_residual_scale_mean=float(np.mean(mha_scales)),
                    mha_residual_scale_final=float(mha_scales[-1]),
                    mha_context_delta_ratio_quantiles=np.quantile(delta_all, [0.5, 0.9, 0.99]).round(6).tolist(),
                    mean_mha_attention=np.stack(mha_attention_means).mean(0).round(6).tolist(),
                )
            view_weight_history[str(ep)] = row
            with open(view_weight_path, "w", encoding="utf-8") as f:
                json.dump({"epochs": [
                    view_weight_history[k]
                    for k in sorted(view_weight_history, key=int)
                ]}, f, indent=2)
            if ep == args.ssl_epochs:
                np.savez_compressed(
                    os.path.splitext(out_path)[0] + "_view_samples.npz",
                    indices=np.concatenate(view_sample_ids),
                    weights=weights_all.astype(np.float32),
                    teacher_similarity=alignment_all.astype(np.float32),
                )
        print(f"ssl epoch={ep} loss={tot / steps:.4f} z_std={zstd / steps:.4f} sec={time.time() - t:.1f}"
              + (f" view_weights={mean_view_weights} max_w_p50/p90/p99={row['max_weight_quantiles']} "
                 f"entropy_p10/p50/p90={row['normalized_entropy_quantiles']} "
                 f"gate_kl={row['gate_kl']:.5f}" if view_weight_batches else "")
              + (f" mask={stripes or 'checker'}" if curriculum else "") + (f" Lt={tl / steps:.4f}" if triplet_w else ""), flush=True)
        torch.save(dict(ep=ep, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        lsampler=lsampler if triplet_w else None), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    torch.save(model.encoder.state_dict(), out_path)
    if model.view_agg is not None or model.view_gate_mode is not None:
        torch.save({
            "predictor": model.predictor.state_dict(),
            "view_agg": model.view_agg.state_dict() if model.view_agg is not None else None,
            "view_gate_mode": model.view_gate_mode,
        }, os.path.splitext(out_path)[0] + "_view_module.pt")
    os.remove(ckpt_path)
