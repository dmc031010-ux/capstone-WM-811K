"""ver3: 자기지도(SimCLR) → 군집화 → labeled 로 군집 이름 붙이기 → 미세조정 + 재군집 반복.

발전 옵션: --v3_weight purity (군집 순도 가중치 + 순도 20% 미만 군집 제외), --v3_nopl (군집 pseudo-label 끔, ablation),
          --v3_multicrop (자기지도에 국소 crop view 추가), --v3_clusters K (군집 수)

- 자기지도 데이터: unlabeled 중 불량 비율 >= r0 (라벨 미사용) 에서 무작위 n_sub 장. labeled 이미지는 미포함.
- 군집 이름: 5% labeled 를 가장 가까운 군집에 배정, 클래스 빈도로 정규화한 비율이 가장 큰 클래스.
  (labeled 가 min_lab 장 이상 & 정규화 비율 >= min_conf 인 군집만 이름 붙임, 나머지는 사용 안 함)
- 미세조정: labeled(클래스 균형, weak) CE + lambda_c * 군집 pseudo-label(pseudo-label 기준 클래스 균형, strong) CE.
  라운드마다 현재 모델 특징으로 다시 군집화/이름 붙이기 ("이리저리 옮기기").
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

from . import augment as A
from .data import CLASSES, NUM_CLASSES
from .engine import BalancedSampler, _Tracker, set_seed
from . import models as wmodels
from .models import ClassPool, FiLMFeatures, FiLMResNet, SizeHead, build_model, load_size, to_input

V3 = dict(
    r0=0.02, n_sub=100_000,             # 자기지도 데이터: 불량 비율 >= 2% 인 unlabeled 중 10만 장 (사용자 결정)
    ssl_lr=1e-3, ssl_temp=0.2,          # SimCLR (AdamW + cosine)
    min_lab=3, min_conf=0.6,            # 군집 수는 --v3_clusters (기본 100)
    min_purity=0.2,                     # --v3_weight purity: 순도 20% 미만 군집은 이름 안 붙임
    ft_rounds=3, ft_lr=1e-3, lambda_c=0.5,
)


def fail_ratio(ds):
    """웨이퍼별 불량 die 비율 (96x96 리사이즈 기준)."""
    vals = np.array([[(b >> s) & 3 for s in (0, 2, 4, 6)] for b in range(256)])
    nf, nd = (vals == 2).sum(1), (vals > 0).sum(1)
    out = np.empty(len(ds.labels))
    for i in range(0, len(out), 50000):
        blk = ds.packed[i:i + 50000].reshape(-1, ds.packed.shape[1] * ds.packed.shape[2])
        out[i:i + 50000] = nf[blk].sum(1) / np.maximum(nd[blk].sum(1), 1)
    return out


def select_subset(ds, r0, n_sub, seed):
    fr = fail_ratio(ds)[ds.unlabeled]
    cand = ds.unlabeled[fr >= r0]
    return np.sort(np.random.default_rng(seed).choice(cand, size=min(n_sub, len(cand)), replace=False)), len(cand)


def _ops(args):
    return A.STRONG_OPS_NO_CUTOUT if args.no_cutout else A.STRONG_OPS


def _view(x, ops):
    return A.strong(A.rotate(x), None, 0.0, keep=False, ops=ops)


def _encoder(arch):
    m = build_model(arch, NUM_CLASSES)
    dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, dim


def pretrain(ds, sub, args, device, out_path):
    """SimCLR (NT-Xent). 인코더(state_dict, fc 제외) 저장."""
    set_seed(args.seed)
    enc, dim = _encoder(args.arch)
    proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 128))
    model = nn.Sequential(enc, proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=V3["ssl_lr"], weight_decay=1e-4)
    B = args.batch_size
    steps = len(sub) // B
    total = steps * args.ssl_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / total)))
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(args.seed)
    labels = torch.arange(B, device=device)
    for ep in range(1, args.ssl_epochs + 1):
        model.train()
        t, tot = time.time(), 0.0
        perm = rng.permutation(sub)
        for s in range(steps):
            with torch.no_grad():
                x = ds.fetch(np.sort(perm[s * B:(s + 1) * B]), device)
                v = to_input(torch.cat([_view(x, _ops(args)), _view(x, _ops(args))]))
            with torch.autocast("cuda", dtype=torch.float16):
                z = F.normalize(model(v).float(), dim=1)
            sim = z @ z.T / V3["ssl_temp"]
            sim.fill_diagonal_(-1e4)
            target = torch.cat([labels + B, labels])
            loss = F.cross_entropy(sim, target)
            if args.v3_multicrop:
                # 발전방향 4: 국소 패턴(Scratch/Loc 등)용 작은 crop 2장(48x48)이 같은 웨이퍼의 전체 view 와 가까워지도록
                with torch.no_grad():
                    xl = to_input(torch.cat([A.local_crop(A.rotate(x), 48), A.local_crop(A.rotate(x), 48)]))
                with torch.autocast("cuda", dtype=torch.float16):
                    zl = F.normalize(model(xl).float(), dim=1)
                loss = loss + F.cross_entropy(zl @ z[:B].T / V3["ssl_temp"], torch.cat([labels, labels]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"ssl epoch={ep} loss={tot / steps:.4f} sec={time.time() - t:.1f}", flush=True)
    torch.save(enc.state_dict(), out_path)


@torch.no_grad()
def embed(enc, ds, idx, device, bs=1024):
    enc.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            x = to_input(ds.fetch(idx[i:i + bs], device))
            f = enc(x, torch.as_tensor(idx[i:i + bs], device=device)) if getattr(enc, "needs_idx", False) else enc(x)
            out.append(F.normalize(f.float(), dim=1))
    return torch.cat(out)


def kmeans(x, k, iters=30, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k)
        c = torch.where(n[:, None] > 0, F.normalize(s, dim=1), c)
    return c


def name_clusters(c, f_lab, y_lab, min_lab, min_conf, min_purity=0.0):
    """군집 → 클래스. 반환: (k,) 클래스 또는 -1, (k,) 순도.

    순도 = 군집에 들어간 labeled 중 붙은 이름의 클래스 비율 (클래스 빈도 보정 전 원래 개수, 작은 군집은 +1 로 축소).
    min_purity > 0 이면 순도가 그보다 낮은 군집은 이름을 붙이지 않음 (예: 'none' 이 대부분인데 Scratch 로 붙는 군집).
    """
    a = (f_lab @ c.T).argmax(1).cpu().numpy()
    cnt = np.zeros((len(c), NUM_CLASSES))
    np.add.at(cnt, (a, y_lab), 1)
    r = cnt / np.maximum(np.bincount(y_lab, minlength=NUM_CLASSES), 1)        # 클래스 빈도 정규화
    lab = r.argmax(1)
    conf = r.max(1) / np.maximum(r.sum(1), 1e-12)
    purity = cnt[np.arange(len(c)), lab] / (cnt.sum(1) + 1)
    lab[(cnt.sum(1) < min_lab) | (conf < min_conf) | (purity < min_purity)] = -1
    return lab, purity


def cluster_report(c, f, y):
    """labeled 분석용 (학습/선택에 사용 안 함): NMI, 순도, 클래스별 '자기 클래스가 다수인 군집에 들어간 비율'."""
    a = (f @ c.T).argmax(1).cpu().numpy()
    cnt = np.zeros((len(c), NUM_CLASSES)); np.add.at(cnt, (a, y), 1)
    maj = cnt.argmax(1)
    per = {CLASSES[k]: float((maj[a[y == k]] == k).mean()) if (y == k).any() else None for k in range(NUM_CLASSES)}
    return dict(nmi=float(normalized_mutual_info_score(y, a)), purity=float(cnt.max(1).sum() / cnt.sum()),
                frac_in_own_majority_cluster=per)


def batch_hard_triplet(f, y, margin=0.3):
    """ver7.4: 정규화 특징에서 앵커마다 가장 먼 같은 클래스 vs 가장 가까운 다른 클래스."""
    f = F.normalize(f.float(), dim=1)
    d = torch.cdist(f, f)
    same = y[:, None] == y[None]
    pos = (d * same).max(1).values
    neg = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(pos - neg + margin).mean()


def focal_loss(logits, y, gamma=2.0):
    """ver7.5: CE * (1 - p_t)^gamma — 이미 잘 맞추는 샘플의 손실을 줄이고 헷갈리는 샘플에 집중."""
    ce = F.cross_entropy(logits, y, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


def finetune(ds, lab_idx, sub, enc_path, args, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    model = build_model(args.arch, NUM_CLASSES)
    if args.film:   # ver8.2: 사전학습 때와 같은 FiLM ResNet (인코더 + FiLM Generator 를 함께 불러옴)
        assert args.v3_nopl and not args.triplet and not args.size_head
        model = FiLMResNet(model, load_size(args.cache))
        assert len(model.size) == len(ds.labels)
    model = model.to(device)
    missing = model.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True), strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    actual_missing = set(missing.missing_keys)
    sa_mode = getattr(wmodels, "USE_SA", False)
    if sa_mode:
        if isinstance(sa_mode, tuple) and len(sa_mode) == 4 and sa_mode[0] == "combo":
            _, l_m, b_m, p_m = sa_mode
            prefixes = []
            if l_m is not None:
                prefixes.extend(["line_branch.", "line_scale"])
            if b_m is not None:
                prefixes.append("layer3_sa.")
            if p_m is not None:
                prefixes.append("avgpool.")
            sa_prefixes = tuple(prefixes)
        else:
            sa_prefixes = {
                "layer3": ("layer3.sa.",),
                "layer2": ("layer2.sa.",),
                "ver9.7": ("layer3.sa.", "avgpool."),
                "ver9.7_1step": ("layer3.sa.", "avgpool."),
                "ver9.7_belief": ("layer3.sa.",),
                "ver9.7_pool": ("avgpool.",),
                "ver9.8": ("layer2.sa.", "layer3.sa.", "avgpool."),
                "ver9.9": ("line_branch.", "line_scale", "layer3_sa.", "avgpool."),
                "ver9.9_pyr_max": ("line_branch.", "line_scale", "layer3_sa.", "avgpool."),
                "ver9.9_pyr_avg": ("line_branch.", "line_scale", "layer3_sa.", "avgpool."),
                "ver9.9_learn": ("line_branch.", "line_scale", "layer3_sa.", "avgpool."),
                "ver9.9_scale": ("line_branch.", "line_scale", "layer3_sa.", "avgpool."),
                "ver9.9A": ("line_branch.", "line_scale"),
                "ver9.9A_pyr_max": ("line_branch.", "line_scale"),
                "ver9.9A_pyr_avg": ("line_branch.", "line_scale"),
                "ver9.9A_learn": ("line_branch.", "line_scale"),
                "ver9.9A_scale": ("line_branch.", "line_scale"),
                "ver9.9B": ("line_branch.", "line_scale", "layer3_sa."),
                "ver9.9C": ("line_branch.", "line_scale", "avgpool."),
            }.get(sa_mode, ("avgpool.0.",))
        assert actual_missing - expected_missing == {k for k in actual_missing if k.startswith(sa_prefixes)}, missing
    else:
        assert actual_missing == expected_missing, missing
    if args.size_head:  # ver8.1: 분류기 직전에 크기 임베딩 (die 수 → log → 표준화)
        assert args.v3_nopl and not args.triplet
        d = np.log(np.load(os.path.join(args.cache, "wm811k_shapes_dies.npy"))[:, 2].astype(np.float64))
        assert len(d) == len(ds.labels)
        model = SizeHead(model, torch.tensor((d - d.mean()) / d.std(), dtype=torch.float32)).to(device)
    if args.class_pool:  # ver8.3: 평균 풀링 → 클래스별 학습 풀링 (r 은 lr x10, weight decay 없음 — 90 epoch 안에 평균↔최대 사이를 움직일 수 있게)
        assert args.v3_nopl and not args.triplet and not args.size_head and not args.film
        model = ClassPool(model).to(device)
    y_lab = ds.labels[lab_idx]
    B = args.batch_size
    steps = math.ceil(len(lab_idx) / B)
    epochs = V3["ft_rounds"] * args.ft_epochs
    attn_wd = getattr(args, "attn_wd", None)
    if args.class_pool:
        opt = torch.optim.AdamW([{"params": model.base.parameters()},
                                 {"params": [model.r], "lr": V3["ft_lr"] * 10, "weight_decay": 0.0}],
                                lr=V3["ft_lr"], weight_decay=args.wd)
    elif attn_wd is not None:
        bb_decay, bb_no_decay, sa_decay, sa_no_decay = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_new = name.startswith(("layer3.sa.", "layer2.sa.", "layer3_sa.", "line_branch.", "line_scale", "avgpool.", "fc."))
            no_decay = (param.ndim < 2) or name.endswith(".bias") or ("rel_pos_bias" in name) or ("pos_emb" in name)
            if is_new:
                (sa_no_decay if no_decay else sa_decay).append(param)
            else:
                (bb_no_decay if no_decay else bb_decay).append(param)
        opt = torch.optim.AdamW([
            {"params": bb_decay, "weight_decay": args.wd},
            {"params": bb_no_decay, "weight_decay": 0.0},
            {"params": sa_decay, "weight_decay": attn_wd},
            {"params": sa_no_decay, "weight_decay": 0.0},
        ], lr=V3["ft_lr"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=V3["ft_lr"], weight_decay=args.wd)
    # 미세조정 lr: warmup + cosine (ver9.7), --cosine 이면 lr*cos(7*pi*k/16K), 아니면 고정
    total_steps = steps * epochs
    warmup_steps = getattr(args, "warmup_epochs", 0) * steps
    if warmup_steps > 0 and args.cosine:
        def _lr_lambda(k):
            if k < warmup_steps:
                return 0.1 + 0.9 * (k / max(1, warmup_steps))
            progress = (k - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, (lambda k: math.cos(7 * math.pi * k / (16 * total_steps))) if args.cosine else (lambda k: 1.0))
    scaler = torch.amp.GradScaler()
    lsampler = BalancedSampler(y_lab, args.seed, args.sample_pow)
    # ver7.8 / 7.81: none 클래스 손실 가중치만 낮춤, ver7.82 / 7.83: + Scratch 가중치 올림 (나머지 1.0), 모두 1.0 이면 일반 CE
    cw = torch.ones(NUM_CLASSES, device=device)
    cw[CLASSES.index("none")] = args.none_w
    cw[CLASSES.index("Scratch")] = args.scratch_w
    tracker = _Tracker(out_dir, model, ds, device, args, dict(n_labeled=len(lab_idx), n_sub=len(sub), stage="ver3", v3=V3, rounds=[]))

    class _FeatWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.forward_features(x)

    feat_model = (model.base if args.size_head else FiLMFeatures(model) if args.film
                  else nn.Sequential(*list(model.base.children())[:-1], nn.Flatten()) if args.class_pool
                  else _FeatWrap(model) if hasattr(model, "forward_features")
                  else nn.Sequential(*list(model.children())[:-1], nn.Flatten()))   # fc 앞 특징 (가중치 공유)
    ep = 0
    for r in range(V3["ft_rounds"]):
        # (재)군집화 + 이름 붙이기: 라운드 0 은 자기지도 특징, 이후는 미세조정된 특징
        f_sub = embed(feat_model, ds, sub, device)
        f_lab = embed(feat_model, ds, lab_idx, device)
        c = kmeans(f_sub, args.v3_clusters, seed=args.seed + r)
        purity_w = args.v3_weight == "purity"
        names, purity = name_clusters(c, f_lab, y_lab, V3["min_lab"], V3["min_conf"],
                                      V3["min_purity"] if purity_w else 0.0)
        a = (f_sub @ c.T).argmax(1).cpu().numpy()
        pl = names[a]
        keep = pl >= 0
        p_idx, p_lab = sub[keep], pl[keep]
        p_w = purity[a][keep] if purity_w else np.ones(int(keep.sum()))     # 군집 순도 가중치 (발전방향 3)
        rep = cluster_report(c, embed(feat_model, ds, ds.train, device), ds.labels[ds.train])
        rep.update(round=r, n_named_clusters=int((names >= 0).sum()), n_pseudo=int(keep.sum()),
                   pseudo_class_counts={CLASSES[k]: int((p_lab == k).sum()) for k in range(NUM_CLASSES)},
                   pseudo_mean_weight={CLASSES[k]: round(float(p_w[p_lab == k].mean()), 3) if (p_lab == k).any() else None
                                       for k in range(NUM_CLASSES)})
        tracker.extra["rounds"].append(rep)
        print(f"[ver3 round {r}] " + str({k: v for k, v in rep.items() if k != 'frac_in_own_majority_cluster'}), flush=True)
        print(f"   frac_in_own_majority_cluster: {rep['frac_in_own_majority_cluster']}", flush=True)
        del f_sub
        # --v3_nopl: 군집 pseudo-label 을 쓰지 않음 (ablation = 자기지도 사전학습 + labeled 미세조정, WaPIRL 방식)
        psampler = BalancedSampler(p_lab, args.seed + r) if len(p_lab) and not args.v3_nopl else None
        for _ in range(args.ft_epochs):
            ep += 1
            model.train()
            aux_w = getattr(args, "aux_belief_w", 0.0)
            ls = getattr(args, "label_smoothing", 0.0)
            grad_clip = getattr(args, "grad_clip", 0.0)
            t, acc = time.time(), dict(
                Ls=0.0, Lc=0.0,
                **({"Lt": 0.0} if args.triplet else {}),
                **({"La": 0.0} if aux_w > 0.0 else {}),
            )
            for _ in range(steps):
                li = lab_idx[lsampler(B)]
                yl = torch.from_numpy(ds.labels[li]).to(device)
                with torch.no_grad():
                    xl = A.weak(ds.fetch(li, device), "rotate")
                    if getattr(args, "use_hflip", False):
                        xl = A.hflip(xl)
                    if psampler:
                        j = psampler(args.mu * B // 2)
                        xp = A.strong(ds.fetch(p_idx[j], device), None, 0.0, keep=False, ops=_ops(args))
                        yp = torch.from_numpy(p_lab[j]).to(device).long()
                        wp = torch.from_numpy(p_w[j]).to(device).float()
                        xin = torch.cat([xl, xp])
                    else:
                        xin = xl
                with torch.autocast("cuda", dtype=torch.float16):
                    if args.triplet:
                        feat = feat_model(to_input(xin))
                        logits = model.fc(feat).float()
                    elif args.size_head or args.film:
                        logits = model(to_input(xin), torch.as_tensor(li, device=device)).float()
                    else:
                        logits = model(to_input(xin)).float()
                Ls = focal_loss(logits[:B], yl, args.focal) if args.focal else \
                    F.cross_entropy(logits[:B], yl, weight=cw if (cw != 1.0).any() else None, label_smoothing=ls)
                if args.triplet:   # ver7.4: 같은 클래스는 당기고 다른 클래스(None 포함)는 밀기
                    Lt = batch_hard_triplet(feat[:B], yl)
                    acc["Lt"] += Lt.item()
                if psampler:   # 가중 평균 (가중치가 모두 1 이면 일반 평균과 같음)
                    Lc = (F.cross_entropy(logits[B:], yp, reduction="none") * wp).sum() / wp.sum().clamp_min(1e-12)
                else:
                    Lc = torch.zeros((), device=device)
                loss = Ls + V3["lambda_c"] * Lc
                if args.triplet:
                    loss = loss + args.triplet * Lt
                if aux_w > 0.0:
                    sa_mod = getattr(model, "layer3_sa", None) or getattr(model.layer3, "sa", None)
                    step_logits = getattr(sa_mod, "last_step_logits", None)
                    if step_logits:
                        n_st = len(step_logits)
                        w_sum = n_st * (n_st + 1) / 2.0
                        La = sum(((i + 1) / w_sum) * F.cross_entropy(sl[:B], yl, weight=cw if (cw != 1.0).any() else None, label_smoothing=ls)
                                 for i, sl in enumerate(step_logits))
                        loss = loss + aux_w * La
                        acc["La"] += La.item()
                        sa_mod.last_step_logits = None
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                acc["Ls"] += Ls.item(); acc["Lc"] += Lc.item()
            log = {k: v / steps for k, v in acc.items()}
            log["round"] = r
            log["sec"] = round(time.time() - t, 1)
            tracker.step(ep, log)
    return tracker.result
