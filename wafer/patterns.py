"""ver2: 클래스별 패턴 사전 + 패턴 점수 (학습 전 한 번, 고정 모델).

1) supervised 모델의 layer3 국소 특징 (96 입력 → 6x6 칸, 256차원, L2 정규화) 을 labeled 에서 추출
2) 클래스마다 spherical k-means 로 K 개 패턴
3) 패턴 중요도 = (자기 클래스 웨이퍼에서의 증거) - (다른 클래스들 증거의 클래스 평균), 음수는 0, 클래스 내 합 1
   증거 e(x, p) = 웨이퍼 x 의 36칸 중 패턴 p 와 가장 닮은 칸의 코사인 유사도
4) 클래스 점수 S_c(x) = sum_k imp_ck * e(x, p_ck),  q = softmax(S / T)
5) T 는 val NLL 최소, tau_p 는 클래스별로 val 에서 '그 클래스로 예측 & q_max >= tau_p' 의 precision >= 0.9 인
   최소값 (불가능한 클래스와 'none' 은 1.01 = 통로 닫힘)
"""
import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASSES, NUM_CLASSES
from .models import to_input

NONE = CLASSES.index("none")


@torch.no_grad()
def local_features(model, x):
    """resnet 계열 layer3 출력 → (B, 36, C) L2 정규화."""
    m = model
    h = m.maxpool(m.relu(m.bn1(m.conv1(to_input(x)))))
    h = m.layer3(m.layer2(m.layer1(h)))
    return F.normalize(h.flatten(2).transpose(1, 2).float(), dim=-1)


@torch.no_grad()
def _features(model, ds, idx, device, bs=512):
    model.eval()
    return torch.cat([local_features(model, ds.fetch(idx[i:i + bs], device)).half().cpu()
                      for i in range(0, len(idx), bs)])


def _kmeans(x, k, iters=30, seed=0):
    """spherical k-means. x: (N, C) 정규화."""
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            s = x[a == j]
            if len(s):
                c[j] = F.normalize(s.mean(0), dim=0)
    return c


def evidence(feat, P, device):
    """feat: (N, 36, C) half cpu, P: (M, C) → (N, M) 칸별 최대 유사도."""
    out = []
    Pd = P.to(device)
    for i in range(0, len(feat), 4096):
        f = feat[i:i + 4096].to(device).float()
        out.append((f @ Pd.T).amax(1).cpu())
    return torch.cat(out)


def build(model, ds, lab_idx, device, K=8):
    """패턴 사전 + 중요도 + T / tau_p 보정. 반환: dict(P, W, T, tau_p, report)."""
    y = ds.labels[lab_idx]
    feat = _features(model, ds, lab_idx, device)
    P, owner = [], []
    for c in range(NUM_CLASSES):
        xc = feat[torch.from_numpy(y == c)].reshape(-1, feat.shape[-1]).float()
        P.append(_kmeans(xc, min(K, len(xc))))
        owner += [c] * len(P[-1])
    P, owner = torch.cat(P), torch.tensor(owner)

    E = evidence(feat, P, device)                                     # (N, M)
    yt = torch.from_numpy(y)
    cls_mean = torch.stack([E[yt == c].mean(0) for c in range(NUM_CLASSES)])   # (9, M)
    W = torch.zeros(len(P), NUM_CLASSES)
    for c in range(NUM_CLASSES):
        j = owner == c
        others = [o for o in range(NUM_CLASSES) if o != c]
        imp = (cls_mean[c, j] - cls_mean[others][:, j].mean(0)).clamp(min=0)
        W[j, c] = imp / imp.sum() if imp.sum() > 0 else 1.0 / j.sum()

    # val 에서 보정
    fv = _features(model, ds, ds.val, device)
    Sv = evidence(fv, P, device) @ W
    yv = torch.from_numpy(ds.labels[ds.val])
    T = min([0.005, 0.01, 0.02, 0.05, 0.1, 0.2], key=lambda t: F.cross_entropy(Sv / t, yv).item())
    qv = torch.softmax(Sv / T, 1)
    qmax, qarg = qv.max(1)
    # 클래스별 tau_p: 그 클래스로 예측 & q_max >= t 의 val precision >= 0.9 (최소 5개) 인 최소 t, 불가능하면 통로 닫음
    tau_p, rep = np.full(NUM_CLASSES, 1.01), {}
    for c in range(NUM_CLASSES):
        if c == NONE:
            continue
        for t in np.arange(0.50, 1.00, 0.01):
            sel = (qarg == c) & (qmax >= t)
            if sel.sum() >= 5 and (yv[sel] == c).float().mean() >= 0.9:
                tau_p[c] = round(float(t), 2)
                break
    for c in range(NUM_CLASSES):
        s = (qarg == c) & (qmax >= tau_p[c])
        rep[CLASSES[c]] = dict(tau_p=float(tau_p[c]), n_selected=int(s.sum()),
                               precision=float((yv[s] == c).float().mean()) if s.any() else None,
                               recall_of_class=float((s & (yv == c)).sum() / max((yv == c).sum(), 1)))
    report = dict(T=T, val_pattern_top1_acc=float((qarg == yv).float().mean()),
                  val_pattern_macro_recall=float(np.mean([((qarg == c) & (yv == c)).sum().item() / max((yv == c).sum().item(), 1)
                                                          for c in range(NUM_CLASSES)])),
                  per_class_on_route=rep, n_patterns=len(P))
    return dict(P=P, W=W, T=T, tau_p=tau_p, report=report)


def score(model, ds, idx, pat, device, bs=512):
    """idx 웨이퍼들의 패턴 확률 → (q_arg int8, q_max float16) numpy."""
    model.eval()
    args, maxs = [], []
    for i in range(0, len(idx), 8192):
        f = _features(model, ds, idx[i:i + 8192], device, bs)
        q = torch.softmax(evidence(f, pat["P"], device) @ pat["W"] / pat["T"], 1)
        m, a = q.max(1)
        args.append(a.to(torch.int8)); maxs.append(m.half())
    return torch.cat(args).numpy(), torch.cat(maxs).numpy()
