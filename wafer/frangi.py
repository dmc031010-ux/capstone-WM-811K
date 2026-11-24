"""y9.2: 2D Frangi vesselness (Frangi et al. 1998, 식 14·15) 를 die 격자 위 불량 지시 맵에 적용.

입력: 원본 die 격자 맵 {0 웨이퍼 밖, 1 정상, 2 불량} → 불량 지시 이진 맵 B (불량 1, 나머지 0).
σ 마다 Gaussian 2차 미분 (경계 constant 0) × σ² (scale 정규화) 로 Hessian, 고유값 |λ1| ≤ |λ2|.
  V_σ = 0 (λ2 > 0, 어두운 구조), 아니면 exp(−R_B² / 2β²) · (1 − exp(−S² / 2c²)),  R_B = |λ1|/|λ2|, S = √(λ1² + λ2²)
  V = max_σ V_σ. c 는 이미지마다 바꾸지 않는 전역 고정값 (C_FIXED, 합성 1~3 die 직선의 중심선 S 최댓값의 절반).
출력은 이미 [0,1] (두 항의 곱) → 추가 정규화 없음. 웨이퍼 밖은 0.
"""
import os
import numpy as np
from scipy import ndimage

SIGMAS = (1.0, 2.0, 3.0)   # die 단위
BETA = 0.5
C_FIXED = None             # calibrate() 로 정함 (아래)


def _eig(B, s):
    hrr = ndimage.gaussian_filter(B, s, order=(2, 0), mode="constant", cval=0.0) * s * s
    hcc = ndimage.gaussian_filter(B, s, order=(0, 2), mode="constant", cval=0.0) * s * s
    hrc = ndimage.gaussian_filter(B, s, order=(1, 1), mode="constant", cval=0.0) * s * s
    tmp = np.sqrt((hrr - hcc) ** 2 + 4 * hrc ** 2)
    a, b = (hrr + hcc + tmp) / 2, (hrr + hcc - tmp) / 2
    swap = np.abs(a) > np.abs(b)
    l1, l2 = np.where(swap, b, a), np.where(swap, a, b)   # |l1| <= |l2|
    return l1, l2


def vesselness(nat, sigmas=SIGMAS, beta=BETA, c=None, per_scale=False):
    """nat: (H, W) 원본 die 격자 {0,1,2}. 반환 V (H, W) float32 in [0,1] (per_scale=True 면 (len(sigmas), H, W))."""
    c = C_FIXED if c is None else c
    B = (nat == 2).astype(np.float64)
    out = []
    for s in sigmas:
        l1, l2 = _eig(B, s)
        Rb2 = (l1 / np.where(l2 == 0, 1, l2)) ** 2
        S2 = l1 ** 2 + l2 ** 2
        v = np.exp(-Rb2 / (2 * beta ** 2)) * (1 - np.exp(-S2 / (2 * c ** 2)))
        v[(l2 >= 0)] = 0.0                                  # 밝은 구조만 (불량 = 밝음 → λ2 < 0)
        out.append(v)
    out = np.stack(out)
    out[:, nat == 0] = 0.0                                  # 웨이퍼 밖
    return (out if per_scale else out.max(0)).astype(np.float32)


def line_map(h, w, thick, angle_deg, length=0.6, center=(0.5, 0.5)):
    """합성: h×w 원형 웨이퍼 (모두 정상) 에 thick die 두께 직선. 선 = 중심선에서 수직 거리 < thick/2 인 die."""
    r, c = np.mgrid[0:h, 0:w] + 0.5
    yy, xx = r / h - 0.5, c / w - 0.5
    nat = np.where(xx ** 2 + yy ** 2 < 0.25, 1, 0).astype(np.uint8)
    th = np.radians(angle_deg); u = np.array([np.cos(th), np.sin(th)])
    pr, pc = r - center[0] * h - 0.25, c - center[1] * w - 0.25   # 0.25 die 어긋나게: 0°·90° 에서 die 중심이 경계에 걸리지 않도록
    along = pc * u[0] + pr * u[1]; perp = -pc * u[1] + pr * u[0]
    L = length * min(h, w) / 2
    on = (np.abs(perp) < thick / 2) & (np.abs(along) <= L) & (nat > 0)
    nat[on] = 2
    return nat, (np.abs(perp) < 0.5) & (np.abs(along) <= L * 0.8) & (nat > 0)   # 중심선 (끝 20% 제외)


def calibrate(h=60, w=60):
    """C_FIXED = 0.5 × max_{두께 1~3, σ, 각도 0/30/45} 중심선 S 중앙값."""
    best = 0.0
    for t in (1, 2, 3):
        for ang in (0, 30, 45):
            nat, cl = line_map(h, w, t, ang)
            B = (nat == 2).astype(np.float64)
            for s in SIGMAS:
                l1, l2 = _eig(B, s)
                best = max(best, float(np.median(np.sqrt(l1 ** 2 + l2 ** 2)[cl])))
    return 0.5 * best


C_FIXED = calibrate()


def cache_paths(cache, kind="frangi"):
    """kind: "frangi" (y9.2, V) / "length" (y9.3a, V·w) / "scaled" (y9.3b, k·V)."""
    if kind == "frangi":
        d = os.path.join(cache, "y9.2"); return os.path.join(d, "frangi_u8.npy"), os.path.join(d, "frangi_idx.npy")
    d = os.path.join(cache, "y9.3"); return os.path.join(d, f"{kind}_u8.npy"), os.path.join(d, f"{kind}_idx.npy")


def build_labeled_cache(cache, pkl, size=96, extra=()):
    """라벨 있는 웨이퍼 전부 + extra (fine-tune 의 군집 진단이 읽는 SSL 표본) 의 Frangi 채널 → (n, 96, 96) uint8 (V × 255, 반올림) + 전역 인덱스. 1회만.
    원본 격자 ≤ 96 은 캐시에서 원본 격자를 정확히 복원, > 96 은 pkl 원본에서 계산. 둘 다 캐시와 같은 nearest 로 96."""
    import pickle, sys as _sys, time
    from .data import WM811K, _resize_nearest
    out_u8, out_idx = cache_paths(cache)
    os.makedirs(os.path.dirname(out_u8), exist_ok=True)
    ds = WM811K(cache); sd = np.load(os.path.join(cache, "wm811k_shapes_dies.npy"))
    idx = np.union1d(np.nonzero(ds.labels >= 0)[0], np.asarray(extra, np.int64))
    big = idx[sd[idx, :2].max(1) > size]
    t = time.time()
    nat_big = {}
    if len(big):
        import pandas.core.indexes
        _sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
        with open(pkl, "rb") as f:
            maps = pickle.load(f, encoding="latin1")["waferMap"].values
        for i in big:
            nat_big[int(i)] = np.minimum(np.asarray(maps[i]).astype(np.uint8), 2)
            assert nat_big[int(i)].shape == tuple(sd[i, :2])
        del maps
        print(f"[frangi] pkl 에서 원본 격자 > {size} 인 {len(big)}장 읽음 ({time.time() - t:.0f}s)", flush=True)
    U = np.zeros((len(idx), size, size), np.uint8)
    for k0 in range(0, len(idx), 4096):
        ch = idx[k0:k0 + 4096]
        img = ds.fetch(ch, "cpu")[:, 0].numpy().astype(np.uint8)
        for j, i in enumerate(ch):
            h, w = sd[i, :2]
            if int(i) in nat_big:
                nat = nat_big[int(i)]
            else:
                ri = ((np.arange(size) + 0.5) * h / size).astype(np.int64); ci = ((np.arange(size) + 0.5) * w / size).astype(np.int64)
                nat = img[j][np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
                assert nat.shape == (h, w)
            v = vesselness(nat)
            assert np.isfinite(v).all()
            U[k0 + j] = np.rint(_resize_nearest(v[None], size)[0] * 255).astype(np.uint8)
        if k0 % 40960 == 0:
            print(f"[frangi] {k0 + len(ch)}/{len(idx)} ({time.time() - t:.0f}s)", flush=True)
    np.save(out_u8, U); np.save(out_idx, idx)
    print(f"[frangi] 저장 {out_u8} {U.nbytes / 1e9:.2f}GB ({time.time() - t:.0f}s)")


# ---------------- y9.3: 선 길이 (die) 가중치 ----------------
LMAX = 30.0   # 포화점 (die)


def line_weight(nat, mode="defect", V=None, tau=None, reach=2.0, lmax=LMAX, return_parts=False):
    """원본 die 격자 nat 의 die 별 길이 가중치 w ∈ [0,1] (y9.3, debate/y9.3/03).
    성분: mode="defect" 불량 die 8-연결, mode="ridge" Frangi 능선 {V ≥ tau} (웨이퍼 안) 8-연결.
    성분 중 픽셀 ≥ 3 이고 길쭉함 (주/부 표준편차 비) ≥ 3 인 것만 g = min(L, lmax)/lmax, L = 주축 투영 범위 + 1 (die). 나머지 성분 g = 0.
    각 die 의 w = 거리 ≤ reach (die) 안의 가장 가까운 성분 픽셀이 속한 성분의 g, 아니면 0. 웨이퍼 밖 0."""
    S = (nat == 2) if mode == "defect" else ((V >= tau) & (nat > 0))
    lab, n = ndimage.label(S, structure=np.ones((3, 3)))
    w = np.zeros(nat.shape, np.float32)
    if n == 0:
        return (w, lab, np.zeros(1)) if return_parts else w
    r, c = np.nonzero(lab); k = lab[r, c]; r = r.astype(np.float64); c = c.astype(np.float64)
    cnt = np.bincount(k, minlength=n + 1).astype(np.float64); cs = np.maximum(cnt, 1)
    mr, mc = np.bincount(k, r, n + 1) / cs, np.bincount(k, c, n + 1) / cs
    dr, dc = r - mr[k], c - mc[k]
    a = np.bincount(k, dr * dr, n + 1) / cs; d = np.bincount(k, dc * dc, n + 1) / cs; b = np.bincount(k, dr * dc, n + 1) / cs
    t = np.sqrt(((a - d) / 2) ** 2 + b ** 2); lmax_ev, lmin_ev = (a + d) / 2 + t, (a + d) / 2 - t
    elong = np.sqrt(lmax_ev / np.maximum(lmin_ev, 1e-6))
    vr, vc = np.where(np.abs(b) > 1e-12, b, np.where(a >= d, 1.0, 0.0)), np.where(np.abs(b) > 1e-12, lmax_ev - a, np.where(a >= d, 0.0, 1.0))
    nv = np.hypot(vr, vc); vr, vc = vr / np.maximum(nv, 1e-12), vc / np.maximum(nv, 1e-12)
    proj = dr * vr[k] + dc * vc[k]
    pmax = np.full(n + 1, -np.inf); pmin = np.full(n + 1, np.inf)
    np.maximum.at(pmax, k, proj); np.minimum.at(pmin, k, proj)
    L = np.where(cnt > 0, pmax - pmin + 1, 0.0)
    g = np.where((cnt >= 3) & (elong >= 3), np.minimum(L, lmax) / lmax, 0.0); g[0] = 0.0
    dist, (ri, ci) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    w = g[lab[ri, ci]].astype(np.float32); w[dist > reach] = 0.0; w[nat == 0] = 0.0
    return (w, lab, np.where(cnt > 0, L, 0.0) * ((cnt >= 3) & (elong >= 3))) if return_parts else w


TAU = 0.2   # y9.3 0단계에서 고른 능선 임계 (runs/y9.3_step0/verify_length.txt)


def build_y93_cache(cache, pkl, size=96):
    """y9.3: y9.2 캐시와 같은 웨이퍼 집합에 대해 length (V·w, 능선 τ=TAU) 와 scaled (k·V) 두 표를 만든다.
    k = 5% 라벨 학습 세트 (split_seed 0) 의 불량 die 위 mean(V·w) / mean(V) (원본 격자). 원본 격자 얻는 방법은 build_labeled_cache 와 같음."""
    import pickle, sys as _sys, time
    from .data import WM811K, _resize_nearest
    ds = WM811K(cache); sd = np.load(os.path.join(cache, "wm811k_shapes_dies.npy"))
    idx = np.load(cache_paths(cache)[1])
    big = idx[sd[idx, :2].max(1) > size]; t = time.time(); nat_big = {}
    if len(big):
        import pandas.core.indexes
        _sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
        with open(pkl, "rb") as f:
            maps = pickle.load(f, encoding="latin1")["waferMap"].values
        for i in big:
            nat_big[int(i)] = np.minimum(np.asarray(maps[i]).astype(np.uint8), 2)
            assert nat_big[int(i)].shape == tuple(sd[i, :2])
        del maps

    def natives(ch):
        img = ds.fetch(ch, "cpu")[:, 0].numpy().astype(np.uint8)
        for j, i in enumerate(ch):
            h, w = sd[i, :2]
            if int(i) in nat_big:
                yield nat_big[int(i)]; continue
            ri = ((np.arange(size) + 0.5) * h / size).astype(np.int64); ci = ((np.arange(size) + 0.5) * w / size).astype(np.int64)
            nat = img[j][np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
            assert nat.shape == (h, w); yield nat

    lab = ds.labeled_subset(0.05, seed=0); s_vw = s_v = 0.0
    for k0 in range(0, len(lab), 4096):
        for nat in natives(lab[k0:k0 + 4096]):
            v = vesselness(nat); w = line_weight(nat, "ridge", V=v, tau=TAU); b = nat == 2
            s_vw += float((v * w)[b].sum()); s_v += float(v[b].sum())
    k = s_vw / s_v
    print(f"[y9.3] k = mean(V·w)/mean(V) (5% 학습, 불량 die 위) = {k:.4f}", flush=True)
    UL = np.zeros((len(idx), size, size), np.uint8); US = np.zeros((len(idx), size, size), np.uint8)
    for k0 in range(0, len(idx), 4096):
        ch = idx[k0:k0 + 4096]
        for j, nat in enumerate(natives(ch)):
            v = vesselness(nat); w = line_weight(nat, "ridge", V=v, tau=TAU)
            assert np.isfinite(v).all()
            UL[k0 + j] = np.rint(_resize_nearest((v * w)[None], size)[0] * 255).astype(np.uint8)
            US[k0 + j] = np.rint(_resize_nearest((k * v)[None], size)[0] * 255).astype(np.uint8)
        if k0 % 40960 == 0:
            print(f"[y9.3] {k0 + len(ch)}/{len(idx)} ({time.time() - t:.0f}s)", flush=True)
    for kind, U in (("length", UL), ("scaled", US)):
        pu, pi = cache_paths(cache, kind); os.makedirs(os.path.dirname(pu), exist_ok=True)
        np.save(pu, U); np.save(pi, idx)
    with open(os.path.join(os.path.dirname(cache_paths(cache, "length")[0]), "k.txt"), "w") as f:
        f.write(f"{k:.6f}\n")
    print(f"[y9.3] 저장 ({time.time() - t:.0f}s)")
