"""y9.2 0단계: Frangi 채널 검증 (학습 없음). 합성 + 실제 val (원본 격자 ≤ 96) 만 사용, test 안 씀.

통과 기준 (debate/y9.2/03_response.md 에 미리 고정):
 (1) 1/2/3 die 직선의 중심선 최종 응답이 서로 2배 이상 차이 나지 않음
 (2) 3×3 덩어리 중심부 응답 < 선 중심선 응답의 절반
 (3) 불량 없는 웨이퍼 최댓값 < 0.05, 실제 웨이퍼 전체에서 NaN/inf 0
 링은 값만 보고.
사용: python runs/y9.2_step0/verify_frangi.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print(f"설정: σ = {Fr.SIGMAS} die, β = {Fr.BETA}, c (전역 고정) = {Fr.C_FIXED:.4f}, 정규화 없음 (V ∈ [0,1])\n")
ok = lambda b: "통과" if b else "**실패**"


def circle(h, w):
    r, c = np.mgrid[0:h, 0:w] + 0.5
    return np.where(((c / w - 0.5) ** 2 + (r / h - 0.5) ** 2) < 0.25, 1, 0).astype(np.uint8), r / h - 0.5, c / w - 0.5


# ---- (1) 직선 두께 ----
print("(1) 직선 중심선 응답 중앙값 [σ=1 / σ=2 / σ=3 → 최종 max]")
fin = {}
for g in (40, 70):
    for ang in (0, 30, 45):
        row = []
        for t in (1, 2, 3):
            nat, cl = Fr.line_map(g, g, t, ang)
            ps = Fr.vesselness(nat, per_scale=True); v = ps.max(0)
            fin[(g, ang, t)] = np.median(v[cl])
            row.append(" / ".join(f"{np.median(p[cl]):.2f}" for p in ps) + f" → {fin[(g, ang, t)]:.2f}")
        print(f"  격자 {g}×{g} 각도 {ang:2d}°  " + "   ".join(f"{t} die: {r}" for t, r in zip((1, 2, 3), row)))
ratio = max(max(fin[(g, a, t)] for t in (1, 2, 3)) / max(min(fin[(g, a, t)] for t in (1, 2, 3)), 1e-9) for g in (40, 70) for a in (0, 30, 45))
print(f"  두께 간 최대/최소 비율의 최댓값 {ratio:.2f} → (1) {ok(ratio < 2)}")

# ---- (2) 덩어리 ----
print("\n(2) 덩어리 중심부 응답 (70×70 격자, 선 비교값 = 같은 격자 1~3 die 선 최종 응답의 최솟값)")
line_ref = min(fin[(70, a, t)] for a in (0, 30, 45) for t in (1, 2, 3))
res2 = {}
for k in (2, 3, 5):
    nat, _, _ = circle(70, 70); nat[35 - k // 2:35 - k // 2 + k, 35 - k // 2:35 - k // 2 + k] = 2
    v = Fr.vesselness(nat); res2[k] = v[35, 35]
    print(f"  {k}×{k} 덩어리 중심 {v[35, 35]:.3f}, 덩어리 안 최댓값 {v[35 - k // 2:35 - k // 2 + k, 35 - k // 2:35 - k // 2 + k].max():.3f}, 주변 포함 최댓값 {v.max():.3f}")
nat, _, _ = circle(70, 70); nat[35, 35] = 2; v1 = Fr.vesselness(nat)[35, 35]
print(f"  고립 1 die 불량 {v1:.3f}  | 선 기준 {line_ref:.3f} → (2) 3×3 < 선/2: {ok(res2[3] < line_ref / 2)}")

# ---- 링 ----
print("\n링 (값만 보고, 70×70)")
for name, r0, r1 in [("Edge-Ring 모양 (반지름 0.44~0.48)", 0.44, 0.48), ("Donut 모양 (반지름 0.20~0.24)", 0.20, 0.24), ("Donut 두꺼움 (0.15~0.28)", 0.15, 0.28)]:
    nat, yy, xx = circle(70, 70); rr = np.sqrt(yy ** 2 + xx ** 2); ring = (rr >= r0) & (rr < r1) & (nat > 0); nat[ring] = 2
    v = Fr.vesselness(nat)
    print(f"  {name}: 링 위 응답 중앙값 {np.median(v[ring]):.3f}, 최댓값 {v.max():.3f}")

# ---- (3) 경계 ----
print("\n(3) 경계")
mx = max(Fr.vesselness(circle(g, g)[0]).max() for g in (20, 40, 70, 96))
nat, _, _ = circle(70, 70); nf = nat.copy(); nf[nat > 0] = 2; vnf = Fr.vesselness(nf)
nat, yy, xx = circle(70, 70); el = (np.sqrt(yy ** 2 + xx ** 2) > 0.40) & (xx > 0.25) & (nat > 0); nat[el] = 2; vel = Fr.vesselness(nat)
print(f"  불량 없는 웨이퍼 최댓값 {mx:.4f} | 전부 불량 (Near-full 모양) 최댓값 {vnf.max():.3f} | 가장자리 덩어리 (Edge-Loc 모양) 최댓값 {vel.max():.3f}")

# ---- 실제 val ----
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]


def native(i):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])], ri, ci


va = ds.val; yv = ds.labels[va]; small = sd[va, :2].max(1) <= 96
rng = np.random.default_rng(0)
print(f"\n실제 val (원본 격자 ≤ 96, 클래스당 최대 300장): 웨이퍼별 [불량 die 위 평균 V | 웨이퍼 안 상위 1% V] 의 중앙값")
bad = 0; EX = {}
for k, cname in enumerate(CLASSES):
    ids = va[(yv == k) & small]; ids = rng.permutation(ids)[:300]
    a, b = [], []
    for i in ids:
        nat, ri, ci = native(i); v = Fr.vesselness(nat)
        bad += int(~np.isfinite(v)).sum() if False else int((~np.isfinite(v)).sum())
        d = nat == 2
        a.append(v[d].mean() if d.any() else 0.0); b.append(np.percentile(v[nat > 0], 99))
        EX.setdefault(cname, []).append((i, nat, v))
    print(f"  {cname:10s} n={len(ids):3d}  불량 위 평균 {np.median(a):.3f}  상위 1% {np.median(b):.3f}")
print(f"  NaN/inf 개수 {bad} | (3) 불량 없는 웨이퍼 < 0.05 & NaN/inf 0: {ok(mx < 0.05 and bad == 0)}")

# ---- C6: 같은 배경·중심선, 1/2/3 die ----
print("\nC6: val none 웨이퍼 (die ≥ 1200, 50장) 에 같은 중심선으로 1/2/3 die 선 → 중심선 응답 / 배경 (선에서 4 die 넘게 떨어진 곳) 상위 1%")
cands = va[(yv == CLASSES.index("none")) & small & (dies[va] >= 1200)]
cl_r = {1: [], 2: [], 3: []}; bg_r = {1: [], 2: [], 3: []}
for i in rng.permutation(cands)[:50]:
    nat0, _, _ = native(i); h, w = nat0.shape; ang = rng.uniform(0, 180); cen = (0.5 + rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.15, 0.15))
    for t in (1, 2, 3):
        ln, cl = Fr.line_map(h, w, t, ang, length=0.5, center=cen)
        nat = nat0.copy(); m = (ln == 2) & (nat0 > 0); nat[m] = 2; cl = cl & (nat0 > 0)
        v = Fr.vesselness(nat)
        far = ndimage_far = None
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(~m)
        cl_r[t].append(np.median(v[cl]) if cl.any() else np.nan); bg_r[t].append(np.percentile(v[(nat0 > 0) & (dist > 4)], 99))
for t in (1, 2, 3):
    print(f"  {t} die: 중심선 {np.nanmedian(cl_r[t]):.3f}  배경 상위 1% {np.median(bg_r[t]):.3f}")

# ---- 그림 ----
CM = ListedColormap(["#ffffff", "#d9dde3", "#c0392b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
show = ["Scratch", "Loc", "none", "Edge-Ring", "Donut", "Center", "Edge-Loc", "Random"]
fig, axes = plt.subplots(4, len(show), figsize=(2.3 * len(show), 9.6))
for j, cname in enumerate(show):
    for r in range(2):
        i, nat, v = EX[cname][r]
        axes[2 * r, j].imshow(nat, cmap=CM, vmin=0, vmax=2, interpolation="nearest")
        axes[2 * r + 1, j].imshow(v, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[2 * r, j].set_title(f"{cname} die {dies[i]}", fontsize=9)
        axes[2 * r + 1, j].set_title(f"Frangi 최댓값 {v.max():.2f}", fontsize=8.5)
for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("y9.2 Frangi 채널 (원본 die 격자, val 예시): 위 = 웨이퍼맵, 아래 = vesselness (0~1)", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "frangi_examples.png"), dpi=115)
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
for j, t in enumerate((1, 2, 3)):
    nat, _ = Fr.line_map(40, 40, t, 30); axes[0, j].imshow(nat, cmap=CM, vmin=0, vmax=2, interpolation="nearest")
    axes[1, j].imshow(Fr.vesselness(nat), cmap="magma", vmin=0, vmax=1, interpolation="nearest"); axes[0, j].set_title(f"{t} die 선 (40×40, 30°)")
for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "frangi_lines.png"), dpi=115)
print("\n그림:", HERE)
