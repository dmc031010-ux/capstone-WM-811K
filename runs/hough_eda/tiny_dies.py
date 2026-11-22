"""die 수가 아주 적은 웨이퍼: 분포 + 이미지 (원본 격자 / 모델 입력 96x96)."""
import os, sys
import numpy as np
proj, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
d = sd[:, 2]; lab = ds.labels
for t in (10, 50, 100, 200, 300):
    g = d < t
    print(f"die < {t}: {g.sum()}장 (라벨 있는 것 {(g & (lab >= 0)).sum()}장, 클래스 {dict(zip(*np.unique([CLASSES[c] for c in lab[g & (lab >= 0)]], return_counts=True))) if (g & (lab >= 0)).any() else {}})")
vals = np.unique(d)[:12]
print("가장 작은 die 수 12종:", vals.tolist(), "| 각 장수:", [int((d == v).sum()) for v in vals])
pick = [np.nonzero(d == v)[0][0] for v in vals]       # die 수 종류별 첫 웨이퍼 1장씩 (라벨 있는 것 우선)
pick = [(np.nonzero((d == v) & (lab >= 0))[0][:1].tolist() or [p])[0] for v, p in zip(vals, pick)]

cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
fig, axes = plt.subplots(4, 6, figsize=(15, 11.5))
for n, i in enumerate(pick):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(int); ci = ((np.arange(96) + 0.5) * w / 96).astype(int)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    lb = CLASSES[lab[i]] if lab[i] >= 0 else "라벨 없음"
    a0, a1 = axes.ravel()[2 * n], axes.ravel()[2 * n + 1]
    a0.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    for yy in range(h + 1): a0.axhline(yy - 0.5, color="#b9b8b1", lw=0.5)
    for xx in range(w + 1): a0.axvline(xx - 0.5, color="#b9b8b1", lw=0.5)
    a0.set_title(f"die {d[i]}개 · 격자 {h}×{w}\n원본 격자 · {lb} (같은 die 수 {int((d == d[i]).sum())}장)", fontsize=9)
    a1.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    a1.set_title(f"모델 입력 96×96\n(die 1개 = 약 {96 / w:.0f}×{96 / h:.0f} 픽셀)", fontsize=9)
    for a in (a0, a1):
        a.set_xticks([]); a.set_yticks([])
fig.suptitle("die 수가 가장 적은 웨이퍼 12종 (종류별 1장, 라벨 있는 것 우선). 흰색 = 웨이퍼 밖, 회색 = 정상 die, 검정 = 불량 die",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=110)
print("저장:", out)

