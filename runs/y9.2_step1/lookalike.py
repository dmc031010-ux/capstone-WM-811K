"""seed 2 Frangi 오탐 예시 (val none, die 518, 불량 1021 px) 와 닮은 '라벨된 학습 웨이퍼 (5%)' 는 어떤 라벨인가.
모델이 실제로 본 5% 라벨 (split_seed 0) 의 Scratch·none 만 사용. test 안 씀.
특징 (96 격자): L 가장 큰 불량 덩어리 길이, T 굵기 (넓이/길이), E 길쭉함, bg 그 덩어리 밖 불량 비율.
사용: python runs/y9.2_step1/lookalike.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from wafer.data import WM811K, CLASSES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")


def feats(idx):
    img = ds.fetch(idx, "cpu")[:, 0].numpy().astype(np.uint8); out = np.zeros((len(idx), 4))
    for k in range(len(idx)):
        b = img[k] == 2; lab, n = ndimage.label(b, structure=np.ones((3, 3)))
        if n == 0: continue
        sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); big = np.argmax(sz) + 1
        p = np.argwhere(lab == big).astype(float); A = len(p)
        ev = np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1] if A >= 3 else np.array([A, 1.0])
        L = 4 * np.sqrt(max(ev[0], 1e-6))
        out[k] = (L, A / max(L, 1), np.sqrt(ev[0] / max(ev[1], 1e-6)), (b.sum() - A) / max((img[k] > 0).sum(), 1))
    return img, out


lab = ds.labeled_subset(0.05, seed=0); ly = ds.labels[lab]
tr = lab[(ly == SC) | (ly == NO)]; ty = ds.labels[tr]
vn = ds.val[ds.labels[ds.val] == NO]
cand = vn[dies[vn] == 518]
ci, cf = feats(cand); nb = (ci == 2).reshape(len(cand), -1).sum(1)
from wafer import frangi as Fr
pu, pi = Fr.cache_paths(CACHE); U = np.load(pu, mmap_mode="r"); gi = np.load(pi)
c2 = cand[nb == 1021]; fm = np.array([U[np.searchsorted(gi, i)].max() for i in c2])
print(f"die 518·불량 1021 px 인 val none {len(c2)}장, Frangi 최댓값 {np.round(fm / 255, 2)}")
tgt = c2[np.argmax(fm)]   # 그림의 웨이퍼: Frangi 최댓값 0.70
timg, tf = feats(np.array([tgt]))
img, F = feats(tr)
names = ["길이 L", "굵기 T", "길쭉함 E", "배경 불량 bg"]
print(f"대상: val none 전역 index {tgt}, die {dies[tgt]}, " + ", ".join(f"{n} {v:.3g}" for n, v in zip(names, tf[0])))
print(f"5% 라벨 학습 세트: Scratch {np.sum(ty == SC)}장, none {np.sum(ty == NO)}장")
for c, nm in ((SC, "Scratch"), (NO, "none")):
    m = ty == c
    print(f"  {nm:7s} 중앙값 [25%, 75%]: " + " | ".join(f"{n} {np.median(F[m, j]):.3g} [{np.percentile(F[m, j], 25):.3g}, {np.percentile(F[m, j], 75):.3g}]" for j, n in enumerate(names)))
# 대상보다 선이 같거나 더 뚜렷한 (길이·길쭉함 모두 대상 이상) 학습 웨이퍼의 라벨
m = (F[:, 0] >= tf[0, 0]) & (F[:, 2] >= tf[0, 2])
print(f"\n길이 ≥ {tf[0, 0]:.1f} 이고 길쭉함 ≥ {tf[0, 2]:.2f} 인 학습 웨이퍼: Scratch {np.sum(m & (ty == SC))}장, none {np.sum(m & (ty == NO))}장")
m2 = m & (F[:, 3] >= 0.5 * tf[0, 3])
print(f"  그중 배경 불량 ≥ {0.5 * tf[0, 3]:.3f} (대상의 절반 이상): Scratch {np.sum(m2 & (ty == SC))}장, none {np.sum(m2 & (ty == NO))}장")
# 특징 공간 최근접 (표준화)
Z = (F - F.mean(0)) / F.std(0); zt = (tf - F.mean(0)) / F.std(0)
d = np.sqrt(((Z - zt) ** 2).sum(1))
nn_s = np.where(ty == SC)[0][np.argsort(d[ty == SC])[:6]]; nn_n = np.where(ty == NO)[0][np.argsort(d[ty == NO])[:6]]
nn_all = np.argsort(d)[:20]
print(f"가장 닮은 학습 웨이퍼 20장의 라벨: Scratch {np.sum(ty[nn_all] == SC)}, none {np.sum(ty[nn_all] == NO)}")

CM = ListedColormap(["#ffffff", "#d9dde3", "#c0392b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
fig, ax = plt.subplots(3, 6, figsize=(13.5, 7.4))
ax[0, 0].imshow(timg[0], cmap=CM, vmin=0, vmax=2, interpolation="nearest"); ax[0, 0].set_title(f"대상 (val, 라벨 none)\nL {tf[0, 0]:.0f} T {tf[0, 1]:.1f} E {tf[0, 2]:.1f} bg {tf[0, 3]:.3f}", fontsize=8.5)
for j in range(1, 6): ax[0, j].axis("off")
for r, (sel, nm) in enumerate(((nn_s, "Scratch"), (nn_n, "none")), start=1):
    for j, k in enumerate(sel):
        ax[r, j].imshow(img[k], cmap=CM, vmin=0, vmax=2, interpolation="nearest")
        ax[r, j].set_title(f"학습 라벨 {nm}\nL {F[k, 0]:.0f} T {F[k, 1]:.1f} E {F[k, 2]:.1f} bg {F[k, 3]:.3f}", fontsize=8.5)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("대상과 가장 닮은 5% 라벨 학습 웨이퍼 (2행 Scratch, 3행 none)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "lookalike.png"), dpi=110)
print("그림:", os.path.join(HERE, "lookalike.png"))
