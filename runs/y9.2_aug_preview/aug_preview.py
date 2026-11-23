"""y9.2 후보 증강을 실제 웨이퍼로 보여 주는 그림 (학습 없음). train 라벨 Scratch 와 라벨 없는 웨이퍼만 사용, val/test 안 씀.

fig_E.png : 라벨 없는 웨이퍼 6만 장을 y9 seed 0 모델로 채점 → Scratch 확률 상위 / 경계 (0.5~0.7) 예시 + 확률별 장수
fig_B.png : 실제 Scratch (die 많은 것) 를 원본 die 격자에서 2×2, 3×3 칸씩 묶어 die 수를 줄임 (칸 안에 불량 하나라도 있으면 불량)
fig_A.png : 실제 Scratch 의 가장 큰 불량 덩어리 (= 선) 만 떼어, 회전해서 실제 none 웨이퍼에 붙임 (상대 좌표)
사용: python runs/y9.2_aug_preview/aug_preview.py
"""
import os, sys
import numpy as np, torch
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
rng = np.random.default_rng(0)
CM = ListedColormap(["#ffffff", "#d9dde3", "#c0392b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
lab = ds.labeled_subset(0.05, seed=0); ylab = ds.labels[lab]


def native(i):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]


def up96(nat):
    h, w = nat.shape
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return nat[np.ix_(ri, ci)]


def show(ax, m, title, fs=9):
    ax.imshow(m, cmap=CM, vmin=0, vmax=2, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#c8ccd2")
    ax.set_title(title, fontsize=fs)


# ---------- E: pseudo-label ----------
m = build_model("resnet18", NUM_CLASSES)
m.load_state_dict(torch.load("runs/resnet18/p0.05/seed0/y9_ssl50_ft30/best.pt", map_location="cpu", weights_only=True)); m = m.to(dev).eval()
ul = np.sort(rng.choice(ds.unlabeled, 60000, replace=False))
P = []
with torch.no_grad():
    for i in range(0, len(ul), 1024):
        P.append(torch.softmax(m(to_input(ds.fetch(ul[i:i + 1024], dev))), 1).cpu())
P = torch.cat(P).numpy(); ps = P[:, SC]; top = P.argmax(1)
cnt = {t: int(((top == SC) & (ps >= t)).sum()) for t in (0.5, 0.7, 0.9, 0.95, 0.99)}
print("E: 라벨 없는 6만 장 중 Scratch 로 예측 & 확률 ≥ t:", cnt, f"(전체 라벨 없는 웨이퍼 {len(ds.unlabeled):,}장)")
hi = ul[np.argsort(-ps)[:10]]
mid_pool = np.where((top == SC) & (ps >= 0.5) & (ps < 0.7))[0]
mid = ul[rng.choice(mid_pool, min(10, len(mid_pool)), replace=False)]
fig, axes = plt.subplots(2, 10, figsize=(20, 4.9))
for ax, i in zip(axes[0], hi):
    show(ax, ds.fetch([i], "cpu")[0, 0].numpy(), f"확률 {ps[np.searchsorted(ul, i)]:.3f}\ndie {dies[i]}")
for ax, i in zip(axes[1], mid):
    show(ax, ds.fetch([i], "cpu")[0, 0].numpy(), f"확률 {ps[np.searchsorted(ul, i)]:.3f}\ndie {dies[i]}")
axes[0, 0].set_ylabel("확신 높음\n(상위 10장)", fontsize=10); axes[1, 0].set_ylabel("경계\n(0.5~0.7)", fontsize=10)
fig.suptitle("E. pseudo-label: 라벨 없는 실제 웨이퍼 중 y9 가 Scratch 라고 본 것  "
             f"(6만 장 중 확률 ≥0.9: {cnt[0.9]}장, ≥0.95: {cnt[0.95]}장, ≥0.99: {cnt[0.99]}장)  → 위쪽처럼 확신 높은 것만 Scratch 라벨로 학습에 추가",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_E.png"), dpi=120); plt.close(fig)


# ---------- B: die 수 줄이기 ----------
def coarsen(nat, k):
    h, w = nat.shape; H, W = -(-h // k), -(-w // k)
    pad = np.zeros((H * k, W * k), np.uint8); pad[:h, :w] = nat
    blk = pad.reshape(H, k, W, k).transpose(0, 2, 1, 3).reshape(H, W, k * k)
    out = np.where((blk > 0).any(-1), 1, 0).astype(np.uint8)
    out[(blk == 2).any(-1)] = 2
    return out

sc_lab = lab[(ylab == SC) & (sd[lab, :2].max(1) <= 96)]
big = sc_lab[np.argsort(-dies[sc_lab])][:4]
fig, axes = plt.subplots(4, 3, figsize=(8.4, 11.6))
for r, i in enumerate(big):
    nat = native(i)
    for c, k in enumerate((1, 2, 3)):
        n2 = coarsen(nat, k) if k > 1 else nat
        show(axes[r, c], up96(n2), ("원본" if k == 1 else f"{k}×{k} 묶음") + f"  격자 {n2.shape[0]}×{n2.shape[1]}, die {(n2 > 0).sum()}")
fig.suptitle("B. die 수 바꾸기: 실제 Scratch 를 거친 격자로 다시 샘플링\n(칸 안에 불량 die 가 하나라도 있으면 불량) → die 적은 Scratch 예시를 만듦",
             x=0.02, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_B.png"), dpi=120); plt.close(fig)


# ---------- A: 선 이식 ----------
def line_part(nat):
    lb, n = ndimage.label(nat == 2, structure=np.ones((3, 3)))
    if n == 0: return None
    sz = ndimage.sum(np.ones_like(lb), lb, range(1, n + 1)); return lb == (np.argmax(sz) + 1)

def paste(line, src_shape, tgt, ang):
    h, w = src_shape; H, W = tgt.shape; out = tgt.copy()
    r, c = np.nonzero(line)
    u, v = (c + 0.5) / w - 0.5, (r + 0.5) / h - 0.5                       # 웨이퍼 중심 기준 상대 좌표
    ca, sa = np.cos(ang), np.sin(ang); u2, v2 = ca * u - sa * v, sa * u + ca * v
    # 대상 격자에서 1 die 두께로 다시 잇기: 상대 좌표를 촘촘히 보간
    rr = np.floor((v2 + 0.5) * H).astype(int); cc = np.floor((u2 + 0.5) * W).astype(int)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W); rr, cc = rr[ok], cc[ok]
    inw = tgt[rr, cc] > 0; out[rr[inw], cc[inw]] = 2
    return out

none_lab = lab[(ylab == NO) & (sd[lab, :2].max(1) <= 96)]
none_lab = np.array([i for i in rng.permutation(none_lab)[:300] if (native(i) == 2).sum() <= 0.02 * dies[i]])
src = [i for i in sc_lab[rng.permutation(len(sc_lab))] if (lambda L: L is not None and L.sum() >= 8)(line_part(native(i)))][:3]
fig, axes = plt.subplots(3, 5, figsize=(15, 9.6))
for r, i in enumerate(src):
    nat = native(i); L = line_part(nat)
    lineonly = np.where(nat > 0, 1, 0).astype(np.uint8); lineonly[L] = 2
    show(axes[r, 0], up96(nat), f"원본 Scratch (train)\ndie {dies[i]}")
    show(axes[r, 1], up96(lineonly), "가장 큰 불량 덩어리 = 선\n(나머지 불량은 버림)")
    for c, ang in enumerate(rng.uniform(-np.pi, np.pi, 3)):
        t = none_lab[(r * 3 + c) % len(none_lab)]
        show(axes[r, 2 + c], up96(paste(L, nat.shape, native(t), ang)), f"none 웨이퍼 (die {dies[t]}) 에\n{np.degrees(ang):.0f}° 회전해 붙임")
fig.suptitle("A. 선 이식: 실제 Scratch 의 선만 떼어 → 회전 → 다른 실제 none 웨이퍼에 붙임 (선과 배경은 진짜, 조합은 새것)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_A.png"), dpi=120); plt.close(fig)
print("그림:", HERE)
