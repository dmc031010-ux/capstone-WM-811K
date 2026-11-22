"""ver8: test Scratch 를 곡선 / 직선 / 선 아님 으로 나눠 혼동행렬 (seed 0·1). 그림 + 예시 갤러리.

주 덩어리 (외톨이 제거, 1 die 틈 연결) 를 die 비율 보정 좌표 (가로 × h/w) 에서:
  길쭉함 < 2.0                       → 선 아님 (덩어리형)
  휘어짐 비율 = sagitta / 길이 ≥ 0.06 그리고 곡선 맞춤 이득 ≥ 0.3 → 곡선
  나머지                             → 직선
"""
import os, sys
import numpy as np, torch
proj, hdir, outd = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(proj); sys.path.insert(0, proj); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
from scipy import ndimage
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
_a = sys.argv; sys.argv = [_a[0], proj, hdir]
import hough2 as h2
sys.argv = _a
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC = CLASSES.index("Scratch")
K8 = np.ones((3, 3), bool)


def main_cluster(nat):
    den = h2.denoise(nat) == 2
    lab, k = ndimage.label(ndimage.binary_dilation(den, K8) & (nat > 0), structure=K8)
    cnt = np.bincount(lab[den], minlength=k + 1); cnt[0] = 0
    return np.nonzero(den & (lab == cnt.argmax())) if cnt.max() > 0 else (np.array([]), np.array([]))


def curve_feats(i):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    nat = h2.native(img, h, w)
    y, x = main_cluster(nat)
    if len(y) < 5:
        return np.nan, np.nan, np.nan, np.nan, nat, None
    fx = h / w if (h <= 96 and w <= 96) else 1.0          # die 비율 보정 (96 지도 그대로 쓰는 큰 웨이퍼는 이미 원형)
    P = np.c_[x * fx, y].astype(float); mu = P.mean(0); Q = P - mu
    ev, V = np.linalg.eigh(Q.T @ Q / len(P))
    t, u = Q @ V[:, 1], Q @ V[:, 0]
    elong = np.sqrt(ev[1] / max(ev[0], 1 / 12))
    L = t.max() - t.min() + 1
    r1 = u - np.polyval(np.polyfit(t, u, 1), t)
    c2 = np.polyfit(t, u, 2); r2 = u - np.polyval(c2, t)
    sag = abs(c2[0]) * (L / 2) ** 2
    gain = 1 - (r2 ** 2).sum() / max((r1 ** 2).sum(), 1e-9)
    tt = np.linspace(t.min(), t.max(), 50)                 # 그림용 맞춤 곡선 (원래 격자 좌표로 되돌림)
    curve = mu + np.outer(tt, V[:, 1]) + np.outer(np.polyval(c2, tt), V[:, 0])
    curve[:, 0] /= fx
    return elong, sag / L, gain, L, nat, (x, y, curve)


te = ds.test
sc_ids = te[ds.labels[te] == SC]
F = [curve_feats(i) for i in sc_ids]
elong = np.array([f[0] for f in F]); bow = np.array([f[1] for f in F]); gain = np.array([f[2] for f in F])
grp = np.where(np.isnan(elong) | (elong < 2.0), "선 아님", np.where((bow >= 0.06) & (gain >= 0.3), "곡선", "직선"))
GROUPS = ["직선", "곡선", "선 아님"]
print("test Scratch", len(sc_ids), "장:", {g: int((grp == g).sum()) for g in GROUPS})


@torch.no_grad()
def predict(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m(to_input(ds.fetch(sc_ids, "cpu"))).argmax(1).numpy()


P = {s: predict(s) for s in (0, 1)}
CM = {}
for s in (0, 1, "합"):
    cm = np.zeros((3, NUM_CLASSES), int)
    for a, g in enumerate(GROUPS):
        for p in (P[0], P[1]) if s == "합" else (P[s],):
            cm[a] += np.bincount(p[grp == g], minlength=NUM_CLASSES)
    CM[s] = cm
    print(f"\n[ver8 seed {s}] 행 = Scratch 모양, 열 = 예측")
    print(" " * 10 + "".join(f"{c[:8]:>9s}" for c in CLASSES) + f"{'재현율':>9s}")
    for a, g in enumerate(GROUPS):
        print(f"{g:8s}" + "".join(f"{v:9d}" for v in cm[a]) + f"{cm[a, SC] / max(cm[a].sum(), 1):9.3f}")

# ---- 그림 1: 혼동행렬 (seed 0, seed 1, 합)
blue = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
short = [c.replace("Edge-", "E-").replace("Near-full", "N-full") for c in CLASSES]
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
for ax, s in zip(axes, (0, 1, "합")):
    cm = CM[s]; rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap=blue, vmin=0, vmax=1, aspect="auto")
    for a in range(3):
        for j in range(NUM_CLASSES):
            if cm[a, j]:
                ax.text(j, a, f"{cm[a, j]}\n({rn[a, j]:.0%})", ha="center", va="center", fontsize=9,
                        color="#fcfcfb" if rn[a, j] > 0.55 else "#0b0b0b", fontweight="bold" if j == SC else "normal")
    n = cm.sum(1)
    ax.set_yticks(range(3), [f"{g} Scratch ({k}{'' if s != '합' else ', ×2'})" for g, k in zip(GROUPS, n)], fontsize=10)
    ax.set_xticks(range(NUM_CLASSES), short, fontsize=9.5)
    ax.set_title(f"ver8 seed {s}" + (" (두 seed 합)" if s == "합" else "") + "  · 칸 = 장수 (행 기준 비율), 굵은 열 = Scratch 로 맞힘", loc="left", fontsize=11)
    for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle("ver8: Scratch 모양별 예측 (test Scratch 120장, 열 = 예측 클래스)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig10_curve_cm.png"), dpi=130); plt.close(fig)

# ---- 그림 2: 분류가 맞는지 확인용 예시 (그룹별 6장, seed 0 예측 표시)
rng = np.random.default_rng(0)
cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
fig, axes = plt.subplots(3, 6, figsize=(15, 8.6))
for a, g in enumerate(GROUPS):
    idx = np.nonzero(grp == g)[0]
    for j, k in enumerate(rng.choice(idx, min(6, len(idx)), replace=False)):
        ax = axes[a, j]
        e, b, gn, L, nat, extra = F[k]
        ax.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        if extra is not None:
            x, y, curve = extra
            ax.scatter(x, y, s=max(4, 700 / max(nat.shape)), color="#eb6834", marker="s", linewidths=0)
            if g != "선 아님":
                ax.plot(curve[:, 0], curve[:, 1], color="#2a78d6", lw=1.5)
        p0, p1 = CLASSES[P[0][k]], CLASSES[P[1][k]]
        ax.set_title(f"{g} · 휘어짐 {b:.2f} · 길쭉함 {e:.1f}\n예측 s0 {p0} / s1 {p1}", fontsize=8.5,
                     color="#0b0b0b" if (p0 == "Scratch") else "#d03b3b")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    for j in range(min(6, len(idx)), 6):
        axes[a, j].axis("off")
fig.suptitle("모양 분류 확인 (test Scratch). 주황 = 주 덩어리, 파랑 = 2차 곡선 맞춤, 빨간 제목 = seed 0 이 틀림", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig11_curve_examples.png"), dpi=125); plt.close(fig)
print("\n그림:", outd)
