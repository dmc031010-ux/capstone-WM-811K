"""y9.1 0단계: 기존 모델이 '흩어진 랜덤 불량' 에 얼마나 흔들리는가 (학습 없음, val 만 사용).

모델: y9 seed 0·1 (= ver7 마스킹 사전학습), 참고 ver3 seed 0 (SimCLR, 증강에 픽셀 노이즈 포함).
노이즈: 원본 die 격자에서 정상 die 를 확률 p 로 불량으로 (die 단위, 회전 없음) → 캐시와 같은 nearest 로 96×96.
  p = 0, 2, 4, 8, 16%. 웨이퍼마다 난수는 p 와 무관하게 고정 (u < p 로 켜므로 p 가 커질수록 같은 die 가 누적).
대상
 (A) 실제 val Scratch 웨이퍼 (원본 격자 ≤ 96): Scratch 재현율과 none / Loc 으로 가는 비율
 (B) 합성선: val none 웨이퍼 (정상 지도) 에 같은 상대 위치·길이의 1 die 직선을 그린 뒤 노이즈. die 수 적음(300~1199) / 많음(1200+)
 (C) 선 없는 val none 웨이퍼: 노이즈만 넣었을 때 none 을 유지하는지 (불량 쪽 오탐)
사용: python runs/y9.1_step0/noise_sensitivity.py
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO, LO = CLASSES.index("Scratch"), CLASSES.index("none"), CLASSES.index("Loc")
PS = [0.0, 0.02, 0.04, 0.08, 0.16]
rng = np.random.default_rng(0)
MODELS = [("y9 seed 0 (마스킹)", "runs/resnet18/p0.05/seed0/y9_ssl50_ft30/best.pt"),
          ("y9 seed 1 (마스킹)", "runs/resnet18/p0.05/seed1/y9_ssl50_ft30/best.pt"),
          ("ver3 seed 0 (SimCLR)", "runs/resnet18/p0.05/seed0/ver3_ssl50_ft30/best.pt")]


def native(i):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    assert (nat[np.ix_(ri, ci)] == img).all()
    return nat, ri, ci


def draw(nat, seg):
    h, w = nat.shape; out = nat.copy(); (u0, v0), (u1, v1) = seg
    t = np.linspace(0, 1, int(np.hypot((u1 - u0) * w, (v1 - v0) * h) * 4) + 2)
    r = np.floor((v0 + t * (v1 - v0)) * h).astype(int); c = np.floor((u0 + t * (u1 - u0)) * w).astype(int)
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w); r, c = r[ok], c[ok]; inw = nat[r, c] > 0
    out[r[inw], c[inw]] = 2
    return out


def segment():
    while True:
        rr, a0 = 0.35 * np.sqrt(rng.random()), rng.random() * 2 * np.pi
        cx, cy = 0.5 + rr * np.cos(a0), 0.5 + rr * np.sin(a0)
        th, L = rng.random() * np.pi, 0.3 + 0.3 * rng.random()
        p0 = (cx - L / 2 * np.cos(th), cy - L / 2 * np.sin(th)); p1 = (cx + L / 2 * np.cos(th), cy + L / 2 * np.sin(th))
        if all(np.hypot(x - 0.5, y - 0.5) < 0.47 for x, y in (p0, p1)):
            return p0, p1


def build(ids, line):
    """웨이퍼마다 (p 별 96 지도 리스트). 노이즈 난수 u 는 웨이퍼마다 고정."""
    out = {p: [] for p in PS}
    for i in ids:
        nat, ri, ci = native(i)
        if line:
            nat = draw(nat, segment())
        u = rng.random(nat.shape)
        for p in PS:
            noisy = np.where((nat == 1) & (u < p), 2, nat).astype(np.uint8)
            out[p].append(noisy[np.ix_(ri, ci)])
    return {p: np.stack(v) for p, v in out.items()}


va = ds.val; yv = ds.labels[va]; small_grid = sd[va, :2].max(1) <= 96
def normal_none(lo, hi, n):
    c = va[(yv == NO) & small_grid & (dies[va] >= lo) & (dies[va] < hi)]
    good = []
    for i in rng.permutation(c):
        if (native(i)[0] == 1).sum() >= 0.8 * dies[i]:
            good.append(i)
        if len(good) == n: break
    return np.array(good)


GROUPS = {"(A) 실제 val Scratch": build(va[(yv == SC) & small_grid], line=False),
          "(B) 합성선 · die 적음": build(normal_none(300, 1200, 200), line=True),
          "(B) 합성선 · die 많음": build(normal_none(1200, 10 ** 9, 200), line=True),
          "(C) 선 없는 none": build(normal_none(0, 10 ** 9, 400), line=False)}


@torch.no_grad()
def probs(path, X):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    o = []
    for i in range(0, len(X), 512):
        o.append(torch.softmax(m(to_input(torch.as_tensor(X[i:i + 512], dtype=torch.float32, device=dev)[:, None])), 1).cpu())
    return torch.cat(o).numpy()


R = {}
print(f"{'대상':22s} {'모델':22s} " + " ".join(f"p={int(p * 100):>2d}%" for p in PS) + "   (값: Scratch 판정 비율 / none / Loc;  (C) 는 none 유지 비율)")
for g, D in GROUPS.items():
    for mn, path in MODELS:
        rows = []
        for p in PS:
            a = probs(path, D[p]).argmax(1)
            R[(g, mn, p)] = (np.mean(a == SC), np.mean(a == NO), np.mean(a == LO))
            rows.append(R[(g, mn, p)])
        if g.startswith("(C)"):
            print(f"{g:22s} {mn:22s} " + " ".join(f"{r[1]:6.2f}" for r in rows) + f"   n={len(D[0.0])}")
        else:
            print(f"{g:22s} {mn:22s} " + " ".join(f"{r[0]:.2f}/{r[1]:.2f}/{r[2]:.2f}" for r in rows) + f"   n={len(D[0.0])}")

# 그림
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})
COL = {"y9 seed 0 (마스킹)": "#2a78d6", "y9 seed 1 (마스킹)": "#86b6ef", "ver3 seed 0 (SimCLR)": "#eb6834"}
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
x = [p * 100 for p in PS]
for ax, g in zip(axes, GROUPS):
    k = 1 if g.startswith("(C)") else 0
    for mn, _ in MODELS:
        ax.plot(x, [R[(g, mn, p)][k] for p in PS], "-o", color=COL[mn], lw=2, label=mn)
        if k == 0:
            ax.plot(x, [R[(g, mn, p)][1] for p in PS], ":", color=COL[mn], lw=1.3)
    ax.set_xticks(x); ax.set_xlabel("추가한 랜덤 불량 (정상 die 중 %)"); ax.set_ylim(0, 1.02); ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_title(f"{g} (n={len(GROUPS[g][0.0])})\n" + ("실선 = none 유지 비율" if k else "실선 = Scratch 판정, 점선 = none 판정"), loc="left", fontsize=10.5)
axes[0].set_ylabel("비율"); axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("0단계: 흩어진 랜덤 불량을 늘릴 때 기존 모델의 판정 변화 (val, die 단위 노이즈, 학습 없음)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "noise_sensitivity.png"), dpi=140)
print("그림:", os.path.join(HERE, "noise_sensitivity.png"))
