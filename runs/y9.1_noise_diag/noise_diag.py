"""y9.1 노이즈 진단 보강 (Codex 06 R3·R4, 07 다음 단계 1). 학습 없음, val 만.

0단계 (runs/y9.1_step0) 와 같은 방식에 다음을 추가:
  - 파일럿 체크포인트 ①~④ + y9 + SimCLR 전부 (모두 seed 0 fine-tune 모델)
  - 노이즈 난수 5세트 (웨이퍼·선은 고정, die 별 난수만 바꿈) → 평균 ± 표준편차
  - 같은 웨이퍼 짝비교: 웨이퍼별 Scratch 판정률 (5세트 평균) 의 차이, 웨이퍼 부트스트랩 95% 구간
  - 실제 Scratch 를 die 수 구간 (<1200 / ≥1200) 으로 나눔, Scratch→none/Loc 전이, none→Scratch 오탐
그룹: (A) 실제 val Scratch (원본 격자 ≤ 96), (B) val none 위 합성 1-die 직선 (die 적음 / 많음), (C) 선 없는 val none
사용: python runs/y9.1_noise_diag/noise_diag.py
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
PS = [0.0, 0.02, 0.04, 0.08, 0.16]; NSEED = 5
R0 = "runs/resnet18/p0.05/seed0"
MODELS = [("① 마스킹", f"{R0}/y9.1a_ssl10_ft30/best.pt"), ("② 마스킹+proj", f"{R0}/y9.1b_ssl10_ft30/best.pt"),
          ("③ 회전만+proj", f"{R0}/y9.1c_ssl10_ft30/best.pt"), ("④ 노이즈+proj", f"{R0}/y9.1d_ssl10_ft30/best.pt"),
          ("y9 (50ep)", f"{R0}/y9_ssl50_ft30/best.pt"), ("SimCLR (50ep)", f"{R0}/ver3_ssl50_ft30/best.pt")]
rng = np.random.default_rng(0)


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
    """X[seed][p] = (n, 96, 96) uint8. 웨이퍼·선은 고정, 노이즈 난수만 seed 별로 다름."""
    X = [{p: [] for p in PS} for _ in range(NSEED)]
    for i in ids:
        nat, ri, ci = native(i)
        if line:
            nat = draw(nat, segment())
        for s in range(NSEED):
            u = rng.random(nat.shape)
            for p in PS:
                X[s][p].append(np.where((nat == 1) & (u < p), 2, nat).astype(np.uint8)[np.ix_(ri, ci)])
    return [{p: np.stack(v) for p, v in d.items()} for d in X]


va = ds.val; yv = ds.labels[va]; small_grid = sd[va, :2].max(1) <= 96
def normal_none(lo, hi, n):
    c = va[(yv == NO) & small_grid & (dies[va] >= lo) & (dies[va] < hi)]
    good = []
    for i in rng.permutation(c):
        if (native(i)[0] == 1).sum() >= 0.8 * dies[i]:
            good.append(i)
        if len(good) == n: break
    return np.array(good)


ids_A = va[(yv == SC) & small_grid]
GROUPS = {"(A) 실제 Scratch": (ids_A, build(ids_A, False)),
          "(B) 합성선 · die 적음": (None, build(normal_none(300, 1200, 200), True)),
          "(B) 합성선 · die 많음": (None, build(normal_none(1200, 10 ** 9, 200), True)),
          "(C) 선 없는 none": (None, build(normal_none(0, 10 ** 9, 400), False))}
A_small = dies[ids_A] < 1200


@torch.no_grad()
def run(path, X):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    out = []
    for i in range(0, len(X), 512):
        out.append(torch.softmax(m(to_input(torch.as_tensor(X[i:i + 512], dtype=torch.float32, device=dev)[:, None])), 1).cpu())
    return torch.cat(out).numpy()


# P[(model, group)] = (NSEED, len(PS), n, C) 확률
P = {}
for mn, path in MODELS:
    for g, (_, X) in GROUPS.items():
        P[(mn, g)] = np.stack([np.stack([run(path, X[s][p]) for p in PS]) for s in range(NSEED)])

pct = " ".join(f"{int(p * 100):>11d}%" for p in PS)
def ms(a):  # a: (NSEED,) → "평균±sd"
    return f"{a.mean():.2f}±{a.std():.2f}"

for g in GROUPS:
    n = P[(MODELS[0][0], g)].shape[2]
    if g.startswith("(C)"):
        print(f"\n{g} (n={n})  [none 유지율 / none→Scratch 오탐률, 노이즈 5세트 평균±sd]\n{'':16s}{pct}")
        for mn, _ in MODELS:
            a = P[(mn, g)].argmax(-1)
            print(f"{mn:16s}" + " ".join(f"{ms((a[:, k] == NO).mean(1))}/{(a[:, k] == SC).mean():.3f}" for k in range(len(PS))))
        continue
    print(f"\n{g} (n={n})  [Scratch 판정률 평균±sd | 평균 Scratch 확률 | →none | →Loc]\n{'':16s}{pct}")
    for mn, _ in MODELS:
        pr = P[(mn, g)]; a = pr.argmax(-1)
        print(f"{mn:16s}" + " ".join(f"{ms((a[:, k] == SC).mean(1)):>12s}" for k in range(len(PS))))
        print(f"{'  Scratch 확률':16s}" + " ".join(f"{pr[:, k, :, SC].mean():12.2f}" for k in range(len(PS))))
        print(f"{'  →none / →Loc':16s}" + " ".join(f"{(a[:, k] == NO).mean():5.2f}/{(a[:, k] == LO).mean():.2f} " for k in range(len(PS))))

# (A) die 수 구간
print(f"\n(A) 실제 Scratch, die 수 구간별 Scratch 판정률 (5세트 평균)  die<1200 n={A_small.sum()}, ≥1200 n={(~A_small).sum()}\n{'':16s}{pct}")
for mn, _ in MODELS:
    a = P[(mn, "(A) 실제 Scratch")].argmax(-1) == SC
    print(f"{mn:16s}" + " ".join(f"{a[:, k, A_small].mean():5.2f}/{a[:, k, ~A_small].mean():.2f} " for k in range(len(PS))))

# 같은 웨이퍼 짝비교: 웨이퍼별 판정률 (5세트 평균) 차이, 웨이퍼 부트스트랩
def paired(g, m1, m2):
    d = (P[(m2, g)].argmax(-1) == SC).mean(0) - (P[(m1, g)].argmax(-1) == SC).mean(0)  # (len(PS), n)
    n = d.shape[1]; bi = np.random.default_rng(1).integers(0, n, (2000, n))
    out = []
    for k in range(len(PS)):
        bs = d[k][bi].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(f"{d[k].mean():+.3f}[{lo:+.2f},{hi:+.2f}]")
    return out

PAIRS = [("③ 회전만+proj", "④ 노이즈+proj"), ("① 마스킹", "④ 노이즈+proj"), ("y9 (50ep)", "④ 노이즈+proj"), ("y9 (50ep)", "③ 회전만+proj")]
for g in ["(A) 실제 Scratch", "(B) 합성선 · die 적음", "(B) 합성선 · die 많음"]:
    print(f"\n짝비교 {g}: (뒤 − 앞) Scratch 판정률 차, 웨이퍼 부트스트랩 95% 구간\n{'':28s}" + " ".join(f"{int(p * 100):>19d}%" for p in PS))
    for m1, m2 in PAIRS:
        print(f"{m2[:6]} − {m1[:12]:14s}      " + " ".join(f"{s:>20s}" for s in paired(g, m1, m2)))

# 그림
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})
COL = {"① 마스킹": "#9aa3ad", "② 마스킹+proj": "#c9b07a", "③ 회전만+proj": "#2f9e6a", "④ 노이즈+proj": "#d6452a",
       "y9 (50ep)": "#2a78d6", "SimCLR (50ep)": "#8a5ad6"}
fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True); x = [p * 100 for p in PS]
for ax, g in zip(axes, GROUPS):
    cls = NO if g.startswith("(C)") else SC
    for mn, _ in MODELS:
        r = (P[(mn, g)].argmax(-1) == cls).mean(2)  # (NSEED, len(PS))
        ax.errorbar(x, r.mean(0), yerr=r.std(0), fmt="-o", ms=4, capsize=3, color=COL[mn], lw=1.8, label=mn)
    ax.set_xticks(x); ax.set_xlabel("추가한 랜덤 불량 (정상 die 중 %)"); ax.set_ylim(0, 1.02); ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_title(f"{g} (n={P[(MODELS[0][0], g)].shape[2]})\n" + ("none 유지율" if cls == NO else "Scratch 판정률"), loc="left", fontsize=10.5)
axes[0].set_ylabel("비율 (노이즈 5세트 평균 ± sd)"); axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("y9.1 노이즈 진단 보강: 파일럿 ①~④ 와 참고 모델 (val, seed 0, 학습 없음)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "noise_diag.png"), dpi=140)
print("\n그림:", os.path.join(HERE, "noise_diag.png"))
