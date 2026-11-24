"""재현율 0.7 이후에 남는 val Scratch 진단 (val 만, test 안 봄, 학습 없음).
모델 (y9.2a, y9.3a) × seed 0·1·2 각각: P(Scratch) 로 val 전체를 줄 세워 Scratch 재현이 0.7 이 되는 임계 아래에 있는 Scratch = '어려운' Scratch.
본 집합: y9.2a 에서 3 seed 중 2 이상 어려움. 대조: y9.2a 에서 3 seed 모두 임계 위 ('쉬운').
특징: die 수, 불량 비율, 불량 die 성분 최대 길이, Frangi 능선 (τ=0.2) 성분 최대 길이, Frangi 최댓값, 예측 클래스, 위에 있는 none 장수.
사용: python runs/y9.3_step1/hard_scratch.py
"""
import os, sys, gc
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input, FrangiInput

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
va = ds.val; yv = ds.labels[va]; SC, NO = CLASSES.index("Scratch"), CLASSES.index("none"); s = yv == SC
vs = va[s]; nS = len(vs)
MODELS = {"y9.2a": "frangi", "y9.3a": "length"}


@torch.no_grad()
def probs(seed, meth, tab):
    m = FrangiInput(build_model("resnet18", NUM_CLASSES), *tab)
    d = f"runs/resnet18/p0.05/seed{seed}/{meth}_ssl50_ft30"
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    out = []
    for i in range(0, len(va), 1024):
        j = va[i:i + 1024]; out.append(torch.softmax(m(to_input(ds.fetch(j, dev)), torch.as_tensor(j, device=dev)).float(), 1).cpu())
    del m; torch.cuda.empty_cache(); return torch.cat(out).numpy()


HARD, PS, PRED, NABOVE = {}, {}, {}, {}
for meth, kind in MODELS.items():
    pu, pi = Fr.cache_paths(CACHE, kind); gi = np.load(pi)
    row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi)); tab = (torch.from_numpy(np.load(pu)), row)
    for seed in (0, 1, 2):
        p = probs(seed, meth, tab); ps = p[:, SC]
        th = np.sort(ps[s])[::-1][int(np.ceil(0.7 * nS)) - 1]          # Scratch 재현 0.7 이 되는 임계
        HARD[(meth, seed)] = ps[s] < th; PS[(meth, seed)] = ps[s]; PRED[(meth, seed)] = p[s].argmax(1)
        NABOVE[(meth, seed)] = np.array([int(((yv == NO) & (ps > x)).sum()) for x in ps[s]])   # 이 Scratch 보다 위에 선 none 장수
    del tab; gc.collect()

# 원본 격자 특징 (> 96 격자는 nan)
img96 = ds.fetch(vs, "cpu")[:, 0].numpy().astype(np.uint8)
F = {k: np.full(nS, np.nan) for k in ("dies", "frac", "Ldef", "Lridge", "Vmax", "ncomp")}
for j, i in enumerate(vs):
    h, w = sd[i, :2]; F["dies"][j] = sd[i, 2]
    if max(h, w) > 96: continue
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img96[j][np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    v = Fr.vesselness(nat)
    F["frac"][j] = (nat == 2).sum() / max((nat > 0).sum(), 1)
    F["Ldef"][j] = Fr.line_weight(nat, return_parts=True)[2].max()
    F["Lridge"][j] = Fr.line_weight(nat, "ridge", V=v, tau=Fr.TAU, return_parts=True)[2].max()
    F["Vmax"][j] = v[nat > 0].max(); F["ncomp"][j] = Fr.ndimage.label(nat == 2, structure=np.ones((3, 3)))[1]

nh = sum(HARD[("y9.2a", sd_)].astype(int) for sd_ in (0, 1, 2)); nh3 = sum(HARD[("y9.3a", sd_)].astype(int) for sd_ in (0, 1, 2))
hard = nh >= 2; easy = nh == 0
print(f"val Scratch {nS}장 (원본 격자 ≤96: {np.isfinite(F['frac']).sum()}장). 재현 0.7 임계 아래 = 모델·seed 당 {int(HARD[('y9.2a', 0)].sum())}장 안팎.")
print(f"y9.2a 어려움 seed 수 분포 (0/1/2/3): {[int((nh == k).sum()) for k in range(4)]},  y9.3a: {[int((nh3 == k).sum()) for k in range(4)]}")
print(f"본 집합 (y9.2a 2 seed 이상 어려움) {hard.sum()}장, 그중 y9.3a 에서도 2 seed 이상 어려움 {int((hard & (nh3 >= 2)).sum())}장, y9.3a 에서 쉬움 (0 seed) {int((hard & (nh3 == 0)).sum())}장")
print(f"y9.3a 에서만 어려움 (y9.2a 0 seed, y9.3a ≥2) {int(((nh == 0) & (nh3 >= 2)).sum())}장\n")

med = lambda x, m: np.nanmedian(x[m])
q = lambda x, m: f"{np.nanpercentile(x[m], 25):.2f}/{np.nanmedian(x[m]):.2f}/{np.nanpercentile(x[m], 75):.2f}"
print(f"{'특징 (25%/중앙/75%)':24s} {'어려운 n=' + str(hard.sum()):>22s} {'쉬운 n=' + str(easy.sum()):>22s}")
for k, name in (("dies", "die 수"), ("frac", "불량 die 비율"), ("Ldef", "불량 성분 최대 길이(die)"), ("Lridge", "능선 성분 최대 길이(die)"), ("Vmax", "Frangi 최댓값"), ("ncomp", "불량 덩어리 수")):
    print(f"{name:24s} {q(F[k], hard):>22s} {q(F[k], easy):>22s}")
ps_a = np.mean([PS[("y9.2a", x)] for x in (0, 1, 2)], 0); na_a = np.mean([NABOVE[("y9.2a", x)] for x in (0, 1, 2)], 0)
print(f"{'y9.2a P(Scratch) 평균':24s} {q(ps_a, hard):>22s} {q(ps_a, easy):>22s}")
print(f"{'위에 선 none 장수 평균':24s} {q(na_a, hard):>22s} {q(na_a, easy):>22s}")

print("\n길이 구간별 어려운 비율 (y9.2a 2 seed 이상, 분모 = 그 구간 val Scratch)")
for key, nm in (("Ldef", "불량 성분"), ("Lridge", "능선 성분")):
    for lo, hi in ((0, 10), (10, 15), (15, 999)):
        m = (F[key] >= lo) & (F[key] < hi)
        if m.sum(): print(f"  {nm} L {lo}~{min(hi, 999) - 1 if hi < 999 else '+'}: {int((hard & m).sum())}/{int(m.sum())} = {(hard & m).sum() / m.sum():.2f}")
for nm, m in (("die<1200", F["dies"] < 1200), ("die≥1200", F["dies"] >= 1200)):
    print(f"  {nm}: {int((hard & m).sum())}/{int(m.sum())} = {(hard & m).sum() / m.sum():.2f}")
for lo, hi in ((0, 0.05), (0.05, 0.10), (0.10, 1.01)):
    m = (F["frac"] >= lo) & (F["frac"] < hi)
    if m.sum(): print(f"  불량 비율 {lo:.2f}~{hi:.2f}: {int((hard & m).sum())}/{int(m.sum())} = {(hard & m).sum() / m.sum():.2f}")

print("\n어려운 Scratch 의 argmax 예측 (y9.2a 세 seed 합)")
c = np.zeros(NUM_CLASSES, int)
for x in (0, 1, 2): c += np.bincount(PRED[("y9.2a", x)][hard], minlength=NUM_CLASSES)
print("  " + " ".join(f"{CLASSES[i]} {c[i]}" for i in np.argsort(-c) if c[i]))

print("\n어려운 Scratch 목록 (P(S) 낮은 순): 전역 idx | die | 불량비율 | L불량 | L능선 | Vmax | y9.2a P(S) 평균 | 위 none | 예측 (seed0/1/2) | y9.3a 어려움 seed 수")
order = np.where(hard)[0][np.argsort(ps_a[hard])]
for j in order:
    pr = "/".join(CLASSES[PRED[('y9.2a', x)][j]] for x in (0, 1, 2))
    print(f"  {vs[j]:7d} | {int(F['dies'][j]):5d} | {F['frac'][j]:.3f} | {F['Ldef'][j]:4.0f} | {F['Lridge'][j]:4.0f} | {F['Vmax'][j]:.2f} | {ps_a[j]:.3f} | {na_a[j]:6.0f} | {pr} | {nh3[j]}")

cm = ListedColormap(["white", "#d9d9d9", "#c0392b"])
def grid(idx_list, fname, title):
    n = len(idx_list); cols = 8; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4))
    for a in np.ravel(ax): a.axis("off")
    for a, j in zip(np.ravel(ax), idx_list):
        a.imshow(img96[j], cmap=cm, vmin=0, vmax=2, interpolation="nearest")
        a.set_title(f"{vs[j]} d{int(F['dies'][j])}\nL{F['Ldef'][j]:.0f}/{F['Lridge'][j]:.0f} P{ps_a[j]:.2f}", fontsize=7)
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(os.path.join(HERE, fname), dpi=100); plt.close(fig)

grid(order, "hard_scratch.png", "Hard val Scratch (y9.2a, below recall-0.7 threshold in >=2 seeds). title: idx, dies, L defect/ridge, mean P(S)")
rng = np.random.default_rng(0); e = np.where(easy)[0]; e = e[np.argsort(-ps_a[e])]
grid(rng.choice(e, min(24, len(e)), replace=False), "easy_scratch.png", "Easy val Scratch (above threshold in all 3 seeds), random 24")
