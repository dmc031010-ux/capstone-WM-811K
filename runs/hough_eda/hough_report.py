"""Hough 특징 분석 + 회의용 그림. 통계 = train split, 조합 특징은 train 으로 맞추고 val 로 평가."""
import os, sys
import numpy as np
proj, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY, ACC, INK2 = "#b9b8b1", "#eb6834", "#52514e"

H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
z = np.load(os.path.join(out_dir, "hough2_feats.npz"))
F = np.full((len(ds.labels), 6), np.nan); F[z["idx"]] = z["f"]
names = list(z["names"])
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
LABEL = {"r96": "기존 (96×96)", "rnat": "원본 die 해상도", "rden": "원본 + 외톨이 불량 제거",
         "length": "선 길이 / 지름", "thick": "선 굵기", "comb": "조합 (rden+길이+굵기)"}
SIGN = {"thick": -1}   # 굵기는 Scratch 가 작을수록 → 부호 반대로 AUC


def split(idx):
    f = F[idx]; ok = ~np.isnan(f[:, :5]).any(1)
    return idx[ok], f[ok]


tr, ftr = split(ds.train); va, fva = split(ds.val)
print(f"train {len(ds.train)} 중 불량 0 개 등 제외 후 {len(tr)} / val {len(va)}")
ytr, yva = ds.labels[tr], ds.labels[va]

def X(f):
    return np.c_[f[:, 2], f[:, 3], np.log(f[:, 4])]
clf = LogisticRegression(max_iter=1000).fit((X(ftr) - X(ftr).mean(0)) / X(ftr).std(0), ytr == SC)
mu, sd = X(ftr).mean(0), X(ftr).std(0)
comb_tr = clf.decision_function((X(ftr) - mu) / sd); comb_va = clf.decision_function((X(fva) - mu) / sd)
print("조합 계수 (rden, length, log thick):", clf.coef_.round(2))

feat_tr = {n: SIGN.get(n, 1) * ftr[:, i] for i, n in enumerate(names[:5])}; feat_tr["comb"] = comb_tr
feat_va = {n: SIGN.get(n, 1) * fva[:, i] for i, n in enumerate(names[:5])}; feat_va["comb"] = comb_va
small_tr, small_va = dies[tr] < 1200, dies[va] < 1200


def auc(y, a, k, g=None):
    g = np.ones(len(y), bool) if g is None else g
    m = g & ((y == SC) | (np.ones_like(y, bool) if k is None else (y == k)))
    return roc_auc_score(y[m] == SC, a[m])

# --- 표 1: 클래스별 중앙값
print("\n클래스별 중앙값 (train):")
print(f"{'class':10s} " + " ".join(f"{n:>8s}" for n in names[:5]))
for k, c in enumerate(CLASSES):
    g = ytr == k
    print(f"{c:10s} " + " ".join(f"{np.median(ftr[g, i]):8.3f}" for i in range(5)))

# --- 표 2: AUC
others = [c for c in CLASSES if c != "Scratch"]
print("\nScratch AUC (train; 'val' 열 = 조합 특징을 val 에서 평가):")
hdr = ["전체"] + others + ["작은:none", "큰:none"]
print(f"{'feature':8s} " + " ".join(f"{h:>9s}" for h in hdr))
AUC = {}
for n, a in feat_tr.items():
    row = [auc(ytr, a, None)] + [auc(ytr, a, CLASSES.index(c)) for c in others] + \
          [auc(ytr, a, NO, small_tr), auc(ytr, a, NO, ~small_tr)]
    AUC[n] = row
    print(f"{n:8s} " + " ".join(f"{v:9.3f}" for v in row))
vrow = [auc(yva, comb_va, None), auc(yva, comb_va, NO), auc(yva, comb_va, NO, small_va), auc(yva, comb_va, NO, ~small_va)]
print(f"comb@val 전체 {vrow[0]:.3f} vs none {vrow[1]:.3f} 작은:none {vrow[2]:.3f} 큰:none {vrow[3]:.3f}")
from sklearn.ensemble import HistGradientBoostingClassifier
gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(np.c_[ftr[:, :4], np.log(ftr[:, 4])], ytr == SC)
gva = gb.predict_proba(np.c_[fva[:, :4], np.log(fva[:, 4])])[:, 1]
print(f"부스팅 (Hough 5개, train 학습) @val: 전체 {auc(yva, gva, None):.3f} vs none {auc(yva, gva, NO):.3f} vs Loc {auc(yva, gva, CLASSES.index('Loc')):.3f}"
      f" vs Edge-Loc {auc(yva, gva, CLASSES.index('Edge-Loc')):.3f} 작은:none {auc(yva, gva, NO, small_va):.3f} 큰:none {auc(yva, gva, NO, ~small_va):.3f}"
      f" | val Scratch {(yva == SC).sum()}장")
AUC["gb"] = [auc(yva, gva, None)] + [auc(yva, gva, CLASSES.index(c)) for c in others] + \
            [auc(yva, gva, NO, small_va), auc(yva, gva, NO, ~small_va)]
LABEL["gb"] = "5개 조합 (부스팅, val 평가)"
for n in ("r96", "rden", "comb"):
    print(f"{n} Mann-Whitney (Scratch > 나머지) p = {mannwhitneyu(feat_tr[n][ytr == SC], feat_tr[n][ytr != SC], alternative='greater').pvalue:.1e}")
th = ftr[:, 5]
for nm, g in (("none", ytr == NO), ("Scratch", ytr == SC)):
    print(f"최고 직선 각도 {nm}: 0/90 도 비율 {np.isin(th[g], [0, 90]).mean():.3f} (각도가 고르면 0.011)")

# ===== 그림 1: 클래스별 분포 (기존 vs 개선)
order = sorted(range(len(CLASSES)), key=lambda k: np.median(ftr[ytr == k, 2]))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
for ax, i, title in ((axes[0], 0, "기존: 96×96 지도의 ratio"), (axes[1], 2, "변형: 원본 die 해상도 + 외톨이 불량 제거 (none 과 오히려 더 겹침)")):
    data = [ftr[ytr == k, i] for k in order]
    bp = ax.boxplot(data, orientation="horizontal", showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color="#0b0b0b", lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for b, k in zip(bp["boxes"], order):
        b.set_facecolor(ACC if k == SC else GRAY); b.set_edgecolor("none")
    ax.set_yticks(range(1, len(order) + 1), [f"{CLASSES[k]} ({(ytr == k).sum():,})" for k in order])
    for t, k in zip(ax.get_yticklabels(), order):
        if k == SC: t.set_color(ACC); t.set_fontweight("bold")
    ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel("가장 강한 직선 위 불량 수 ÷ 전체 불량 수")
    ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    a = AUC["r96" if i == 0 else "rden"]
    ax.text(0.99, 0.02, f"Scratch vs 나머지 AUC {a[0]:.3f}\nScratch vs none AUC {a[1 + others.index('none')]:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=9)
fig.suptitle("Hough 직선 집중도: 클래스별 분포 (train 138k장, 상자 = 25~75%, 수염 = 1.5 IQR)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig1_class_dist.png"), dpi=150); plt.close(fig)

# ===== 그림 2: Scratch vs 각 클래스 AUC (점 그림)
show = ["r96", "rnat", "rden", "length", "gb"]
cols = ["전체"] + others
fig, ax = plt.subplots(figsize=(9, 5.2))
for j, n in enumerate(show):
    ys = np.arange(len(cols)) + (j - 2) * 0.13
    ax.scatter(AUC[n][:len(cols)], ys, s=36, color=SER[j], label=LABEL[n], zorder=3, edgecolor="#fcfcfb", linewidth=1)
ax.set_yticks(range(len(cols)), [f"vs {c}" for c in cols]); ax.invert_yaxis()
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3))); ax.text(0.505, len(cols) - 0.45, "0.5 = 구분 못함", color=INK2, fontsize=8)
ax.set_xlim(0.3, 1.02);ax.set_xlabel("AUC (Scratch 가 더 크게 나올 확률)")
ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.set_title("Scratch 를 다른 클래스와 얼마나 구분하나 (단일 특징 = train, 조합 = val)", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_auc.png"), dpi=150); plt.close(fig)

# ===== 그림 3: 작은 / 큰 웨이퍼, Scratch vs none
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.15
for j, n in enumerate(show):
    vals = AUC[n][-2:]
    bars = ax.bar(np.arange(2) + (j - 2) * (w + 0.01), vals, w, color=SER[j], label=LABEL[n])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
ax.set_xticks(range(2), [f"작은 웨이퍼 (die<1200)\nScratch {((ytr == SC) & small_tr).sum()}장", f"큰 웨이퍼\nScratch {((ytr == SC) & ~small_tr).sum()}장"])
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC: Scratch vs none")
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
ax.set_title("웨이퍼 크기별 Scratch vs none 구분력 (단일 특징 = train, 조합 = val)", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_size.png"), dpi=150); plt.close(fig)

# ===== 그림 4: 예시 웨이퍼 + 찾은 직선 (원본 해상도, 외톨이 제거 후)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hough2 as h2
sd_hw = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, :2]
rng = np.random.default_rng(0)
picks = []
for c, sm in (("Scratch", True), ("Scratch", False), ("none", True), ("none", False), ("Loc", True), ("Edge-Ring", False)):
    g = np.nonzero((ytr == CLASSES.index(c)) & (small_tr == sm))[0]
    picks += [(c, sm, tr[i]) for i in rng.choice(g, 3, replace=False)]
fig, axes = plt.subplots(3, 6, figsize=(15, 8))
cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
for j, (c, sm, i) in enumerate(picks):
    ax = axes[j % 3, j // 3]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    nat = h2.denoise(h2.native(img, *sd_hw[i]))
    ax.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    v, n, t, dist, along = h2.best_line(nat)
    hh, ww = nat.shape
    on = np.abs(dist) <= 0.5
    yy, xx = np.nonzero(nat == 2)
    ax.scatter(xx[on], yy[on], s=max(4, 900 / max(hh, ww)), color=ACC, marker="s", linewidths=0)
    f = F[i]
    ax.set_title(f"{c} · {'작은' if sm else '큰'} ({dies[i]} die)\nratio {f[2]:.2f} · 길이 {f[3]:.2f} · 굵기 {f[4]:.1f}", fontsize=8.5,
                 color=ACC if c == "Scratch" else "#0b0b0b")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("예시: 외톨이 불량을 지운 원본 die 지도 (회색 = 정상 die, 검정 = 불량, 주황 = 가장 강한 직선 위 불량)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig4_examples.png"), dpi=130); plt.close(fig)
print("그림 저장:", out_dir)
