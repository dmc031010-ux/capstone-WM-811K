"""ver8 (seed 0·1) 모델 EDA — 학습 없음.
[1] test Scratch 120장: 맞힘/틀림을 가르는 모양 특징 (선 픽셀 굵기, die 굵기, 길쭉함, 길이, 집중도, 배경 불량 밀도, die 수, Hough 비율)
    단일 특징 AUC + 로지스틱 회귀 (표준화 계수, 교차검증 AUC), 굵기 구간별 재현율
[2] ver8 seed 0 의 분류기 직전 특징 (512) t-SNE: 클래스 배치 + Scratch 맞힘/틀림 위치
[3] 특징 공간 이웃: 틀린 Scratch 의 가장 가까운 학습 라벨 웨이퍼 10장의 클래스 구성
"""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
proj, hdir, outd = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(proj); sys.path.insert(0, proj); os.makedirs(outd, exist_ok=True)
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 10.5})
dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC, NO, LO = CLASSES.index("Scratch"), CLASSES.index("none"), CLASSES.index("Loc")
txt = []
def say(s=""):
    print(s); txt.append(s)


def load(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    return m.to(dev).eval()


@torch.no_grad()
def run(m, idx):
    feat = {}
    h = m.avgpool.register_forward_hook(lambda mod, i, o: feat.__setitem__("f", o.flatten(1)))
    P, Fs = [], []
    for i in range(0, len(idx), 1024):
        P.append(torch.softmax(m(to_input(ds.fetch(idx[i:i + 1024], dev))), 1).cpu()); Fs.append(feat["f"].cpu())
    h.remove()
    return torch.cat(P).numpy(), torch.cat(Fs).numpy()


M = {s: load(s) for s in (0, 1)}
te = ds.test; yte = ds.labels[te]
lab = ds.labeled_subset(0.05, seed=0); ylab = ds.labels[lab]
Pte, Fte = {}, {}
for s in (0, 1):
    Pte[s], Fte[s] = run(M[s], te)
_, Flab = run(M[0], lab)

# ===== [1] Scratch 맞힘/틀림을 가르는 특징
zs = np.load(os.path.join(hdir, "shape_feats.npz")); zh = np.load(os.path.join(hdir, "hough2_feats.npz"))
S = np.full((len(ds.labels), zs["f"].shape[1]), np.nan); S[zs["idx"]] = zs["f"]; SN = list(zs["names"])
Hh = np.full((len(ds.labels), 6), np.nan); Hh[zh["idx"]] = zh["f"]
sc = te[yte == SC]
h_, w_ = sd[sc, 0].astype(float), sd[sc, 1].astype(float)
px_per_die = np.where(np.maximum(h_, w_) <= 96, 96 / np.sqrt(h_ * w_), 1.0)   # 큰 격자는 특징이 96 지도에서 계산됨 (이미 픽셀 단위)
X96 = ds.fetch(sc, "cpu")[:, 0].numpy()
dens = (X96 == 2).sum((1, 2)) / np.maximum((X96 > 0).sum((1, 2)), 1)
F = {"선 폭 (픽셀)": S[sc, SN.index("width")] * px_per_die, "선 폭 (die)": S[sc, SN.index("width")], "길쭉함": S[sc, SN.index("elong")],
     "길이/지름": S[sc, SN.index("length")], "주 덩어리 집중도": S[sc, SN.index("conc")], "불량 밀도 (웨이퍼 전체)": dens,
     "log die 수": np.log(sd[sc, 2].astype(float)), "Hough 직선 집중도": Hh[sc, 0], "die 1개 픽셀 크기": px_per_die}
ok = ~np.isnan(np.c_[[v for v in F.values()]].T).any(1)
say(f"== [1] test Scratch {len(sc)}장 (특징 계산 가능 {ok.sum()}장): 맞힘 vs 틀림 (seed 0 / seed 1) ==")
hit = {s: Pte[s][yte == SC].argmax(1) == SC for s in (0, 1)}
say(f"맞힘: seed 0 {hit[0].sum()} / seed 1 {hit[1].sum()} · 둘 다 맞힘 {(hit[0] & hit[1]).sum()} · 둘 다 틀림 {(~hit[0] & ~hit[1]).sum()}")
say(f"{'특징':22s} {'맞힘 중앙값':>10s} {'틀림 중앙값':>10s} {'AUC s0':>7s} {'AUC s1':>7s} {'p (s0)':>9s}")
AUC = {}
for k, v in F.items():
    a = [roc_auc_score(hit[s][ok], v[ok]) for s in (0, 1)]; AUC[k] = a
    p = mannwhitneyu(v[ok & hit[0]], v[ok & ~hit[0]]).pvalue
    say(f"{k:22s} {np.median(v[ok & hit[0]]):10.3f} {np.median(v[ok & ~hit[0]]):10.3f} {a[0]:7.2f} {a[1]:7.2f} {p:9.1e}")
say("  (AUC: 값이 클수록 '맞힐' 가능성이 높음. 0.5 = 무관, 0.5 미만 = 값이 작을수록 맞힘)")
Xf = np.c_[[v for v in F.values()]].T[ok]; Xf = (Xf - Xf.mean(0)) / Xf.std(0)
for s in (0, 1):
    clf = LogisticRegression(max_iter=2000, C=0.5)
    cvp = cross_val_predict(clf, Xf, hit[s][ok], cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    clf.fit(Xf, hit[s][ok])
    say(f"로지스틱 (seed {s}): 5겹 교차검증 AUC {roc_auc_score(hit[s][ok], cvp):.2f} · 표준화 계수 " +
        ", ".join(f"{k} {c:+.2f}" for k, c in sorted(zip(F, clf.coef_[0]), key=lambda t: -abs(t[1]))))

# 굵기 구간별 재현율 그림
wpx = F["선 폭 (픽셀)"]
bins = [0, 2, 3, 4, 6, 100]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ax = axes[0]
for s, c in ((0, "#2a78d6"), (1, "#eb6834")):
    r, n = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = ok & (wpx >= lo) & (wpx < hi); r.append(hit[s][g].mean() if g.any() else np.nan); n.append(g.sum())
    ax.plot(range(len(r)), r, "-o", color=c, lw=2, label=f"ver8 seed {s}")
ax.set_xticks(range(len(n)), [f"{lo}~{hi}px\n({k}장)" if hi < 100 else f"{lo}px 이상\n({k}장)" for lo, hi, k in zip(bins[:-1], bins[1:], n)])
ax.set_ylim(0, 1); ax.set_ylabel("Scratch 재현율"); ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False)
ax.set_title("주 선의 평균 폭 (96×96 입력 픽셀) 별 재현율", loc="left", fontsize=11.5)
ax = axes[1]
for cond, mk, col, lab_ in ((hit[0] & hit[1], "o", "#1a7f37", "두 seed 모두 맞힘"), (hit[0] ^ hit[1], "s", "#eda100", "seed 하나만 맞힘"),
                             (~hit[0] & ~hit[1], "x", "#d03b3b", "두 seed 모두 틀림")):
    g = ok & cond
    ax.scatter(wpx[g], F["길쭉함"][g], marker=mk, s=34, color=col, label=f"{lab_} ({g.sum()})", linewidths=1.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("주 선의 평균 폭 (픽셀, 로그 축)"); ax.set_ylabel("길쭉함 (로그 축)")
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=9); ax.set_title("test Scratch 120장: 굵기 × 길쭉함", loc="left", fontsize=11.5)
fig.suptitle("ver8 의 Scratch 재현율: 선이 6픽셀 이상으로 굵어지면 크게 떨어짐 (가늘수록 대체로 잘 맞힘)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "1_scratch_hit_by_shape.png"), dpi=150); plt.close(fig)

# ===== [2] t-SNE (seed 0 특징)
rng = np.random.default_rng(0)
sel = np.r_[np.nonzero(yte != NO)[0], rng.choice(np.nonzero(yte == NO)[0], 3000, replace=False)]
Z = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(Fn.normalize(torch.from_numpy(Fte[0][sel]), dim=1).numpy())
ys = yte[sel]; p0 = Pte[0][sel].argmax(1)
PAL = ["#2a78d6", "#e87ba4", "#1baf7a", "#6250d6", "#eda100", "#008300", "#eb6834", "#e34948", "#c9c8c1"]
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
for k in [NO] + [k for k in range(NUM_CLASSES) if k != NO]:
    g = ys == k
    axes[0].scatter(Z[g, 0], Z[g, 1], s=5 if k == NO else 7, color=PAL[k], label=f"{CLASSES[k]} ({g.sum()})", alpha=0.6 if k == NO else 0.85, linewidths=0)
axes[0].legend(frameon=False, fontsize=8.5, markerscale=2.5, loc="upper left", bbox_to_anchor=(1, 1)); axes[0].set_xticks([]); axes[0].set_yticks([])
axes[0].set_title("정답 클래스로 색칠 (none 은 3,000장만 표시)", loc="left", fontsize=11.5)
ax = axes[1]
ax.scatter(Z[:, 0], Z[:, 1], s=4, color="#dcdbd5", linewidths=0)
for k, col, nm in ((NO, "#9a9992", "none"), (LO, "#2a78d6", "Loc")):
    g = ys == k; ax.scatter(Z[g, 0], Z[g, 1], s=5, color=col, alpha=0.35, linewidths=0, label=f"{nm} (정답)")
g = (ys == SC) & (p0 == SC); ax.scatter(Z[g, 0], Z[g, 1], s=46, marker="o", color="#1a7f37", edgecolor=BG, label=f"Scratch 맞힘 ({g.sum()})")
for k, mk, nm in ((NO, "X", "Scratch → none"), (LO, "P", "Scratch → Loc")):
    g = (ys == SC) & (p0 == k); ax.scatter(Z[g, 0], Z[g, 1], s=60, marker=mk, color="#d03b3b", edgecolor=BG, label=f"{nm} ({g.sum()})")
g = (ys == SC) & (p0 != SC) & (p0 != NO) & (p0 != LO); ax.scatter(Z[g, 0], Z[g, 1], s=50, marker="D", color="#eda100", edgecolor=BG, label=f"Scratch → 기타 ({g.sum()})")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1, 1)); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("test Scratch 120장의 위치 (ver8 seed 0 예측)", loc="left", fontsize=11.5)
for a_ in axes:
    for s_ in a_.spines.values(): s_.set_visible(False)
fig.suptitle("ver8 seed 0 의 분류기 직전 특징 (512차원) 을 t-SNE 로 2차원에 펼침 · test", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "2_tsne.png"), dpi=140); plt.close(fig)

# ===== [3] 특징 공간 이웃 (학습 라벨 6,919장 중 가장 가까운 10장)
fl = Fn.normalize(torch.from_numpy(Flab), dim=1); ft = Fn.normalize(torch.from_numpy(Fte[0][yte == SC]), dim=1)
nn10 = (ft @ fl.T).topk(10, dim=1).indices.numpy(); nb = ylab[nn10]
p_sc = Pte[0][yte == SC].argmax(1)
say("\n== [3] test Scratch 의 가장 가까운 학습 라벨 웨이퍼 10장 (ver8 seed 0 특징, 코사인) ==")
for nm, g in (("맞힘", p_sc == SC), ("→ none 틀림", p_sc == NO), ("→ Loc 틀림", p_sc == LO)):
    comp = {CLASSES[k]: (nb[g] == k).mean() for k in range(NUM_CLASSES)}
    say(f"  {nm:10s} {g.sum():3d}장 | 이웃 중 " + " · ".join(f"{k} {v:.0%}" for k, v in sorted(comp.items(), key=lambda t: -t[1]) if v >= 0.02))
say(f"  학습 라벨 Scratch 는 6,919장 중 {(ylab == SC).sum()}장 ({(ylab == SC).mean():.1%})")
open(os.path.join(outd, "numbers.txt"), "w", encoding="utf-8").write("\n".join(txt))
print("저장:", outd)

