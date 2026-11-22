"""모양 특징 EDA: Scratch vs Loc / Edge-Loc / none. 단일 특징 = train, 조합 (부스팅) = train 학습 → val 평가.
+ ver8 이 test 에서 Scratch→Loc 로 틀린 웨이퍼의 모양 + 클래스별 평균 불량 지도 (위치 쏠림)."""
import os, sys
import numpy as np, torch
proj, hdir, outd = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(proj); sys.path.insert(0, proj); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_a = sys.argv; sys.argv = [_a[0], proj, hdir]
import hough2 as h2
sys.argv = _a

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})
COL = {"Scratch": "#eb6834", "Loc": "#2a78d6", "Edge-Loc": "#1baf7a", "none": "#b9b8b1"}
H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
N = len(ds.labels)
zs = np.load(os.path.join(hdir, "shape_feats.npz")); zh = np.load(os.path.join(hdir, "hough2_feats.npz"))
S = np.full((N, zs["f"].shape[1]), np.nan); S[zs["idx"]] = zs["f"]; SN = list(zs["names"])
Hf = np.full((N, 6), np.nan); Hf[zh["idx"]] = zh["f"]
HX = lambda i: np.c_[Hf[i, :4], np.log(Hf[i, 4])]
SX = lambda i: np.c_[S[i, :10], S[i, 10:]]
C = {c: CLASSES.index(c) for c in CLASSES}
SC = C["Scratch"]
KO = {"n_main": "주 덩어리 크기", "conc": "한곳 집중도", "elong": "길쭉함", "length": "길이/지름", "width": "평균 폭",
      "width_cv": "폭 변동", "fill": "끊김 없음", "sagitta": "휘어짐(die)", "curv_gain": "곡선 맞춤 이득", "rms_perp": "직선에서 벗어남",
      "r_center": "중심→테두리 위치", "edge_frac": "테두리 접촉 비율"}

ok = lambda i: i[~np.isnan(S[i, 0]) & ~np.isnan(Hf[i, :5]).any(1)]
tr, va, te = ok(ds.train), ok(ds.val), ok(ds.test)
ytr, yva = ds.labels[tr], ds.labels[va]
print(f"train {len(tr)} / val {len(va)} (불량 3개 미만 제외)")

print("\n=== 클래스별 중앙값 (train) ===")
show_c = ["Scratch", "Loc", "Edge-Loc", "none", "Center", "Donut", "Random"]
print(f"{'특징':16s}" + "".join(f"{c:>10s}" for c in show_c))
for j, n in enumerate(SN):
    print(f"{KO[n]:16s}" + "".join(f"{np.nanmedian(S[tr[ytr == C[c]], j]):10.3f}" for c in show_c))

print("\n=== 단일 특징 AUC (train). 0.5 에서 멀수록 구분됨, <0.5 는 Scratch 쪽이 작음 ===")
pairs = ["Loc", "Edge-Loc", "none"]
print(f"{'특징':16s}" + "".join(f"{'vs ' + p:>12s}" for p in pairs))
A1 = {}
for j, n in enumerate(SN):
    row = []
    for p in pairs:
        g = (ytr == SC) | (ytr == C[p])
        row.append(roc_auc_score(ytr[g] == SC, S[tr[g], j]))
    A1[n] = row
    print(f"{KO[n]:16s}" + "".join(f"{v:12.3f}" for v in row))


def pair_gb(p, X):
    g = (ytr == SC) | (ytr == C[p]); gv = (yva == SC) | (yva == C[p])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X(tr[g]), ytr[g] == SC)
    return m, roc_auc_score(yva[gv] == SC, m.predict_proba(X(va[gv]))[:, 1])


BX = lambda i: np.c_[HX(i), SX(i)]
print("\n=== 조합 (부스팅, 둘씩 짝지어 학습, val AUC) ===")
A2, models = {}, {}
for p in pairs:
    res = {}
    for name, X in (("Hough", HX), ("모양", SX), ("Hough+모양", BX)):
        m, a = pair_gb(p, X); res[name] = a
        if name == "Hough+모양":
            models[p] = m
    A2[p] = res
    print(f"Scratch vs {p:9s} (val Scratch {(yva == SC).sum()}장, {p} {(yva == C[p]).sum()}장): " + " | ".join(f"{k} {v:.3f}" for k, v in res.items()))

# --- ver8 이 test 에서 Scratch→Loc 로 틀린 것: 모양은 Scratch 같은가 Loc 같은가
@torch.no_grad()
def predict(seed, idx):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m.eval(); o = []
    for i in range(0, len(idx), 512):
        o.append(m(to_input(ds.fetch(idx[i:i + 512], "cpu"))).argmax(1))
    return torch.cat(o).numpy()


yte = ds.labels[te]
sub = te[(yte == SC) | (yte == C["Loc"])]; ysub = ds.labels[sub]
score = models["Loc"].predict_proba(BX(sub))[:, 1]
err_ids = []
print("\n=== ver8 test: Scratch vs Loc 짝 모델 (Hough+모양) 의 Scratch 확률 ===")
for seed in (0, 1):
    p = predict(seed, sub)
    G = {"Scratch 맞힘": (ysub == SC) & (p == SC), "Scratch→Loc 틀림": (ysub == SC) & (p == C["Loc"]),
         "Loc 맞힘": (ysub == C["Loc"]) & (p == C["Loc"]), "Loc→Scratch 틀림": (ysub == C["Loc"]) & (p == SC)}
    print(f"[seed {seed}] " + " | ".join(f"{k} {g.sum()}장 중앙값 {np.median(score[g]) if g.any() else np.nan:.2f}" for k, g in G.items()))
    e, l = G["Scratch→Loc 틀림"], G["Loc 맞힘"]
    print(f"   틀린 Scratch vs 맞힌 Loc AUC {roc_auc_score(np.r_[np.ones(e.sum()), np.zeros(l.sum())], np.r_[score[e], score[l]]):.3f}"
          f" | 틀린 Scratch 중 확률 > 0.5: {int((score[e] > 0.5).sum())}/{e.sum()}")
    for j in ("elong", "width", "width_cv", "length"):
        k = SN.index(j)
        print(f"   {KO[j]}: 틀린 Scratch {np.median(S[sub[e], k]):.2f} / 맞힌 Scratch {np.median(S[sub[G['Scratch 맞힘']], k]):.2f} / 맞힌 Loc {np.median(S[sub[l], k]):.2f}")
    err_ids += list(sub[e])
err_ids = list(dict.fromkeys(err_ids))
sc_of = dict(zip(sub, score))

# ===== 그림 6: 주요 특징 분포
feat_show = ["elong", "length", "width", "width_cv", "fill", "sagitta", "r_center", "edge_frac"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
cls4 = ["Scratch", "Loc", "Edge-Loc", "none"]
for ax, n in zip(axes.ravel(), feat_show):
    k = SN.index(n)
    data = [S[tr[ytr == C[c]], k] for c in cls4]
    data = [d[~np.isnan(d)] for d in data]
    bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True, medianprops=dict(color="#0b0b0b", lw=1.5))
    for b, c in zip(bp["boxes"], cls4):
        b.set_facecolor(COL[c]); b.set_edgecolor("none")
    ax.set_xticks(range(1, 5), cls4, fontsize=9)
    a = A1[n]
    ax.set_title(f"{KO[n]}\nAUC vs Loc {a[0]:.2f} · E-Loc {a[1]:.2f} · none {a[2]:.2f}", fontsize=10, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
    if n in ("elong", "sagitta", "width"):
        ax.set_yscale("log")
fig.suptitle("주 불량 덩어리의 모양 특징 (train, 원본 die 해상도, 외톨이 제거 후). AUC: Scratch 가 클수록 1, 작을수록 0, 0.5 = 구분 못함",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig6_shape_feats.png"), dpi=130); plt.close(fig)

# ===== 그림 7: 클래스별 평균 불량 지도 (96x96, train) — 특정 위치 쏠림?
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
mean_none = None
for ax, c in zip(axes, ["Scratch", "Loc", "Edge-Loc", "none"]):
    ids = tr[ytr == C[c]][:20000]
    acc = np.zeros((96, 96)); wf = np.zeros((96, 96))
    for i in range(0, len(ids), 2000):
        x = ds.fetch(ids[i:i + 2000], "cpu")[:, 0].numpy()
        acc += (x == 2).sum(0); wf += (x > 0).sum(0)
    m = acc / np.maximum(wf, 1)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=np.percentile(m[wf > 0], 99.5))
    ax.set_title(f"{c} ({len(ids):,}장): die 위치별 불량 비율", fontsize=10, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("클래스별 평균 불량 지도 (train, 96×96). 한 자리에 진하게 몰리면 장비 고정 위치 흔적일 수 있음", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig7_mean_maps.png"), dpi=130); plt.close(fig)

# ===== 그림 8: ver8 이 Loc 으로 틀린 Scratch (test) + 맞힌 Loc 비교
rng = np.random.default_rng(0)
loc_ok = sub[(ysub == C["Loc"])]
pick_err = list(rng.choice(err_ids, min(12, len(err_ids)), replace=False))
pick_loc = list(rng.choice(loc_ok, 6, replace=False))
cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
fig, axes = plt.subplots(3, 6, figsize=(15, 8.4))
for ax, i, tag in zip(axes.ravel(), pick_err + pick_loc, ["Scratch→Loc 오답"] * len(pick_err) + ["실제 Loc"] * 6):
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    nat = h2.native(img, *dies[i, :2])
    ax.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    f = S[i]
    ax.set_title(f"{tag} · {dies[i, 2]} die\n길쭉함 {f[SN.index('elong')]:.1f} · 폭 {f[SN.index('width')]:.1f} · 모양 점수 {sc_of[i]:.2f}",
                 fontsize=8.5, color=COL["Scratch"] if "Scratch" in tag else COL["Loc"])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
for ax in axes.ravel()[len(pick_err) + 6:]:
    ax.axis("off")
fig.suptitle("ver8 이 Loc 으로 틀린 Scratch (위 두 줄, test, seed 0·1) vs 실제 Loc (아래). 모양 점수 = Hough+모양 짝 모델의 Scratch 확률",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(outd, "fig8_scratch_as_loc.png"), dpi=130); plt.close(fig)
print("\n그림:", outd)
