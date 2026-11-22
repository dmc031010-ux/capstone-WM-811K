"""ResNet-34 ImageNet 초기화 미팅 자료 (새 학습 없음: 저장된 history + best.pt 로 분석)."""
import json, os, sys
import numpy as np, torch
proj, outd = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj); os.makedirs(outd, exist_ok=True)
from sklearn.metrics import confusion_matrix, f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 10.5})
RUNS = [("R34 ImageNet 초기화", "resnet34", "runs/resnet34/p0.05/seed0/supervised_imagenet", "#eb6834"),
        ("R34 무작위 초기화", "resnet34", "runs/resnet34/p0.05/seed0/supervised", "#2a78d6"),
        ("R18 무작위 초기화", "resnet18", "runs/resnet18/p0.05/seed0/supervised", "#9a9992")]
M = {n: json.load(open(os.path.join(p, "metrics.json"), encoding="utf-8")) for n, _, p, _ in RUNS}
V8 = [json.load(open(f"runs/resnet18/p0.05/seed{s}/ver8_ssl50_ft30/metrics.json", encoding="utf-8")) for s in (0, 1)]
txt = []
def say(s=""):
    print(s); txt.append(s)

# ---------- 1. 학습 곡선 (val macro, 학습 손실)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
say("== 학습 과정 (val macro F1, 100 epoch, lr 0.01 고정, seed 0) ==")
for n, _, _, c in RUNS:
    h = M[n]["history"]; e = np.array([r["epoch"] for r in h]); v = np.array([r["val_macro_f1"] for r in h]); l = np.array([r["loss"] for r in h])
    sm = np.convolve(v, np.ones(5) / 5, mode="same")
    axes[0].plot(e, v, color=c, lw=0.8, alpha=0.35); axes[0].plot(e[2:-2], sm[2:-2], color=c, lw=2.2, label=n)
    be = M[n]["best_epoch"]; axes[0].scatter([be], [v[be - 1]], color=c, s=40, zorder=4, edgecolor=BG)
    axes[1].plot(e, l, color=c, lw=2, label=n)
    first = next((int(x) for x, y in zip(e, v) if y >= 0.75), None)
    say(f"{n:14s} 처음 val ≥ 0.75: epoch {first} · 최고 val {v.max():.3f} (epoch {be}) · 마지막 30 epoch val 평균 {v[-30:].mean():.3f} 표준편차 {v[-30:].std():.3f}"
        f" · 1 epoch 손실 {l[0]:.2f} · 마지막 손실 {l[-1]:.3f}")
axes[0].axhline(0.75, color=INK2, lw=0.8, ls=(0, (3, 3))); axes[0].text(101, 0.75, "0.75", va="center", fontsize=8.5, color=INK2)
axes[0].set_ylim(0.3, 0.85); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val macro F1")
axes[0].grid(color=GRID, lw=0.8); axes[0].legend(frameon=False, loc="lower right"); axes[0].set_title("val macro F1 · 굵은 선 = 5 epoch 이동평균, 점 = 최종 선택된 best epoch", loc="left", fontsize=11)
axes[1].set_yscale("log"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("학습 손실 (로그 축)"); axes[1].grid(color=GRID, lw=0.8)
axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[1].set_title("학습 손실", loc="left", fontsize=11.5)
fig.suptitle("학습 곡선: ImageNet 초기화는 빨리 올라오지만, val 은 끝까지 크게 흔들림", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "2_learning_curves.png"), dpi=150); plt.close(fig)

# ---------- 2. 예측 (test)
dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
te = ds.test; y = ds.labels[te]; small = dies[te] < 1200
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")


@torch.no_grad()
def predict(arch, path):
    m = build_model(arch, NUM_CLASSES)
    m.load_state_dict(torch.load(os.path.join(path, "best.pt"), map_location="cpu", weights_only=True))
    m = m.to(dev).eval(); o = []
    for i in range(0, len(te), 1024):
        o.append(m(to_input(ds.fetch(te[i:i + 1024], dev))).argmax(1).cpu())
    return torch.cat(o).numpy()


P = {}
for n, arch, path, _ in RUNS:
    P[n] = predict(arch, path)
    assert abs(f1_score(y, P[n], labels=range(NUM_CLASSES), average="macro", zero_division=0) - M[n]["test"]["macro_f1"]) < 1e-3

# ---------- 3. 요약표 (+ ver8 참고)
rows = []
for n, _, _, _ in RUNS:
    t, v = M[n]["test"], M[n]["val"]
    rows.append([n, "지도학습 100 ep", f"{t['macro_f1']:.3f}", f"{t['acc']:.3f}", f"{v['macro_f1']:.3f}", f"{t['per_class_f1']['Scratch']:.3f}",
                 f"{np.mean([t['per_class_f1'][c] for c in ['Near-full', 'Donut', 'Random', 'Scratch', 'Loc']]):.3f}", str(M[n]["best_epoch"])])
for s, m in enumerate(V8):
    t, v = m["test"], m["val"]
    rows.append([f"참고: ver8 seed {s} (R18)", "자기지도 사전학습\n+ fine-tune", f"{t['macro_f1']:.3f}", f"{t['acc']:.3f}", f"{v['macro_f1']:.3f}",
                 f"{t['per_class_f1']['Scratch']:.3f}", f"{np.mean([t['per_class_f1'][c] for c in ['Near-full', 'Donut', 'Random', 'Scratch', 'Loc']]):.3f}", str(m.get("best_epoch"))])
fig, ax = plt.subplots(figsize=(13, 3.9)); ax.axis("off")
cols = ["모델", "학습 방식", "test macro", "test 정확도", "val macro", "Scratch F1", "소수 5개 평균", "best epoch"]
cc = [["#fbe3d6"] * 8, [BG] * 8, [BG] * 8, ["#f0efec"] * 8, ["#f0efec"] * 8]
tb = ax.table(cellText=rows, colLabels=cols, cellColours=cc, colColours=["#f0efec"] * 8, loc="center", cellLoc="center", colWidths=[0.22, 0.15, 0.09, 0.09, 0.09, 0.09, 0.1, 0.09])
tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 2.3)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0: cell.set_text_props(fontweight="bold")
ax.set_title("결과 요약 · test 17,296장 / val 17,295장 · 라벨 5% (6,919장) · seed 0 (ver8 만 seed 0, 1)", loc="left", fontsize=12.5, fontweight="bold", pad=8)
fig.savefig(os.path.join(outd, "1_summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
say("\n== 요약 ==")
for r in rows: say(" | ".join(x.replace("\n", " ") for x in r))

# ---------- 4. 클래스별 F1: ImageNet - 무작위 (R34)
a, b = M["R34 ImageNet 초기화"]["test"]["per_class_f1"], M["R34 무작위 초기화"]["test"]["per_class_f1"]
n_cls = np.bincount(y, minlength=NUM_CLASSES)
d = np.array([a[c] - b[c] for c in CLASSES])
fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(range(NUM_CLASSES), d, color=["#eb6834" if v > 0 else "#2a78d6" for v in d], width=0.6, zorder=2)
for i, v in enumerate(d):
    ax.text(i, v + (0.006 if v >= 0 else -0.006), f"{v:+.3f}\n({b[CLASSES[i]]:.2f}→{a[CLASSES[i]]:.2f})", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(NUM_CLASSES), [f"{c}\n({k})" for c, k in zip(CLASSES, n_cls)], fontsize=9.5)
ax.set_ylabel("test F1 차이 (ImageNet - 무작위)"); ax.set_ylim(-0.22, 0.2); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_title("같은 R34 에서 초기화만 바꿨을 때 클래스별 F1 변화 (괄호 = test 장수)", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "3_per_class_gain.png"), dpi=150); plt.close(fig)
say("\n== 클래스별 F1 (test) ==")
for n, _, _, _ in RUNS:
    say(f"{n:14s} " + " ".join(f"{c} {M[n]['test']['per_class_f1'][c]:.3f}" for c in CLASSES))

# ---------- 5. die 수 구간별
say("\n== die 수 적은 (<1200) / 많은 (≥1200) 웨이퍼 (test) ==")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
for ax, (key, ttl) in zip(axes, [("macro", "macro F1"), ("hit", "Scratch 맞힌 수 (적음 61 · 많음 59장 중)"), ("fp", "none 을 불량으로 오탐한 수")]):
    for j, (n, _, _, c) in enumerate(RUNS):
        for gi, gm in enumerate((small, ~small)):
            yt, yp = y[gm], P[n][gm]
            val = dict(macro=f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0),
                       hit=int(((yt == SC) & (yp == SC)).sum()), fp=int(((yt == NO) & (yp != NO)).sum()))[key]
            xx = gi + (j - 1) * 0.27
            ax.bar(xx, val, 0.25, color=c, label=n if gi == 0 else None, zorder=2)
            ax.text(xx, val, f"{val:.3f}" if key == "macro" else f"{val}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks([0, 1], ["die 수 적음 (<1200)", "die 수 많음 (≥1200)"]); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title(ttl, loc="left", fontsize=11)
    if key == "macro": ax.set_ylim(0.6, 0.92)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("die 수 구간별 (test, seed 0)", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "4_by_die_count.png"), dpi=150); plt.close(fig)
for n, _, _, _ in RUNS:
    r_ = []
    for g, gm in (("적음", small), ("많음", ~small)):
        yt, yp = y[gm], P[n][gm]
        r_.append(f"{g}: macro {f1_score(yt, yp, labels=np.unique(yt), average='macro', zero_division=0):.3f} · Scratch {((yt == SC) & (yp == SC)).sum()}/{(yt == SC).sum()} · none 오탐 {((yt == NO) & (yp != NO)).sum()}")
    say(f"{n:14s} " + " | ".join(r_))

# ---------- 6. 혼동행렬 + 차이
blue = LinearSegmentedColormap.from_list("b", [BG, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
div = LinearSegmentedColormap.from_list("d", ["#d95926", "#f3c4ad", "#f0efec", "#9ec5f4", "#256abf"])
short = [c.replace("Edge-", "E-").replace("Near-full", "N-full") for c in CLASSES]
CM = {n: confusion_matrix(y, P[n], labels=range(NUM_CLASSES)) for n, _, _, _ in RUNS}
for n, fname in (("R34 ImageNet 초기화", "5_confusion_r34_imagenet.png"), ("R34 무작위 초기화", "5_confusion_r34_random.png")):
    cm = CM[n]; rn = cm / cm.sum(1, keepdims=True); f1 = f1_score(y, P[n], labels=range(NUM_CLASSES), average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2)); ax.imshow(rn, cmap=blue, vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9.5, fontweight="bold" if i == j else "normal", color=BG if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(NUM_CLASSES), short, rotation=40, ha="right"); ax.set_yticks(range(NUM_CLASSES), [f"{s} ({k})  F1 {v:.2f}" for s, k, v in zip(short, n_cls, f1)], fontsize=9.5)
    ax.set_xlabel("예측"); ax.set_ylabel("정답 (장수) · 클래스 F1")
    ax.set_title(f"{n} (ResNet-34, seed 0) 혼동행렬 · test macro F1 {f1.mean():.3f}\n칸 숫자 = 장수, 색 = 행(정답) 기준 비율", loc="left", fontsize=11.5)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outd, fname), dpi=150); plt.close(fig)
dcm = CM["R34 ImageNet 초기화"] - CM["R34 무작위 초기화"]
sgn = dcm.astype(float); off = ~np.eye(NUM_CLASSES, dtype=bool); sgn[off] *= -1
fig, ax = plt.subplots(figsize=(8.2, 7.2)); lim = max(np.abs(sgn).max(), 1)
ax.imshow(sgn, cmap=div, norm=TwoSlopeNorm(0, -lim, lim))
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if dcm[i, j]:
            ax.text(j, i, f"{dcm[i, j]:+d}", ha="center", va="center", fontsize=9.5, fontweight="bold" if i == j else "normal", color=INK)
ax.set_xticks(range(NUM_CLASSES), short, rotation=40, ha="right"); ax.set_yticks(range(NUM_CLASSES), [f"{s} ({k})" for s, k in zip(short, n_cls)])
ax.set_xlabel("예측"); ax.set_ylabel("정답 (장수)")
ax.set_title("혼동행렬 차이: R34 ImageNet - R34 무작위 (장수)\n파랑 = 좋아짐 (대각선 증가 / 오답 감소), 주황 = 나빠짐", loc="left", fontsize=11.5)
for sp_ in ax.spines.values(): sp_.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(outd, "5_confusion_diff.png"), dpi=150); plt.close(fig)
say("\n== 혼동행렬 주요 칸 (test) ==")
for n, _, _, _ in RUNS:
    cm = CM[n]
    say(f"{n:14s} Scratch 맞힘 {cm[SC, SC]}/120 · Scratch→none {cm[SC, NO]} · Scratch→Loc {cm[SC, CLASSES.index('Loc')]} · none 오탐 {cm[NO].sum() - cm[NO, NO]}")
open(os.path.join(outd, "numbers.txt"), "w", encoding="utf-8").write("\n".join(txt))
print("저장:", outd)


