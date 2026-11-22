"""ver8.2 (FiLM) 미팅 자료: 구조 그림, 요약표, 클래스별 F1, die 수 구간별, 사전학습 손실, 혼동행렬 (seed 0·1). PNG 따로 + 요약 txt."""
import json, os, re, sys
import numpy as np, torch
proj, outd = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj); os.makedirs(outd, exist_ok=True)
from sklearn.metrics import confusion_matrix, f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import FiLMResNet, SizeHead, build_model, load_size, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COL = {"ver8": "#9a9992", "ver8.1": "#2a78d6", "ver8.2": "#eb6834"}
NAME = {"ver8": "ver8 (기준)", "ver8.1": "ver8.1 (분류기 직전 크기)", "ver8.2": "ver8.2 (FiLM)"}
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "font.size": 10.5})
dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); size = load_size(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
te = ds.test; y = ds.labels[te]; small = dies[te] < 1200
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]
RUNS = ["ver8", "ver8.1", "ver8.2"]
txt = []
def say(s=""):
    print(s); txt.append(s)

# ---------- 데이터
M = {(r, s): json.load(open(f"runs/resnet18/p0.05/seed{s}/{r}_ssl50_ft30/metrics.json", encoding="utf-8")) for r in RUNS for s in (0, 1)}


@torch.no_grad()
def predict(run, seed):
    m = build_model("resnet18", NUM_CLASSES)
    m = SizeHead(m, size) if run == "ver8.1" else FiLMResNet(m, size) if run == "ver8.2" else m
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/{run}_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m = m.to(dev).eval(); o = []
    for i in range(0, len(te), 1024):
        j = te[i:i + 1024]; x = to_input(ds.fetch(j, dev))
        o.append((m(x, torch.as_tensor(j, device=dev)) if run != "ver8" else m(x)).argmax(1).cpu())
    return torch.cat(o).numpy()


P = {(r, s): predict(r, s) for r in RUNS for s in (0, 1)}
for k, p in P.items():
    assert abs(f1_score(y, p, labels=range(NUM_CLASSES), average="macro", zero_division=0) - M[k]["test"]["macro_f1"]) < 1e-3

# ---------- 1. FiLM 구조 그림
fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, yy, w, h, t, fc, ec, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, yy + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")
def arrow(x0, y0, x1, y1, c=INK2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, shrinkA=0, shrinkB=0))
gy = 2.55
box(0.1, gy, 1.25, 0.8, "웨이퍼 지도\n96×96", "#f0efec", INK2)
box(1.65, gy, 1.2, 0.8, "conv1·BN\nmaxpool", "#f0efec", INK2)
xs = [3.15, 5.15, 7.15, 9.15]
for i, x in enumerate(xs):
    box(x, gy, 1.05, 0.8, f"layer{i + 1}\n({[64, 128, 256, 512][i]}ch)", "#f0efec", INK2)
    box(x + 1.15, gy + 0.1, 0.6, 0.6, "×(1+γ)\n+β", "#fbe3d6", COL["ver8.2"], fs=8.5, bold=True)
    arrow(x + 1.05, gy + 0.4, x + 1.15, gy + 0.4)
    if i < 3: arrow(x + 1.75, gy + 0.4, xs[i + 1], gy + 0.4)
arrow(1.35, gy + 0.4, 1.65, gy + 0.4); arrow(2.85, gy + 0.4, 3.15, gy + 0.4)
box(11.15, gy, 1.7, 0.8, "평균 풀링\n→ 분류기 (9)", "#f0efec", INK2)
arrow(10.9, gy + 0.4, 11.15, gy + 0.4)
box(0.1, 0.45, 1.35, 0.85, "크기 s\n표준화\nlog(die 수)", "#fbe3d6", COL["ver8.2"], fs=9)
box(1.65, 0.45, 2.2, 0.85, "FiLM Generator\nLinear(1→16)·ReLU", "#fbe3d6", COL["ver8.2"], fs=9.5)
arrow(1.45, 0.87, 1.65, 0.87, COL["ver8.2"])
ax.plot([3.85, xs[-1] + 1.45], [0.87, 0.87], color=COL["ver8.2"], lw=1.2)
for i, x in enumerate(xs):
    arrow(x + 1.45, 0.87, x + 1.45, gy + 0.1, COL["ver8.2"])
    ax.text(x + 1.52, 1.75, f"Linear(16→{2 * [64, 128, 256, 512][i]})\n→ γ{i + 1}, β{i + 1}", ha="left", va="center", fontsize=8.3, color=COL["ver8.2"])
ax.text(0.1, 4.25, "ver8.2 · FiLM: stage 출력 h 를 h × (1 + γ) + β 로 조절 (γ, β 는 크기 s 에서 생성, 채널마다 다름)", fontsize=12.5, fontweight="bold", color=INK)
ax.text(0.1, 3.82, "γ, β 를 만드는 층은 0 으로 시작 → 학습 시작 시 일반 ResNet 과 동일. 사전학습(온라인·타깃 인코더 모두)부터 크기를 받음. 그 외 조건은 ver8 과 같음",
        fontsize=9.5, color=INK2)
fig.savefig(os.path.join(outd, "1_film_structure.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 2. 요약표
rows = [("test macro F1", lambda m: m["test"]["macro_f1"]), ("val macro F1", lambda m: m["val"]["macro_f1"]),
        ("소수 클래스 5개 평균", lambda m: np.mean([m["test"]["per_class_f1"][c] for c in MINOR])),
        ("Scratch F1", lambda m: m["test"]["per_class_f1"]["Scratch"]), ("사전학습 후 군집 NMI", lambda m: m["rounds"][0]["nmi"])]
cells, colors = [], []
say("== 요약 (test 17,296장 / seed 0, seed 1, 평균) ==")
for name, f in rows:
    r_ = [name]; c_ = ["#f0efec"]
    for run in RUNS:
        a, b = f(M[(run, 0)]), f(M[(run, 1)])
        r_.append(f"{a:.3f} / {b:.3f}\n평균 {(a + b) / 2:.3f}"); c_.append(BG)
    best = int(np.argmax([(f(M[(r, 0)]) + f(M[(r, 1)])) / 2 for r in RUNS]))
    c_[1 + best] = "#e5effb"
    cells.append(r_); colors.append(c_)
    say(f"{name:14s} " + " | ".join(f"{run} {f(M[(run, 0)]):.3f}/{f(M[(run, 1)]):.3f} (평균 {(f(M[(run, 0)]) + f(M[(run, 1)])) / 2:.3f})" for run in RUNS))
fig, ax = plt.subplots(figsize=(11, 3.9)); ax.axis("off")
t = ax.table(cellText=cells, colLabels=["", *[NAME[r] for r in RUNS]], cellColours=colors, colColours=["#f0efec"] * 4, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 2.6)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0: cell.set_text_props(fontweight="bold")
ax.set_title("ver8.2 (FiLM) 결과 요약 · 칸 = seed 0 / seed 1 · 파란 칸 = 두 seed 평균이 가장 높음", loc="left", fontsize=12.5, fontweight="bold", pad=8)
fig.savefig(os.path.join(outd, "2_summary_table.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 3. 클래스별 F1 (두 seed 평균 + seed 범위)
fig, ax = plt.subplots(figsize=(12, 4.4))
x = np.arange(NUM_CLASSES); wbar = 0.26
for j, run in enumerate(RUNS):
    v = np.array([[M[(run, s)]["test"]["per_class_f1"][c] for c in CLASSES] for s in (0, 1)])
    mean = v.mean(0)
    ax.bar(x + (j - 1) * (wbar + 0.02), mean, wbar, color=COL[run], label=NAME[run], zorder=2)
    ax.errorbar(x + (j - 1) * (wbar + 0.02), mean, yerr=[mean - v.min(0), v.max(0) - mean], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=3)
ax.set_xticks(x, CLASSES); ax.set_ylim(0.3, 1.0); ax.set_ylabel("test F1 (두 seed 평균, 막대 끝 선 = seed 범위)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
for lbl in ax.get_xticklabels():
    if lbl.get_text() == "Scratch": lbl.set_color(COL["ver8.2"]); lbl.set_fontweight("bold")
ax.set_title("클래스별 F1", loc="left", fontsize=12.5, fontweight="bold", pad=26)
fig.savefig(os.path.join(outd, "3_per_class_f1.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
say("\n== 클래스별 F1 (두 seed 평균) ==")
for run in RUNS:
    say(f"{run:7s} " + " ".join(f"{c} {np.mean([M[(run, s)]['test']['per_class_f1'][c] for s in (0, 1)]):.3f}" for c in CLASSES))

# ---------- 4. die 수 적은 / 많은 웨이퍼
say("\n== die 수 적은 (<1200, test 12,499장, Scratch 61) / 많은 (4,797장, Scratch 59) 웨이퍼 ==")
stat = {}
for run in RUNS:
    for s in (0, 1):
        p = P[(run, s)]
        for g, gm in (("적음", small), ("많음", ~small)):
            yt, yp = y[gm], p[gm]
            stat[(run, s, g)] = dict(macro=f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0),
                                     hit=int(((yt == SC) & (yp == SC)).sum()), fp=int(((yt == NO) & (yp != NO)).sum()))
        a, b = stat[(run, s, "적음")], stat[(run, s, "많음")]
        say(f"{run:7s} seed {s} | 적음: macro {a['macro']:.3f} · Scratch 맞힘 {a['hit']}/61 · none 오탐 {a['fp']} | 많음: macro {b['macro']:.3f} · Scratch 맞힘 {b['hit']}/59 · none 오탐 {b['fp']}")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (key, ttl) in zip(axes, [("macro", "macro F1"), ("hit", "Scratch 맞힌 수 (적음 61 · 많음 59장 중)"), ("fp", "none 을 불량으로 오탐한 수")]):
    for j, run in enumerate(RUNS):
        for gi, g in enumerate(("적음", "많음")):
            v = [stat[(run, s, g)][key] for s in (0, 1)]
            xx = gi + (j - 1) * 0.27
            ax.bar(xx, np.mean(v), 0.25, color=COL[run], label=NAME[run] if gi == 0 else None, zorder=2)
            ax.scatter([xx, xx], v, s=14, color=INK, zorder=3)
    ax.set_xticks([0, 1], ["die 수 적음\n(<1200)", "die 수 많음\n(≥1200)"]); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title(ttl, loc="left", fontsize=11)
    if key == "macro": ax.set_ylim(0.6, 0.9)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("die 수 구간별 (막대 = 두 seed 평균, 점 = 각 seed)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "4_by_die_count.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 5. 사전학습 손실 (seed 0·1, 마지막 50 epoch = 최종 실행)
def losses(f):
    v = [float(m.group(1)) for m in re.finditer(r"ssl epoch=\d+ loss=([\d.]+)", open(f, encoding="utf-8", errors="ignore").read())]
    return v[-50:]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, s in zip(axes, (0, 1)):
    for run in ("ver8", "ver8.2"):
        v = losses(f"runs/log_{run}_p0.05_s{s}.txt"); ax.plot(range(1, 51), v, color=COL[run], lw=2, label=NAME[run])
        ax.scatter([50], [v[-1]], color=COL[run], s=20, zorder=3)
    ax.set_yscale("log"); ax.set_xlabel("사전학습 epoch")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.grid(color=GRID, lw=0.8); ax.set_title(f"seed {s}", loc="left", fontsize=11)
axes[0].set_ylabel("일관성 손실 (1 - cos, 로그 축)"); axes[0].legend(frameon=False)
fig.suptitle("사전학습 손실: FiLM 을 넣어도 사전학습은 정상 진행 (붕괴 없음)", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(outd, "5_pretrain_loss.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
say("\n== 사전학습 마지막 손실 ==")
for run in ("ver8", "ver8.2"):
    say(f"{run}: " + " / ".join(f"seed {s} {losses(f'runs/log_{run}_p0.05_s{s}.txt')[-1]:.4f}" for s in (0, 1)))

# ---------- 6. 혼동행렬 ver8.2 seed 0·1
blue = LinearSegmentedColormap.from_list("b", [BG, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
short = [c.replace("Edge-", "E-").replace("Near-full", "N-full") for c in CLASSES]
n_cls = np.bincount(y, minlength=NUM_CLASSES)
for s in (0, 1):
    cm = confusion_matrix(y, P[("ver8.2", s)], labels=range(NUM_CLASSES)); rn = cm / cm.sum(1, keepdims=True)
    f1 = f1_score(y, P[("ver8.2", s)], labels=range(NUM_CLASSES), average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2)); ax.imshow(rn, cmap=blue, vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9.5, fontweight="bold" if i == j else "normal", color=BG if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(NUM_CLASSES), short, rotation=40, ha="right"); ax.set_yticks(range(NUM_CLASSES), [f"{a} ({n})  F1 {v:.2f}" for a, n, v in zip(short, n_cls, f1)], fontsize=9.5)
    ax.set_xlabel("예측"); ax.set_ylabel("정답 (장수) · 클래스 F1")
    ax.set_title(f"ver8.2 (FiLM) seed {s} 혼동행렬 · test macro F1 {f1.mean():.3f}\n칸 숫자 = 장수, 색 = 행(정답) 기준 비율", loc="left", fontsize=11.5)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(outd, f"6_confusion_ver8.2_seed{s}.png"), dpi=150); plt.close(fig)

open(os.path.join(outd, "numbers.txt"), "w", encoding="utf-8").write("\n".join(txt))
print("\n저장:", outd)

