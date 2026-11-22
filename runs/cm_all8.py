"""ver8 이후 모든 실험의 test 혼동행렬 (seed 별). 그림 1장 + 텍스트."""
import os, sys
import numpy as np, torch
proj, out = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj)
from sklearn.metrics import confusion_matrix, f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import ClassPool, FiLMResNet, SizeHead, build_model, load_size, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); size = load_size(CACHE)
te = ds.test; y = ds.labels[te]
RUNS = [("ver8", "기준 (ver7 과 동일)"), ("ver8.1", "분류기 직전 크기 임베딩"), ("ver8.2", "FiLM (사전학습부터)"), ("ver8.3", "클래스별 학습 풀링")]


@torch.no_grad()
def predict(run, seed):
    m = build_model("resnet18", NUM_CLASSES)
    m = SizeHead(m, size) if run == "ver8.1" else FiLMResNet(m, size) if run == "ver8.2" else ClassPool(m) if run == "ver8.3" else m
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/{run}_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m = m.to(dev).eval(); o = []
    for i in range(0, len(te), 1024):
        j = te[i:i + 1024]; x = to_input(ds.fetch(j, dev))
        o.append((m(x, torch.as_tensor(j, device=dev)) if run in ("ver8.1", "ver8.2") else m(x)).argmax(1).cpu())
    return torch.cat(o).numpy()


cells = []
for run, desc in RUNS:
    for seed in (0, 1):
        if os.path.exists(f"runs/resnet18/p0.05/seed{seed}/{run}_ssl50_ft30/best.pt"):
            p = predict(run, seed)
            cells.append((run, desc, seed, confusion_matrix(y, p, labels=range(NUM_CLASSES)), f1_score(y, p, labels=range(NUM_CLASSES), average=None, zero_division=0)))
        else:
            cells.append((run, desc, seed, None, None))

w = 9
for run, desc, seed, cm, f1 in cells:
    if cm is None:
        print(f"\n[{run} seed {seed}] 실험 없음"); continue
    print(f"\n[{run} seed {seed}] {desc} · test macro {f1.mean():.3f} · 행 = 정답, 열 = 예측")
    print(" " * 10 + "".join(f"{c[:8]:>{w}s}" for c in CLASSES) + f"{'재현율':>{w}s}{'F1':>{w}s}")
    for i, c in enumerate(CLASSES):
        print(f"{c:10s}" + "".join(f"{v:{w}d}" for v in cm[i]) + f"{cm[i, i] / cm[i].sum():{w}.3f}{f1[i]:{w}.3f}")

blue = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
short = [c.replace("Edge-", "E-").replace("Near-full", "N-full") for c in CLASSES]
n_cls = np.bincount(y, minlength=NUM_CLASSES)
odir = os.path.splitext(out)[0]; os.makedirs(odir, exist_ok=True)
for run, desc, seed, cm, f1 in cells:
    if cm is None:
        continue
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap=blue, vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9.5, fontweight="bold" if i == j else "normal",
                        color="#fcfcfb" if rn[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(NUM_CLASSES), short, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(NUM_CLASSES), [f"{s} ({n})  F1 {v:.2f}" for s, n, v in zip(short, n_cls, f1)], fontsize=9.5)
    ax.set_xlabel("예측", fontsize=10); ax.set_ylabel("정답 (장수) · 클래스 F1", fontsize=10)
    ax.set_title(f"{run} seed {seed} · {desc}\ntest 17,296장 · macro F1 {f1.mean():.3f} · Scratch F1 {f1[CLASSES.index('Scratch')]:.3f}\n"
                 f"칸 숫자 = 장수, 색 = 행(정답) 기준 비율", loc="left", fontsize=11)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    o = os.path.join(odir, f"cm_{run}_seed{seed}.png"); fig.savefig(o, dpi=130); plt.close(fig)
    print("저장:", o)
