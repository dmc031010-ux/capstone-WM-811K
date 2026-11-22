"""seed 0 · ver8.1 / ver8.2 혼동행렬을 die 수 적은 (<1200) / 많은 (>=1200) 웨이퍼로 따로. PNG 4장 + 텍스트."""
import os, sys
import numpy as np, torch
proj, outd = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj); os.makedirs(outd, exist_ok=True)
from sklearn.metrics import confusion_matrix, f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import FiLMResNet, SizeHead, build_model, load_size, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); size = load_size(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
te = ds.test; y = ds.labels[te]; small = dies[te] < 1200
DESC = {"ver8.1": "분류기 직전 크기 임베딩", "ver8.2": "FiLM"}


@torch.no_grad()
def predict(run):
    m = build_model("resnet18", NUM_CLASSES)
    m = SizeHead(m, size) if run == "ver8.1" else FiLMResNet(m, size)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed0/{run}_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m = m.to(dev).eval(); o = []
    for i in range(0, len(te), 1024):
        j = te[i:i + 1024]
        o.append(m(to_input(ds.fetch(j, dev)), torch.as_tensor(j, device=dev)).argmax(1).cpu())
    return torch.cat(o).numpy()


blue = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
short = [c.replace("Edge-", "E-").replace("Near-full", "N-full") for c in CLASSES]
for run in ("ver8.1", "ver8.2"):
    p = predict(run)
    assert abs(f1_score(y, p, labels=range(NUM_CLASSES), average="macro", zero_division=0) -
               {"ver8.1": 0.8224, "ver8.2": 0.8026}[run]) < 2e-3
    for gname, tag, g in (("die 수 적은 웨이퍼 (<1200)", "small", small), ("die 수 많은 웨이퍼 (≥1200)", "large", ~small)):
        yt, yp = y[g], p[g]
        present = np.unique(yt)
        cm = confusion_matrix(yt, yp, labels=range(NUM_CLASSES))
        f1 = f1_score(yt, yp, labels=range(NUM_CLASSES), average=None, zero_division=0)
        macro = f1[present].mean(); acc = (yt == yp).mean()
        print(f"\n[{run} seed 0 · {gname}] {g.sum()}장 · macro F1 {macro:.3f} (정답에 있는 {len(present)}개 클래스 평균) · 정확도 {acc:.3f}")
        print(" " * 10 + "".join(f"{c[:8]:>9s}" for c in CLASSES) + f"{'재현율':>9s}{'F1':>9s}")
        for i, c in enumerate(CLASSES):
            n = cm[i].sum()
            print(f"{c:10s}" + "".join(f"{v:9d}" for v in cm[i]) + (f"{cm[i, i] / n:9.3f}{f1[i]:9.3f}" if n else f"{'-':>9s}{'-':>9s}"))
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8.2, 7.2)); ax.imshow(rn, cmap=blue, vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j]:
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9.5, fontweight="bold" if i == j else "normal",
                            color="#fcfcfb" if rn[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(NUM_CLASSES), short, rotation=40, ha="right", fontsize=10)
        ax.set_yticks(range(NUM_CLASSES), [f"{s} ({cm[i].sum()})  " + (f"F1 {f1[i]:.2f}" if cm[i].sum() else "F1 -") for i, s in enumerate(short)], fontsize=9.5)
        ax.set_xlabel("예측", fontsize=10); ax.set_ylabel("정답 (장수) · 클래스 F1", fontsize=10)
        ax.set_title(f"{run} seed 0 · {DESC[run]} · {gname}\ntest {g.sum():,}장 · macro F1 {macro:.3f} · 정확도 {acc:.3f}\n칸 숫자 = 장수, 색 = 행(정답) 기준 비율",
                     loc="left", fontsize=11)
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.tight_layout(); o = os.path.join(outd, f"cm_{run}_seed0_{tag}.png"); fig.savefig(o, dpi=130); plt.close(fig)
        print("저장:", o)
