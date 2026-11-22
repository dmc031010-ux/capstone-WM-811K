"""조작 실험 예시: 같은 선분 3개 × 조건 6개 (모델 입력 96x96 그대로) + seed 0 예측."""
import os, sys
import numpy as np, torch
proj, out = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fewdie_conds.npz"))
names = list(z["names"]); C = [z[f"c{j}"] for j in range(len(names))]
show = [2, 3, 4, 5, 6]
m = build_model("resnet18", NUM_CLASSES)
m.load_state_dict(torch.load("runs/resnet18/p0.05/seed0/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True)); m.eval()
ks = [0, 1, 2, 3]
cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
fig, axes = plt.subplots(len(ks), len(show), figsize=(3.0 * len(show), 3.3 * len(ks) + 0.8))
for c, j in enumerate(show):
    with torch.no_grad():
        p = torch.softmax(m(to_input(torch.as_tensor(C[j][ks], dtype=torch.float32)[:, None])), 1).numpy()
    for r, k in enumerate(ks):
        ax = axes[r, c]
        ax.imshow(C[j][k], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        a = p[r].argmax()
        ax.set_title((f"{names[j]}\n" if r == 0 else "") + f"선분 {k + 1} → {CLASSES[a]} {p[r, a]:.0%} (Scr {p[r, CLASSES.index('Scratch')]:.0%})",
                     fontsize=9, color="#1a7f37" if CLASSES[a] == "Scratch" else "#d03b3b")
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("조작 실험 예시 (ver8 seed 0): 같은 줄 = 같은 선분 (상대 위치·길이·각도 동일). 적음 바탕 k 와 많음 바탕 k 는 서로 다른 실제 none 웨이퍼",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / (3.3 * len(ks) + 0.8))); fig.savefig(out, dpi=105)
print("저장:", out)
