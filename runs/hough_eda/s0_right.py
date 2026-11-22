"""ver8 seed 0 이 맞힌 test Scratch 중 20장 (무작위, seed 0 고정) — fig12 와 같은 형식 (모델 입력 96x96 그대로)."""
import os, sys
import numpy as np, torch
proj, out = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj)
torch.set_num_threads(4)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC = CLASSES.index("Scratch")
ids = ds.test[ds.labels[ds.test] == SC]
x = ds.fetch(ids, "cpu")


@torch.no_grad()
def predict(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m.eval()
    return torch.softmax(m(to_input(x)), 1).numpy()


p0, p1 = predict(0), predict(1)
y0, y1 = p0.argmax(1), p1.argmax(1)
right = np.nonzero(y0 == SC)[0]
pick = np.random.default_rng(0).choice(right, 20, replace=False)
pick = sorted(pick, key=lambda k: -p0[k, SC])
print(f"seed 0 이 맞힌 Scratch {len(right)}장 중 무작위 20장 (seed 1 도 맞힘 {int((y1[pick] == SC).sum())}장)")
print(f"맞힌 {len(right)}장 Scratch 확률 중앙값 {np.median(p0[right, SC]):.2f}, 작은 웨이퍼(<1200 die) 비율 {(sd[ids[right], 2] < 1200).mean():.2f}"
      f" | 틀린 것의 작은 웨이퍼 비율 {(sd[ids[y0 != SC], 2] < 1200).mean():.2f}")

cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
ncol, nrow = 5, 4
fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.4 * nrow + 0.8))
for n, k in enumerate(pick):
    ax = axes.ravel()[n]
    ax.imshow(x[k, 0].numpy(), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ok1 = y1[k] == SC
    ax.set_title(f"정답 #{n + 1} · {sd[ids[k], 2]} die\ns0: Scratch {p0[k, SC]:.0%}\ns1: {CLASSES[y1[k]]}" + (" (s1 도 맞힘)" if ok1 else " (s1 은 틀림)"),
                 fontsize=9.5, color="#1a7f37" if ok1 else "#b06d00")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#1a7f37" if ok1 else "#b06d00"); sp.set_linewidth(1.5)
fig.suptitle(f"ver8 seed 0 이 맞힌 test Scratch {len(right)}장 중 무작위 20장 (모델 입력 96×96 그대로, Scratch 확률 높은 순)\n"
             f"초록 = seed 1 도 맞힘, 갈색 = seed 1 은 틀림", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (3.4 * nrow + 0.8)))
fig.savefig(out, dpi=105)
print("저장:", out)
