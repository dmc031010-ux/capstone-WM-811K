"""ver8 seed 0 이 test 에서 틀린 Scratch 전부 (모델 입력 96x96 그대로). 예측 클래스별로 묶고, seed 1 결과 표시."""
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
wrong = np.nonzero(y0 != SC)[0]
order = sorted(wrong, key=lambda k: (CLASSES[y0[k]] != "none", CLASSES[y0[k]], -p0[k, y0[k]]))
only0 = int((y1[wrong] == SC).sum())
print(f"test Scratch {len(ids)}장 중 seed 0 오답 {len(wrong)}장 (seed 1 은 맞힘 {only0}장, 둘 다 틀림 {len(wrong) - only0}장)")
print("seed 0 예측 분포:", {CLASSES[c]: int((y0[wrong] == c).sum()) for c in np.unique(y0[wrong])})

cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
pages = [("none", [k for k in order if CLASSES[y0[k]] == "none"]), ("Loc 등", [k for k in order if CLASSES[y0[k]] != "none"])]
num = {k: n + 1 for n, k in enumerate(order)}
for pi, (pname, ks) in enumerate(pages):
    ncol = 7; nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.4 * nrow + 0.8), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for n, k in enumerate(ks):
        ax = axes.ravel()[n]; ax.axis("on")
        ax.imshow(x[k, 0].numpy(), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        both = y1[k] != SC
        ax.set_title(f"#{num[k]} · {sd[ids[k], 2]} die\ns0: {CLASSES[y0[k]]} {p0[k, y0[k]]:.0%} (Scr {p0[k, SC]:.0%})\n"
                     f"s1: {CLASSES[y1[k]]}" + (" (둘 다 틀림)" if both else " (s1 맞힘)"),
                     fontsize=9, color="#d03b3b" if both else "#2a78d6")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d03b3b" if both else "#2a78d6"); sp.set_linewidth(1.5)
    nb = sum(y1[k] != SC for k in ks)
    fig.suptitle(f"[{pi + 1}/2] ver8 seed 0 이 '{pname}' 으로 틀린 test Scratch {len(ks)}장 (전체 오답 {len(wrong)}장 중, 모델 입력 96×96 그대로)\n"
                 f"빨강 = seed 1 도 틀림 {nb}장, 파랑 = seed 1 은 맞힘 {len(ks) - nb}장. 괄호 Scr = seed 0 의 Scratch 확률",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (3.4 * nrow + 0.8)))
    o = out.replace(".png", f"_{pi + 1}.png"); fig.savefig(o, dpi=105); plt.close(fig)
    print("저장:", o)
for n, k in enumerate(order):
    print(f"#{n + 1} idx {ids[k]} die {sd[ids[k], 2]} s0 {CLASSES[y0[k]]} s1 {CLASSES[y1[k]]}")


