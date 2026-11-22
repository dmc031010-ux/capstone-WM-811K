"""ver8 seed 0: test 전체에서 Loc 확률과 Scratch 확률이 둘 다 높은 웨이퍼 (min(p_Loc, p_Scr) >= TH). 정답 / 최종 예측 / seed 1 예측 표시."""
import os, sys
import numpy as np, torch
proj, out = sys.argv[1], sys.argv[2]
TH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
os.chdir(proj); sys.path.insert(0, proj)
torch.set_num_threads(6)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC, LO = CLASSES.index("Scratch"), CLASSES.index("Loc")
te = ds.test; y = ds.labels[te]


@torch.no_grad()
def probs(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m.eval(); o = []
    for i in range(0, len(te), 1024):
        o.append(torch.softmax(m(to_input(ds.fetch(te[i:i + 1024], "cpu"))), 1))
    return torch.cat(o).numpy()


p0, p1 = probs(0), probs(1)
both = np.minimum(p0[:, LO], p0[:, SC])
for t in (0.1, 0.2, 0.3, 0.4):
    print(f"min(Loc, Scratch) >= {t}: {(both >= t).sum()}장")
sel = np.nonzero(both >= TH)[0]
sel = sel[np.argsort(-both[sel])]
print(f"\n기준 {TH}: {len(sel)}장 | 정답 분포 {dict(zip(*np.unique([CLASSES[c] for c in y[sel]], return_counts=True)))}"
      f" | 최종 예측 분포 {dict(zip(*np.unique([CLASSES[c] for c in p0[sel].argmax(1)], return_counts=True)))}")

cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
ncol = 6; nrow = int(np.ceil(len(sel) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 3.55 * nrow + 0.9), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for n, k in enumerate(sel):
    ax = axes.ravel()[n]; ax.axis("on")
    ax.imshow(ds.fetch([te[k]], "cpu")[0, 0].numpy(), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    pred = p0[k].argmax(); ok = pred == y[k]
    ax.set_title(f"#{n + 1} 정답 {CLASSES[y[k]]} · {sd[te[k], 2]} die\nLoc {p0[k, LO]:.0%} · Scratch {p0[k, SC]:.0%}\n"
                 f"→ 예측 {CLASSES[pred]} ({'맞음' if ok else '틀림'}) · s1 {CLASSES[p1[k].argmax()]}",
                 fontsize=9, color="#1a7f37" if ok else "#d03b3b")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#1a7f37" if ok else "#d03b3b"); sp.set_linewidth(1.5)
fig.suptitle(f"ver8 seed 0 이 Loc 과 Scratch 사이에서 망설인 test 웨이퍼 {len(sel)}장 (Loc 확률, Scratch 확률 모두 {TH:.0%} 이상, 둘 중 작은 값이 큰 순)\n"
             f"초록 = seed 0 최종 예측이 정답, 빨강 = 오답. s1 = seed 1 예측. 모델 입력 96×96 그대로",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (3.55 * nrow + 0.9)))
fig.savefig(out, dpi=105)
print("저장:", out)
for n, k in enumerate(sel):
    print(f"#{n + 1} idx {te[k]} 정답 {CLASSES[y[k]]} Loc {p0[k, LO]:.2f} Scr {p0[k, SC]:.2f} 예측 {CLASSES[p0[k].argmax()]} s1 {CLASSES[p1[k].argmax()]}")
