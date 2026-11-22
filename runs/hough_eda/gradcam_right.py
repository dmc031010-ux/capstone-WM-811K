"""ver8 seed 0 이 맞힌 Scratch 20장 (fig14 와 같은 20장) Grad-CAM: 입력 / Scratch 근거 / 2등 클래스 근거.
+ 120장 전체 정량: Scratch 근거가 불량 die 위에 얼마나 몰렸나 (맞힘 vs 틀림)."""
import os, sys
import numpy as np, torch, torch.nn.functional as Fn
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


def load(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    return m.eval()


m, m1 = load(0), load(1)
with torch.no_grad():
    p0 = torch.softmax(m(to_input(x)), 1).numpy(); y1 = m1(to_input(x)).argmax(1).numpy()
y0 = p0.argmax(1)
act = {}
m.layer3.register_forward_hook(lambda mod, i, o: act.__setitem__("a", o))


def gradcam(cls):
    m.zero_grad()
    logits = m(to_input(x))
    a = act["a"]; a.retain_grad()
    logits.gather(1, cls[:, None]).sum().backward()
    cam = Fn.relu((a.grad.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    return Fn.interpolate(cam, size=96, mode="bilinear", align_corners=False)[:, 0].detach().numpy()


second = np.argsort(-p0, 1)[:, 1]
cam_scr = gradcam(torch.full((len(ids),), SC))
cam_2nd = gradcam(torch.as_tensor(second))

# 정량: 불량 die 쏠림 = (근거 가중 평균으로 본 불량 비율) / (웨이퍼 안 불량 비율). 1 = 아무 데나, >1 = 불량 위에 몰림
img = x[:, 0].numpy(); defect = img == 2; wafer = img > 0
def enrich(cam, k):
    c = cam[k] * wafer[k]
    return (c * defect[k]).sum() / max(c.sum(), 1e-9) / max(defect[k].sum() / wafer[k].sum(), 1e-9)
def rim(cam, k):   # 근거 중 웨이퍼 바깥 (배경) 에 놓인 비율
    return (cam[k] * ~wafer[k]).sum() / max(cam[k].sum(), 1e-9)
G = {"맞힘": y0 == SC, "틀림→none": y0 == CLASSES.index("none"), "틀림→Loc 등": (y0 != SC) & (y0 != CLASSES.index("none"))}
print("Scratch 근거 (seed 0, test Scratch 120장):")
for g, mask in G.items():
    ks = np.nonzero(mask)[0]
    print(f"  {g:10s} {len(ks):3d}장 | 불량 위 쏠림 중앙값 {np.median([enrich(cam_scr, k) for k in ks]):.2f}"
          f" | 웨이퍼 바깥 비율 중앙값 {np.median([rim(cam_scr, k) for k in ks]):.2f} | 세기 중앙값 {np.median([cam_scr[k].max() for k in ks]):.2f}")

right = np.nonzero(y0 == SC)[0]
pick = sorted(np.random.default_rng(0).choice(right, 20, replace=False), key=lambda k: -p0[k, SC])   # fig14 와 같은 20장
gray = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
ncolw, nrow = 4, 5
fig, axes = plt.subplots(nrow, ncolw * 3, figsize=(1.95 * ncolw * 3, 2.4 * nrow + 0.9))
for ax in axes.ravel():
    ax.axis("off")
for n, k in enumerate(pick):
    r, c0 = n // ncolw, (n % ncolw) * 3
    vmax = max(cam_scr[k].max(), cam_2nd[k].max(), 1e-9)
    for j, (cam, ttl, cm_) in enumerate([(None, f"정답 #{n + 1} · {sd[ids[k], 2]} die\ns1: {CLASSES[y1[k]]}", None),
                                        (cam_scr[k], f"Scratch 근거\n{p0[k, SC]:.0%}", "Blues"),
                                        (cam_2nd[k], f"2등 {CLASSES[second[k]]} 근거\n{p0[k, second[k]]:.1%}", "Oranges")]):
        ax = axes[r, c0 + j]; ax.axis("on")
        if cam is None:
            ax.imshow(img[k], cmap=gray, vmin=0, vmax=2, interpolation="nearest")
        else:
            ax.imshow(img[k], cmap=gray, vmin=0, vmax=2, interpolation="nearest", alpha=0.55)
            ax.imshow(cam, cmap=cm_, vmin=0, vmax=vmax, alpha=0.6)
        ax.set_title(ttl, fontsize=8.3, color="#b06d00" if (j == 0 and y1[k] != SC) else "#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
fig.suptitle("ver8 seed 0 이 맞힌 Scratch 20장 Grad-CAM (layer3, 6×6 → 96 확대, fig14 와 같은 20장·번호) · 웨이퍼마다 3칸: 입력 / Scratch 근거 (파랑) / "
             "2등 클래스 근거 (주황)\n두 지도는 같은 색 범위. 갈색 s1 = seed 1 은 틀림", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (2.4 * nrow + 0.9)))
fig.savefig(out, dpi=105)
print("저장:", out)
