"""ver8 seed 0 이 틀린 test Scratch 64장 Grad-CAM: (가) 모델이 고른 클래스, (나) Scratch.
층: layer3 출력 (96 입력 → 6x6, layer4 의 3x3 보다 덜 뭉툭). 같은 웨이퍼의 두 지도는 같은 색 범위 (세기 비교 가능).
"""
import os, sys
import numpy as np, torch, torch.nn.functional as Fn
proj, outd = sys.argv[1], sys.argv[2]
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

m = build_model("resnet18", NUM_CLASSES)
m.load_state_dict(torch.load("runs/resnet18/p0.05/seed0/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
m.eval()
m1 = build_model("resnet18", NUM_CLASSES)
m1.load_state_dict(torch.load("runs/resnet18/p0.05/seed1/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
m1.eval()
with torch.no_grad():
    y1 = m1(to_input(x)).argmax(1).numpy()

act = {}
m.layer3.register_forward_hook(lambda mod, i, o: act.__setitem__("a", o))


def gradcam(cls):
    """cls: (B,) 클래스 → (B,96,96) Grad-CAM (ReLU, 정규화 안 함) + 로짓."""
    xin = to_input(x).clone().requires_grad_(False)
    logits = m(xin)
    a = act["a"]; a.retain_grad()
    logits.gather(1, cls[:, None]).sum().backward()
    w = a.grad.mean((2, 3), keepdim=True)
    cam = Fn.relu((w * a).sum(1, keepdim=True))
    cam = Fn.interpolate(cam, size=96, mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().numpy(), logits.detach()


with torch.no_grad():
    prob = torch.softmax(m(to_input(x)), 1).numpy()
y0 = prob.argmax(1)
cam_pred, logits = gradcam(torch.as_tensor(y0))
m.zero_grad()
cam_scr, _ = gradcam(torch.full((len(ids),), SC))
L = logits.numpy()

wrong = np.nonzero(y0 != SC)[0]
order = sorted(wrong, key=lambda k: (CLASSES[y0[k]] != "none", CLASSES[y0[k]], -prob[k, y0[k]]))
num = {k: n + 1 for n, k in enumerate(order)}   # fig12 과 같은 번호

# 정량: 불량 die 위에 놓인 CAM 비율 (모델이 불량을 보고 있나), Scratch CAM 이 아예 0 인 웨이퍼 수
defect = (x[:, 0].numpy() == 2)
on_def = lambda c, k: (c[k] * defect[k]).sum() / max(c[k].sum(), 1e-9)
zero_scr = [k for k in order if cam_scr[k].max() < 1e-6]
print(f"seed 0 오답 {len(order)}장")
print(f"Scratch Grad-CAM 이 완전히 0 (Scratch 근거가 그림 어디에도 없음): {len(zero_scr)}장 → 번호 {[num[k] for k in zero_scr]}")
print(f"예측 클래스 CAM 세기 중앙값 {np.median([cam_pred[k].max() for k in order]):.3f} vs Scratch CAM 세기 중앙값 {np.median([cam_scr[k].max() for k in order]):.3f}")
print(f"로짓 차이 (예측 - Scratch) 중앙값 {np.median([L[k, y0[k]] - L[k, SC] for k in order]):.2f}")
for k in order:
    print(f"#{num[k]} idx {ids[k]} s0 {CLASSES[y0[k]]} {prob[k, y0[k]]:.2f} Scr {prob[k, SC]:.3f} | 로짓 {L[k, y0[k]]:.1f} vs Scr {L[k, SC]:.1f} | "
          f"CAM 최대 예측 {cam_pred[k].max():.3f} Scr {cam_scr[k].max():.3f}")

gray = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
per = 16; ncolw = 4
for pg in range(int(np.ceil(len(order) / per))):
    ks = order[pg * per:(pg + 1) * per]
    nrow = int(np.ceil(len(ks) / ncolw))
    fig, axes = plt.subplots(nrow, ncolw * 3, figsize=(1.9 * ncolw * 3, 2.35 * nrow + 0.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for n, k in enumerate(ks):
        r, c0 = n // ncolw, (n % ncolw) * 3
        vmax = max(cam_pred[k].max(), cam_scr[k].max(), 1e-9)
        img = x[k, 0].numpy()
        for j, (cam, ttl, cm_) in enumerate([(None, f"#{num[k]} 입력 · {sd[ids[k], 2]} die\ns1: {CLASSES[y1[k]]}", None),
                                            (cam_pred[k], f"(가) {CLASSES[y0[k]]} 근거\n{prob[k, y0[k]]:.0%}", "Oranges"),
                                            (cam_scr[k], f"(나) Scratch 근거\n{prob[k, SC]:.1%}" + (" · 없음" if cam_scr[k].max() < 1e-6 else ""), "Blues")]):
            ax = axes[r, c0 + j]; ax.axis("on")
            if cam is None:
                ax.imshow(img, cmap=gray, vmin=0, vmax=2, interpolation="nearest")
            else:
                ax.imshow(img, cmap=gray, vmin=0, vmax=2, interpolation="nearest", alpha=0.55)
                ax.imshow(cam, cmap=cm_, vmin=0, vmax=vmax, alpha=0.6)
            ax.set_title(ttl, fontsize=8.3, color="#2a78d6" if (j == 0 and y1[k] == SC) else "#0b0b0b")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle(f"[{pg + 1}/{int(np.ceil(len(order) / per))}] ver8 seed 0 Grad-CAM (layer3, 6×6 → 96 확대) · 웨이퍼마다 3칸: 입력 / (가) 모델이 고른 클래스의 근거 (주황) / "
                 f"(나) Scratch 의 근거 (파랑)\n두 지도는 같은 색 범위 → 색이 연할수록 근거가 약함. 번호 = fig12 와 같음. 파란 s1 = seed 1 은 맞힘",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (2.35 * nrow + 0.9)))
    o = os.path.join(outd, f"fig13_gradcam_s0_{pg + 1}.png"); fig.savefig(o, dpi=105); plt.close(fig)
    print("저장:", o)
