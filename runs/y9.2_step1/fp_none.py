"""y9.2 1단계: val 의 none → Scratch 오탐 분석 (val 만, test 안 봄).
seed 0·1·2 × {b 0 채널, a Frangi}. 오탐 웨이퍼와 맞게 판정된 none 비교:
 불량 die 수, die 수, Frangi 최댓값·불량 위 평균, 가장 큰 불량 연결 덩어리의 길이·길쭉함 (주성분 고유값 비).
a 오탐이 b·ver8 에서도 오탐이었는지 겹침. 그림: seed 2 a 오탐 예시.
사용: python runs/y9.2_step1/fp_none.py
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input, FrangiInput
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
pu, pi = Fr.cache_paths(CACHE); table = torch.from_numpy(np.load(pu)); gi = np.load(pi)
row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi))
va = ds.val; yv = ds.labels[va]; vn = va[yv == NO]


def load(seed, kind):
    d = f"runs/resnet18/p0.05/seed{seed}/" + {"ver8": "ver8_ssl50_ft30", "b": "y9.2b_ssl50_ft30", "a": "y9.2a_ssl50_ft30"}[kind]
    m = build_model("resnet18", NUM_CLASSES)
    if kind != "ver8":
        m = FrangiInput(m, table if kind == "a" else None, row if kind == "a" else None)
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True))
    return m.to(dev).eval()


@torch.no_grad()
def probs(m, idx):
    out = []
    for i in range(0, len(idx), 1024):
        x = to_input(ds.fetch(idx[i:i + 1024], dev)); j = torch.as_tensor(idx[i:i + 1024], device=dev)
        out.append(torch.softmax((m(x, j) if getattr(m, "needs_idx", False) else m(x)).float(), 1).cpu())
    return torch.cat(out).numpy()


# 웨이퍼별 특징 (none 전체, 96 격자 기준: 캐시의 Frangi 와 같은 격자)
img = ds.fetch(vn, "cpu")[:, 0].numpy().astype(np.uint8); fr = table[row[vn]].numpy() / 255.0
feat = {}
nbad = (img == 2).reshape(len(vn), -1).sum(1); ndie_px = (img > 0).reshape(len(vn), -1).sum(1)
fmax = fr.reshape(len(vn), -1).max(1)
fbad = np.array([fr[k][img[k] == 2].mean() if nbad[k] else 0.0 for k in range(len(vn))])
lng, elo = np.zeros(len(vn)), np.zeros(len(vn))
for k in range(len(vn)):
    lab, n = ndimage.label(img[k] == 2, structure=np.ones((3, 3)))
    if n == 0: continue
    sz = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)); big = np.argmax(sz) + 1
    p = np.argwhere(lab == big).astype(float)
    if len(p) < 3: lng[k] = len(p); continue
    ev = np.sort(np.linalg.eigvalsh(np.cov(p.T)))[::-1]
    lng[k] = 4 * np.sqrt(ev[0]); elo[k] = np.sqrt(ev[0] / max(ev[1], 1e-6))
F = dict(nbad=nbad, bad_ratio=nbad / np.maximum(ndie_px, 1), dies=dies[vn], fmax=fmax, fbad=fbad, length=lng, elong=elo)


def summ(mask):
    return " ".join(f"{k} {np.median(v[mask]):.3g}" for k, v in F.items())


PRED = {}
for seed in (0, 1, 2):
    for kind in ("ver8", "b", "a"):
        m = load(seed, kind); PRED[(seed, kind)] = probs(m, vn); del m; torch.cuda.empty_cache()

print(f"val none {len(vn)}장. 값은 중앙값 (96 격자 기준: nbad 불량 픽셀 수, length 가장 큰 불량 덩어리 길이 px, elong 길쭉함)")
print(f"  맞게 판정된 none 전체 (seed 2 a 기준): {summ(PRED[(2, 'a')].argmax(1) == NO)}")
print(f"  불량 있는 none (nbad ≥ 5)            : {summ(nbad >= 5)}")
for seed in (0, 1, 2):
    pa, pb, p8 = (PRED[(seed, k)].argmax(1) == SC for k in ("a", "b", "ver8"))
    print(f"\n[seed {seed}] none→Scratch: ver8 {p8.sum()}, b {pb.sum()}, a {pa.sum()} | a 오탐 중 b 도 오탐 {(pa & pb).sum()}, ver8 도 오탐 {(pa & p8).sum()}, a 에만 {(pa & ~pb & ~p8).sum()}")
    print(f"  a 오탐 전체      : {summ(pa)}")
    print(f"  a 에만 새 오탐   : {summ(pa & ~pb)}")
    print(f"  b 오탐 (비교)    : {summ(pb)}")
    print(f"  a 오탐 Scratch 확률 중앙값 {np.median(PRED[(seed, 'a')][pa, SC]):.2f}, ≥0.9 인 비율 {np.mean(PRED[(seed, 'a')][pa, SC] >= 0.9):.2f}")
all3 = np.logical_and.reduce([PRED[(s, 'a')].argmax(1) == SC for s in (0, 1, 2)])
any_b = np.logical_or.reduce([PRED[(s, k)].argmax(1) == SC for s in (0, 1, 2) for k in ("b", "ver8")])
print(f"\n세 seed a 모두 오탐: {all3.sum()}장, 그중 b·ver8 어느 하나라도 오탐 {(all3 & any_b).sum()}장")
print(f"  {summ(all3)}")

# 그림: seed 2 a 에만 새로 생긴 오탐 (Scratch 확률 높은 순) + 세 seed 공통 오탐
CM = ListedColormap(["#ffffff", "#d9dde3", "#c0392b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
pa2, pb2 = PRED[(2, 'a')].argmax(1) == SC, PRED[(2, 'b')].argmax(1) == SC
new = np.nonzero(pa2 & ~pb2)[0]; new = new[np.argsort(-PRED[(2, 'a')][new, SC])][:8]
com = np.nonzero(all3)[0][:8]
for name, sel in (("fp_new_seed2.png", new), ("fp_common.png", com)):
    if not len(sel): continue
    fig, ax = plt.subplots(2, len(sel), figsize=(2.2 * len(sel), 4.8), squeeze=False)
    for j, k in enumerate(sel):
        ax[0, j].imshow(img[k], cmap=CM, vmin=0, vmax=2, interpolation="nearest")
        ax[1, j].imshow(fr[k], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax[0, j].set_title(f"die {dies[vn[k]]} 불량 {nbad[k]}\nP(S) a {PRED[(2, 'a')][k, SC]:.2f} b {PRED[(2, 'b')][k, SC]:.2f}", fontsize=8)
        ax[1, j].set_title(f"Frangi max {fmax[k]:.2f}", fontsize=8)
    for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle({"fp_new_seed2.png": "seed 2: Frangi 모델에서만 새로 Scratch 로 판정된 val none (위 웨이퍼맵, 아래 Frangi)",
                  "fp_common.png": "세 seed Frangi 모델 모두 Scratch 로 판정한 val none"}[name], x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=110); plt.close(fig)
print("\n그림:", HERE)
