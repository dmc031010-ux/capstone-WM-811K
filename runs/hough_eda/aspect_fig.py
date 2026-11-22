"""직사각형 die 웨이퍼: 원본 격자 (1 die = 정사각 픽셀) → 타원, 실제 비율 → 원, 96x96 → 원. + Scratch 각도 왜곡과 좌표 보정."""
import os, sys
import numpy as np
proj, hdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, proj); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wafer.data import WM811K, CLASSES
_a = sys.argv; sys.argv = [_a[0], proj, hdir]
import hough2 as h2
sys.argv = _a
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
lab = ds.labels
cmap = matplotlib.colors.ListedColormap(["#fcfcfb", "#dcdbd5", "#2b2b2b"])
ORG = "#eb6834"

# 예시: 비율이 큰 Scratch (선이 뚜렷한 것 = 불량이 적당히 적은 것)
asp = sd[:, 1] / sd[:, 0]
cand = np.nonzero((lab == CLASSES.index("Scratch")) & (np.maximum(asp, 1 / asp) > 1.5) & (sd[:, :2].max(1) <= 96))[0]
from scipy import ndimage
K8 = np.ones((3, 3), bool)


def main_pts(nat):
    den = h2.denoise(nat) == 2
    lab_, k_ = ndimage.label(ndimage.binary_dilation(den, K8) & (nat > 0), structure=K8)
    cnt = np.bincount(lab_[den], minlength=k_ + 1); cnt[0] = 0
    return np.nonzero(den & (lab_ == cnt.argmax()))


def axis(px, py):
    P = np.c_[px, py].astype(float); P -= P.mean(0); ev, V = np.linalg.eigh(P.T @ P / len(P))
    t = P @ V[:, 1]
    return np.degrees(np.arctan2(-V[1, 1], V[0, 1])) % 180, np.sqrt(ev[1] / max(ev[0], 1 / 12)), t.max() - t.min()


best = None
for c in cand:
    hh, ww = sd[c, :2]
    yy, xx = main_pts(h2.native(ds.fetch([c], "cpu")[0, 0].numpy().astype(np.uint8), hh, ww))
    if len(yy) < 8:
        continue
    a0, e0, _ = axis(xx, yy); a1, e1, _ = axis(xx * hh / ww, yy)
    if e0 > 2.5 and 15 < a0 < 75 and (best is None or abs(a1 - a0) > best[0]):
        best = (abs(a1 - a0), c)
i = best[1]
h, w = sd[i, :2]
print("예시 웨이퍼", i, "h,w", h, w, "die 수", sd[i, 2], "가로/세로 칸 비율", w / h)
img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
nat = h2.native(img, h, w)

fig = plt.figure(figsize=(17, 9.8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
# (1) 원본 격자, 1 die = 정사각 픽셀
ax = fig.add_subplot(gs[0, 0:2])
ax.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
ax.set_title(f"① 원본 격자를 1 die = 정사각형 1칸으로 그림\n세로 {h}줄 × 가로 {w}칸 → {'옆으로 납작한' if w > h else '위아래로 긴'} 타원", loc="left", fontsize=11)
ax.set_xticks([0, w - 1], ["1", f"{w}칸"]); ax.set_yticks([0, h - 1], ["1", f"{h}줄"])
# (2) 실제 물리 모양: die 를 실제 비율 (가로:세로 = h:w) 직사각형으로
ax = fig.add_subplot(gs[0, 2])
ax.imshow(nat, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect=w / h)
ax.set_title(f"② 실제 모양: die 1개 = 가로:세로 {h}:{w}\n({'폭이 좁고 키 큰' if w > h else '가로로 넓고 납작한'} 칩) → 원", loc="left", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])
# (3) 96x96 모델 입력
ax = fig.add_subplot(gs[0, 3])
ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
ax.set_title(f"③ 모델 입력 96×96: 가로 ×{96 / w:.2f}, 세로 ×{96 / h:.2f}\n따로 늘림 → 원 (die 1개 = 약 {96 / w:.1f}×{96 / h:.1f} 픽셀)", loc="left", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# (4) 좌표 보정: 불량 점 좌표, 보정 전/후 + 주축 각도
y, x = main_pts(nat)
wy, wx = np.nonzero(nat > 0)
dy, dx = np.nonzero(nat == 2)
for col, (title, f) in enumerate([("④ 특징 계산 (보정 전): 원본 격자 좌표 그대로", 1.0),
                                  (f"⑤ 보정 후: 가로 좌표 × (세로 줄 수 / 가로 칸 수 = {h}/{w} = {h / w:.2f})", h / w)]):
    ax = fig.add_subplot(gs[1, col * 2:col * 2 + 2])
    ax.scatter(wx * f, wy, s=7, color="#dcdbd5", marker="s", linewidths=0)
    ax.scatter(dx * f, dy, s=7, color="#8a8984", marker="s", linewidths=0)
    ax.scatter(x * f, y, s=16, color=ORG, marker="s", linewidths=0)
    ang, el, L = axis(x * f, y)
    D = 2 * np.sqrt(len(wx) * f / np.pi)     # 같은 면적 원의 지름
    ax.set_title(f"{title}\nScratch 선 (주황) 각도 {ang:.0f}° · 길쭉함 {el:.1f} · 길이/지름 {L / D:.2f}", loc="left", fontsize=11)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle(f"직사각형 die 웨이퍼가 타원으로 보이는 이유 (Scratch 예시, {sd[i, 2]} die). 회색 = 정상 die, 검정 = 불량",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(out, dpi=125)
print("저장:", out)




