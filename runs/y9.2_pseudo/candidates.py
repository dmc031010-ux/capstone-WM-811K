"""E 후보: 라벨 없는 실제 웨이퍼 전부 (638,507장) 를 채점해 Scratch pseudo-label 후보를 뽑고 그림으로 보여 줌. 학습 없음, val/test 안 씀.

채점 모델: y9 seed 0 (마스킹 50ep) 와 SimCLR seed 0 (ver3 50ep) 의 fine-tune 체크포인트.
후보 조건: y9 의 예측이 Scratch 이고 확률 ≥ 0.99.
자동 제외 (사람이 보기 전 거르기):
  - 줄무늬: 웨이퍼 안 열 (또는 행) 중 불량 비율 80% 넘는 줄이 10% 이상
  - 불량이 너무 많음 (웨이퍼 die 의 30% 초과) / 너무 적음 (불량 die 5개 미만)
출력: candidates.npz (인덱스·확률·SimCLR 동의 여부·제외 사유), fig_kept_*.png (남은 후보 무작위 60장씩), fig_excluded.png
사용: python runs/y9.2_pseudo/candidates.py
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
SC = CLASSES.index("Scratch")
R0 = "runs/resnet18/p0.05/seed0"
CM = ListedColormap(["#ffffff", "#d9dde3", "#c0392b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
ul = ds.unlabeled


def load(p):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); return m.to(dev).eval()


@torch.no_grad()
def score(m, idx):
    o = []
    for i in range(0, len(idx), 2048):
        o.append(torch.softmax(m(to_input(ds.fetch(idx[i:i + 2048], dev))), 1).cpu())
    return torch.cat(o).numpy()


Py = score(load(f"{R0}/y9_ssl50_ft30/best.pt"), ul)
cand = ul[(Py.argmax(1) == SC) & (Py[:, SC] >= 0.99)]
pc = Py[np.searchsorted(ul, cand), SC]
Ps = score(load(f"{R0}/ver3_ssl50_ft30/best.pt"), cand)
agree = (Ps.argmax(1) == SC) & (Ps[:, SC] >= 0.99)

X = ds.fetch(cand, "cpu")[:, 0].numpy()
reason = np.array([""] * len(cand), dtype=object)
for k, x in enumerate(X):
    w = x > 0; d = x == 2; nd = int(d.sum()); nw = int(w.sum())
    colf = d.sum(0)[w.sum(0) > 0] / w.sum(0)[w.sum(0) > 0]; rowf = d.sum(1)[w.sum(1) > 0] / w.sum(1)[w.sum(1) > 0]
    if (colf > 0.8).mean() > 0.1 or (rowf > 0.8).mean() > 0.1: reason[k] = "줄무늬"
    elif nd > 0.3 * nw: reason[k] = "불량 과다"
    elif nd / max(nw, 1) * dies[cand[k]] < 5: reason[k] = "불량 과소"
kept = reason == ""
np.savez(os.path.join(HERE, "candidates.npz"), idx=cand, p_y9=pc, agree_simclr=agree, reason=reason.astype(str))

print(f"라벨 없는 웨이퍼 {len(ul):,}장 중 y9 Scratch 확률 ≥0.99: {len(cand)}장")
for r in ("줄무늬", "불량 과다", "불량 과소"):
    print(f"  자동 제외 {r}: {(reason == r).sum()}장")
print(f"  남은 후보: {kept.sum()}장 (SimCLR 도 ≥0.99 Scratch 로 동의: {(kept & agree).sum()}장)")
print(f"  남은 후보 die 수: <1200 {(dies[cand[kept]] < 1200).sum()}장, ≥1200 {(dies[cand[kept]] >= 1200).sum()}장")
print(f"  참고: train 라벨 Scratch 48장")


def grid(ids, title, path, tags=None, n=60, cols=12):
    ids = ids[:n]; rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.85 + 0.6))
    for ax in np.atleast_1d(axes).ravel(): ax.axis("off")
    for ax, (k, i) in zip(np.atleast_1d(axes).ravel(), enumerate(ids)):
        ax.imshow(ds.fetch([i], "cpu")[0, 0].numpy(), cmap=CM, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"#{k + 1} die {dies[i]}" + (f"\n{tags[k]}" if tags is not None else ""), fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


rng = np.random.default_rng(0)
kk = np.where(kept)[0]
for name, sel in [("agree", kk[agree[kk]]), ("y9only", kk[~agree[kk]])]:
    pick = rng.permutation(sel)[:60]; pick = pick[np.argsort(dies[cand[pick]])]
    tags = [f"p {pc[j]:.3f}" for j in pick]
    grid(cand[pick], f"E 후보 ({'y9·SimCLR 둘 다 ≥0.99' if name == 'agree' else 'y9 만 ≥0.99, SimCLR 은 동의 안 함'}): "
         f"{len(sel)}장 중 무작위 {len(pick)}장, die 수 순서", os.path.join(HERE, f"fig_kept_{name}.png"), tags)
ex = np.where(~kept)[0]; pick = rng.permutation(ex)[:24]
grid(cand[pick], f"자동 제외된 예 ({len(ex)}장 중 24장)", os.path.join(HERE, "fig_excluded.png"), [reason[j] for j in pick], n=24)
print("그림:", HERE)
