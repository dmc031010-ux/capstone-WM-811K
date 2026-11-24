"""사용자 가설 검증: "선으로 이어져 있어도 die 수가 일정 개수를 넘지 않으면 none 으로 라벨" 인가.
원본 die 격자 (≤ 96 만) 에서 웨이퍼마다 가장 긴 '선 모양' 불량 덩어리의 길이 (die) 를 재고, 길이 구간별 Scratch 비율.
선 모양 = 8-연결 덩어리 중 길쭉함 (주축/부축 표준편차 비) ≥ 3, 길이 = 주축 방향 투영 범위 (die).
'이어붙임' = 1 die 틈을 메운 맵 (binary closing 3×3) 에서 덩어리를 찾되 길이는 원래 불량 die 로 잼.
데이터: 5% 라벨 학습 세트 (split_seed 0) 의 Scratch·none, val 의 Scratch·none. test 안 씀.
사용: python runs/y9.2_step1/line_length_rule.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from wafer.data import WM811K, CLASSES

CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
S8 = np.ones((3, 3))


def native(img, h, w):
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]


def longest_line(b, bridge):
    lab, n = ndimage.label(ndimage.binary_closing(b, S8) & (b | True) if bridge else b, structure=S8)
    best = (0, 0)   # (길이 die, 덩어리 불량 die 수)
    for k in range(1, n + 1):
        p = np.argwhere((lab == k) & b).astype(float)
        if len(p) < 3: continue
        c = p - p.mean(0); ev, V = np.linalg.eigh(np.cov(c.T))
        if np.sqrt(ev[1] / max(ev[0], 1e-6)) < 3: continue
        proj = c @ V[:, 1]; L = proj.max() - proj.min() + 1
        if L > best[0]: best = (L, len(p))
    return best


def measure(idx):
    idx = idx[sd[idx, :2].max(1) <= 96]
    img = ds.fetch(idx, "cpu")[:, 0].numpy().astype(np.uint8)
    out = np.zeros((len(idx), 4))
    for j, i in enumerate(idx):
        nat = native(img[j], *sd[i, :2]); b = nat == 2
        out[j, :2] = longest_line(b, False); out[j, 2:] = longest_line(b, True)
    return idx, out


lab = ds.labeled_subset(0.05, seed=0)
sets = {"5% 학습": lab, "val": ds.val}
bins = [0, 1, 4, 6, 8, 10, 12, 15, 20, 30, 1000]
for name, base in sets.items():
    y = ds.labels[base]; base = base[(y == SC) | (y == NO)]
    idx, M = measure(base); yy = ds.labels[idx]
    print(f"\n===== {name}: Scratch {np.sum(yy == SC)}장, none {np.sum(yy == NO)}장 (원본 격자 ≤ 96) =====")
    for col, tag in ((0, "그대로"), (2, "1 die 틈 이어붙임")):
        L = M[:, col]
        print(f"[{tag}] 가장 긴 선 모양 덩어리 길이 (die). Scratch 중앙값 {np.median(L[yy == SC]):.1f} "
              f"[10% {np.percentile(L[yy == SC], 10):.1f}, 25% {np.percentile(L[yy == SC], 25):.1f}], "
              f"none 중앙값 {np.median(L[yy == NO]):.1f} [90% {np.percentile(L[yy == NO], 90):.1f}, 99% {np.percentile(L[yy == NO], 99):.1f}]")
        print(f"  {'길이 구간':>10s} {'Scratch':>8s} {'none':>7s} {'Scratch 비율':>12s}")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (L >= lo) & (L < hi)
            if m.sum(): print(f"  {f'{lo}~{hi - 1}' if hi < 1000 else f'{lo}+':>10s} {np.sum(m & (yy == SC)):8d} {np.sum(m & (yy == NO)):7d} {np.mean(yy[m] == SC):12.3f}")
print("\n대상 웨이퍼 (val none 754928):", end=" ")
_, Mt = measure(np.array([754928])); print(f"그대로 길이 {Mt[0, 0]:.0f} die ({Mt[0, 1]:.0f}개), 이어붙임 길이 {Mt[0, 2]:.0f} die ({Mt[0, 3]:.0f}개)")
