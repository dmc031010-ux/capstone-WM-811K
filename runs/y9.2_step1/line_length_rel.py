"""라벨 기준이 '절대 길이 (die)' 인가 '웨이퍼 지름 대비 상대 길이' 인가.
line_length_rule.py 와 같은 선 측정 (원본 격자 ≤ 96, 8-연결, 길쭉함 ≥ 3, 주축 투영 범위, 이어붙임 없음).
지름 = 원본 격자 max(H, W) (die). 웨이퍼를 die 수로 3그룹 (<600, 600~1199, ≥1200) 으로 나눠,
각 기준의 구간별 Scratch 비율 곡선이 그룹 간에 겹치는지 비교. 겹치는 쪽이 라벨 기준에 가깝다.
데이터: 5% 라벨 학습 + val 의 Scratch·none (합쳐서 표본 확보). test 안 씀.
사용: python runs/y9.2_step1/line_length_rel.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from wafer.data import WM811K, CLASSES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
S8 = np.ones((3, 3))


def native(img, h, w):
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]


def longest_line(b):
    lab, n = ndimage.label(b, structure=S8); best = 0.0
    for k in range(1, n + 1):
        p = np.argwhere(lab == k).astype(float)
        if len(p) < 3: continue
        c = p - p.mean(0); ev, V = np.linalg.eigh(np.cov(c.T))
        if np.sqrt(ev[1] / max(ev[0], 1e-6)) < 3: continue
        proj = c @ V[:, 1]; best = max(best, proj.max() - proj.min() + 1)
    return best


base = np.concatenate([ds.labeled_subset(0.05, seed=0), ds.val])
base = base[np.isin(ds.labels[base], [SC, NO]) & (sd[base, :2].max(1) <= 96)]
img = ds.fetch(base, "cpu")[:, 0].numpy().astype(np.uint8)
L = np.array([longest_line(native(img[j], *sd[i, :2]) == 2) for j, i in enumerate(base)])
D = sd[base, :2].max(1).astype(float); R = L / D; y = ds.labels[base]; nd = dies[base]
G = [("die <600", nd < 600), ("die 600~1199", (nd >= 600) & (nd < 1200)), ("die ≥1200", nd >= 1200)]
print(f"5% 학습 + val: Scratch {np.sum(y == SC)}, none {np.sum(y == NO)} (원본 격자 ≤ 96)")
for g, m in G:
    print(f"  {g:12s} Scratch {np.sum(m & (y == SC)):3d}, none {np.sum(m & (y == NO)):5d}, 지름 중앙값 {np.median(D[m]):.0f} die, "
          f"Scratch 선 길이 중앙값 {np.median(L[m & (y == SC)]):.1f} die = 지름의 {np.median(R[m & (y == SC)]):.2f}")

AB = [0, 6, 8, 10, 12, 15, 20, 30, 1000]; RB = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 10]
curves = {}
for name, X, bins, fmt in (("절대 길이 (die)", L, AB, "{:.0f}"), ("상대 길이 (지름 대비)", R, RB, "{:.2f}")):
    print(f"\n[{name}] 구간별 Scratch 비율 (Scratch/전체)")
    print(f"  {'구간':>12s} " + " ".join(f"{g:>16s}" for g, _ in G))
    curves[name] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        cells, row = [], []
        for g, m in G:
            k = m & (X >= lo) & (X < hi); s = np.sum(k & (y == SC)); n = k.sum()
            cells.append(f"{s / n:6.3f} ({s:2d}/{n:4d})" if n else f"{'-':>16s}"); row.append((s, n))
        curves[name].append(row)
        lab_ = f"{fmt.format(lo)}~{fmt.format(hi)}" if hi < bins[-1] else f"{fmt.format(lo)}+"
        print(f"  {lab_:>12s} " + " ".join(cells))
    # 그룹 간 불일치: 구간마다 그룹 비율의 최대-최소 (각 그룹 n ≥ 20 인 구간만), 표본 가중 평균
    gaps, ws = [], []
    for row in curves[name]:
        ok = [(s / n) for s, n in row if n >= 20]
        if len(ok) >= 2: gaps.append(max(ok) - min(ok)); ws.append(sum(n for _, n in row))
    print(f"  그룹 간 Scratch 비율 차 (구간별 최대−최소, 가중 평균): {np.average(gaps, weights=ws):.3f}  (작을수록 이 기준이 크기와 무관하게 일관)")

# 로지스틱 비교: 어느 변수가 라벨을 더 잘 설명 (로그 손실, 5-fold)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, X in (("절대 길이", np.c_[np.log1p(L)]), ("상대 길이", np.c_[np.log1p(R * 100)]), ("절대 + 지름", np.c_[np.log1p(L), np.log(D)])):
    s = cross_val_score(LogisticRegression(class_weight=None, max_iter=1000), X, (y == SC).astype(int), cv=cv, scoring="neg_log_loss")
    print(f"  로지스틱 {name:10s} 로그 손실 {-s.mean():.5f}")
lr = LogisticRegression(max_iter=1000).fit(np.c_[np.log1p(L), np.log(D)], (y == SC).astype(int))
print(f"  절대+지름 계수: log 길이 {lr.coef_[0, 0]:+.2f}, log 지름 {lr.coef_[0, 1]:+.2f} "
      f"(지름 계수 ≈ 0 이면 절대 길이 기준, ≈ −(길이 계수) 이면 상대 길이 기준)")

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (name, bins) in zip(ax, (("절대 길이 (die)", AB), ("상대 길이 (지름 대비)", RB))):
    mids = np.arange(len(bins) - 1)
    for (g, _), col in zip(G, ("#3b6fb6", "#c0392b", "#2e8b57")):
        r = [s / n if n >= 10 else np.nan for s, n in [row[G.index((g, _)) if False else [x[0] for x in G].index(g)] for row in curves[name]]]
        a.plot(mids, r, "o-", color=col, label=g)
    a.set_xticks(mids); a.set_xticklabels([f"{bins[i]:g}~" for i in mids], fontsize=8)
    a.set_title(name); a.set_ylabel("Scratch 비율"); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle("가장 긴 선 모양 불량 길이별 Scratch 라벨 비율 (5% 학습 + val, 구간 표본 ≥ 10)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "line_length_rel.png"), dpi=110)
print("그림:", os.path.join(HERE, "line_length_rel.png"))
