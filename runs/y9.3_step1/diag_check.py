"""대각선 검증 (학습 없음, test 안 봄): 1 die 두께 직선의 Frangi 응답이 각도에 따라 달라지는가.
1) 합성: 원형 웨이퍼 (지름 40·60 die) 에 길이 10·15·20 die, 각도 0~90°, 8-연결 1 die 선 (각 열/행에 die 하나, 래스터) 과
   4-연결 계단 선 (대각선을 가로·세로 이동으로만) → 선 위 die 의 V 중앙값·최댓값, σ 별 최댓값.
2) 실제 (5% 학습 + val 의 Scratch 라벨): 가장 긴 길쭉한 불량 성분 (불량 die ≥3, 길쭉함 ≥3, L ≥ 8) 의 방향 (축에서 벗어난 각 0~45°) 과 그 성분 위 V 중앙값.
사용: python runs/y9.3_step1/diag_check.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES


def wafer(D):
    r, c = np.mgrid[0:D, 0:D] + 0.5
    return np.where((r / D - 0.5) ** 2 + (c / D - 0.5) ** 2 < 0.25, 1, 0).astype(np.uint8)


def line8(D, L, ang):
    """8-연결 1 die 선: 주 방향 축으로 L 칸 진행, 다른 축은 반올림 (Bresenham 과 같음)."""
    nat = wafer(D); t = np.radians(ang); c0 = D / 2
    n = L; pts = []
    if abs(np.cos(t)) >= abs(np.sin(t)):
        for k in range(n):
            x = k - n / 2; y = x * np.tan(t); pts.append((int(np.floor(c0 + y)), int(np.floor(c0 + x))))
    else:
        for k in range(n):
            y = k - n / 2; x = y / np.tan(t); pts.append((int(np.floor(c0 + y)), int(np.floor(c0 + x))))
    m = np.zeros_like(nat, bool)
    for r, c in pts: m[r, c] = True
    nat[m & (nat > 0)] = 2; return nat, m


def line4(D, L, ang):
    """4-연결 계단 선: 8-연결 선의 대각 이동마다 die 하나를 더해 변으로 이어지게."""
    nat, m = line8(D, L, ang); pts = sorted(zip(*np.nonzero(m)), key=lambda p: (p[1], p[0]) if abs(np.cos(np.radians(ang))) >= abs(np.sin(np.radians(ang))) else (p[0], p[1]))
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if r0 != r1 and c0 != c1: m[r0, c1] = True
    nat[m & (nat > 0)] = 2; return nat, m


print("=== 1) 합성: 선 위 die 의 V (중앙값 / 최댓값), 각도별. 8-연결 = 실제 대각 Scratch 같은 계단 없는 대각선 ===")
for D in (40, 60):
    for L in (10, 15, 20):
        print(f"\n지름 {D}, 길이 {L} die")
        print(f"  {'각도':>4s} | {'8-연결 V 중앙/최대':>18s} | σ1 σ2 σ3 최대     | {'4-연결 V 중앙/최대':>18s} | 8-연결 die 수")
        for ang in (0, 15, 30, 45, 60, 75, 90):
            n8, m8 = line8(D, L, ang); v8 = Fr.vesselness(n8); ps = Fr.vesselness(n8, per_scale=True)
            n4, m4 = line4(D, L, ang); v4 = Fr.vesselness(n4)
            print(f"  {ang:4d} | {np.median(v8[m8]):8.2f} / {v8[m8].max():.2f}    | {ps[0][m8].max():.2f} {ps[1][m8].max():.2f} {ps[2][m8].max():.2f} | {np.median(v4[m4]):8.2f} / {v4[m4].max():.2f}    | {m8.sum()}")

print("\n=== 2) 실제 Scratch (5% 학습 + val, 원본 격자 ≤ 96): 가장 긴 성분의 축 이탈 각 vs 그 성분 위 V ===")
CACHE = os.path.expanduser("~/wafer_cache"); ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC = CLASSES.index("Scratch"); tr = ds.labeled_subset(0.05, seed=0)
rows = []
for split, idx in (("train", tr[ds.labels[tr] == SC]), ("val", ds.val[ds.labels[ds.val] == SC])):
    img = ds.fetch(idx, "cpu")[:, 0].numpy().astype(np.uint8)
    for j, i in enumerate(idx):
        h, w = sd[i, :2]
        if max(h, w) > 96: continue
        ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
        nat = img[j][np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
        _, lab, L = Fr.line_weight(nat, return_parts=True)
        k = int(np.argmax(L))
        if k == 0 or L[k] < 8: continue
        r, c = np.nonzero(lab == k); r = r - r.mean(); c = c - c.mean()
        ev, evec = np.linalg.eigh(np.cov(np.stack([r, c])))
        vr, vc = evec[:, 1]; a = np.degrees(np.arctan2(abs(vr), abs(vc)))       # 0 = 가로, 90 = 세로
        off = min(a, 90 - a)                                                    # 축에서 벗어난 각 0~45
        v = Fr.vesselness(nat); thick = (lab == k).sum() / L[k]                # die 수 / 길이 ≈ 두께
        rows.append((split, int(i), off, float(np.median(v[lab == k])), float(v[lab == k].max()), L[k], thick))
off = np.array([r[2] for r in rows]); vmed = np.array([r[3] for r in rows]); vmax = np.array([r[4] for r in rows]); th = np.array([r[6] for r in rows])
print(f"대상 {len(rows)}장 (학습 {sum(r[0] == 'train' for r in rows)}, val {sum(r[0] == 'val' for r in rows)}), 가장 긴 성분 L ≥ 8")
print(f"  {'축 이탈 각':>10s} | {'n':>3s} | V 중앙값 (성분 위) | V 최댓값 | die 수/길이 (두께)")
for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 45.1)):
    m = (off >= lo) & (off < hi)
    if m.sum(): print(f"  {lo:4.0f}~{min(hi, 45):<4.0f}° | {m.sum():3d} | {np.median(vmed[m]):18.2f} | {np.median(vmax[m]):8.2f} | {np.median(th[m]):.2f}")
thin = th < 1.3
print(f"\n얇은 선만 (die 수/길이 < 1.3, n={thin.sum()}):")
for lo, hi in ((0, 15), (15, 30), (30, 45.1)):
    m = thin & (off >= lo) & (off < hi)
    if m.sum(): print(f"  {lo:4.0f}~{min(hi, 45):<4.0f}° | {m.sum():3d} | V 중앙값 {np.median(vmed[m]):.2f} | V 최댓값 {np.median(vmax[m]):.2f}")
from scipy.stats import spearmanr
print(f"\nSpearman (축 이탈 각 vs 성분 위 V 중앙값): 전체 {spearmanr(off, vmed)[0]:+.2f}, 얇은 선 {spearmanr(off[thin], vmed[thin])[0]:+.2f}")
print("\nhard_scratch 의 C 유형 예시:")
for r in rows:
    if r[1] in (327901, 245132, 7430, 694518, 177992, 264429, 262955):
        print(f"  {r[1]:7d} ({r[0]}): 축 이탈 {r[2]:.0f}°, V 중앙 {r[3]:.2f}, 최대 {r[4]:.2f}, L {r[5]:.0f}, 두께 {r[6]:.2f}")
