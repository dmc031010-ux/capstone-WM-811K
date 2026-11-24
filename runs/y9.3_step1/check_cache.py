"""y9.3 1단계 학습 전 점검 (Codex C5): 두 캐시의 인덱스 = y9.2 인덱스, 무작위 200장 재계산 일치, 표가 0 채널이 아님.
사용: python runs/y9.3_step1/check_cache.py  (실패 시 exit 1)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer import frangi as Fr
from wafer.data import WM811K, _resize_nearest
C = os.path.expanduser("~/wafer_cache"); ds = WM811K(C); sd = np.load(os.path.join(C, "wm811k_shapes_dies.npy"))
g2 = np.load(Fr.cache_paths(C)[1]); k = float(open(os.path.join(C, "y9.3", "k.txt")).read())
UL, UL_i = np.load(Fr.cache_paths(C, "length")[0], mmap_mode="r"), np.load(Fr.cache_paths(C, "length")[1])
US, US_i = np.load(Fr.cache_paths(C, "scaled")[0], mmap_mode="r"), np.load(Fr.cache_paths(C, "scaled")[1])
good = np.array_equal(UL_i, g2) and np.array_equal(US_i, g2); print(f"인덱스 = y9.2 ({len(g2)}): {good}, k = {k:.4f}")
rng = np.random.default_rng(0); small = g2[sd[g2, :2].max(1) <= 96]; dmax = 0
for i in rng.choice(small, 200, replace=False):
    h, w = sd[i, :2]; img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    v = Fr.vesselness(nat); wt = Fr.line_weight(nat, "ridge", V=v, tau=Fr.TAU); r = int(np.searchsorted(g2, i))
    dmax = max(dmax, int(np.abs(np.rint(_resize_nearest((v * wt)[None], 96)[0] * 255) - UL[r]).max()),
               int(np.abs(np.rint(_resize_nearest((k * v)[None], 96)[0] * 255) - US[r]).max()))
lab = ds.labeled_subset(0.05, seed=0); rows = np.searchsorted(g2, lab[:2000])
nzL = np.mean(UL[rows].reshape(len(rows), -1).max(1) > 0); nzS = np.mean(US[rows].reshape(len(rows), -1).max(1) > 0)
print(f"재계산 200장 최대 차 {dmax} | 0 아닌 웨이퍼 비율 length {nzL:.2f}, scaled {nzS:.2f} | 평균 (불량 위 아님, 전체 픽셀) length {UL[rows].mean() / 255:.4f}, scaled {US[rows].mean() / 255:.4f}")
ok = good and dmax == 0 and nzL > 0.3 and nzS > 0.5
print("점검", "통과" if ok else "**실패**"); sys.exit(0 if ok else 1)
