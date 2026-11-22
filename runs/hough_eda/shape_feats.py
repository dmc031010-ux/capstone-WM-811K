"""모양 특징 (CPU): 원본 die 지도 (hough2.native) → 외톨이 불량 제거 → 주 덩어리 (1 die 틈은 이어 붙여 묶음) 의 모양.

특징 (주 덩어리 기준):
  n_main    주 덩어리 불량 die 수,  conc = 주 덩어리 / 전체 불량 (불량이 한곳에 몰린 정도)
  elong     길쭉함 = sqrt(주축 분산 / 부축 분산)  (선 ≫ 1, 둥근 덩어리 ≈ 1)
  length    주축 방향 길이 / 웨이퍼 지름
  width     주축 1 die 칸마다 수직 폭 (불량 수) 의 평균  (가는 선 ≈ 1~2)
  width_cv  그 폭의 변동계수 (일정한 폭 ≈ 0, 가운데 볼록한 덩어리 ≫)
  fill      주축 길이 중 불량이 있는 칸 비율 (점선 < 1)
  sagitta   2차 곡선 맞춤의 휘어짐 = |a| (L/2)^2 (die),  curv_gain = 2차 맞춤이 1차보다 잔차를 줄인 비율
  rms_perp  직선 맞춤 수직 잔차 RMS (die)
  r_center  주 덩어리 중심의 반지름 위치 (0 = 웨이퍼 중심, 1 = 테두리)
  edge_frac 주 덩어리 중 테두리 2 die 이내 비율
사용: python shape_feats.py <프로젝트> <hough폴더>
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import ndimage

proj, hdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [_argv[0], proj, hdir]
import hough2 as h2
sys.argv = _argv
NAMES = ["n_main", "conc", "elong", "length", "width", "width_cv", "fill", "sagitta", "curv_gain", "rms_perp", "r_center", "edge_frac"]
K8 = np.ones((3, 3), bool)


def shape(nat):
    wafer = nat > 0
    den = h2.denoise(nat) == 2
    n_all = den.sum()
    if n_all < 3:
        return [np.nan] * len(NAMES)
    lab, k = ndimage.label(ndimage.binary_dilation(den, K8) & wafer, structure=K8)   # 1 die 틈은 이어서 묶음
    cnt = np.bincount(lab[den], minlength=k + 1); cnt[0] = 0
    main = den & (lab == cnt.argmax())
    y, x = np.nonzero(main); n = len(y)
    if n < 3:
        return [np.nan] * len(NAMES)
    P = np.c_[x, y].astype(float); mu = P.mean(0); Q = P - mu
    ev, V = np.linalg.eigh(Q.T @ Q / n)                      # 오름차순
    t, u = Q @ V[:, 1], Q @ V[:, 0]                          # 주축 좌표, 수직 좌표
    elong = np.sqrt(ev[1] / max(ev[0], 1 / 12))
    L = t.max() - t.min() + 1
    bins = np.floor(t - t.min()).astype(int)
    wcount = np.bincount(bins)
    occ = wcount[wcount > 0]
    width, width_cv, fill = occ.mean(), occ.std() / occ.mean(), len(occ) / len(wcount)
    r1 = u - np.polyval(np.polyfit(t, u, 1), t)
    if n >= 4 and L >= 4:
        c2 = np.polyfit(t, u, 2); r2 = u - np.polyval(c2, t)
        sag = abs(c2[0]) * (L / 2) ** 2
        gain = 1 - (r2 ** 2).sum() / max((r1 ** 2).sum(), 1e-9)
    else:
        sag, gain = 0.0, 0.0
    hh, ww = nat.shape
    wy, wx = np.nonzero(wafer)
    c = np.array([wx.mean(), wy.mean()]); R = np.sqrt(wafer.sum() / np.pi)
    dist_edge = ndimage.distance_transform_edt(np.pad(wafer, 1))[1:-1, 1:-1]
    return [n, n / n_all, elong, L / (2 * R), width, width_cv, fill, sag, gain, np.sqrt((r1 ** 2).mean()),
            np.linalg.norm(mu - c) / R, (dist_edge[y, x] <= 2).mean()]


def work(a):
    imgs, hw = a
    return np.array([shape(h2.native(m, h, w)) for m, (h, w) in zip(imgs, hw)], dtype=np.float64)


if __name__ == "__main__":
    sys.path.insert(0, proj)
    from wafer.data import WM811K
    H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
    ds = WM811K(CACHE)
    sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
    lab = np.sort(np.concatenate([ds.train, ds.val, ds.test]))
    t0 = time.time()
    chunks = ((ds.fetch(c, "cpu")[:, 0].numpy().astype(np.uint8), sd[c, :2]) for c in np.array_split(lab, 120))
    with Pool(6) as p:
        f = np.concatenate(list(p.imap(work, chunks)))
    np.savez(os.path.join(hdir, "shape_feats.npz"), idx=lab, f=f, names=np.array(NAMES))
    print(f"done {len(lab)} 장 {time.time() - t0:.0f}s, NaN (불량 3개 미만) {np.isnan(f[:, 0]).mean():.3f}")
