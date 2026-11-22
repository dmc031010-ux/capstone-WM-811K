"""Hough 특징 다듬기 (CPU). hough_eda.py 의 96x96 기본 ratio 와 비교.

원본 die 해상도: 96x96 은 원본 (h,w) 을 nearest 로 늘린 것 → h,w <= 96 이면 행/열 첫 샘플만 골라 원본을 정확히 복원
                (pkl 을 다시 읽지 않음). h 또는 w > 96 (큰 웨이퍼, 줄인 것) 은 96 지도를 그대로 사용.
특징 (웨이퍼마다, 직선 칸: 각도 1도, 거리 1 die):
  r96      = 96x96 지도 votes / 불량 수 (기존)
  rnat     = 원본 해상도 votes / 불량 수
  rden     = 외톨이 불량 (8 이웃 중 불량 0개) 을 지운 뒤 votes / (지운 뒤 불량 수)
  length   = 노이즈 제거 지도의 최고 직선 위에서 끊김 2 die 이하로 이어진 가장 긴 줄의 길이 / 웨이퍼 지름
  thick    = 최고 직선 ±2 die 띠 안 불량 수 / 직선 위 (±0.5) 불량 수  (가는 선 ≈ 1~2, 덩어리 ≫)
  theta    = 노이즈 제거 지도 최고 직선 각도 (진단: 0/90 도 쏠림)
사용: python hough2.py <프로젝트> <출력폴더>
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np

proj, out_dir = sys.argv[1], sys.argv[2]
H = os.path.expanduser("~")
CACHE = os.path.join(H, "wafer_cache")
TH = np.deg2rad(np.arange(180)); C, S = np.cos(TH), np.sin(TH)
R = 140
OFF = np.arange(180) * (2 * R + 1)
NAN6 = (np.nan,) * 6


def native(img, h, w):
    if h > 96 or w > 96:
        return img
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    r0 = np.unique(ri, return_index=True)[1]; c0 = np.unique(ci, return_index=True)[1]
    assert len(r0) == h and len(c0) == w
    return img[np.ix_(r0, c0)]


def best_line(m):
    y, x = np.nonzero(m == 2)
    if len(y) == 0:
        return None
    hh, ww = m.shape
    xc, yc = x - (ww - 1) / 2, y - (hh - 1) / 2
    rho = xc[:, None] * C + yc[:, None] * S
    acc = np.bincount((np.rint(rho).astype(np.int64) + R + OFF).ravel(), minlength=180 * (2 * R + 1))
    k = int(acc.argmax()); t, rb = k // (2 * R + 1), k % (2 * R + 1) - R
    return acc[k], len(y), t, rho[:, t] - rb, -xc * S[t] + yc * C[t]


def denoise(m):
    d = (m == 2).astype(np.int8); p = np.pad(d, 1)
    nb = sum(p[1 + a:1 + a + m.shape[0], 1 + b:1 + b + m.shape[1]] for a in (-1, 0, 1) for b in (-1, 0, 1)) - d
    out = m.copy(); out[(d == 1) & (nb == 0)] = 1
    return out


def feats(img, h, w):
    b96 = best_line(img)
    if b96 is None:
        return NAN6
    r96 = b96[0] / b96[1]
    nat = native(img, h, w)
    bn = best_line(nat)
    rnat = bn[0] / bn[1]
    den = denoise(nat)
    bd = best_line(den)
    if bd is None:
        return (r96, rnat, np.nan, np.nan, np.nan, np.nan)
    v, n, t, dist, along = bd
    on = np.sort(along[np.abs(dist) <= 0.5])
    gaps = np.diff(on) > 2.0
    starts = np.r_[0, np.nonzero(gaps)[0] + 1]; ends = np.r_[np.nonzero(gaps)[0], len(on) - 1]
    run = (on[ends] - on[starts]).max() + 1
    diam = np.sqrt(4 * (nat > 0).sum() / np.pi)
    thick = (np.abs(dist) <= 2.5).sum() / max((np.abs(dist) <= 0.5).sum(), 1)
    return (r96, rnat, v / n, run / diam, thick, float(t))


def work(a):
    imgs, hw = a
    return np.array([feats(m, h, w) for m, (h, w) in zip(imgs, hw)], dtype=np.float64)


if __name__ == "__main__":
    sys.path.insert(0, proj)
    from wafer.data import WM811K
    ds = WM811K(CACHE)
    sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
    lab = np.sort(np.concatenate([ds.train, ds.val, ds.test]))
    t0 = time.time()
    chunks = ((ds.fetch(c, "cpu")[:, 0].numpy().astype(np.uint8), sd[c, :2]) for c in np.array_split(lab, 120))
    with Pool(6) as p:
        f = np.concatenate(list(p.imap(work, chunks)))
    np.savez(os.path.join(out_dir, "hough2_feats.npz"), idx=lab, f=f,
             names=np.array(["r96", "rnat", "rden", "length", "thick", "theta"]))
    print(f"done {len(lab)} 장 {time.time() - t0:.0f}s, 원본 복원 가능 (h,w<=96) 비율 {(sd[lab, :2].max(1) <= 96).mean():.3f}")
