"""WM-811K (LSWMD.pkl) 전처리 / 캐시 / 분할.

- 모든 웨이퍼 맵을 96x96 nearest-neighbor 로 리사이즈 (WaPIRL 과 동일)
- 값 {0: 배경, 1: 정상 die, 2: 불량 die} 를 2bit 로 4픽셀/바이트 압축 → 81만장 ≈ 1.9GB
- 라벨 데이터 172,950 장을 stratified 80/10/10 (train/val/test) 로 분할
  (논문 Table 3 의 5%=6,918 → train=138,360 = 0.8 * 172,950)
"""
import os
import pickle
import time

import numpy as np
import torch

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "none"]
NUM_CLASSES = len(CLASSES)


def _resize_nearest(maps, size):
    """maps: (n, h, w) 같은 크기 묶음 → (n, size, size). PIL NEAREST 와 같은 샘플링 위치."""
    _, h, w = maps.shape
    ri = ((np.arange(size) + 0.5) * h / size).astype(np.int64)
    ci = ((np.arange(size) + 0.5) * w / size).astype(np.int64)
    return maps[:, ri][:, :, ci]


def pack2bit(x):
    """(n, H, W) uint8 (값 0..3) → (n, H, W//4) uint8."""
    n, h, w = x.shape
    x = x.reshape(n, h, w // 4, 4).astype(np.uint8)
    return x[..., 0] | (x[..., 1] << 2) | (x[..., 2] << 4) | (x[..., 3] << 6)


def unpack2bit(p):
    """GPU/CPU torch uint8 (B, H, W//4) → float (B, 1, H, W), 값 {0,1,2}."""
    parts = [(p >> s) & 3 for s in (0, 2, 4, 6)]
    return torch.stack(parts, -1).flatten(-2).unsqueeze(1).float()


def cache_paths(cache_dir, size):
    return (os.path.join(cache_dir, f"wm811k_{size}_packed.npy"),
            os.path.join(cache_dir, f"wm811k_labels.npy"))


def prepare(pkl_path, cache_dir, size=96):
    packed_path, label_path = cache_paths(cache_dir, size)
    if os.path.exists(packed_path) and os.path.exists(label_path):
        print(f"[prepare] cache exists: {packed_path}")
        return
    os.makedirs(cache_dir, exist_ok=True)
    import sys
    import pandas.core.indexes
    sys.modules.setdefault("pandas.indexes", pandas.core.indexes)  # 구버전 pandas pickle 호환
    t = time.time()
    with open(pkl_path, "rb") as f:
        df = pickle.load(f, encoding="latin1")  # Python2 시절 pickle
    maps = df["waferMap"].values
    ftype = df["failureType"].values
    del df
    n = len(maps)
    print(f"[prepare] loaded {n} wafers in {time.time() - t:.1f}s")

    labels = np.full(n, -1, np.int64)  # -1 = unlabeled
    for i, a in enumerate(ftype):
        if np.size(a) > 0:
            labels[i] = CLASSES.index(str(np.asarray(a).ravel()[0]))
    del ftype

    shapes = np.array([m.shape for m in maps], dtype=np.int64)
    keys = shapes[:, 0] * 100000 + shapes[:, 1]
    packed = np.empty((n, size, size // 4), np.uint8)
    for k in np.unique(keys):
        idx = np.nonzero(keys == k)[0]
        group = np.stack(maps[idx]).astype(np.uint8)
        packed[idx] = pack2bit(np.minimum(_resize_nearest(group, size), 2))
    del maps

    np.save(packed_path, packed)
    np.save(label_path, labels)
    print(f"[prepare] saved {packed.nbytes / 1e9:.2f}GB in {time.time() - t:.1f}s")
    for c, name in enumerate(CLASSES):
        print(f"  {name:10s} {(labels == c).sum()}")
    print(f"  unlabeled  {(labels == -1).sum()}")


class WM811K:
    """전체 압축 맵을 RAM 에 올려두고 인덱스로 꺼내 쓰는 컨테이너."""

    def __init__(self, cache_dir, size=96, split_seed=0):
        packed_path, label_path = cache_paths(cache_dir, size)
        if not os.path.exists(packed_path):
            raise FileNotFoundError(f"{packed_path} 없음 - 먼저 `python wafixmatch.py prepare` 실행")
        self.size = size
        self.packed = np.load(packed_path)
        self.labels = np.load(label_path)
        self.unlabeled = np.nonzero(self.labels < 0)[0]
        self.train, self.val, self.test = self._split(split_seed)

    def _split(self, seed):
        rng = np.random.default_rng(seed)
        tr, va, te = [], [], []
        for c in range(NUM_CLASSES):
            idx = rng.permutation(np.nonzero(self.labels == c)[0])
            n_tr = int(round(0.8 * len(idx)))
            n_va = int(round(0.1 * len(idx)))
            tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])
        return np.concatenate(tr), np.concatenate(va), np.concatenate(te)

    def labeled_subset(self, frac, seed=0):
        """train 에서 클래스별 비율 frac 만큼 (5% ⊂ 10% ⊂ ... 가 되도록 고정 순서)."""
        if frac >= 1.0:
            return self.train
        rng = np.random.default_rng(seed)
        out = []
        for c in range(NUM_CLASSES):
            idx = rng.permutation(self.train[self.labels[self.train] == c])
            out.append(idx[:max(1, int(round(frac * len(idx))))])
        return np.concatenate(out)

    def fetch(self, idx, device):
        """글로벌 인덱스 → GPU float (B,1,H,W) 값 {0,1,2}."""
        p = torch.from_numpy(self.packed[np.asarray(idx)])
        return unpack2bit(p.to(device, non_blocking=True))
