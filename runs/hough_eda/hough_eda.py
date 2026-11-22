"""Hough transform EDA (CPU). 라벨 웨이퍼 전체 (96x96 모델 입력 지도) 의 불량 픽셀 (값 2) 에 직선 Hough.

특징 (웨이퍼마다):
  votes = 가장 많은 불량 픽셀이 지나는 직선 위의 불량 픽셀 수 (각도 1도, 거리 1픽셀 칸)
  ratio = votes / 불량 픽셀 수 (불량 중 몇 %가 한 직선 위에 있나)
  n_def = 불량 픽셀 수
보고: train split 만 (val/test 는 나중 평가용으로 남김). 클래스별 분포 + Scratch vs 나머지 AUC / Mann-Whitney, 작은/큰 웨이퍼별.
사용: python hough_eda.py <프로젝트> <출력폴더>
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np

proj, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
H = os.path.expanduser("~")
CACHE = os.path.join(H, "wafer_cache")
TH = np.deg2rad(np.arange(180)); C, S = np.cos(TH), np.sin(TH)
R = 70   # 중심 기준 |rho| <= 48*sqrt(2) < 70
OFF = np.arange(180) * (2 * R + 1)


def hough(img):
    y, x = np.nonzero(img == 2)
    n = len(y)
    if n == 0:
        return 0, 0
    rho = np.rint((x[:, None] - 47.5) * C + (y[:, None] - 47.5) * S).astype(np.int32) + R
    return int(np.bincount((rho + OFF).ravel(), minlength=180 * (2 * R + 1)).max()), n


def work(imgs):
    return np.array([hough(m) for m in imgs], dtype=np.int64)


if __name__ == "__main__":
    sys.path.insert(0, proj)
    from wafer.data import WM811K, CLASSES
    ds = WM811K(CACHE)
    lab = np.sort(np.concatenate([ds.train, ds.val, ds.test]))
    feat_path = os.path.join(out_dir, "hough_feats.npz")
    if not os.path.exists(feat_path):
        t = time.time()
        chunks = (ds.fetch(c, "cpu")[:, 0].numpy().astype(np.uint8) for c in np.array_split(lab, 120))
        with Pool(6) as p:   # 웨이퍼 그림은 메인에서만 읽어 넘김 (작업자마다 전체 캐시를 올리지 않게)
            res = list(p.imap(work, chunks))
        f = np.concatenate(res)
        np.savez(feat_path, idx=lab, votes=f[:, 0], n_def=f[:, 1])
        print(f"Hough {len(lab)} 장 {time.time() - t:.0f}s")
    z = np.load(feat_path)
    full_votes = np.zeros(len(ds.labels), np.int64); full_n = np.zeros(len(ds.labels), np.int64)
    full_votes[z["idx"]] = z["votes"]; full_n[z["idx"]] = z["n_def"]
    dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
    np.save(os.path.join(out_dir, "train_idx.npy"), ds.train)

    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    tr = ds.train; y = ds.labels[tr]; v = full_votes[tr]; n = full_n[tr]
    ratio = np.where(n > 0, v / np.maximum(n, 1), 0.0)
    small = dies[tr] < 1200
    S_ = CLASSES.index("Scratch")
    feats = {"votes": v.astype(float), "ratio": ratio}

    def q(a):
        return f"{np.median(a):6.3f} [{np.percentile(a, 25):6.3f}, {np.percentile(a, 75):6.3f}]" if a.max() <= 1 else \
               f"{np.median(a):6.0f} [{np.percentile(a, 25):5.0f}, {np.percentile(a, 75):5.0f}]"

    print(f"\n=== train split {len(tr)} 장: 클래스별 중앙값 [25%, 75%] ===")
    print(f"{'class':10s} {'N':>7s} | {'n_def':>20s} | {'votes':>20s} | {'ratio':>26s}")
    for k, c in enumerate(CLASSES):
        g = y == k
        print(f"{c:10s} {g.sum():7d} | {q(n[g].astype(float)):>20s} | {q(v[g].astype(float)):>20s} | {q(ratio[g]):>26s}")

    print("\n=== Scratch vs 각 클래스: AUC (1 = Scratch 가 항상 더 큼, 0.5 = 구분 못함), Mann-Whitney p (단측, Scratch 가 큼) ===")
    for name, a in feats.items():
        row = []
        for k, c in enumerate(CLASSES):
            if k == S_:
                continue
            g = (y == S_) | (y == k)
            row.append(f"{c} {roc_auc_score(y[g] == S_, a[g]):.2f}")
        p = mannwhitneyu(a[y == S_], a[y != S_], alternative="greater").pvalue
        print(f"{name:6s} vs 나머지 전체 AUC {roc_auc_score(y == S_, a):.3f} p={p:.1e} | " + " ".join(row))

    print("\n=== 작은 (<1200 die) / 큰 웨이퍼 별: Scratch vs none, Scratch vs 나머지 AUC ===")
    N_ = CLASSES.index("none")
    for gname, gm in (("작은", small), ("큰", ~small)):
        for name, a in feats.items():
            ys, aa = y[gm], a[gm]
            g = (ys == S_) | (ys == N_)
            print(f"{gname} {name:6s} Scratch {int((ys == S_).sum()):4d}장 중앙값 {np.median(aa[ys == S_]):.3f} | none 중앙값 {np.median(aa[ys == N_]):.3f}"
                  f" | AUC vs none {roc_auc_score(ys[g] == S_, aa[g]):.3f} | AUC vs 나머지 {roc_auc_score(ys == S_, aa):.3f}")

    # 그림: 클래스별 boxplot (votes, ratio) + Scratch 작은/큰
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a_, (name, a) in zip(ax, feats.items()):
        a_.boxplot([a[y == k] for k in range(len(CLASSES))], showfliers=False)
        a_.set_xticks(range(1, len(CLASSES) + 1), CLASSES, rotation=30)
        a_.set_title({"votes": "max Hough votes (defect px on best line)", "ratio": "max votes / defect px"}[name])
        a_.get_xticklabels()[S_].set_color("crimson")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "hough_boxplot.png"), dpi=130)
    print("\n그림:", os.path.join(out_dir, "hough_boxplot.png"))
