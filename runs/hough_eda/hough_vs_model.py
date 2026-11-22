"""ver8 이 test 작은 웨이퍼 Scratch 에서 맞힌 / 틀린 것의 Hough 점수 비교 (CPU 추론, GPU 학습과 안 겹침).

Hough 점수: 특징 5개 부스팅 (train 으로 학습) 의 Scratch 확률 + 선 길이 단일 특징.
"""
import os, sys
import numpy as np, torch
proj, hdir = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj)
torch.set_num_threads(4)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

H = os.path.expanduser("~"); CACHE = os.path.join(H, "wafer_cache")
ds = WM811K(CACHE)
z = np.load(os.path.join(hdir, "hough2_feats.npz"))
F = np.full((len(ds.labels), 6), np.nan); F[z["idx"]] = z["f"]
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
X = lambda i: np.c_[F[i, :4], np.log(F[i, 4])]

tr = ds.train[~np.isnan(F[ds.train, :5]).any(1)]
gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X(tr), ds.labels[tr] == SC)
te = ds.test; y = ds.labels[te]; small = dies[te] < 1200
ok = ~np.isnan(F[te, :5]).any(1)
score = np.full(len(te), np.nan); score[ok] = gb.predict_proba(X(te[ok]))[:, 1]
length = F[te, 3]
print(f"test 작은 웨이퍼 Scratch {((y == SC) & small).sum()}장 (Hough 계산 불가 = 불량 0개: {((y == SC) & small & ~ok).sum()})")


@torch.no_grad()
def predict(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m.eval(); out = []
    for i in range(0, len(te), 512):
        out.append(m(to_input(ds.fetch(te[i:i + 512], "cpu"))).argmax(1))
    return torch.cat(out).numpy()


def med(a):
    a = a[~np.isnan(a)]
    return f"{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]" if len(a) else "-"


none_s = small & (y == NO) & ok
for seed in (0, 1):
    p = predict(seed)
    sc = small & (y == SC) & ok
    groups = {"Scratch 맞힘": sc & (p == SC), "Scratch→none 틀림": sc & (p == NO), "Scratch→다른 불량 틀림": sc & (p != SC) & (p != NO),
              "none 맞힘": none_s & (p == NO), "none→Scratch 틀림": none_s & (p == SC)}
    print(f"\n=== ver8 seed {seed}: test 작은 웨이퍼 ===")
    print(f"{'그룹':22s} {'장수':>6s} | {'부스팅 Scratch 확률 중앙값 [25%,75%]':>36s} | {'선 길이/지름':>24s}")
    for k, g in groups.items():
        print(f"{k:22s} {g.sum():6d} | {med(score[g]):>36s} | {med(length[g]):>24s}")
    miss = groups["Scratch→none 틀림"]; hit = groups["Scratch 맞힘"]; nn = groups["none 맞힘"]
    for name, a in (("부스팅", score), ("선 길이", length)):
        print(f"[{name}] AUC: 틀린 Scratch vs none {roc_auc_score(np.r_[np.ones(miss.sum()), np.zeros(nn.sum())], np.r_[a[miss], a[nn]]):.3f}"
              f" | 맞힌 Scratch vs none {roc_auc_score(np.r_[np.ones(hit.sum()), np.zeros(nn.sum())], np.r_[a[hit], a[nn]]):.3f}")
    for q in (99, 99.9):
        t = np.percentile(score[nn], q)
        n_fp = int((score[nn] > t).sum())
        print(f"  기준 = none 의 상위 {100 - q:.1f}% ({n_fp}장이 걸림): 틀린 Scratch {int((score[miss] > t).sum())}/{miss.sum()} 걸림,"
              f" 맞힌 Scratch {int((score[hit] > t).sum())}/{hit.sum()} 걸림")
