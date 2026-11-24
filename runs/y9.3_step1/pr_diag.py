"""Scratch 정밀도-재현율 진단 (val 만, test 안 봄, 학습 없음, 앙상블 없음).
모델별 P(Scratch) 로 줄 세웠을 때: AP, 재현율 0.5/0.6/0.7/0.79 에서 정밀도, 재현 0.79 에서 오탐의 실제 클래스.
사용: python runs/y9.3_step1/pr_diag.py
"""
import os, sys, gc
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input, FrangiInput

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); va = ds.val; yv = ds.labels[va]; SC = CLASSES.index("Scratch"); s = yv == SC
KINDS = {"y9.2b 0채널": ("y9.2b", None), "y9.2a V": ("y9.2a", "frangi"), "y9.3a V·w": ("y9.3a", "length"), "y9.3b k·V": ("y9.3b", "scaled")}
RECS = (0.5, 0.6, 0.7, 0.79)


@torch.no_grad()
def scores(seed, meth, tab):
    m = FrangiInput(build_model("resnet18", NUM_CLASSES), *tab)
    d = f"runs/resnet18/p0.05/seed{seed}/{meth}_ssl50_ft30"
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    out = []
    for i in range(0, len(va), 1024):
        j = va[i:i + 1024]; out.append(torch.softmax(m(to_input(ds.fetch(j, dev)), torch.as_tensor(j, device=dev)).float(), 1)[:, SC].cpu())
    del m; torch.cuda.empty_cache(); return torch.cat(out).numpy()


P = {}
for name, (meth, kind) in KINDS.items():
    tab = (None, None)
    if kind:
        pu, pi = Fr.cache_paths(CACHE, kind); gi = np.load(pi)
        row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi)); tab = (torch.from_numpy(np.load(pu)), row)
    for seed in (0, 1, 2):
        P[(name, seed)] = scores(seed, meth, tab)
    del tab; gc.collect()


def prec_at(p, r):
    o = np.argsort(-p, kind="stable"); tp = np.cumsum(s[o]); k = int(np.searchsorted(tp, int(np.ceil(r * s.sum())))) + 1
    return tp[k - 1] / k, o[:k]


print(f"val Scratch {s.sum()}장 / 전체 {len(va)}장. 점수 = P(Scratch), argmax 와 무관한 순위 진단.\n")
print(f"{'':12s} {'seed':>4s} {'AP':>6s} " + " ".join(f"{'P@R' + str(r):>8s}" for r in RECS))
for name in KINDS:
    for seed in (0, 1, 2):
        p = P[(name, seed)]
        print(f"{name:12s} {seed:4d} {average_precision_score(s, p):6.3f} " + " ".join(f"{prec_at(p, r)[0]:8.3f}" for r in RECS))
    print(f"{name:12s} {'평균':>4s} {np.mean([average_precision_score(s, P[(name, sd)]) for sd in (0, 1, 2)]):6.3f} "
          + " ".join(f"{np.mean([prec_at(P[(name, sd)], r)[0] for sd in (0, 1, 2)]):8.3f}" for r in RECS) + "\n")

print("재현 0.79 까지 Scratch 라 불렀을 때 오탐의 실제 클래스 (세 seed 합):")
for name in KINDS:
    c = np.zeros(NUM_CLASSES, int)
    for seed in (0, 1, 2):
        _, top = prec_at(P[(name, seed)], 0.79); c += np.bincount(yv[top][~s[top]], minlength=NUM_CLASSES)
    print(f"  {name:12s} 합 {c.sum():4d} | " + " ".join(f"{CLASSES[i]} {c[i]}" for i in np.argsort(-c) if c[i]))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for seed in (0, 1, 2):
    for name in KINDS:
        pr, rc, _ = precision_recall_curve(s, P[(name, seed)]); ax[seed].plot(rc, pr, label=f"{name} AP {average_precision_score(s, P[(name, seed)]):.3f}")
    ax[seed].axvline(0.79, c="gray", ls=":"); ax[seed].set(title=f"seed {seed} (val Scratch)", xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1)); ax[seed].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "pr_diag.png"), dpi=110)
