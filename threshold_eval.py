"""ver7.55: ver7.5 (Focal) 모델의 Scratch 판단 임계값 실험 (추가 학습 없음), 5% seed 0.

  python threshold_eval.py
규칙: argmax 가 Scratch 이고 p_Scratch >= t 일 때만 Scratch, 아니면 두 번째로 높은 클래스.
t 는 val macro-F1 로 고르고, test 는 평가에만 사용.
"""
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from ensemble_eval import BASE, MINOR, probs
from wafer.data import CLASSES, NUM_CLASSES, WM811K

RUN = "ver7.5_ssl50_ft30"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
S, NONE = CLASSES.index("Scratch"), CLASSES.index("none")


def predict(p, t):
    pred = p.argmax(1)
    if t is None:
        return pred
    low = (pred == S) & (p[:, S] < t)
    pred[low] = np.argsort(p[low], 1)[:, -2]
    return pred


def report(y, pred):
    pc = f1_score(y, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    return dict(macro=pc.mean(), minor=np.mean([pc[CLASSES.index(c)] for c in MINOR]), scratch=pc[S],
                hit=cm[S, S], miss_none=cm[S, NONE], fp_none=cm[NONE, S], pred_s=cm[:, S].sum())


def main():
    import torch
    device = torch.device("cuda")
    ds = WM811K(os.path.join(os.path.expanduser("~"), "wafer_cache"))
    P = {s: probs(RUN, ds, getattr(ds, s), device) for s in ("val", "test")}
    Y = {s: ds.labels[getattr(ds, s)] for s in ("val", "test")}
    saved = json.load(open(os.path.join(BASE, RUN, "metrics.json"), encoding="utf-8"))["test"]["macro_f1"]
    print(f"check: recomputed test macro {report(Y['test'], predict(P['test'].copy(), None))['macro']:.4f} vs saved {saved:.4f}")
    print(f"Scratch 수: val {(Y['val'] == S).sum()}, test {(Y['test'] == S).sum()}")

    cols = ["macro", "minor", "scratch", "hit", "miss_none", "fp_none", "pred_s"]
    print(f"\n{'t':>6s} | {'val macro':>9s} {'val Scr':>8s} | " + " ".join(f"{c:>9s}" for c in cols) + "   (test)")
    res = {}
    for t in [None] + THRESHOLDS:
        v, r = report(Y["val"], predict(P["val"].copy(), t)), report(Y["test"], predict(P["test"].copy(), t))
        res[t] = v["macro"]
        print(f"{'argmax' if t is None else t:>6} | {v['macro']:9.3f} {v['scratch']:8.3f} | "
              + " ".join(f"{r[c]:9.3f}" if isinstance(r[c], float) else f"{r[c]:9d}" for c in cols))
    best = max(THRESHOLDS, key=lambda t: res[t])
    print(f"\nval macro-F1 로 고른 t = {best}")


if __name__ == "__main__":
    main()
