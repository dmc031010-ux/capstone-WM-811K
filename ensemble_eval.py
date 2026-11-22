"""두 모델의 분류 결과 조합 (추가 학습 없음): ver3(SimCLR) + ver6.2(마스킹), 5% seed 0.

  python ensemble_eval.py
조합 규칙: score_c = w_c * p_ver6.2_c + (1 - w_c) * p_ver3_c, 예측 = argmax_c score_c
  1) 단순 평균 (모든 w_c = 0.5)
  2) 패턴별 선택 (Random/Donut/Center 는 w=1, 나머지는 w=0)
  3) val 로 고른 클래스별 w_c (test 는 평가에만 사용)
"""
import itertools
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from wafer.data import CLASSES, NUM_CLASSES, WM811K
from wafer.models import build_model, to_input

BASE = os.path.join("runs", "resnet18", "p0.05", "seed0")
MODELS = {"ver3": "ver3_ssl50_ft30", "ver6.2": "ver6.2_ssl50_ft30"}
MASK_CLASSES = ["Random", "Donut", "Center"]
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]


@torch.no_grad()
def probs(run, ds, idx, device, bs=1024):
    m = build_model("resnet18", NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(os.path.join(BASE, run, "best.pt"), map_location=device, weights_only=True))
    m.eval()
    out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(m(to_input(ds.fetch(idx[i:i + bs], device))).float(), 1).cpu())
    return torch.cat(out).numpy()


def scores(y, p):
    pred = p.argmax(1)
    pc = f1_score(y, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return pc.mean(), dict(zip(CLASSES, pc))


def combine(p3, p6, w):
    return w[None] * p6 + (1 - w[None]) * p3


def main():
    device = torch.device("cuda")
    ds = WM811K(os.path.join(os.path.expanduser("~"), "wafer_cache"))
    P = {s: {k: probs(r, ds, getattr(ds, s), device) for k, r in MODELS.items()} for s in ("val", "test")}
    Y = {s: ds.labels[getattr(ds, s)] for s in ("val", "test")}

    for k, r in MODELS.items():   # 검증: 저장된 test 값과 같아야 함
        saved = json.load(open(os.path.join(BASE, r, "metrics.json"), encoding="utf-8"))["test"]["macro_f1"]
        print(f"check {k}: recomputed test macro {scores(Y['test'], P['test'][k])[0]:.4f} vs saved {saved:.4f}")

    w_avg = np.full(NUM_CLASSES, 0.5)
    w_rule = np.array([1.0 if c in MASK_CLASSES else 0.0 for c in CLASSES])
    # 3) val 로 클래스별 w 선택: 좌표 상승 3바퀴, 후보 {0, .25, .5, .75, 1}
    w_val = w_avg.copy()
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for _, c in itertools.product(range(3), range(NUM_CLASSES)):
        best = max(grid, key=lambda g: scores(Y["val"], combine(P["val"]["ver3"], P["val"]["ver6.2"],
                                                                np.where(np.arange(NUM_CLASSES) == c, g, w_val)))[0])
        w_val[c] = best

    rows = {"ver3 단독": np.zeros(NUM_CLASSES), "ver6.2 단독": np.ones(NUM_CLASSES),
            "1) 단순 평균": w_avg, "2) 패턴별 선택": w_rule, "3) val 클래스별 가중치": w_val}
    print("\nval 로 고른 w (ver6.2 비중):", dict(zip(CLASSES, w_val.tolist())))
    print(f"\n{'':22s}{'val macro':>10s}{'test macro':>11s}{'test minor':>11s}" + "".join(f"{c[:9]:>10s}" for c in CLASSES))
    for name, w in rows.items():
        vm = scores(Y["val"], combine(P["val"]["ver3"], P["val"]["ver6.2"], w))[0]
        tm, pc = scores(Y["test"], combine(P["test"]["ver3"], P["test"]["ver6.2"], w))
        print(f"{name:22s}{vm:10.3f}{tm:11.3f}{np.mean([pc[c] for c in MINOR]):11.3f}" + "".join(f"{pc[c]:10.3f}" for c in CLASSES))


if __name__ == "__main__":
    main()
