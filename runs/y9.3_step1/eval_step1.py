"""y9.3 1단계 평가 (val 만, test 안 봄). seed 0·1·2 × {y9.2b 0 채널, y9.2a Frangi V, y9.3a 길이 가중 V·w, y9.3b 진폭 맞춘 k·V}.
판정 (debate/y9.3/03, 실행 전 고정, y9.3a − y9.2a):
 1 macro 평균 차 ≥ +0.005 이고 3 seed 모두 > 0
 2 none→Scratch 세 seed 합계 20% 이상 감소
 3 Scratch F1 평균 차 ≥ −0.03, Loc→Scratch 합계 증가 20% 이내
 4 소수 5개 각각 평균 F1 차 ≥ −0.03, die<1200 Scratch 재현 평균 차 ≥ −0.05
 y9.3a 가 y9.3b 보다 1·2 에서 낫지 않으면 '길이 정보' 효과로 주장 안 함.
함께: seed 별 짝 차이, 오탐 해소·신규, Scratch AP, 같은 Scratch 재현율 (y9.2a 의 재현율) 에서 none 오탐 수, 길이·크기 구간별 재현율.
사용: python runs/y9.3_step1/eval_step1.py
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from sklearn.metrics import f1_score, average_precision_score
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input, FrangiInput

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO, LO = CLASSES.index("Scratch"), CLASSES.index("none"), CLASSES.index("Loc")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]
va = ds.val; yv = ds.labels[va]
KINDS = {"b": ("y9.2b", None), "a": ("y9.2a", "frangi"), "L": ("y9.3a", "length"), "S": ("y9.3b", "scaled")}
NAME = {"b": "y9.2b 0 채널", "a": "y9.2a V", "L": "y9.3a V·w", "S": "y9.3b k·V"}
TAB = {}   # 표 하나 3.7 GB → 한 번에 하나만 메모리에


def tab(kind):
    if kind not in TAB:
        TAB.clear(); import gc; gc.collect()
        pu, pi = Fr.cache_paths(CACHE, kind); gi = np.load(pi)
        row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi)); TAB[kind] = (torch.from_numpy(np.load(pu)), row)
    return TAB[kind]


@torch.no_grad()
def probs(seed, k):
    meth, kind = KINDS[k]; m = build_model("resnet18", NUM_CLASSES)
    m = FrangiInput(m, *(tab(kind) if kind else (None, None)))
    d = f"runs/resnet18/p0.05/seed{seed}/{meth}_ssl50_ft30"
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    out = []
    for i in range(0, len(va), 1024):
        j = va[i:i + 1024]; out.append(torch.softmax(m(to_input(ds.fetch(j, dev)), torch.as_tensor(j, device=dev)).float(), 1).cpu())
    be = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["best_epoch"]
    del m; torch.cuda.empty_cache(); return torch.cat(out).numpy(), be


# val Scratch 의 선 길이 (원본 격자, 라벨 분석과 같은 불량 die 성분 정의) — 구간별 재현율용
vs = va[yv == SC]; Ls = np.zeros(len(vs))
for j, i in enumerate(vs):
    if sd[i, :2].max() > 96: Ls[j] = np.nan; continue
    h, w = sd[i, :2]; img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    _, _, L = Fr.line_weight(nat, return_parts=True); Ls[j] = L.max()

R = {}
for k in KINDS:
    for seed in (0, 1, 2):
        p, be = probs(seed, k); pred = p.argmax(1); s = yv == SC; ps = pred == SC
        f = f1_score(yv, pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
        R[(seed, k)] = dict(p=p, pred=pred, macro=f.mean(), f1={c: f[CLASSES.index(c)] for c in MINOR}, be=be,
                            rec=(ps & s).sum() / s.sum(), pre=(ps & s).sum() / max(ps.sum(), 1), ap=average_precision_score(s, p[:, SC]),
                            no2sc=int(((yv == NO) & ps).sum()), lo2sc=int(((yv == LO) & ps).sum()),
                            rec_small=(pred[s][dies[vs] < 1200] == SC).mean(), sc_hit=pred[s] == SC)

for seed in (0, 1, 2):
    print(f"\n========== seed {seed} (val) ==========")
    print(f"{'':14s} {'macro':>6s} {'ScrF1':>6s} {'재현':>5s} {'정밀':>5s} {'AP':>5s} {'none→S':>6s} {'Loc→S':>6s} {'재현<1200':>9s} | " + " ".join(f"{c[:5]:>5s}" for c in MINOR) + " | best ep")
    for k in KINDS:
        r = R[(seed, k)]
        print(f"{NAME[k]:14s} {r['macro']:6.3f} {r['f1']['Scratch']:6.3f} {r['rec']:5.2f} {r['pre']:5.2f} {r['ap']:5.3f} {r['no2sc']:6d} {r['lo2sc']:6d} {r['rec_small']:9.2f} | "
              + " ".join(f"{r['f1'][c]:5.2f}" for c in MINOR) + f" | {r['be']}")
    a = R[(seed, "a")]; fa = (yv == NO) & (a["pred"] == SC)
    for k in ("L", "S"):
        r = R[(seed, k)]; fr_ = (yv == NO) & (r["pred"] == SC)
        # 같은 Scratch 재현율: y9.2a 의 재현율이 되도록 P(Scratch) 임계를 정했을 때 none 오탐
        sp = np.sort(r["p"][yv == SC, SC])[::-1]; th = sp[max(int(round(a["rec"] * len(sp))) - 1, 0)]
        sp_a = np.sort(a["p"][yv == SC, SC])[::-1]; th_a = sp_a[max(int(round(a["rec"] * len(sp_a))) - 1, 0)]
        print(f"  {NAME[k]} vs y9.2a: none 오탐 해소 {int((fa & ~fr_).sum())}, 신규 {int((~fa & fr_).sum())} | Scratch 새로 맞힘 {int((~a['sc_hit'] & r['sc_hit']).sum())}, 새로 놓침 {int((a['sc_hit'] & ~r['sc_hit']).sum())} | "
              f"재현 {a['rec']:.2f} 에 맞춘 none 오탐 (P(S) 임계) {int(((yv == NO) & (r['p'][:, SC] >= th)).sum())} 대 y9.2a {int(((yv == NO) & (a['p'][:, SC] >= th_a)).sum())}")

print("\n========== Scratch 재현율: 선 길이 (die) × 웨이퍼 크기 구간 (세 seed 평균, 분모는 val Scratch 장수) ==========")
for lo, hi in ((0, 10), (10, 15), (15, 99)):
    for g, gm in (("die<1200", dies[vs] < 1200), ("die≥1200", dies[vs] >= 1200)):
        m = (Ls >= lo) & (Ls < hi) & gm
        if m.sum():
            print(f"  L {lo:>2}~{hi - 1:<2} {g:9s} n={m.sum():3d}  " + "  ".join(f"{NAME[k]} {np.mean([R[(s, k)]['sc_hit'][m].mean() for s in (0, 1, 2)]):.2f}" for k in KINDS))

avg = lambda f, k1, k2: [f(R[(s, k1)]) - f(R[(s, k2)]) for s in (0, 1, 2)]
for k, ref in (("L", "a"), ("S", "a"), ("L", "S")):
    dm = avg(lambda r: r["macro"], k, ref); n_k = sum(R[(s, k)]["no2sc"] for s in (0, 1, 2)); n_r = sum(R[(s, ref)]["no2sc"] for s in (0, 1, 2))
    l_k = sum(R[(s, k)]["lo2sc"] for s in (0, 1, 2)); l_r = sum(R[(s, ref)]["lo2sc"] for s in (0, 1, 2))
    dsf = np.mean(avg(lambda r: r["f1"]["Scratch"], k, ref)); dmin = {c: np.mean(avg(lambda r: r["f1"][c], k, ref)) for c in MINOR}
    drs = np.mean(avg(lambda r: r["rec_small"], k, ref))
    c1 = np.mean(dm) >= 0.005 and all(d > 0 for d in dm); c2 = n_k <= 0.8 * n_r; c3 = dsf >= -0.03 and l_k <= 1.2 * max(l_r, 1); c4 = min(dmin.values()) >= -0.03 and drs >= -0.05
    print(f"\n========== {NAME[k]} − {NAME[ref]} (seed 0·1·2) ==========")
    print(f"  1) macro 평균 {np.mean(dm):+.4f}, seed별 " + " ".join(f"{d:+.3f}" for d in dm) + f" {'O' if c1 else 'X'}")
    print(f"  2) none→Scratch 합계 {n_r} → {n_k} ({(n_k - n_r) / max(n_r, 1):+.0%}) {'O' if c2 else 'X'}")
    print(f"  3) Scratch F1 평균 {dsf:+.3f}, Loc→Scratch 합계 {l_r} → {l_k} {'O' if c3 else 'X'}")
    print(f"  4) 소수 평균 차 " + " ".join(f"{c} {v:+.3f}" for c, v in dmin.items()) + f", die<1200 재현 {drs:+.3f} {'O' if c4 else 'X'}")
    print(f"  판정: {'통과' if c1 and c2 and c3 and c4 else '미달'}" + ("  (y9.3a − y9.2a 가 본 판정)" if (k, ref) == ("L", "a") else ""))
