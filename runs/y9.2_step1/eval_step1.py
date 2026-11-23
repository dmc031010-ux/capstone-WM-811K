"""y9.2 1단계 평가 (val 만, test 안 봄). seed 0·1 × {ver8 기존, y9.2b (0 채널), y9.2a (Frangi 채널)}.

[A] val: macro, 소수 5개 F1, Scratch F1·재현·정밀, 혼동 (none→Scratch, Loc→Scratch, Scratch→none, Scratch→Loc),
    die 수 <1200 / ≥1200 Scratch 재현율, Scratch 대 나머지 PR-AUC (average precision)
[B] 같은 seed 짝비교 (val Scratch 119장): ver8 대비 새로 맞힘 / 새로 놓침
[C] 굵기 조작 (Codex C4·C6, val none 배경): 같은 배경·같은 중심선에 1/2/3 die 선 → 평균 Scratch 확률.
    Frangi 채널은 조작된 원본 격자에서 다시 계산. 배경 die ≥1200 100장, 300~1199 100장.
진행 기준 (debate/y9.2/03): seed 0·1 둘 다 (a)−(b) val macro > 0, 소수 5개 중 −0.03 넘는 하락 없음, die<1200 Scratch 재현율 상승
사용: python runs/y9.2_step1/eval_step1.py
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.metrics import f1_score, average_precision_score
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES, _resize_nearest
from wafer.models import build_model, to_input, FrangiInput

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO, LO = CLASSES.index("Scratch"), CLASSES.index("none"), CLASSES.index("Loc")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]
pu, pi = Fr.cache_paths(CACHE); table = torch.from_numpy(np.load(pu)); gi = np.load(pi)
row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi))
va = ds.val; yv = ds.labels[va]


def load(seed, kind):
    d = f"runs/resnet18/p0.05/seed{seed}/" + {"ver8": "ver8_ssl50_ft30", "b": "y9.2b_ssl50_ft30", "a": "y9.2a_ssl50_ft30"}[kind]
    m = build_model("resnet18", NUM_CLASSES)
    if kind != "ver8":
        m = FrangiInput(m, table if kind == "a" else None, row if kind == "a" else None)
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=True))
    return m.to(dev).eval(), d


@torch.no_grad()
def probs_idx(m, idx):
    out = []
    for i in range(0, len(idx), 1024):
        x = to_input(ds.fetch(idx[i:i + 1024], dev)); j = torch.as_tensor(idx[i:i + 1024], device=dev)
        out.append(torch.softmax((m(x, j) if getattr(m, "needs_idx", False) else m(x)).float(), 1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def probs_img(m, X, Fm):
    """X: (n,96,96) {0,1,2}, Fm: (n,96,96) Frangi [0,1] (Frangi 모델만 사용)."""
    out = []
    for i in range(0, len(X), 512):
        x = to_input(torch.as_tensor(X[i:i + 512], dtype=torch.float32, device=dev)[:, None])
        if isinstance(m, FrangiInput):
            f = torch.as_tensor(Fm[i:i + 512], dtype=torch.float32, device=dev)[:, None] if m.table is not None else torch.zeros_like(x)
            x = torch.cat([x, f], 1)
        out.append(torch.softmax(m(x).float(), 1).cpu())
    return torch.cat(out).numpy()


# ---------- [C] 굵기 조작 이미지 (모델과 무관, 한 번만) ----------
rng = np.random.default_rng(0)


def native(i):
    h, w = sd[i, :2]; img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    return img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]


none_ok = va[(yv == NO) & (sd[va, :2].max(1) <= 96)]
MAN = {}
for gname, lo, hi in [("die ≥1200", 1200, 10 ** 9), ("die 300~1199", 300, 1200)]:
    ids = rng.permutation(none_ok[(dies[none_ok] >= lo) & (dies[none_ok] < hi)])[:100]
    X = {t: [] for t in (1, 2, 3)}; Fm = {t: [] for t in (1, 2, 3)}
    for i in ids:
        nat0 = native(i); h, w = nat0.shape; ang = rng.uniform(0, 180); cen = (0.5 + rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.15, 0.15))
        for t in (1, 2, 3):
            ln, _ = Fr.line_map(h, w, t, ang, length=0.5, center=cen)
            nat = nat0.copy(); nat[(ln == 2) & (nat0 > 0)] = 2
            X[t].append(_resize_nearest(nat[None], 96)[0]); Fm[t].append(_resize_nearest(Fr.vesselness(nat)[None], 96)[0])
    MAN[gname] = ({t: np.stack(v) for t, v in X.items()}, {t: np.stack(v) for t, v in Fm.items()})

small_sc = dies[va[yv == SC]] < 1200
R = {}
for seed in (0, 1):
    P = {}
    for kind in ("ver8", "b", "a"):
        m, d = load(seed, kind); p = probs_idx(m, va); P[kind] = p; pred = p.argmax(1)
        f = f1_score(yv, pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
        s = yv == SC; ps = pred == SC
        rec = (ps & s).sum() / s.sum(); pre = (ps & s).sum() / max(ps.sum(), 1)
        sc_pred = pred[s]
        man = {g: [float(probs_img(m, X[t], Fm[t])[:, SC].mean()) for t in (1, 2, 3)] for g, (X, Fm) in MAN.items()}
        R[(seed, kind)] = dict(macro=f.mean(), f1={c: f[CLASSES.index(c)] for c in MINOR}, rec=rec, pre=pre,
                               no2sc=int(((yv == NO) & ps).sum()), lo2sc=int(((yv == LO) & ps).sum()),
                               sc2no=int((sc_pred == NO).sum()), sc2lo=int((sc_pred == LO).sum()),
                               rec_small=(sc_pred[small_sc] == SC).mean(), rec_big=(sc_pred[~small_sc] == SC).mean(),
                               ap=average_precision_score(s, p[:, SC]), man=man,
                               best_epoch=json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))["best_epoch"])
        del m; torch.cuda.empty_cache()
    for kind in ("b", "a"):
        a0 = P["ver8"].argmax(1)[yv == SC] == SC; a1 = P[kind].argmax(1)[yv == SC] == SC
        R[(seed, kind)]["new_hit"] = int((~a0 & a1).sum()); R[(seed, kind)]["new_miss"] = int((a0 & ~a1).sum())

NAME = {"ver8": "ver8 (기존)", "b": "y9.2b 0 채널", "a": "y9.2a Frangi"}
for seed in (0, 1):
    print(f"\n========== seed {seed} (val) ==========")
    print(f"{'':16s} {'macro':>6s} {'Scr F1':>6s} {'재현':>5s} {'정밀':>5s} {'AP':>5s} | 재현 die<1200 / ≥1200 | none→S Loc→S S→none S→Loc | " + " ".join(f"{c[:5]:>5s}" for c in MINOR) + " | best ep")
    for kind in ("ver8", "b", "a"):
        r = R[(seed, kind)]
        print(f"{NAME[kind]:16s} {r['macro']:6.3f} {r['f1']['Scratch']:6.3f} {r['rec']:5.2f} {r['pre']:5.2f} {r['ap']:5.3f} | {r['rec_small']:13.2f} / {r['rec_big']:.2f} | "
              f"{r['no2sc']:6d} {r['lo2sc']:5d} {r['sc2no']:6d} {r['sc2lo']:5d} | " + " ".join(f"{r['f1'][c]:5.2f}" for c in MINOR) + f" | {r['best_epoch']}")
    for kind in ("b", "a"):
        r = R[(seed, kind)]; print(f"  {NAME[kind]} vs ver8 짝비교 (val Scratch 119장): 새로 맞힘 {r['new_hit']}, 새로 놓침 {r['new_miss']}")
    print("  굵기 조작 (val none 배경, 평균 Scratch 확률, 선 1 / 2 / 3 die):")
    for g in MAN:
        print(f"    {g:12s} " + "   ".join(f"{NAME[k]} " + " / ".join(f"{v:.2f}" for v in R[(seed, k)]['man'][g]) for k in ("ver8", "b", "a")))
    ra, rb = R[(seed, "a")], R[(seed, "b")]
    drop = {c: ra["f1"][c] - rb["f1"][c] for c in MINOR}
    print(f"  진행 기준 (a − b): macro {ra['macro'] - rb['macro']:+.3f} {'O' if ra['macro'] > rb['macro'] else 'X'} | "
          f"소수 최대 하락 {min(drop.values()):+.3f} ({min(drop, key=drop.get)}) {'O' if min(drop.values()) >= -0.03 else 'X'} | "
          f"die<1200 재현 {ra['rec_small'] - rb['rec_small']:+.3f} {'O' if ra['rec_small'] > rb['rec_small'] else 'X'}")
