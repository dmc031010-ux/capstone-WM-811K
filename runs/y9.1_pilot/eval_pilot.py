"""y9.1 파일럿 평가 (seed 0, val 만). debate/y9.1/04_decision.md 의 4조건 + 참고선.

[1] 사전학습 로그: 마지막 loss, z_std, 뷰별 cos
[2] 고정 인코더 (fine-tune 전): 라벨 6,919장으로 로지스틱 회귀 → val macro / Scratch F1, kNN (k=20, 코사인) val macro, val 특징 유효 랭크
[3] fine-tune 후: val macro / Scratch F1 / 소수 5개 (metrics.json, test 는 보지 않음)
[4] 잡음 곡선 (0단계와 같은 방식): 실제 val Scratch 에 die 단위 랜덤 불량 0/2/4/8% 추가 시 Scratch 판정 비율
사용: python runs/y9.1_pilot/eval_pilot.py
"""
import json, os, re, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache"); ENC = os.path.join(CACHE, "ver3/runs/resnet18/p0.05/seed0")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]
ARMS = [("① 마스킹 (proj 없음)", "y9.1a", "masklat91a_e10.pt"), ("② 마스킹 + proj", "y9.1b", "masklat91b_e10.pt"),
        ("③ 회전만 + proj", "y9.1c", "masklat91c_e10.pt"), ("④ 노이즈 + proj", "y9.1d", "masklat91d_e10.pt")]
REFS = [("참고: y9 (마스킹 50ep)", "y9", "masklat9_e50.pt", "runs/resnet18/p0.05/seed0/y9_ssl50_ft30"),
        ("참고: SimCLR (ver3 50ep)", None, "simclr_random_cut_e50.pt", "runs/resnet18/p0.05/seed0/ver3_ssl50_ft30")]
lab = ds.labeled_subset(0.05, seed=0); ylab = ds.labels[lab]; va = ds.val; yva = ds.labels[va]


def enc(fname):
    m = build_model("resnet18", NUM_CLASSES)
    r = m.load_state_dict(torch.load(os.path.join(ENC, fname), map_location="cpu", weights_only=True), strict=False)
    assert set(r.missing_keys) == {"fc.weight", "fc.bias"} and not r.unexpected_keys, r
    m.fc = nn.Identity(); return m.to(dev).eval()


@torch.no_grad()
def feats(m, idx):
    o = []
    for i in range(0, len(idx), 1024):
        o.append(m(to_input(ds.fetch(idx[i:i + 1024], dev))).float().cpu())
    return torch.cat(o)


def eff_rank(F_):
    s = torch.linalg.svdvals(F_ - F_.mean(0)); p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def log_tail(method):
    f = f"runs/log_{method}_p0.05_s0.txt"
    L = [l for l in open(f, encoding="utf-8", errors="ignore") if l.startswith("ssl epoch=")]
    if not L: return "로그 없음"
    m = re.search(r"loss=([\d.]+) z_std=([\d.]+).*?view_cos=([\d./]+)", L[-1])
    return f"epoch {len(L)}: loss {m.group(1)} z_std {m.group(2)} 뷰별 cos {m.group(3)}" if m else L[-1].strip()


# 잡음 곡선용 데이터 (실제 val Scratch, 원본 격자 ≤ 96)
rng = np.random.default_rng(0)
vs = va[(yva == SC) & (sd[va, :2].max(1) <= 96)]
PS = [0.0, 0.02, 0.04, 0.08]
NOISY = {p: [] for p in PS}
for i in vs:
    h, w = sd[i, :2]; img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(int); ci = ((np.arange(96) + 0.5) * w / 96).astype(int)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    u = rng.random(nat.shape)
    for p in PS:
        NOISY[p].append(np.where((nat == 1) & (u < p), 2, nat).astype(np.uint8)[np.ix_(ri, ci)])
NOISY = {p: torch.as_tensor(np.stack(v), dtype=torch.float32)[:, None] for p, v in NOISY.items()}


@torch.no_grad()
def noise_curve(best):
    m = build_model("resnet18", NUM_CLASSES); m.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    return [float((m(to_input(NOISY[p].to(dev))).argmax(1) == SC).float().mean()) for p in PS]


rows = []
for name, method, fname, *rest in ARMS + REFS:
    if not os.path.exists(os.path.join(ENC, fname)):
        print(f"{name}: 인코더 없음 ({fname})"); continue
    e = enc(fname); ftr, fva = feats(e, lab), feats(e, va)
    sc_ = StandardScaler().fit(ftr.numpy())
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc_.transform(ftr.numpy()), ylab)
    pv = clf.predict(sc_.transform(fva.numpy())); lf = f1_score(yva, pv, labels=range(NUM_CLASSES), average=None, zero_division=0)
    sim = Fn.normalize(fva, dim=1) @ Fn.normalize(ftr, dim=1).T
    nb = sim.topk(20, dim=1).indices.numpy(); kp = np.array([np.bincount(ylab[r], minlength=NUM_CLASSES).argmax() for r in nb])
    kf = f1_score(yva, kp, labels=range(NUM_CLASSES), average=None, zero_division=0)
    rk = eff_rank(fva)
    run_dir = rest[0] if rest else f"runs/resnet18/p0.05/seed0/{method}_ssl10_ft30"
    mt = os.path.join(run_dir, "metrics.json")
    ft = json.load(open(mt, encoding="utf-8")) if os.path.exists(mt) else None
    curve = noise_curve(os.path.join(run_dir, "best.pt")) if ft else None
    rows.append(name)
    print(f"\n{name}")
    if method and method.startswith("y9.1"):
        print(f"  [1] {log_tail(method)}")
    print(f"  [2] 고정 인코더: 선형 val macro {lf.mean():.3f} Scratch {lf[SC]:.3f} | kNN val macro {kf.mean():.3f} Scratch {kf[SC]:.3f} | 유효 랭크 {rk:.1f}")
    if ft:
        v = ft["val"]; pc = v["per_class_f1"]
        print(f"  [3] fine-tune: val macro {v['macro_f1']:.3f} Scratch {pc['Scratch']:.3f} 소수5 {np.mean([pc[c] for c in MINOR]):.3f} | "
              + " ".join(f"{c} {pc[c]:.2f}" for c in MINOR))
        print(f"  [4] 잡음 곡선 (val Scratch {len(vs)}장, 추가 불량 0/2/4/8%): " + " / ".join(f"{c:.2f}" for c in curve))
    else:
        print("  [3][4] fine-tune 결과 없음")
