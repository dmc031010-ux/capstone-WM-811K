"""ver8 vs ver8.3 (seed 0): 학습된 클래스별 r, 전체/소수/클래스별 F1, die 수 적은/많은 웨이퍼, none 오탐, Scratch 오답 64장 중 회수, #13."""
import json, os, sys
import numpy as np, torch
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from sklearn.metrics import f1_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import ClassPool, build_model, to_input

dev = torch.device("cuda")
ds = WM811K(os.path.expanduser("~/wafer_cache"))
dies = np.load(os.path.expanduser("~/wafer_cache/wm811k_shapes_dies.npy"))[:, 2]
te = ds.test; y = ds.labels[te]; small = dies[te] < 1200
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]


@torch.no_grad()
def predict(run):
    m = build_model("resnet18", NUM_CLASSES)
    if run == "ver8.3":
        m = ClassPool(m)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed0/{run}_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    m = m.to(dev).eval(); o = []
    for i in range(0, len(te), 1024):
        o.append(torch.softmax(m(to_input(ds.fetch(te[i:i + 1024], dev))).float(), 1).cpu())
    return torch.cat(o).numpy(), m


P, R = {}, None
for run in ("ver8", "ver8.3"):
    P[run], m = predict(run)
    if run == "ver8.3":
        R = m.r.detach().cpu().numpy()
print("학습된 r (0 = 평균, + 클수록 최대 쪽, - 클수록 최소 쪽):")
print("   " + "  ".join(f"{c} {v:+.2f}" for c, v in zip(CLASSES, R)))
for run in ("ver8", "ver8.3"):
    mt = json.load(open(f"runs/resnet18/p0.05/seed0/{run}_ssl50_ft30/metrics.json", encoding="utf-8"))
    p = P[run].argmax(1); pc = mt["test"]["per_class_f1"]
    assert abs(f1_score(y, p, labels=range(NUM_CLASSES), average="macro", zero_division=0) - mt["test"]["macro_f1"]) < 1e-3
    print(f"\n[{run}] test macro {mt['test']['macro_f1']:.3f} · val {mt['val']['macro_f1']:.3f} · 소수5 {np.mean([pc[c] for c in MINOR]):.3f} · best epoch {mt.get('best_epoch')}")
    print("   " + " ".join(f"{c} {pc[c]:.3f}" for c in CLASSES))
    for g, gm in (("die<1200", small), ("die≥1200", ~small)):
        yt, yp = y[gm], p[gm]; sc = yt == SC
        mac = f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0)
        print(f"   {g}: macro {mac:.3f} · Scratch 맞힘 {(yp[sc] == SC).sum()}/{sc.sum()} · Scr→none {(yp[sc] == NO).sum()} · Scr→Loc {(yp[sc] == CLASSES.index('Loc')).sum()}"
              f" · none→불량 오탐 {((yt == NO) & (yp != NO)).sum()} (그중 →Scratch {((yt == NO) & (yp == SC)).sum()})")
w8 = (y == SC) & (P["ver8"].argmax(1) != SC)
p83 = P["ver8.3"].argmax(1)
print(f"\nver8 이 틀린 Scratch {w8.sum()}장 중 ver8.3 이 맞힘 {((p83 == SC) & w8).sum()}장 / ver8 이 맞힌 {((y == SC) & ~w8).sum()}장 중 ver8.3 이 틀림 {((p83 != SC) & (y == SC) & ~w8).sum()}장")
k = int(np.nonzero(te == 642918)[0][0])
print(f"#13 (idx 642918): ver8 none {P['ver8'][k, NO]:.3f} Scr {P['ver8'][k, SC]:.4f} → ver8.3 none {P['ver8.3'][k, NO]:.3f} Scr {P['ver8.3'][k, SC]:.4f} (예측 {CLASSES[p83[k]]})")
