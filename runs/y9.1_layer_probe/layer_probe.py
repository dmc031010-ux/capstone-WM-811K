"""어느 단계에서 Scratch 정보가 약해지는가: 레이어별 선형 probe (학습 없음, val 만).

모델: fine-tune 된 seed 0 체크포인트 (① 마스킹 10ep, ③ 회전만 10ep, y9 50ep, SimCLR 50ep).
  fine-tune 이 train 라벨을 이미 봤으므로 probe 는 train 이 아니라 val 안에서 5-fold 교차검증 (3회 반복).
특징: stem (maxpool 뒤, 24×24), layer1 (24×24), layer2 (12×12), layer3 (6×6), layer4 (3×3)
  각 지도를 공간 평균 (Avg) / 공간 최대 (Max) 로 벡터화. 참고로 모델 자체의 최종 출력 (fc 확률) 도 같이.
과제: Scratch vs none (none 3,000장 무작위), Scratch vs Loc. 지표 AUC (임계값 무관).
  Scratch 를 die 수 <1200 / ≥1200 으로 나눠 out-of-fold 점수로 AUC 를 따로 계산.
해석: Avg 의 AUC 가 갑자기 떨어지는 레이어 = 정보가 약해지는 곳. 같은 레이어에서 Max ≫ Avg = 평균이 희석.
사용: python runs/y9.1_layer_probe/layer_probe.py
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
dies = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))[:, 2]
SC, NO, LO = CLASSES.index("Scratch"), CLASSES.index("none"), CLASSES.index("Loc")
R0 = "runs/resnet18/p0.05/seed0"
MODELS = [("① 마스킹 10ep", f"{R0}/y9.1a_ssl10_ft30/best.pt"), ("③ 회전만 10ep", f"{R0}/y9.1c_ssl10_ft30/best.pt"),
          ("y9 50ep", f"{R0}/y9_ssl50_ft30/best.pt"), ("SimCLR 50ep", f"{R0}/ver3_ssl50_ft30/best.pt")]
STAGES = ["stem", "layer1", "layer2", "layer3", "layer4"]

va = ds.val; yv = ds.labels[va]; rng = np.random.default_rng(0)
i_sc, i_lo = va[yv == SC], va[yv == LO]
i_no = rng.choice(va[yv == NO], 3000, replace=False)
IDX = np.concatenate([i_sc, i_no, i_lo]); Y = ds.labels[IDX]
small = dies[i_sc] < 1200
print(f"val Scratch {len(i_sc)} (die<1200 {small.sum()}, ≥1200 {(~small).sum()}), none {len(i_no)}, Loc {len(i_lo)}")


@torch.no_grad()
def features(path):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    F = {f"{s}/{p}": [] for s in STAGES for p in ("Avg", "Max")}; out = []
    for i in range(0, len(IDX), 512):
        x = to_input(ds.fetch(IDX[i:i + 512], dev))
        h = m.maxpool(m.relu(m.bn1(m.conv1(x)))); maps = {"stem": h}
        for s in STAGES[1:]:
            h = getattr(m, s)(h); maps[s] = h
        for s, a in maps.items():
            F[f"{s}/Avg"].append(a.mean((2, 3)).float().cpu()); F[f"{s}/Max"].append(a.amax((2, 3)).float().cpu())
        out.append(torch.softmax(m.fc(torch.flatten(m.avgpool(h), 1)), 1).cpu())
        assert torch.allclose(out[-1], torch.softmax(m(x), 1).cpu(), atol=1e-4)
    return {k: torch.cat(v).numpy() for k, v in F.items()}, torch.cat(out).numpy()


def probe(X, neg):
    """Scratch vs neg 클래스. 5-fold × 3회 out-of-fold 점수 → AUC (전체 / die 적음 / die 많음)."""
    sel = (Y == SC) | (Y == neg); Xs, ys = X[sel], (Y[sel] == SC).astype(int)
    pos = np.where(ys == 1)[0]; negs = np.where(ys == 0)[0]
    res = []
    for r in range(3):
        oof = np.zeros(len(ys))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Xs, ys):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            oof[te] = clf.fit(Xs[tr], ys[tr]).decision_function(Xs[te])
        auc = lambda p: roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(negs))], np.r_[oof[p], oof[negs]])
        res.append((auc(pos), auc(pos[small]), auc(pos[~small])))
    return np.array(res)  # (3, 3)


def own(P, neg):
    sel = (Y == SC) | (Y == neg); s = P[sel, SC]; ys = (Y[sel] == SC)
    pos, negs = s[ys], s[~ys]
    f = lambda p: roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(negs))], np.r_[p, negs])
    return f(pos), f(pos[small]), f(pos[~small])


FEAT = {mn: features(path) for mn, path in MODELS}
for neg, nn_ in [(NO, "none"), (LO, "Loc")]:
    print(f"\n===== Scratch vs {nn_}: AUC (5-fold×3 평균±sd)  [전체 | die<1200 | die≥1200] =====")
    for mn, _ in MODELS:
        F, P = FEAT[mn]
        print(f"\n{mn}")
        for s in STAGES:
            line = f"  {s:7s}"
            for p in ("Avg", "Max"):
                r = probe(F[f"{s}/{p}"], neg); m_, sd_ = r.mean(0), r[:, 0].std()
                line += f"  {p} {m_[0]:.3f}±{sd_:.3f} | {m_[1]:.3f} | {m_[2]:.3f}   "
            print(line)
        o = own(P, neg)
        print(f"  {'최종출력':7s}  모델 자신의 Scratch 확률 AUC {o[0]:.3f} | {o[1]:.3f} | {o[2]:.3f}")
