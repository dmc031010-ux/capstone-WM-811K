"""판정층만 바꾸면 나아지는가 (학습 없음 = 인코더 고정, 헤드만 train 라벨로 다시 맞춤). 평가 val 만.

모델: fine-tune 된 seed 0 체크포인트 (① 마스킹 10ep, ③ 회전만 10ep, y9 50ep, SimCLR 50ep). 인코더 고정.
헤드 (모두 train 라벨 5% = 6,919장 으로만 학습, 하이퍼파라미터 고정, val 은 채점만):
  fc 원본       : fine-tune 때 학습된 그대로 (기준)
  L4 선형       : layer4 평균 특징 + 로지스틱 (C=0.1), 클래스 가중 없음 / balanced 두 가지
  L4 프로토타입 : layer4 평균 특징의 클래스 평균과의 코사인 유사도 (가장 가까운 클래스)
  L3 선형       : layer3 평균 특징 (256) + 로지스틱
  L3+L4 선형    : 두 특징 이어 붙임
지표: val macro F1, Scratch F1 / 재현율 / 정밀도, none→Scratch 오탐 수, 소수 5개 F1
주의: 인코더가 이미 train 라벨로 fine-tune 되어 train 특징은 과적합되어 있음 → 같은 라벨로 맞춘 헤드가 fc 와 비슷해질 수 있음.
사용: python runs/y9.1_head_probe/head_probe.py
"""
import os, sys
import numpy as np, torch, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input

dev = torch.device("cuda")
ds = WM811K(os.path.expanduser("~/wafer_cache"))
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]; MI = [CLASSES.index(c) for c in MINOR]
R0 = "runs/resnet18/p0.05/seed0"
MODELS = [("① 마스킹 10ep", f"{R0}/y9.1a_ssl10_ft30/best.pt"), ("③ 회전만 10ep", f"{R0}/y9.1c_ssl10_ft30/best.pt"),
          ("y9 50ep", f"{R0}/y9_ssl50_ft30/best.pt"), ("SimCLR 50ep", f"{R0}/ver3_ssl50_ft30/best.pt")]
tr = ds.labeled_subset(0.05, seed=0); ytr = ds.labels[tr]; va = ds.val; yva = ds.labels[va]
print(f"train {len(tr)}장 (Scratch {(ytr == SC).sum()}), val {len(va)}장 (Scratch {(yva == SC).sum()})")


@torch.no_grad()
def feats(m, idx):
    L3, L4, P = [], [], []
    for i in range(0, len(idx), 1024):
        x = to_input(ds.fetch(idx[i:i + 1024], dev))
        h3 = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
        h4 = m.layer4(h3); f4 = h4.mean((2, 3))
        L3.append(h3.mean((2, 3)).float().cpu()); L4.append(f4.float().cpu()); P.append(m.fc(f4).argmax(1).cpu())
    return torch.cat(L3).numpy(), torch.cat(L4).numpy(), torch.cat(P).numpy()


def report(name, pred):
    f = f1_score(yva, pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    s = (yva == SC); p = (pred == SC)
    rec = recall_score(s, p, zero_division=0); pre = precision_score(s, p, zero_division=0)
    fp = int(((yva == NO) & p).sum())
    print(f"  {name:22s} macro {f.mean():.3f} | Scratch F1 {f[SC]:.3f} 재현 {rec:.2f} 정밀 {pre:.2f} none→Scratch {fp:3d} | "
          + " ".join(f"{c[:4]} {f[i]:.2f}" for c, i in zip(MINOR, MI)))


def lin(Xtr, Xva, cw=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight=cw))
    return clf.fit(Xtr, ytr).predict(Xva)


def proto(Xtr, Xva):
    t = Fn.normalize(torch.as_tensor(Xtr), dim=1); v = Fn.normalize(torch.as_tensor(Xva), dim=1)
    C = Fn.normalize(torch.stack([t[torch.as_tensor(ytr == k)].mean(0) for k in range(NUM_CLASSES)]), dim=1)
    return (v @ C.T).argmax(1).numpy()


for mn, path in MODELS:
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    a3, a4, _ = feats(m, tr); b3, b4, pfc = feats(m, va)
    print(f"\n{mn}")
    report("fc 원본", pfc)
    report("L4 선형", lin(a4, b4))
    report("L4 선형 balanced", lin(a4, b4, "balanced"))
    report("L4 프로토타입", proto(a4, b4))
    report("L3 선형", lin(a3, b3))
    report("L3+L4 선형", lin(np.hstack([a3, a4]), np.hstack([b3, b4])))
