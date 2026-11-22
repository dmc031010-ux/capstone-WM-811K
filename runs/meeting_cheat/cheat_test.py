"""형제 마스킹본 간 치팅 검증. python cheat_test.py <umm 경로> grad|train|eval

grad : 학습된 ver7 인코더 — 1번 뷰 출력의 입력 미분이 형제 뷰 / 상관없는 웨이퍼에 얼마나 큰지 (train BN, eval BN)
train: 같은 seed 로 bn_groups="together" / "split" 5 에폭씩 사전학습 (ckpt 남김)
eval : 두 모델을 같은 고정 배치로 together / split 두 방식 모두 채점
"""
import os, sys, types
import numpy as np, torch
proj, mode = sys.argv[1], sys.argv[2]; os.chdir(proj); sys.path.insert(0, proj)
from wafer import ver6
from wafer.data import WM811K
from wafer.models import to_input
from wafer.selfsup import _encoder

dev = torch.device("cuda")
H = os.path.expanduser("~")
CB = os.path.join(H, "wafer_cache", "ver3", "runs", "resnet18", "p0.05", "seed0")
OUT = os.path.dirname(os.path.abspath(__file__))
ds = WM811K(os.path.join(H, "wafer_cache"))
sub = np.load(os.path.join(CB, "ssl_subset_random.npy"))


def batch(i, b=256):
    torch.manual_seed(1000 + i)  # 채점용 고정 배치 / 마스크
    idx = np.sort(np.random.default_rng(1000 + i).choice(sub, b, replace=False))
    with torch.no_grad():
        return ver6._views(ds.fetch(idx, dev), True, cell=6)


if mode == "grad":
    enc, _ = _encoder("resnet18")
    enc.load_state_dict({k: v for k, v in torch.load(os.path.join(CB, "masklat7_e50.pt"), map_location="cpu", weights_only=True).items() if not k.startswith("fc.")})
    enc = enc.to(dev)
    _, xm = batch(0, 64)                       # 웨이퍼 64장 × 마스킹본 4장 = 256장 한 묶음 (형제 포함)
    n, b = xm.shape[:2]
    u = torch.nn.functional.normalize(torch.randn(512, device=dev), dim=0)
    for bn in ("train", "eval"):
        enc.train() if bn == "train" else enc.eval()
        sib, oth, own = [], [], []
        for w in range(16):                    # 기준 = 웨이퍼 w 의 0번 마스킹본
            v = to_input(xm.flatten(0, 1)).clone().requires_grad_(True)
            (enc(v)[w] @ u).backward()         # 순서: (뷰 0 의 64장, 뷰 1 의 64장, ...) → 0번 뷰 = 인덱스 w
            g = v.grad.abs().flatten(1).sum(1).view(n, b)
            own.append(g[0, w].item())
            sib.append(g[1:, w].mean().item())
            m = torch.ones(n, b, dtype=torch.bool, device=dev); m[:, w] = False
            oth.append(g[m].mean().item())
        print(f"BN {bn:5s}: 자기 입력 {np.mean(own):.3e} | 형제 뷰 3장 평균 {np.mean(sib):.3e} | 상관없는 웨이퍼 평균 {np.mean(oth):.3e}"
              f" | 형제/상관없음 = {np.mean(sib) / max(np.mean(oth), 1e-30):.2f}  (max 형제 {max(sib):.3e})", flush=True)

elif mode == "train":
    arm = sys.argv[3]
    args = types.SimpleNamespace(seed=0, arch="resnet18", batch_size=256, ssl_epochs=5)
    ver6.os.remove = lambda p: None            # 끝나도 ckpt (predictor 포함 전체 모델) 남기기
    print(f"=== bn_groups={arm}", flush=True)
    ver6.pretrain(ds, sub, args, dev, os.path.join(OUT, f"cheat_{arm}.pt"), v61=True, cell=6, bn_groups=arm)

else:
    torch.manual_seed(0)
    for arm in ("together", "split"):
        model = ver6.MaskedLatentModel("resnet18", mlp=True).to(dev)
        model.load_state_dict(torch.load(os.path.join(OUT, f"cheat_{arm}.pt.ckpt"), map_location=dev, weights_only=False)["model"])
        model.train()
        res = {g: [] for g in ("together", "split")}
        with torch.no_grad():
            for i in range(40):
                x, xm = batch(i)
                for g in res:
                    torch.manual_seed(i)
                    res[g].append(ver6.consistency_loss(model, x, xm, bn_groups=g)[0].item())
        t, s = np.array(res["together"]), np.array(res["split"])
        print(f"{arm:8s}로 학습한 모델: 같은 반 채점 {t.mean():.5f} | 다른 반 채점 {s.mean():.5f} | 차이(다른-같은) {(s - t).mean():+.5f} ± {(s - t).std() / np.sqrt(len(t)):.5f}", flush=True)
