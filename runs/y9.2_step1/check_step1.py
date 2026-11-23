"""y9.2 1단계 학습 전 점검 (Codex C1). GPU 수 초~수 분, 학습 없음.
1) 캐시: 무작위 라벨 웨이퍼 200장 (96 초과 포함) 을 다시 계산해 표와 같은지, NaN 없음
2) 초기 모델: ver8 인코더 + FrangiInput (Frangi / 0 채널) 의 출력이 1채널 ver8 모델과 같음 (새 채널 가중치 0)
3) 학습 배치: 회전 후 Frangi 채널이 웨이퍼 밖에 남지 않음 (두 채널이 같은 회전), 값 범위 [0,1]
4) 한 스텝 역전파: Frangi 는 새 채널 가중치 기울기 ≠ 0, 0 채널은 = 0
5) state_dict 저장·로드, evaluate (needs_idx) 동작
사용: python runs/y9.2_step1/check_step1.py
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from wafer import frangi as Fr, augment as A
from wafer.data import WM811K, NUM_CLASSES, _resize_nearest
from wafer.models import build_model, to_input, FrangiInput
from wafer.engine import evaluate

dev = torch.device("cuda"); CACHE = os.path.expanduser("~/wafer_cache")
ok = lambda b: "통과" if b else "**실패**"
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
pu, pi = Fr.cache_paths(CACHE); U = np.load(pu); gi = np.load(pi)
row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi))
subs = [np.load(os.path.join(CACHE, f"ver3/runs/resnet18/p0.05/seed{k}/ssl_subset_random.npy")) for k in (0, 1)]
need = np.union1d(np.nonzero(ds.labels >= 0)[0], np.concatenate(subs))
print(f"캐시 {U.shape} uint8, 라벨 웨이퍼 + seed 0·1 SSL 표본 = {len(need)} 장과 같음: {ok(np.array_equal(gi, need))}")

# 1) 다시 계산 (≤96 만: 96 초과는 pkl 필요 → 값 범위·비어있지 않음만)
rng = np.random.default_rng(0)
small = gi[sd[gi, :2].max(1) <= 96]; big = gi[sd[gi, :2].max(1) > 96]
diff = []
for i in rng.choice(small, 200, replace=False):
    h, w = sd[i, :2]; img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    ref = np.rint(_resize_nearest(Fr.vesselness(nat)[None], 96)[0] * 255).astype(np.uint8)
    diff.append(int(np.abs(ref.astype(int) - U[row[i]].astype(int)).max()))
print(f"1) 재계산 200장 최대 차 {max(diff)} (uint8) → {ok(max(diff) == 0)}")
ub = U[row[big]]
outside = [(U[row[i]] > 0) & (ds.fetch([i], 'cpu')[0, 0].numpy() == 0) for i in big[:200]]
print(f"   96 초과 {len(big)}장: 0 아닌 픽셀이 있는 웨이퍼 {np.mean(ub.reshape(len(big), -1).max(1) > 0):.2f}, "
      f"웨이퍼 밖에 응답 {sum(o.sum() for o in outside)} 픽셀 → {ok(sum(o.sum() for o in outside) == 0)}")

# 2) 초기 모델 출력 같음
enc = os.path.join(CACHE, "ver3/runs/resnet18/p0.05/seed0/masklat8_e50.pt")
torch.manual_seed(0); m1 = build_model("resnet18", NUM_CLASSES)
r = m1.load_state_dict(torch.load(enc, map_location="cpu", weights_only=True), strict=False); assert set(r.missing_keys) == {"fc.weight", "fc.bias"}
table = torch.from_numpy(U)
import copy
mf = FrangiInput(copy.deepcopy(m1), table, row).to(dev).eval(); mz = FrangiInput(copy.deepcopy(m1), None, None).to(dev).eval(); m1 = m1.to(dev).eval()
idx = rng.choice(gi[ds.labels[gi] >= 0], 256, replace=False); x = to_input(ds.fetch(idx, dev))
with torch.no_grad():
    o1, of, oz = m1(x), mf(x, torch.as_tensor(idx, device=dev)), mz(x, torch.as_tensor(idx, device=dev))
print(f"2) 초기 출력 최대 차: Frangi {(o1 - of).abs().max().item():.2e}, 0 채널 {(o1 - oz).abs().max().item():.2e} → {ok(torch.allclose(o1, of, atol=1e-4) and torch.allclose(o1, oz, atol=1e-4))}")

# 3) 학습 배치 (selfsup.finetune 과 같은 순서)
torch.manual_seed(1)
xl = torch.cat([ds.fetch(idx, dev), 2 * mf.frangi(idx, dev)], 1); xr = A.weak(xl, "rotate"); xin = to_input(xr)
leak = ((xin[:, 1] > 0) & (xin[:, 0] == 0)).sum().item()
print(f"3) 회전 후 웨이퍼 밖 Frangi 픽셀 {leak} → {ok(leak == 0)} | Frangi 채널 범위 [{xin[:, 1].min().item():.3f}, {xin[:, 1].max().item():.3f}] → {ok(0 <= xin[:, 1].min().item() and xin[:, 1].max().item() <= 1)}")

# 4) 한 스텝 역전파
y = torch.as_tensor(ds.labels[idx], device=dev)
for name, m in (("Frangi", mf), ("0 채널", mz)):
    m.train(); m.zero_grad()
    xi = to_input(A.weak(torch.cat([ds.fetch(idx, dev), 2 * m.frangi(idx, dev)], 1), "rotate"))
    F.cross_entropy(m(xi), y).backward()
    g = m.base.conv1.weight.grad[:, 1].abs().sum().item()
    print(f"4) {name}: 새 채널 가중치 기울기 합 {g:.3e} → {ok(g > 0 if name == 'Frangi' else g == 0)}")

# 5) 저장·로드, evaluate
p = os.path.join(HERE, "_tmp.pt"); torch.save(mf.state_dict(), p)
m2 = build_model("resnet18", NUM_CLASSES); m2 = FrangiInput(m2, table, row).to(dev)
r = m2.load_state_dict(torch.load(p, weights_only=True)); os.remove(p)
ev = evaluate(m2, ds, ds.val[:2048], dev)
print(f"5) 저장·로드 {ok(not r.missing_keys and not r.unexpected_keys)} | evaluate (val 2048장) macro {ev['macro_f1']:.3f} (fc 학습 전이라 의미 없음, 동작만 확인)")
