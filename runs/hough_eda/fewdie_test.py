"""die 수가 적은 웨이퍼의 Scratch 가 none 으로 가는 이유 검증 (ver8 seed 0·1).

가설
 H1 학습 데이터: 라벨 Scratch 48장에 die 수 적은 웨이퍼가 적다 (보지 못한 유형)
 H2 사전 확률: die 수 적은 웨이퍼는 원래 none 비율이 높다 → 모델이 '블록이 큰 지도 = none' 을 배움
 H3 배경 잡음: die 수 적은 웨이퍼는 배경 불량 블록이 커서 선이 묻힌다 (신호 대 잡음)
 H4 선의 겉모양: die 가 크면 1 die 폭 선이 96 입력에서 굵은 블록 사슬이 됨 → '가는 선' 으로 안 보임
검증
 [1] 데이터 점검: die 수 구간별 라벨 분포 (학습 5% / test), Scratch 재현율 (Wilson 95% 구간)
 [2] 조작 실험 (짝 설계): test 의 실제 none 웨이퍼를 바탕으로, '같은 상대 위치·길이·각도' 의 1 die 폭 직선 Scratch 를 그려 넣음
     요인: 바탕 die 수 (적음 300~1199 / 많음 1200+) × 배경 (원래 잡음 / 불량 모두 지움) + 많음 바탕에 2 die 폭 선 (픽셀 굵기 맞춤)
     같은 선분 k 를 적음 바탕 k, 많음 바탕 k 에 적용 → 차이는 die 크기 (와 배경) 뿐
"""
import os, sys
import numpy as np, torch
proj = sys.argv[1]; os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input

dev = torch.device("cuda")
CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
rng = np.random.default_rng(0)
BINS = [(0, 600), (600, 1200), (1200, 2400), (2400, 10 ** 9)]
bname = lambda lo, hi: f"{lo}~{hi - 1}" if hi < 10 ** 9 else f"{lo}+"


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(seed):
    m = build_model("resnet18", NUM_CLASSES)
    m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
    return m.to(dev).eval()


@torch.no_grad()
def probs(m, imgs):
    o = []
    for i in range(0, len(imgs), 512):
        o.append(torch.softmax(m(to_input(torch.as_tensor(imgs[i:i + 512], dtype=torch.float32, device=dev)[:, None])), 1).cpu())
    return torch.cat(o).numpy()


models = {s: load(s) for s in (0, 1)}

# ===== [1] 데이터 점검
lab = ds.labeled_subset(0.05, seed=0); te = ds.test
print("[1-a] die 수 구간별 분포 (H1, H2)")
print(f"{'die 수':>11s} | {'학습 라벨: 전체':>13s} {'none 비율':>9s} {'Scratch':>8s} | {'test: 전체':>10s} {'none 비율':>9s} {'Scratch':>8s} | {'라벨 전체 none 비율':>14s}")
alllab = np.nonzero(ds.labels >= 0)[0]
for lo, hi in BINS:
    g = lambda idx: idx[(dies[idx] >= lo) & (dies[idx] < hi)]
    a, b, c = g(lab), g(te), g(alllab)
    print(f"{bname(lo, hi):>11s} | {len(a):13d} {np.mean(ds.labels[a] == NO):9.3f} {np.sum(ds.labels[a] == SC):8d} | {len(b):10d} {np.mean(ds.labels[b] == NO):9.3f} {np.sum(ds.labels[b] == SC):8d} | {np.mean(ds.labels[c] == NO):14.3f}")
print("\n[1-b] test Scratch 재현율과 Scratch→none 비율 (seed 0+1 합, Wilson 95% 구간)")
yte = ds.labels[te]; X_te_sc = ds.fetch(te[yte == SC], "cpu")[:, 0].numpy(); d_sc = dies[te[yte == SC]]
P = {s: probs(models[s], X_te_sc) for s in (0, 1)}
for lo, hi in BINS:
    g = (d_sc >= lo) & (d_sc < hi); n = 2 * g.sum()
    k = sum((P[s][g].argmax(1) == SC).sum() for s in (0, 1)); kn = sum((P[s][g].argmax(1) == NO).sum() for s in (0, 1))
    lo_, hi_ = wilson(k, n)
    print(f"  die {bname(lo, hi):>10s}: Scratch {g.sum():3d}장 · 재현율 {k / n:.2f} [{lo_:.2f}, {hi_:.2f}] · →none {kn / n:.2f}")

# ===== [2] 조작 실험
def native(i):
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
    nat = img[np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    assert (nat[np.ix_(ri, ci)] == img).all()          # 다시 그리면 캐시의 96 지도와 정확히 같음
    return nat, ri, ci


def render(nat, ri, ci):
    return nat[np.ix_(ri, ci)]


def draw(nat, seg, width):
    """정규화 좌표 (u=열, v=행, 0~1; 96 입력에서 원) 의 선분을 die 격자에 width die 폭으로."""
    h, w = nat.shape; out = nat.copy()
    (u0, v0), (u1, v1) = seg
    L = np.hypot((u1 - u0) * w, (v1 - v0) * h)
    t = np.linspace(0, 1, int(L * 4) + 2)
    rr = np.floor((v0 + t * (v1 - v0)) * h).astype(int); cc = np.floor((u0 + t * (u1 - u0)) * w).astype(int)
    nu, nv = -(v1 - v0) * h, (u1 - u0) * w; nn = np.hypot(nu, nv); nu, nv = nu / nn, nv / nn   # 선에 수직 방향 (die 단위)
    for o in range(width):
        off = o - (width - 1) / 2
        r2 = np.floor(rr + off * nu).astype(int); c2 = np.floor(cc + off * nv).astype(int)   # 행은 nu, 열은 nv 방향으로 이동
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        r2, c2 = r2[ok], c2[ok]; inw = nat[r2, c2] > 0
        out[r2[inw], c2[inw]] = 2
    return out


te_none = te[yte == NO]
grid_ok = sd[te_none, :2].max(1) <= 96
def pick(lo, hi, n):
    c = te_none[(dies[te_none] >= lo) & (dies[te_none] < hi) & grid_ok]
    good = []
    for i in rng.permutation(c):
        nat, _, _ = native(i)
        if (nat == 1).sum() >= 0.8 * dies[i]:        # 정상 die 가 대부분인 정상 지도만 (깨진 지도 제외)
            good.append(i)
        if len(good) == n: break
    return np.array(good)


N = 300
few, many = pick(300, 1200, N), pick(1200, 10 ** 9, N)
segs = []
for _ in range(N):
    while True:
        r, a0 = 0.35 * np.sqrt(rng.random()), rng.random() * 2 * np.pi
        cx, cy = 0.5 + r * np.cos(a0), 0.5 + r * np.sin(a0)
        th, L = rng.random() * np.pi, 0.3 + 0.3 * rng.random()          # 지름의 30~60%
        p0 = (cx - L / 2 * np.cos(th), cy - L / 2 * np.sin(th)); p1 = (cx + L / 2 * np.cos(th), cy + L / 2 * np.sin(th))
        if all(np.hypot(x - 0.5, y - 0.5) < 0.47 for x, y in (p0, p1)):
            segs.append((p0, p1)); break
print(f"\n[2] 조작 실험: 바탕 none 웨이퍼 die 적음 {len(few)}장 (중앙 {int(np.median(dies[few]))} die, die 1개 ≈ {np.median(96 / sd[few, 1]):.1f}×{np.median(96 / sd[few, 0]):.1f} 픽셀)"
      f" / 많음 {len(many)}장 (중앙 {int(np.median(dies[many]))} die, die 1개 ≈ {np.median(96 / sd[many, 1]):.1f}×{np.median(96 / sd[many, 0]):.1f} 픽셀)")
bg_def = lambda ids: np.median([(native(i)[0] == 2).sum() / dies[i] for i in ids])
print(f"    배경 불량 die 비율 중앙값: 적음 {bg_def(few):.3f} / 많음 {bg_def(many):.3f}")

conds = {}
def build(ids, clean, width, line=True):
    out = []
    for k, i in enumerate(ids):
        nat, ri, ci = native(i)
        if clean: nat = np.where(nat == 2, 1, nat).astype(np.uint8)
        out.append(render(draw(nat, segs[k], width) if line else nat, ri, ci))
    return np.stack(out)
for name, ids, clean, width, line in [("적음 · 원래 배경 · 선 없음", few, False, 1, False), ("많음 · 원래 배경 · 선 없음", many, False, 1, False),
                                      ("적음 · 원래 배경 · 1 die 선", few, False, 1, True), ("많음 · 원래 배경 · 1 die 선", many, False, 1, True),
                                      ("적음 · 배경 지움 · 1 die 선", few, True, 1, True), ("많음 · 배경 지움 · 1 die 선", many, True, 1, True),
                                      ("많음 · 원래 배경 · 2 die 선", many, False, 2, True), ("많음 · 배경 지움 · 2 die 선", many, True, 2, True)]:
    conds[name] = build(ids, clean, width, line)
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fewdie_conds.npz"), **{f"c{j}": v for j, v in enumerate(conds.values())},
                    names=np.array(list(conds)), few=few, many=many)

print(f"\n{'조건':28s} | seed 0: P(Scratch) 평균 · Scratch 판정 · none 판정 · Loc 판정 | seed 1: 같은 순서")
res = {}
for name, X in conds.items():
    row = []
    for s in (0, 1):
        p = probs(models[s], X); a = p.argmax(1); res[(name, s)] = p
        row.append(f"{p[:, SC].mean():.3f} · {np.mean(a == SC):.2f} · {np.mean(a == NO):.2f} · {np.mean(a == CLASSES.index('Loc')):.2f}")
    print(f"{name:28s} | " + " | ".join(row))

print("\n짝 비교 (같은 선분, Scratch 판정 비율 차이, 부트스트랩 95% 구간, seed 0·1 각각)")
def boot(a, b):
    d = a.astype(float) - b.astype(float); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for s in (0, 1):
    hit = lambda n: res[(n, s)].argmax(1) == SC
    for label, a, b in [("die 적음 - 많음 (원래 배경)", "적음 · 원래 배경 · 1 die 선", "많음 · 원래 배경 · 1 die 선"),
                        ("die 적음 - 많음 (배경 지움)", "적음 · 배경 지움 · 1 die 선", "많음 · 배경 지움 · 1 die 선"),
                        ("적음: 원래 배경 - 배경 지움", "적음 · 원래 배경 · 1 die 선", "적음 · 배경 지움 · 1 die 선"),
                        ("많음: 원래 배경 - 배경 지움", "많음 · 원래 배경 · 1 die 선", "많음 · 배경 지움 · 1 die 선"),
                        ("많음: 2 die 선 - 1 die 선 (원래 배경)", "많음 · 원래 배경 · 2 die 선", "많음 · 원래 배경 · 1 die 선")]:
        m_, l_, h_ = boot(hit(a), hit(b))
        print(f"  seed {s} · {label:34s} {m_:+.2f} [{l_:+.2f}, {h_:+.2f}]")
