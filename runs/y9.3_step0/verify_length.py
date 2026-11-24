"""y9.3 0단계: 길이 가중 Frangi 채널 검증 (학습 없음, test 안 씀). 기준은 debate/y9.3/03_response.md 에 고정.
 a 합성: 채널 최댓값 (96 리사이즈 + uint8 후) 의 중앙값이 길이 5 < 10 < 20 < 30 die 로 단조 증가
 b 합성: 배경 불량만 있는 맵의 최댓값 중앙값 < 0.1
 c 합성: 끊긴 20 die 선 (4 die 마다 1 die 틈) 최댓값 ≥ 연속 20 die 선의 0.5 배 (같은 조건 짝의 비율 중앙값)
 d 5% 라벨 학습: 웨이퍼 점수 max(V·g) 의 AP > max(V) 의 AP (Scratch 대 나머지 전체)
후보 1 (불량 die 성분) 이 c 를 통과하면 후보 2 (Frangi 능선 τ 0.2 / 0.3) 는 안 봄.
진단 (기준 아님): y9.2a val 오탐의 Frangi 피크 성분 길이, y9.2a 체크포인트에 V 대신 V·g 를 넣었을 때 Scratch 확률 변화.
사용: python runs/y9.3_step0/verify_length.py
"""
import os, sys, pickle, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from scipy import ndimage
from sklearn.metrics import average_precision_score
from wafer import frangi as Fr
from wafer.data import WM811K, CLASSES, NUM_CLASSES, _resize_nearest
from wafer.models import build_model, to_input, FrangiInput

CACHE = os.path.expanduser("~/wafer_cache")
ds = WM811K(CACHE); sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy")); dies = sd[:, 2]
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
ok = lambda b: "통과" if b else "**실패**"
q = lambda a: np.rint(_resize_nearest(a[None], 96)[0] * 255) / 255   # 캐시와 같은 96 + uint8


def channel(nat, cand, V=None):
    V = Fr.vesselness(nat) if V is None else V
    w = Fr.line_weight(nat) if cand == "defect" else Fr.line_weight(nat, "ridge", V=V, tau=cand)
    return V, w


# ---------------- 합성 ----------------
def synth(D, T, t, ang, p, rng, broken=False):
    r, c = np.mgrid[0:D, 0:D] + 0.5; yy, xx = r / D - 0.5, c / D - 0.5
    nat = np.where(xx ** 2 + yy ** 2 < 0.25, 1, 0).astype(np.uint8)
    nat[(rng.random(nat.shape) < p) & (nat > 0)] = 2
    if T:
        th = np.radians(ang); pr, pc = r - D / 2 - 0.25, c - D / 2 - 0.25
        along = pc * np.cos(th) + pr * np.sin(th); perp = -pc * np.sin(th) + pr * np.cos(th)
        on = (np.abs(perp) < t / 2) & (np.abs(along) <= T / 2) & (nat > 0)
        if broken: on &= np.floor(along + T / 2) % 5 != 4
        nat[on] = 2
    return nat


def synth_tests(cand):
    rng = np.random.default_rng(0); res = {T: [] for T in (5, 10, 20, 30)}; ratio = []; bg = []
    for D in (30, 45, 70):
        for p in (0, 0.05, 0.10):
            for draw in range(3 if p else 1):
                seed = rng.integers(1 << 30)
                for t in (1, 2):
                    for ang in (0, 30, 45, 90):
                        for T in (5, 10, 20, 30):
                            nat = synth(D, T, t, ang, p, np.random.default_rng(seed)); V, w = channel(nat, cand)
                            res[T].append(q(V * w).max())
                        nat = synth(D, 20, t, ang, p, np.random.default_rng(seed), broken=True); V, w = channel(nat, cand)
                        ratio.append(q(V * w).max() / max(res[20][-1], 1e-6))
            if p:
                for draw in range(5):
                    nat = synth(D, 0, 1, 0, p, np.random.default_rng(rng.integers(1 << 30))); V, w = channel(nat, cand); bg.append(q(V * w).max())
    med = [np.median(res[T]) for T in (5, 10, 20, 30)]
    A = all(x < y for x, y in zip(med, med[1:])); B = np.median(bg) < 0.1; C = np.median(ratio) >= 0.5
    print(f"  a 길이 5/10/20/30 die 채널 최댓값 중앙값 {' / '.join(f'{m:.3f}' for m in med)} → {ok(A)}")
    print(f"  b 배경 불량만 (5%·10%, 지름 30/45/70) 최댓값 중앙값 {np.median(bg):.3f} (90% {np.percentile(bg, 90):.3f}) → {ok(B)}")
    print(f"  c 끊긴 20 die / 연속 20 die 비율 중앙값 {np.median(ratio):.2f} (25% {np.percentile(ratio, 25):.2f}) → {ok(C)}")
    return A and B and C


# ---------------- 원본 격자 (5% 라벨 학습) ----------------
def natives(idx):
    idx = np.asarray(idx); big = idx[sd[idx, :2].max(1) > 96]; out = {}
    if len(big):
        import pandas.core.indexes
        sys.modules.setdefault("pandas.indexes", pandas.core.indexes)
        with open("LSWMD.pkl", "rb") as f:
            maps = pickle.load(f, encoding="latin1")["waferMap"].values
        for i in big: out[int(i)] = np.minimum(np.asarray(maps[i]).astype(np.uint8), 2)
        del maps
    img = ds.fetch(idx, "cpu")[:, 0].numpy().astype(np.uint8)
    for j, i in enumerate(idx):
        if int(i) in out: continue
        h, w = sd[i, :2]
        ri = ((np.arange(96) + 0.5) * h / 96).astype(np.int64); ci = ((np.arange(96) + 0.5) * w / 96).astype(np.int64)
        out[int(i)] = img[j][np.ix_(np.unique(ri, return_index=True)[1], np.unique(ci, return_index=True)[1])]
    return out


t0 = time.time()
lab = ds.labeled_subset(0.05, seed=0); ly = ds.labels[lab]
NAT = natives(lab); print(f"5% 라벨 학습 {len(lab)}장 원본 격자 준비 ({time.time() - t0:.0f}s, 96 초과 {np.sum(sd[lab, :2].max(1) > 96)}장)")
VV = {int(i): Fr.vesselness(NAT[int(i)]) for i in lab}


def real_tests(cand):
    sV, sg, sVg, Lm = [], [], [], []
    for i in lab:
        nat = NAT[int(i)]; V, w = channel(nat, cand, VV[int(i)])
        sV.append(q(V).max()); sg.append(w.max()); sVg.append(q(V * w).max()); Lm.append(w.max() * Fr.LMAX)
    sV, sg, sVg, Lm = map(np.array, (sV, sg, sVg, Lm))
    ys = (ly == SC).astype(int); m_sn = (ly == SC) | (ly == NO)
    ap = {n: (average_precision_score(ys, s), average_precision_score(ys[m_sn], s[m_sn])) for n, s in (("max V", sV), ("max g", sg), ("max V·g", sVg))}
    for n, (a1, a2) in ap.items():
        print(f"  {n:8s} AP Scratch 대 나머지 {a1:.3f} | Scratch 대 none {a2:.3f}")
    D = ap["max V·g"][0] > ap["max V"][0]
    print(f"  d AP(V·g) > AP(V) (Scratch 대 나머지) → {ok(D)}")
    print("  이 정의의 길이 구간별 Scratch 비율 (5% 학습, 가장 긴 선 성분 L, 30 에서 포화):")
    for lo, hi in ((0, 1), (1, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 20), (20, 30), (30, 99)):
        k = (Lm >= lo) & (Lm < hi)
        print(f"    {lo:>2}~{hi - 1:<2} die  Scratch {np.sum(k & (ly == SC)):3d} / 전체 {k.sum():5d}  (none {np.sum(k & (ly == NO)):5d})")
    short = np.nonzero((ly == SC) & (Lm < 10))[0]
    print(f"  짧은 학습 Scratch (L < 10 die) {len(short)}장: max V {np.round(sV[short], 2).tolist()}")
    print(f"                                    max V·g {np.round(sVg[short], 2).tolist()}")
    print(f"  학습 Scratch 전체 max V·g 중앙값 {np.median(sVg[ly == SC]):.3f}, none {np.median(sVg[ly == NO]):.3f} | max V 중앙값 Scratch {np.median(sV[ly == SC]):.3f}, none {np.median(sV[ly == NO]):.3f}")
    return D, ap["max V·g"][0]


chosen = None
for cand in ("defect", 0.2, 0.3):
    name = "후보 1 (불량 die 성분)" if cand == "defect" else f"후보 2 (Frangi 능선 τ={cand})"
    print(f"\n===== {name} =====")
    passed = synth_tests(cand); D, apd = real_tests(cand); passed = passed and D
    print(f"  → {name}: a~d {'모두 통과' if passed else '미달'}")
    if cand == "defect" and passed:
        chosen = cand; break
    if cand != "defect" and passed and (chosen is None or apd > chosen_ap):
        chosen, chosen_ap = cand, apd
print(f"\n선택: {chosen if chosen is not None else '없음 (학습 안 함)'}  (후보 2 의 τ 는 5% 학습 AP 로 고른 탐색 결과)")

# ---------------- 진단 (기준 아님, val) ----------------
if chosen is not None:
    dev = torch.device("cuda")
    pu, pi = Fr.cache_paths(CACHE); table = torch.from_numpy(np.load(pu)); gi = np.load(pi)
    row = np.full(len(ds.labels), -1, np.int64); row[gi] = np.arange(len(gi))
    va = ds.val; yv = ds.labels[va]; vn = va[yv == NO]; vs = va[yv == SC]
    fp = set()
    models = {}
    for seed in (0, 1, 2):
        m = FrangiInput(build_model("resnet18", NUM_CLASSES), table, row)
        m.load_state_dict(torch.load(f"runs/resnet18/p0.05/seed{seed}/y9.2a_ssl50_ft30/best.pt", map_location="cpu", weights_only=True)); m = m.to(dev).eval(); models[seed] = m
        with torch.no_grad():
            for i in range(0, len(vn), 1024):
                j = vn[i:i + 1024]; p = m(to_input(ds.fetch(j, dev)), torch.as_tensor(j, device=dev)).argmax(1).cpu().numpy()
                fp |= set(j[p == SC].tolist())
    fp = np.array(sorted(fp)); fp_s = fp[sd[fp, :2].max(1) <= 96]; vs_s = vs[sd[vs, :2].max(1) <= 96]
    NV = natives(np.concatenate([fp_s, vs_s]))
    Lpk = []
    for i in fp_s:
        nat = NV[int(i)]; V = Fr.vesselness(nat); pk = np.unravel_index(np.argmax(V), V.shape)
        labd, n = ndimage.label(nat == 2, structure=np.ones((3, 3)))
        dist, (ri, ci) = ndimage.distance_transform_edt(labd == 0, return_indices=True); k = labd[ri[pk], ci[pk]]
        p = np.argwhere(labd == k).astype(float); c = p - p.mean(0)
        if len(p) >= 2:
            ev, Vv = np.linalg.eigh(np.cov(c.T)); pr = c @ Vv[:, 1]; Lpk.append(pr.max() - pr.min() + 1)
        else: Lpk.append(1.0)
    Lpk = np.array(Lpk)
    print(f"\n[진단] y9.2a (seed 0·1·2 합집합) val none→Scratch 오탐 {len(fp)}장 (원본 ≤ 96: {len(fp_s)}장). Frangi 피크가 속한 불량 성분의 길이 (die):")
    print(f"  중앙값 {np.median(Lpk):.1f}, <10 die {np.mean(Lpk < 10):.2f}, <15 die {np.mean(Lpk < 15):.2f}, ≥20 die {np.mean(Lpk >= 20):.2f}")

    @torch.no_grad()
    def probe(ids, use_w):
        X = np.stack([_resize_nearest(NV[int(i)][None], 96)[0] for i in ids]).astype(np.float32)
        Fm = []
        for i in ids:
            nat = NV[int(i)]; V, w = channel(nat, chosen); Fm.append(q(V * w) if use_w else q(V))
        Fm = np.stack(Fm).astype(np.float32); out = {}
        for s, m in models.items():
            x = torch.cat([to_input(torch.as_tensor(X, device=dev)[:, None]), torch.as_tensor(Fm, device=dev)[:, None]], 1)
            out[s] = torch.softmax(m(x).float(), 1)[:, SC].cpu().numpy()
        return out
    for nm, ids in (("val none 오탐", fp_s), ("val Scratch", vs_s)):
        a, b = probe(ids, False), probe(ids, True)
        print(f"  {nm} {len(ids)}장, y9.2a 에 V → V·g 로 바꿔 넣음 (재학습 없음): 평균 P(Scratch) " +
              " | ".join(f"seed {s} {a[s].mean():.2f}→{b[s].mean():.2f} (Scratch 판정 {np.mean(a[s] >= 0.5):.2f}→{np.mean(b[s] >= 0.5):.2f})" for s in a))
print(f"\n총 {time.time() - t0:.0f}s")
