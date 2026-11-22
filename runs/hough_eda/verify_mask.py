"""가설 '바둑판 마스킹이 Scratch 의 가는 선을 지워서 선 특징을 못 배운다' 직접 검증 (사전학습 인코더만, fine-tune 없음 → seed 잡음 없음).

인코더 (모두 seed 0, 같은 10만 장 사전학습 데이터 계열):
  SimCLR (ver3, 바둑판 없음) / 12px·50% (ver6.1) / 6px·50% (ver8) / 6px·30% (ver6.4) / 4px·50% (ver6.5) / 바둑판2+줄무늬2 (ver7.7)
검증 A — 마스킹이 선에 실제로 하는 일: Scratch 선 픽셀 중 가려지는 비율, 남은 선 조각 수, 가장 긴 조각 길이
검증 B — 선 민감도: Scratch 웨이퍼에서 '선만 지웠을 때' 특징 변화 vs '같은 개수의 다른 불량 픽셀을 지웠을 때' 변화
         선 특징을 배웠다면 선 지우기 >> 무작위 지우기. 마스킹이 선 학습을 막았다면 SimCLR 보다 이 비가 작아야 함
검증 C — 고정 특징 선형 분류 (linear probe): 라벨 6,919장으로 로지스틱 회귀 → test Scratch F1 / 재현율
"""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
proj, hdir = sys.argv[1], sys.argv[2]
os.chdir(proj); sys.path.insert(0, proj); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input
from wafer import augment as A
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
_a = sys.argv; sys.argv = [_a[0], proj, hdir]
import hough2 as h2
sys.argv = _a

dev = torch.device("cuda")
torch.manual_seed(0); rng = np.random.default_rng(0)
CACHE = os.path.expanduser("~/wafer_cache"); ENC = os.path.join(CACHE, "ver3/runs/resnet18/p0.05/seed0")
ds = WM811K(CACHE)
sd = np.load(os.path.join(CACHE, "wm811k_shapes_dies.npy"))
SC = CLASSES.index("Scratch")
ENCS = [("SimCLR (바둑판 없음, ver3)", "simclr_random_cut_e50.pt", None),
        ("바둑판 12px·50% (ver6.1)", "masklat61_e50.pt", ("checker", 12, 0.5)),
        ("바둑판 6px·50% (ver8)", "masklat8_e50.pt", ("checker", 6, 0.5)),
        ("바둑판 6px·30% (ver6.4)", "masklat64_e50.pt", ("checker", 6, 0.3)),
        ("바둑판 4px·50% (ver6.5)", "masklat65_e50.pt", ("checker", 4, 0.5)),
        ("바둑판2+줄무늬2 6px (ver7.7)", "masklat77_e50.pt", ("stripe", 6, 0.5))]

# ---- Scratch 선 마스크 (96x96): 원본 die 격자의 주 덩어리 (외톨이 제거, 1 die 틈 연결) 를 96 좌표로, 길쭉한 것만
K8 = np.ones((3, 3), bool)
zs = np.load(os.path.join(hdir, "shape_feats.npz")); S = np.full((len(ds.labels), zs["f"].shape[1]), np.nan); S[zs["idx"]] = zs["f"]
EL = list(zs["names"]).index("elong")
cand = np.nonzero((ds.labels == SC) & (S[:, EL] >= 3) & (sd[:, :2].max(1) <= 96))[0]
X, LINE = [], []
for i in cand:
    h, w = sd[i, :2]
    img = ds.fetch([i], "cpu")[0, 0].numpy().astype(np.uint8)
    nat = h2.native(img, h, w)
    den = h2.denoise(nat) == 2
    lab, k = ndimage.label(ndimage.binary_dilation(den, K8) & (nat > 0), structure=K8)
    cnt = np.bincount(lab[den], minlength=k + 1); cnt[0] = 0
    main = den & (lab == cnt.argmax())
    ri = ((np.arange(96) + 0.5) * h / 96).astype(int); ci = ((np.arange(96) + 0.5) * w / 96).astype(int)
    line96 = main[np.ix_(ri, ci)] & (img == 2)
    other = (img == 2) & ~line96
    if line96.sum() >= 8 and other.sum() >= line96.sum():
        X.append(img); LINE.append(line96)
X = np.stack(X); LINE = np.stack(LINE)
print(f"검증용 Scratch (길쭉함 ≥ 3, 다른 불량이 선보다 많은 것): {len(X)}장 · 선 픽셀 중앙값 {int(np.median(LINE.sum((1, 2))))}")

# ---- 검증 A: 마스킹이 선에 하는 일 (회전 없이, 마스크만)
print("\n[A] 마스킹 1회가 Scratch 선에 하는 일 (마스크 무작위 50번 평균)")
xt = torch.from_numpy(LINE.astype(np.float32))[:, None]
for name, _, cfg in ENCS[1:]:
    hid, nseg, longest = [], [], []
    for _ in range(50):
        if cfg[0] == "checker":
            vis = A.checker_mask(xt, cfg[1], cfg[2])[:, 0].numpy() > 0
        else:
            vis = (A.stripe_mask(xt, cfg[1], "v") if rng.random() < 0.5 else A.stripe_mask(xt, cfg[1], "h"))[:, 0].numpy() > 0
        for j in range(len(LINE)):
            hid.append(1 - vis[j].sum() / LINE[j].sum())
            lb, n = ndimage.label(vis[j], structure=K8)
            nseg.append(n)
            longest.append(np.bincount(lb.ravel())[1:].max() / LINE[j].sum() if n else 0)
    print(f"  {name:28s} 가려지는 선 비율 {np.mean(hid):.2f} · 남은 조각 수 중앙값 {np.median(nseg):.0f} · 가장 긴 조각 = 선의 {np.median(longest):.0%}")
print(f"  (참고: 원래 선 조각 수 중앙값 {np.median([ndimage.label(l, structure=K8)[1] for l in LINE]):.0f})")


def load_enc(fname):
    m = build_model("resnet18", NUM_CLASSES)
    miss = m.load_state_dict(torch.load(os.path.join(ENC, fname), map_location="cpu", weights_only=True), strict=False)
    assert set(miss.missing_keys) == {"fc.weight", "fc.bias"}, miss
    m.fc = nn.Identity()
    return m.to(dev).eval()


@torch.no_grad()
def feats(m, imgs):
    out = []
    for i in range(0, len(imgs), 512):
        out.append(Fn.normalize(m(to_input(torch.as_tensor(imgs[i:i + 512], dtype=torch.float32, device=dev)[:, None])), dim=1).cpu())
    return torch.cat(out)


# 선 지운 것 / 같은 개수 다른 불량 지운 것 (지운 자리는 정상 die = 1)
X_line = X.copy(); X_line[LINE] = 1
X_rand = X.copy()
for j in range(len(X)):
    oy, ox = np.nonzero((X[j] == 2) & ~LINE[j]); n = LINE[j].sum()
    s = rng.choice(len(oy), n, replace=False); X_rand[j, oy[s], ox[s]] = 1

lab_idx = ds.labeled_subset(0.05, seed=0); te = ds.test
print("\n[B] 선 민감도 (특징 변화 = 1 - cos, 사전학습 인코더)   [C] 고정 특징 선형 분류 (라벨 6,919장 → test)")
for name, fname, _ in ENCS:
    m = load_enc(fname)
    f0, fl, fr = feats(m, X), feats(m, X_line), feats(m, X_rand)
    dl = 1 - (f0 * fl).sum(1).numpy(); dr = 1 - (f0 * fr).sum(1).numpy()
    ftr = feats(m, ds.fetch(lab_idx, "cpu")[:, 0].numpy()); fte = feats(m, ds.fetch(te, "cpu")[:, 0].numpy())
    sc = StandardScaler().fit(ftr.numpy())
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(ftr.numpy()), ds.labels[lab_idx])
    p = clf.predict(sc.transform(fte.numpy())); yt = ds.labels[te]
    f1s = f1_score(yt, p, labels=range(NUM_CLASSES), average=None, zero_division=0)
    print(f"  {name:28s} [B] 선 지움 {np.median(dl):.4f} · 다른 불량 지움 {np.median(dr):.4f} · 비 {np.median(dl) / np.median(dr):.2f} · 선>다른 {np.mean(dl > dr):.0%}"
          f" | [C] macro {f1s.mean():.3f} · Scratch F1 {f1s[SC]:.3f} · 재현 {(p[yt == SC] == SC).mean():.2f}")
    del m; torch.cuda.empty_cache()
