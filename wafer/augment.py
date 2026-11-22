"""GPU 배치 증강. 입력/출력 모두 (B,1,H,W) float, 값 {0: 배경, 1: 정상, 2: 불량}.

기하 변환은 nearest 샘플링 + 바깥은 0(배경) 으로 채워 값 집합이 유지된다.
논문에 수치가 없는 파라미터(크롭 비율, 이동량, 노이즈 확률, cutout 크기/개수)는
AUG_CFG 에 모아두었다 (Fig.4/5 와 WaPIRL 설정을 참고한 추정값).
"""
import math

import torch
import torch.nn.functional as F

AUG_CFG = dict(
    crop_scale=(0.5, 1.0),      # random-resized-crop 면적 비율
    crop_ratio=(3 / 4, 4 / 3),
    shift_strong=0.2,           # strong shift: 폭 대비 최대 이동 비율
    shift_weak=0.125,           # FixMatch weak (w/o Rotation 변형) 의 translate
    noise_p=0.05,               # die 픽셀을 정상<->불량 뒤집을 확률
    cutout_size=20,             # 정사각 cutout 한 변 (픽셀, 96 기준)
    cutout_holes=4,             # Fig.4/5 처럼 여러 개
)

STRONG_OPS = ("crop", "cutout", "noise", "rotate", "shift")
STRONG_OPS_NO_CUTOUT = ("crop", "noise", "rotate", "shift")   # ver 계열 --no_cutout (cutout 유무 비교용)


def _warp(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def _eye(b, device):
    return torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=device).repeat(b, 1, 1)


def rotate(x):
    b = x.shape[0]
    a = (torch.rand(b, device=x.device) * 2 - 1) * math.pi
    th = _eye(b, x.device)
    th[:, 0, 0] = a.cos(); th[:, 0, 1] = -a.sin()
    th[:, 1, 0] = a.sin(); th[:, 1, 1] = a.cos()
    return _warp(x, th)


def shift(x, frac):
    b = x.shape[0]
    th = _eye(b, x.device)
    th[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * 2 * frac  # normalized 좌표는 폭=2
    return _warp(x, th)


def hflip(x, p=0.5):
    flip = torch.rand(x.shape[0], device=x.device) < p
    return torch.where(flip[:, None, None, None], x.flip(-1), x)


def crop(x):
    b, dev = x.shape[0], x.device
    s0, s1 = AUG_CFG["crop_scale"]
    r0, r1 = (math.log(v) for v in AUG_CFG["crop_ratio"])
    area = torch.empty(b, device=dev).uniform_(s0, s1)
    ratio = torch.empty(b, device=dev).uniform_(r0, r1).exp()
    w = (area * ratio).sqrt().clamp(max=1.0)
    h = (area / ratio).sqrt().clamp(max=1.0)
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = w; th[:, 1, 1] = h
    th[:, 0, 2] = (torch.rand(b, device=dev) * 2 - 1) * (1 - w)
    th[:, 1, 2] = (torch.rand(b, device=dev) * 2 - 1) * (1 - h)
    return _warp(x, th)


def local_crop(x, size, scale=(0.15, 0.4)):
    """작은 국소 영역(면적 비율 scale)을 잘라 size x size 로 (ver3 multi-crop 자기지도용)."""
    b, dev = x.shape[0], x.device
    area = torch.empty(b, device=dev).uniform_(*scale)
    side = area.sqrt()
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = side; th[:, 1, 1] = side
    th[:, :, 2] = (torch.rand(b, 2, device=dev) * 2 - 1) * (1 - side)[:, None]
    grid = F.affine_grid(th, [b, 1, size, size], align_corners=False)
    return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)


def noise(x):
    flip = (x > 0) & (torch.rand_like(x) < AUG_CFG["noise_p"])
    return torch.where(flip, 3 - x, x)  # 1 <-> 2


def _cut(x, ys, xs, k):
    """ys, xs: (B, n_holes) 좌상단 좌표 → 해당 k x k 영역을 배경(0)으로."""
    h, w = x.shape[-2:]
    ar_h = torch.arange(h, device=x.device)[None, None, :, None]
    ar_w = torch.arange(w, device=x.device)[None, None, None, :]
    ys, xs = ys[:, :, None, None], xs[:, :, None, None]
    m = ((ar_h >= ys) & (ar_h < ys + k) & (ar_w >= xs) & (ar_w < xs + k)).any(1, keepdim=True)
    return x.masked_fill(m, 0.0)


def cutout(x):
    b, (h, w) = x.shape[0], x.shape[-2:]
    k, n = AUG_CFG["cutout_size"], AUG_CFG["cutout_holes"]
    ys = torch.randint(0, h - k + 1, (b, n), device=x.device)
    xs = torch.randint(0, w - k + 1, (b, n), device=x.device)
    return _cut(x, ys, xs, k)


def multi_mask(x, n=4, area=0.2):
    """ver6: 각 이미지마다 면적 area 의 정사각형 하나를 가린(0) 마스킹본 n 장. (B,1,H,W) → (n, B, 1, H, W)."""
    b, (h, w) = x.shape[0], x.shape[-2:]
    k = round(math.sqrt(area * h * w))
    ys = torch.randint(0, h - k + 1, (n, b, 1), device=x.device)
    xs = torch.randint(0, w - k + 1, (n, b, 1), device=x.device)
    return torch.stack([_cut(x, ys[i], xs[i], k) for i in range(n)])


def add_dilated(x, k=5):
    """ver7.3: (B,1,H,W) → (B,2,H,W). 채널 2 = 불량(2)을 k x k 로 팽창한 웨이퍼 맵 (웨이퍼 바깥으로는 번지지 않음)."""
    d = F.max_pool2d((x == 2).float(), k, stride=1, padding=k // 2) > 0
    return torch.cat([x, torch.where(d & (x > 0), torch.full_like(x, 2.0), x)], 1)


def checker_mask(x, cell=12, ratio=0.5):
    """ver6.1~: cell x cell 칸을 바둑판 모양으로 번갈아 0 으로 (50%). 이미지마다 바둑판 시작 위치를 무작위로 이동.

    ratio < 0.5 이면 가릴 칸(바둑판의 한쪽 색) 중 ratio/0.5 비율만 무작위로 골라 가림 (ver6.4: 0.3 → 칸의 60%).
    """
    b, (h, w) = x.shape[0], x.shape[-2:]
    oy = torch.randint(0, 2 * cell, (b, 1, 1, 1), device=x.device)
    ox = torch.randint(0, 2 * cell, (b, 1, 1, 1), device=x.device)
    cy = (torch.arange(h, device=x.device)[None, None, :, None] + oy) // cell
    cx = (torch.arange(w, device=x.device)[None, None, None, :] + ox) // cell
    m = ((cy + cx) % 2) == 0
    if ratio < 0.5:
        n = (max(h, w) + 2 * cell) // cell + 1
        keep = torch.rand(b, n, n, device=x.device) < ratio / 0.5
        bi = torch.arange(b, device=x.device)[:, None, None, None]
        m = m & keep[bi, cy, cx]
    return x.masked_fill(m, 0.0)


def stripe_mask(x, cell, axis):
    """ver7.7: cell 픽셀 두께 줄을 번갈아 0 으로 (50%). axis="v" 세로줄, "h" 가로줄. 이미지마다 시작 위치 무작위."""
    b, n = x.shape[0], x.shape[-1 if axis == "v" else -2]
    o = torch.randint(0, 2 * cell, (b, 1), device=x.device)
    m = ((torch.arange(n, device=x.device)[None] + o) // cell) % 2 == 0
    return x.masked_fill(m[:, None, None, :] if axis == "v" else m[:, None, :, None], 0.0)


def keep_cutout(x, sal, tau_c):
    """Algorithm 1: 영역 중요도 score <= tau_c 인 위치에서만 cutout.

    score = 영역 평균 saliency / 이미지 전체 평균 saliency.
    'repeat ... until score <= tau_c' 의 rejection sampling 은 허용 위치 중 균등 샘플링과
    같은 분포이므로 그렇게 구현 (무한루프 없음). 허용 위치가 없으면 최소 score 위치 사용.
    sal: (B,1,h',w') 비음수, 이미지 해상도의 정수배 축소본이어도 됨.
    """
    b, (h, w) = x.shape[0], x.shape[-2:]
    k, n = AUG_CFG["cutout_size"], AUG_CFG["cutout_holes"]
    if sal.shape[-2:] != (h, w):
        sal = F.interpolate(sal, size=(h, w), mode="nearest")
    score = F.avg_pool2d(sal, k, stride=1).flatten(1)               # (B, (h-k+1)*(w-k+1))
    score = score / sal.flatten(1).mean(1, keepdim=True).clamp_min(1e-12)
    thr = torch.maximum(torch.full_like(score[:, :1], tau_c), score.amin(1, keepdim=True))
    ok = (score <= thr).float()
    pos = torch.multinomial(ok, n, replacement=True)                 # (B, n)
    wc = w - k + 1
    return _cut(x, pos // wc, pos % wc, k)


def randaugment_geo(x, n_ops=2):
    """표준 FixMatch RandAugment 중 기하 연산만 (색 연산은 {0,1,2} 맵에 무의미, Cutout 제외).

    샘플별로 서로 다른 n_ops 개 선택, 강도는 [0, max] 균등 + 랜덤 부호 (FixMatch 증강 풀 범위).
    모두 affine 이라 하나의 행렬로 합성해 한 번만 warp.
    """
    b, dev = x.shape[0], x.device
    pick = torch.rand(b, 6, device=dev).argsort(1)[:, :n_ops]   # 0 identity 1 rotate 2 shearX 3 shearY 4 transX 5 transY
    th = torch.eye(3, device=dev).repeat(b, 1, 1)
    for k in range(n_ops):
        op = pick[:, k]
        v = torch.rand(b, device=dev) * torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        m = torch.eye(3, device=dev).repeat(b, 1, 1)
        a = v * math.radians(30)
        rot = op == 1
        m[rot, 0, 0] = a[rot].cos(); m[rot, 0, 1] = -a[rot].sin()
        m[rot, 1, 0] = a[rot].sin(); m[rot, 1, 1] = a[rot].cos()
        m[op == 2, 0, 1] = 0.3 * v[op == 2]
        m[op == 3, 1, 0] = 0.3 * v[op == 3]
        m[op == 4, 0, 2] = 0.3 * 2 * v[op == 4]   # normalized 좌표는 폭=2
        m[op == 5, 1, 2] = 0.3 * 2 * v[op == 5]
        th = m @ th
    return _warp(x, th[:, :2])


# ---------------------------------------------------------------- 조합

def weak(x, mode="rotate"):
    """WaFixMatch weak = 회전만. 'flipshift' = 기존 FixMatch weak (ablation 'w/o Rotation')."""
    if mode == "rotate":
        return rotate(x)
    if mode == "flipshift":
        return shift(hflip(x), AUG_CFG["shift_weak"])
    raise ValueError(mode)


def strong(x, sal, tau_c, keep, ops=STRONG_OPS):
    """5개 중 서로 다른 2개를 샘플별로 뽑아 적용. cutout 이 뽑히면 먼저 적용 (Algorithm 1).

    keep=True 이면 cutout 은 saliency 기반 KeepCutout, False 이면 일반 랜덤 cutout.
    """
    b = x.shape[0]
    pick = torch.rand(b, len(ops), device=x.device).argsort(1)[:, :2]
    first, second = pick[:, 0], pick[:, 1]
    if "cutout" in ops:
        swap = second == ops.index("cutout")
        first, second = torch.where(swap, second, first), torch.where(swap, first, second)
    x = x.clone()
    for stage in (first, second):
        for j, op in enumerate(ops):
            m = stage == j
            if not m.any():
                continue
            xs = x[m]
            if op == "cutout":
                xs = keep_cutout(xs, sal[m], tau_c) if keep else cutout(xs)
            elif op == "crop":
                xs = crop(xs)
            elif op == "noise":
                xs = noise(xs)
            elif op == "rotate":
                xs = rotate(xs)
            elif op == "shift":
                xs = shift(xs, AUG_CFG["shift_strong"])
            else:
                raise ValueError(op)
            x[m] = xs
    return x
