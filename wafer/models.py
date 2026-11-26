import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import augment as A

# 논문 4.E 의 아키텍처별 배치 크기
DEFAULT_BATCH = {"resnet18": 256, "resnet34": 256, "resnet50": 128, "densenet121": 64}
# ver7.3: 0 이 아니면 입력을 2채널 (원본 + 불량 k x k 팽창본) 로 (CLI 가 설정)
INPUT_DILATE = 0
# True 면 ImageNet 사전학습 가중치로 시작 (CLI --imagenet 이 설정). 첫 conv 는 RGB 가중치를 채널 합으로 1채널화
IMAGENET = False
# ver9.2: True/"layer4" 이면 layer4 뒤(3x3), "layer3" 이면 6x6, "layer2" 이면 12x12 에 SpatialSelfAttention 추가
USE_SA = False


class PositionalEncoding2D(nn.Module):
    """(B, C, H, W) 특징 맵에 더해주는 학습 가능한 2D 위치 임베딩."""

    def __init__(self, channels, h=3, w=3):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, channels, h, w) * 0.02)

    def forward(self, x):
        pos_emb = self.pos_emb
        if pos_emb.shape[-2:] != x.shape[-2:]:
            pos_emb = F.interpolate(pos_emb, size=x.shape[-2:], mode="bicubic", align_corners=False)
        return x + pos_emb


class SpatialSelfAttention(nn.Module):
    """ver9.2: (B, C, H, W) 특징 맵의 spatial grid(H*W 토큰) 간 Multi-Head Self-Attention."""

    def __init__(self, in_dim=512, h=3, w=3, num_heads=4, dropout=0.0, use_gate=False):
        super().__init__()
        self.use_gate = use_gate
        self.pos_enc = PositionalEncoding2D(in_dim, h, w)
        self.norm1 = nn.LayerNorm(in_dim)
        self.mha = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(in_dim)
        ffn_layers = [nn.Linear(in_dim, in_dim * 2), nn.GELU()]
        if dropout > 0.0:
            ffn_layers.append(nn.Dropout(dropout))
        ffn_layers.append(nn.Linear(in_dim * 2, in_dim))
        self.ffn = nn.Sequential(*ffn_layers)
        if use_gate:
            self.drop = nn.Dropout(dropout)
            self.attn_scale = nn.Parameter(torch.zeros(()))
            self.ffn_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        B, C, H, W = x.shape
        if self.use_gate:
            base = x.flatten(2).transpose(1, 2)  # (B, H*W, C), 초기에는 사전학습 특징 그대로 보존
            x_pos = self.pos_enc(x).flatten(2).transpose(1, 2)
            norm_tokens = self.norm1(x_pos)
            attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
            tokens = base + self.attn_scale * self.drop(attn_out)
            tokens = tokens + self.ffn_scale * self.drop(self.ffn(self.norm2(tokens)))
            return tokens.transpose(1, 2).reshape(B, C, H, W)
        x = self.pos_enc(x)
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def _layer_out_channels(layer):
    last = layer[-1]
    return getattr(last, "bn3", getattr(last, "bn2")).num_features


class OrientedLineBranch(nn.Module):
    """ver9.4 (line_base): 원본 96x96 웨이퍼 맵에서 얇은 선형(Scratch) 리지를 추출해 6x6으로 전달하는 보조 스트림."""

    def __init__(self, out_dim=256, dropout=0.1):
        super().__init__()
        self.register_buffer("k9", self._make_oriented_kernels(9, n_ori=8), persistent=False)
        self.register_buffer("k15", self._make_oriented_kernels(15, n_ori=8), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(6, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),  # 96 -> 48
            nn.Conv2d(24, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),  # 48 -> 24
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),  # 24 -> 12
            nn.Conv2d(96, 96, 3, padding=1, groups=96, bias=False),
            nn.Conv2d(96, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),  # 12 -> 6
        )

    @staticmethod
    def _make_oriented_kernels(ks, sigma_v=0.14, v0=0.35, sigma_neg=0.18, sigma_u=0.75, n_ori=8):
        coords = torch.linspace(-1.0, 1.0, ks)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernels = []
        for k in range(n_ori):
            th = k * math.pi / n_ori
            c, s_ = math.cos(th), math.sin(th)
            u = xx * c + yy * s_
            v = -xx * s_ + yy * c
            line = (
                torch.exp(-(v ** 2) / (2 * sigma_v ** 2))
                - 0.5 * torch.exp(-((v - v0) ** 2) / (2 * sigma_neg ** 2))
                - 0.5 * torch.exp(-((v + v0) ** 2) / (2 * sigma_neg ** 2))
            )
            env = torch.exp(-(u ** 2) / (2 * sigma_u ** 2))
            kern = line * env
            kern = kern - kern.mean()
            kern = kern / (kern.abs().sum() + 1e-8)
            kernels.append(kern)
        return torch.stack(kernels).unsqueeze(1)

    @staticmethod
    def _anom(defect, wafer):
        mean_d = defect.sum(dim=(2, 3), keepdim=True) / wafer.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (defect - mean_d) * wafer

    @staticmethod
    def _resp(anom, wafer, kern):
        pad = kern.shape[-1] // 2
        r = F.conv2d(anom, kern, padding=pad)
        mx = r.max(dim=1, keepdim=True).values
        an = mx - r.mean(dim=1, keepdim=True)
        return mx * wafer, an * wafer

    def forward(self, x):
        defect = (x > 0.75).to(x.dtype)
        wafer = (x > 0.25).to(x.dtype)
        anom = self._anom(defect, wafer)
        max9, aniso9 = self._resp(anom, wafer, self.k9.to(x.dtype))
        max15, aniso15 = self._resp(anom, wafer, self.k15.to(x.dtype))
        feats = torch.cat((defect, anom, max9, aniso9, max15, aniso15), dim=1)
        return self.net(feats)


class IterativeSpatialBeliefAttention(nn.Module):
    """ver9.5 (belief_base): 2D 상대 위치 연속성 + 칸별 클래스 belief 점등 + 가중치 공유 반복 추론(3-step)."""

    def __init__(self, in_dim=256, num_classes=9, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim({in_dim}) must be divisible by num_heads({num_heads})")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.h, self.w = h, w
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_steps = num_steps

        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, (2 * h - 1) * (2 * w - 1)))
        self.register_buffer("rel_pos_idx", self._build_rel_idx(h, w), persistent=False)

        self.belief_norm = nn.LayerNorm(in_dim)
        self.belief_head = nn.Linear(in_dim, num_classes)
        self.belief_proj = nn.Linear(num_classes, in_dim, bias=False)
        self.cell_gate = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.cell_gate.weight)
        nn.init.zeros_(self.cell_gate.bias)

        self.norm1 = nn.LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.proj = nn.Linear(in_dim, in_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn_fc1 = nn.Linear(in_dim, in_dim * 2)
        self.ffn_dwconv = nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=3, padding=1, groups=in_dim * 2)
        self.ffn_act = nn.GELU()
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_fc2 = nn.Linear(in_dim * 2, in_dim)

        self.attn_scales = nn.Parameter(torch.full((num_steps,), 0.05))
        self.ffn_scales = nn.Parameter(torch.full((num_steps,), 0.05))
        self.last_step_logits = None

    @staticmethod
    def _build_rel_idx(h, w, device=None):
        cy = torch.arange(h, device=device)
        cx = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=0).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel_y = rel[0] + (h - 1)
        rel_x = rel[1] + (w - 1)
        return rel_y * (2 * w - 1) + rel_x

    def _get_rel_bias(self, H, W, device, dtype):
        if H == self.h and W == self.w:
            bias = self.rel_pos_bias[:, self.rel_pos_idx]
        else:
            table = self.rel_pos_bias.view(1, self.num_heads, 2 * self.h - 1, 2 * self.w - 1)
            table = F.interpolate(table, size=(2 * H - 1, 2 * W - 1), mode="bicubic", align_corners=False)
            idx = self._build_rel_idx(H, W, device=device)
            bias = table.view(self.num_heads, -1)[:, idx]
        return bias.unsqueeze(0).to(dtype=dtype)

    def _rel_mha(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_bias(H, W, x.device, torch.float32)
        attn = self.attn_drop(attn.softmax(dim=-1)).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def _conv_ffn(self, x, H, W):
        B = x.shape[0]
        h = self.ffn_fc1(x)
        h_2d = h.transpose(1, 2).reshape(B, -1, H, W)
        h = self.ffn_dwconv(h_2d).flatten(2).transpose(1, 2)
        h = self.ffn_drop(self.ffn_act(h))
        return self.ffn_fc2(h)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        step_logits = [] if self.training else None

        for step in range(self.num_steps):
            bn = self.belief_norm(tokens)
            cell_logits = self.belief_head(bn)
            belief = F.softmax(cell_logits.float(), dim=-1).to(tokens.dtype)
            ctx_in = self.norm1(tokens) + self.belief_proj(belief)
            attn_out = self._rel_mha(ctx_in, H, W)
            tokens = tokens + self.attn_scales[step] * self.drop(attn_out)
            tokens = tokens + self.ffn_scales[step] * self.drop(self._conv_ffn(self.norm2(tokens), H, W))
            if step_logits is not None:
                post_bn = self.belief_norm(tokens)
                post_logits = self.belief_head(post_bn).float()
                w = F.softmax(self.cell_gate(post_bn).float(), dim=1)
                step_logits.append((post_logits * w).sum(dim=1))

        self.last_step_logits = step_logits
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class SpatialEvidencePooling(nn.Module):
    """ver9.6 (pool_base): 균등 AvgPool 대신 결함 신호가 켜진 공간 칸에 가중치를 집중하는 0-초기화 어텐션 풀링."""

    def __init__(self, in_dim=512, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        scores = self.gate(self.norm(tokens)).float()
        weights = F.softmax(scores, dim=1).to(x.dtype)
        pooled = (tokens * weights).sum(dim=1)
        return pooled.unsqueeze(-1).unsqueeze(-1)


class AddDilated(nn.Module):
    """1채널 입력이면 팽창 채널을 붙여 2채널로 (미세조정/평가). 이미 2채널이면 그대로 (사전학습: 팽창 후 마스킹)."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x if x.shape[1] == 2 else to_input(A.add_dilated(x / 0.5, self.k))


class SizeHead(nn.Module):
    """ver8.1: 백본 특징 (fc 앞) + 웨이퍼 크기 임베딩을 이어 붙여 분류. forward(x, idx): idx = 웨이퍼 번호 (크기 표에서 찾음).

    size: (웨이퍼 수,) 표준화된 log(die 수)
    """

    def __init__(self, base, size, dim=16):
        super().__init__()
        self.fc = nn.Linear(base.fc.in_features + dim, base.fc.out_features)
        base.fc = nn.Identity()
        self.base = base
        self.emb = nn.Sequential(nn.Linear(1, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.register_buffer("size", size)

    needs_idx = True

    def forward(self, x, idx):
        return self.fc(torch.cat([self.base(x), self.emb(self.size[idx].unsqueeze(1))], 1))


class ClassPool(nn.Module):
    """ver8.3: 클래스별 학습 풀링. 칸별 로짓 z_k(칸) = w_k · a(칸) (layer4, 96 입력이면 3x3 칸) 을
    클래스마다 LSE 풀링 (1/r_k) log mean_칸 exp(r_k z_k) + b_k 로 합침.
    r = 0 → 평균 (= 원래 avgpool + fc), r → +∞ → 최대 ('한 칸이라도 강하면'), r → -∞ → 최소 ('한 칸이라도 약하면').
    r 은 0.001 (≈ 평균, ver8 과 같은 출발점) 에서 시작해 학습됨. base.fc 가중치를 그대로 칸별 분류기로 씀.
    """

    def __init__(self, base, r0=1e-3):
        super().__init__()
        self.base = base
        self.r = nn.Parameter(torch.full((base.fc.out_features,), r0))

    def forward(self, x):
        b = self.base
        a = b.layer4(b.layer3(b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))))
        z = torch.einsum("kc,bchw->bkhw", b.fc.weight, a).flatten(2).float()           # (B, K, 칸)
        r = self.r.float()[None, :, None]
        r = torch.where(r >= 0, r.clamp(min=1e-4), r.clamp(max=-1e-4))                 # 0 나눗셈 방지
        m = z.mean(2, keepdim=True)
        pooled = m + (torch.logsumexp(r * (z - m), 2, keepdim=True) - np.log(z.shape[2])) / r
        return pooled.squeeze(2) + b.fc.bias.float()


def load_size(cache):
    """웨이퍼별 표준화된 log(die 수) (웨이퍼 수,) — ver8.1 / ver8.2 의 크기 조건."""
    d = np.log(np.load(os.path.join(cache, "wm811k_shapes_dies.npy"))[:, 2].astype(np.float64))
    return torch.tensor((d - d.mean()) / d.std(), dtype=torch.float32)


class FiLMResNet(nn.Module):
    """ver8.2: FiLM — 크기 s → FiLM Generator (MLP) → stage 별 γ, β (채널 수만큼) → layer1~4 출력 h 를 h*(1+γ)+β 로 조절.

    γ, β 를 만드는 마지막 층은 0 으로 시작 (= 처음엔 일반 ResNet 과 같음). forward(x, idx): idx = 웨이퍼 번호.
    base.fc 는 self.fc 로 옮김 (사전학습: Identity, 미세조정: 분류기). size 표는 저장하지 않음 (persistent=False).
    """
    needs_idx = True

    def __init__(self, base, size, dim=16):
        super().__init__()
        self.fc, base.fc = base.fc, nn.Identity()
        self.base = base
        self.emb = nn.Sequential(nn.Linear(1, dim), nn.ReLU(inplace=True))
        chans = [l[-1].bn2.num_features for l in (base.layer1, base.layer2, base.layer3, base.layer4)]  # BasicBlock (resnet18/34)
        self.film = nn.ModuleList(nn.Linear(dim, 2 * c) for c in chans)
        for f in self.film:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
        self.register_buffer("size", size, persistent=False)

    def features(self, x, idx):
        b = self.base
        e = self.emb(self.size[idx].unsqueeze(1))
        h = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for layer, f in zip((b.layer1, b.layer2, b.layer3, b.layer4), self.film):
            g, beta = f(e)[:, :, None, None].chunk(2, 1)
            h = layer(h) * (1 + g) + beta
        return torch.flatten(b.avgpool(h), 1)

    def forward(self, x, idx):
        return self.fc(self.features(x, idx))


class FiLMFeatures(nn.Module):
    """FiLMResNet 의 fc 앞 특징 (가중치 공유) — 군집화 / 임베딩용."""
    needs_idx = True

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, idx):
        return self.m.features(x, idx)


class FrangiInput(nn.Module):
    """ver9.1 (yeeun 브랜치 y9.2a): 입력 2채널 (원본, Frangi vesselness). conv1 을 2채널로 늘리고 원래 가중치는 채널 0, 새 채널 1 은 0 으로 시작.
    forward(x, idx): x 가 1채널이면 idx (전역 웨이퍼 번호) 로 표에서 Frangi 를 찾아 붙임 (평가·군집 진단).
    이미 2채널이면 그대로 (학습: 회전 전에 붙여서 두 채널이 같은 회전을 받음).
    table: (n, 96, 96) uint8 (V×255) CPU, row: (전체 웨이퍼 수,) 전역 번호 → 표의 행 (-1 = 없음). table=None 이면 0 채널 (대조군)."""

    needs_idx = True

    def __init__(self, base, table, row):
        super().__init__()
        w = base.conv1.weight.data
        base.conv1 = nn.Conv2d(2, w.shape[0], 7, 2, 3, bias=False).to(w.device)
        base.conv1.weight.data.zero_()
        base.conv1.weight.data[:, :1] = w
        self.base, self.table, self.row = base, table, row

    def frangi(self, idx, device):
        idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx)
        if self.table is None:
            return torch.zeros(len(idx), 1, 96, 96, device=device)
        r = self.row[idx]
        assert (r >= 0).all(), "Frangi 표에 없는 웨이퍼"
        return self.table[torch.from_numpy(r)].to(device, non_blocking=True).float().unsqueeze(1) / 255.0

    def add(self, x, idx):
        return x if x.shape[1] == 2 else torch.cat([x, self.frangi(idx, x.device).to(x.dtype)], 1)

    def forward(self, x, idx=None):
        return self.base(self.add(x, idx))


class FrangiFeatures(nn.Module):
    """ver9.1: FrangiInput 의 fc 앞 특징 (군집 진단용, 가중치 공유)."""

    needs_idx = True

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.body = nn.Sequential(*list(m.base.children())[:-1], nn.Flatten())

    def forward(self, x, idx=None):
        return self.body(self.m.add(x, idx))


def build_model(arch, num_classes, in_ch=1):
    """WBM 은 단일 채널 → 첫 conv 를 1채널로 교체 (기본: ImageNet 사전학습 없음, WaPIRL 과 동일)."""
    if IMAGENET and arch.startswith("resnet") and in_ch == 1 and not INPUT_DILATE:
        m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
        w = m.conv1.weight.detach().sum(1, keepdim=True)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.conv1.weight.data.copy_(w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    m = getattr(torchvision.models, arch)(weights=None, num_classes=num_classes)
    if arch.startswith("resnet") and INPUT_DILATE:
        m.conv1 = nn.Sequential(AddDilated(INPUT_DILATE), nn.Conv2d(2, 64, 7, 2, 3, bias=False))
    elif arch.startswith("resnet"):
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    elif arch.startswith("densenet"):
        m.features.conv0 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    else:
        raise ValueError(arch)
    if USE_SA and arch.startswith("resnet"):
        if USE_SA in (True, "layer4"):
            dim = m.fc.in_features
            m.avgpool = nn.Sequential(
                SpatialSelfAttention(in_dim=dim, h=3, w=3, num_heads=4),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif USE_SA == "layer3":
            dim = _layer_out_channels(m.layer3)
            m.layer3.add_module("sa", SpatialSelfAttention(in_dim=dim, h=6, w=6, num_heads=4, dropout=0.1, use_gate=True))
        elif USE_SA == "layer2":
            dim = _layer_out_channels(m.layer2)
            m.layer2.add_module("sa", SpatialSelfAttention(in_dim=dim, h=12, w=12, num_heads=4, dropout=0.1, use_gate=True))
        elif USE_SA == "line_base":
            dim3 = _layer_out_channels(m.layer3)
            m.line_branch = OrientedLineBranch(out_dim=dim3, dropout=0.1)
            m.line_scale = nn.Parameter(torch.tensor(0.05))

            def _forward_features_line(x):
                f_line = m.line_branch(x)
                h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                h = m.layer3(m.layer2(m.layer1(h)))
                if f_line.shape[-2:] != h.shape[-2:]:
                    f_line = F.adaptive_max_pool2d(f_line, h.shape[-2:])
                h = h + m.line_scale * f_line
                h = m.avgpool(m.layer4(h))
                return torch.flatten(h, 1)

            m.forward_features = _forward_features_line
            m._forward_impl = lambda x: m.fc(_forward_features_line(x))
        elif USE_SA == "belief_base":
            dim3 = _layer_out_channels(m.layer3)
            m.layer3.add_module(
                "sa", IterativeSpatialBeliefAttention(in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1)
            )
        elif USE_SA == "pool_base":
            m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1)
        else:
            raise ValueError(f"Unknown USE_SA: {USE_SA}")
    return m


def to_input(x):
    """{0,1,2} 맵 → 모델 입력 [0,1]."""
    return x * 0.5
