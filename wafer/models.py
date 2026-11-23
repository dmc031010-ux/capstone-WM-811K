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
# ver9.1: True/"layer4" 이면 layer4 뒤(3x3), "layer3"(ver9.5) 이면 6x6, "layer2"(ver9.6) 이면 12x12 에 Spatial Self-Attention 추가
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
    """(B, C, H, W) 특징 맵의 spatial grid(H*W 토큰) 간 Multi-Head Self-Attention."""

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


class IterativeSpatialBeliefAttention(nn.Module):
    """ver9.7: 2D 상대 위치 연속성 + 자리별 클래스 belief 점등 + 가중치 공유 반복 추론(3-step).

    mode:
      base        : 기존 3-step 반복 추론
      topk_gate   : ver9.8A - receiver gate(2*sigmoid) + Top-6 칸 보조손실(빈칸 오염 방지)
      radial_geom : ver9.8B - 중심 거리 r 임베딩 + 쌍별 기하 바이어스 (d_ij, |dr|, mean_r)
      temp_ent    : ver9.8C - 스텝별 온도 냉각 (2.0->1.0->0.5) + 엔트로피 기반 업데이트 게이팅
      consensus   : ver9.8D - 3x3 이웃 평균 로짓 합의 업데이트로 고립 오탐 억제
    """

    def __init__(self, in_dim=256, num_classes=9, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1, mode="base"):
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
        self.mode = mode

        # 2D Relative Position Bias table: (num_heads, (2h-1)*(2w-1))
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, (2 * h - 1) * (2 * w - 1)))
        self.register_buffer("rel_pos_idx", self._build_rel_idx(h, w), persistent=False)

        # 자리별 label belief 점등 및 특징 주입 (반복 단계 간 가중치 공유)
        self.belief_norm = nn.LayerNorm(in_dim)
        self.belief_head = nn.Linear(in_dim, num_classes)
        self.belief_proj = nn.Linear(num_classes, in_dim, bias=False)
        self.cell_gate = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.cell_gate.weight)
        nn.init.zeros_(self.cell_gate.bias)

        if mode == "radial_geom":
            rf, pg = self._build_geom(h, w)
            self.register_buffer("rad_feat", rf, persistent=False)
            self.register_buffer("pair_geom", pg, persistent=False)
            self.rad_proj = nn.Linear(2, in_dim)
            nn.init.zeros_(self.rad_proj.weight)
            nn.init.zeros_(self.rad_proj.bias)
            self.geom_bias = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, num_heads))
            nn.init.zeros_(self.geom_bias[-1].weight)
            nn.init.zeros_(self.geom_bias[-1].bias)
        elif mode == "consensus":
            self.consensus_scale = nn.Parameter(torch.tensor(0.25))
            nk = torch.ones(1, 1, 3, 3) / 8.0
            nk[0, 0, 1, 1] = 0.0
            self.register_buffer("neigh_kernel", nk.repeat(num_classes, 1, 1, 1), persistent=False)

        # Weight-shared Relative Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(in_dim)
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.proj = nn.Linear(in_dim, in_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        # Weight-shared Continuity FFN (3x3 depthwise conv 로 인접 X/Y/대각 연속성 보강)
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
    def _build_geom(h, w, device=None):
        cy = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2.0) / max(1.0, (h - 1) / 2.0)
        cx = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2.0) / max(1.0, (w - 1) / 2.0)
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        coords = torch.stack((gy.flatten(), gx.flatten()), dim=1)  # (N, 2)
        r = torch.sqrt((coords ** 2).sum(dim=1) / 2.0).clamp(0.0, 1.0)  # (N,)
        rad_feat = torch.stack((r, r ** 2), dim=1).unsqueeze(0)  # (1, N, 2)
        diff = coords[:, None, :] - coords[None, :, :]
        dij = torch.sqrt((diff ** 2).sum(dim=-1) / 8.0).clamp(0.0, 1.0)
        dr = (r[:, None] - r[None, :]).abs()
        mr = 0.5 * (r[:, None] + r[None, :])
        pair_geom = torch.stack((dij, dr, mr), dim=-1)  # (N, N, 3)
        return rad_feat, pair_geom

    @staticmethod
    def _build_rel_idx(h, w, device=None):
        cy = torch.arange(h, device=device)
        cx = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), dim=0).flatten(1)  # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
        rel_y = rel[0] + (h - 1)
        rel_x = rel[1] + (w - 1)
        return rel_y * (2 * w - 1) + rel_x  # (N, N)

    def _get_rel_bias(self, H, W, device, dtype):
        if H == self.h and W == self.w:
            bias = self.rel_pos_bias[:, self.rel_pos_idx]  # (heads, N, N)
            pg = self.pair_geom if self.mode == "radial_geom" else None
        else:
            table = self.rel_pos_bias.view(1, self.num_heads, 2 * self.h - 1, 2 * self.w - 1)
            table = F.interpolate(table, size=(2 * H - 1, 2 * W - 1), mode="bicubic", align_corners=False)
            idx = self._build_rel_idx(H, W, device=device)
            bias = table.view(self.num_heads, -1)[:, idx]
            pg = self._build_geom(H, W, device=device)[1] if self.mode == "radial_geom" else None
        out = bias.unsqueeze(0).to(dtype=dtype)
        if pg is not None:
            gb = self.geom_bias(pg.to(dtype=self.geom_bias[0].weight.dtype)).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
            out = out + gb
        return out

    def _rel_mha(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_bias(H, W, x.device, torch.float32)
        attn = self.attn_drop(attn.softmax(dim=-1)).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def _conv_ffn(self, x, H, W):
        B, N, C = x.shape
        h = self.ffn_fc1(x)
        h_2d = h.transpose(1, 2).reshape(B, -1, H, W)
        h = self.ffn_dwconv(h_2d).flatten(2).transpose(1, 2)
        h = self.ffn_drop(self.ffn_act(h))
        return self.ffn_fc2(h)

    def _apply_consensus(self, cell_logits, B, H, W):
        cl_2d = cell_logits.float().transpose(1, 2).reshape(B, self.num_classes, H, W)
        nk = self.neigh_kernel.to(device=cl_2d.device, dtype=torch.float32)
        neigh_l = F.conv2d(cl_2d, nk, padding=1, groups=self.num_classes).flatten(2).transpose(1, 2)
        return cell_logits.float() + self.consensus_scale.float() * neigh_l

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        step_logits = [] if self.training else None
        rf = None
        if self.mode == "radial_geom":
            rf = self.rad_feat if (H == self.h and W == self.w) else self._build_geom(H, W, device=x.device)[0]
            rf = self.rad_proj(rf.to(dtype=self.rad_proj.weight.dtype)).to(tokens.dtype)
        taus = (2.0, 1.0, 0.5)

        for step in range(self.num_steps):
            bn = self.belief_norm(tokens)
            cell_logits = self.belief_head(bn)  # (B, N, num_classes)
            if self.mode == "consensus":
                cell_logits = self._apply_consensus(cell_logits, B, H, W).to(tokens.dtype)
            tau = taus[min(step, 2)] if self.mode == "temp_ent" else 1.0
            belief_f = F.softmax(cell_logits.float() / tau, dim=-1)
            belief = belief_f.to(tokens.dtype)
            ctx_in = self.norm1(tokens) + self.belief_proj(belief)
            if rf is not None:
                ctx_in = ctx_in + rf
            attn_out = self._rel_mha(ctx_in, H, W)
            if self.mode == "topk_gate":
                attn_out = attn_out * (2.0 * torch.sigmoid(self.cell_gate(bn)))
            elif self.mode == "temp_ent":
                ent = -(belief_f * (belief_f + 1e-8).log()).sum(dim=-1, keepdim=True) / math.log(self.num_classes)
                attn_out = attn_out * (0.5 + ent).to(tokens.dtype)
            tokens = tokens + self.attn_scales[step] * self.drop(attn_out)
            tokens = tokens + self.ffn_scales[step] * self.drop(self._conv_ffn(self.norm2(tokens), H, W))
            if step_logits is not None:
                post_bn = self.belief_norm(tokens)
                post_logits = self.belief_head(post_bn).float()
                if self.mode == "consensus":
                    post_logits = self._apply_consensus(post_logits, B, H, W)
                gate_raw = self.cell_gate(post_bn).float()
                if self.mode == "topk_gate":
                    k_top = min(6, gate_raw.shape[1])
                    top_s, top_idx = gate_raw.squeeze(-1).topk(k_top, dim=1)
                    top_l = torch.gather(post_logits, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.num_classes))
                    w = F.softmax(top_s, dim=1).unsqueeze(-1)
                    step_logits.append((top_l * w).sum(dim=1))
                else:
                    w = F.softmax(gate_raw, dim=1)
                    step_logits.append((post_logits * w).sum(dim=1))

        self.last_step_logits = step_logits
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class RelSpatialAttention(nn.Module):
    """ver9.8 (layer2, 12x12): 2D 상대 위치 바이어스 + 3x3 DWConv FFN + LayerScale 경량 공간 어텐션.

    layer3 축소 전에 얇은 1~2px Scratch 선을 12x12(8px/칸) 해상도에서 먼저 연결해 보존하는 역할.
    """

    def __init__(self, in_dim=128, h=12, w=12, num_heads=4, dropout=0.1, init_scale=0.05):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim({in_dim}) must be divisible by num_heads({num_heads})")
        self.h, self.w = h, w
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, (2 * h - 1) * (2 * w - 1)))
        self.register_buffer("rel_pos_idx", IterativeSpatialBeliefAttention._build_rel_idx(h, w), persistent=False)

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

        self.attn_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def _get_rel_bias(self, H, W, device, dtype):
        if H == self.h and W == self.w:
            bias = self.rel_pos_bias[:, self.rel_pos_idx]
        else:
            table = self.rel_pos_bias.view(1, self.num_heads, 2 * self.h - 1, 2 * self.w - 1)
            table = F.interpolate(table, size=(2 * H - 1, 2 * W - 1), mode="bicubic", align_corners=False)
            idx = IterativeSpatialBeliefAttention._build_rel_idx(H, W, device=device)
            bias = table.view(self.num_heads, -1)[:, idx]
        return bias.unsqueeze(0).to(dtype=dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        h_in = self.norm1(tokens)
        qkv = self.qkv(h_in).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_bias(H, W, x.device, torch.float32)
        attn = self.attn_drop(attn.softmax(dim=-1)).to(v.dtype)
        attn_out = self.proj((attn @ v).transpose(1, 2).reshape(B, H * W, C))
        tokens = tokens + self.attn_scale * self.drop(attn_out)

        f = self.ffn_fc1(self.norm2(tokens))
        f = self.ffn_dwconv(f.transpose(1, 2).reshape(B, -1, H, W)).flatten(2).transpose(1, 2)
        f = self.ffn_fc2(self.ffn_drop(self.ffn_act(f)))
        tokens = tokens + self.ffn_scale * self.drop(f)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class OrientedLineBranch(nn.Module):
    """ver9.9: 원본 96x96 웨이퍼 맵에서 얇은 선형(Scratch) 리지를 손실 없이 추출해 6x6으로 전달하는 보조 스트림.

    - 웨이퍼 내부 평균 불량 밀도를 제거한 anomaly 맵에 다중 스케일(9x9, 15x15) 8방향 zero-mean 선형 필터 적용
    - 회전 불변 방향성 이방성(max - mean)으로 등방성 랜덤 노이즈(none)와 뭉친 결함(Loc)을 억제하고 얇은 선(Scratch)만 강조
    - stride-2 평균 희석 대신 MaxPool2d(2)로 96→48→24→12→6 까지 1~2px 선형 피크 신호를 보존

    mode (ver9.7 필터 변형, 모두 self.net 입력은 6채널로 동일):
      base      : k9 + k15 (기존)
      pyr_max   : 해법 A  - MaxPool 피라미드 (96/48/32) + 단일 k9
      pyr_avg   : 신규    - AvgPool 피라미드 (96/48/32) + 단일 k9 (최근접 업샘플링의 역연산)
      learn     : 해법 B  - 폭/억제날개/길이(sigma_v, v0, sigma_neg, sigma_u)를 학습 파라미터로
      scale     : 해법 C  - 웨이퍼별 die 블록 크기 s 추정 → k9 을 dilation 1/2/3 으로 찍고 s 로 선형 보간
    """

    def __init__(self, out_dim=256, dropout=0.1, mode="base"):
        super().__init__()
        self.mode = mode
        self.register_buffer("k9", self._make_oriented_kernels(9, n_ori=8), persistent=False)
        self.register_buffer("k15", self._make_oriented_kernels(15, n_ori=8), persistent=False)
        if mode == "learn":
            # 정규화 좌표 [-1, 1] 기준 초기값 = 기존 상수 (0.14, 0.35, 0.18, 0.75). log 공간 학습
            init = torch.log(torch.tensor([0.14, 0.35, 0.18, 0.75]))
            self.log_p9 = nn.Parameter(init.clone())
            self.log_p15 = nn.Parameter(init.clone())
        elif mode == "pyr_avg_learn":
            init = torch.log(torch.tensor([0.14, 0.35, 0.18, 0.75]))
            self.log_p9_pyr = nn.Parameter(init.unsqueeze(0).repeat(3, 1).clone())
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
    def _kernel_from_params(ks, sigma_v, v0, sigma_neg, sigma_u, n_ori=8):
        coords = torch.linspace(-1.0, 1.0, ks, device=sigma_v.device, dtype=sigma_v.dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernels = []
        for k in range(n_ori):
            th = k * math.pi / n_ori
            c, s_ = math.cos(th), math.sin(th)
            u = xx * c + yy * s_
            v = -xx * s_ + yy * c
            line = (torch.exp(-(v ** 2) / (2 * sigma_v ** 2))
                    - 0.5 * torch.exp(-((v - v0) ** 2) / (2 * sigma_neg ** 2))
                    - 0.5 * torch.exp(-((v + v0) ** 2) / (2 * sigma_neg ** 2)))
            env = torch.exp(-(u ** 2) / (2 * sigma_u ** 2))
            kern = line * env
            kern = kern - kern.mean()
            kern = kern / (kern.abs().sum() + 1e-8)
            kernels.append(kern)
        return torch.stack(kernels).unsqueeze(1)  # (n_ori, 1, ks, ks)

    @staticmethod
    def _make_oriented_kernels(ks, n_ori=8):
        p = torch.tensor([0.14, 0.35, 0.18, 0.75])
        return OrientedLineBranch._kernel_from_params(ks, p[0], p[1], p[2], p[3], n_ori)

    @staticmethod
    def _resp(anom, wafer, kern, dilation=1):
        pad = (kern.shape[-1] // 2) * dilation
        r = F.conv2d(anom, kern, padding=pad, dilation=dilation)
        mx = r.max(dim=1, keepdim=True).values
        an = mx - r.mean(dim=1, keepdim=True)
        return mx * wafer, an * wafer

    @staticmethod
    def _anom(defect, wafer):
        mean_d = defect.sum(dim=(2, 3), keepdim=True) / wafer.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return (defect - mean_d) * wafer

    @staticmethod
    def estimate_die_scale(x):
        """입력 맵만으로 웨이퍼별 1-die 픽셀 크기 s 추정 (1~3). 최근접 업샘플링이면 die 경계 열/행에서만 값이 바뀜."""
        B, _, H, W = x.shape
        tx = ((x[..., 1:] - x[..., :-1]).abs() > 1e-6).any(dim=2).any(dim=1).float().sum(dim=-1)   # (B,) 변화 있는 열 수
        ty = ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs() > 1e-6).any(dim=3).any(dim=1).float().sum(dim=-1)
        sx = W / (tx + 1.0)
        sy = H / (ty + 1.0)
        return (0.5 * (sx + sy)).clamp(1.0, 3.0).view(B, 1, 1, 1)

    def forward(self, x):
        # x: (B, 1, H, W) in {0.0, 0.5, 1.0}
        defect = (x > 0.75).to(x.dtype)
        wafer = (x > 0.25).to(x.dtype)
        anom = self._anom(defect, wafer)
        H, W = x.shape[-2:]
        k9 = self.k9.to(x.dtype)
        if self.mode == "base":
            max9, aniso9 = self._resp(anom, wafer, k9)
            max15, aniso15 = self._resp(anom, wafer, self.k15.to(x.dtype))
            feats = torch.cat((defect, anom, max9, aniso9, max15, aniso15), dim=1)
        elif self.mode == "learn":
            p9, p15 = torch.exp(self.log_p9), torch.exp(self.log_p15)
            kk9 = self._kernel_from_params(9, p9[0], p9[1], p9[2], p9[3]).to(x.dtype)
            kk15 = self._kernel_from_params(15, p15[0], p15[1], p15[2], p15[3]).to(x.dtype)
            max9, aniso9 = self._resp(anom, wafer, kk9)
            max15, aniso15 = self._resp(anom, wafer, kk15)
            feats = torch.cat((defect, anom, max9, aniso9, max15, aniso15), dim=1)
        elif self.mode in ("pyr_max", "pyr_avg", "pyr_avg_learn"):
            mx_all, ans = [], []
            for idx, f in enumerate((1, 2, 3)):
                if f == 1:
                    d_s, w_s = defect, wafer
                elif self.mode == "pyr_max":
                    d_s, w_s = F.max_pool2d(defect, f, f), F.max_pool2d(wafer, f, f)
                else:
                    d_s, w_s = F.avg_pool2d(defect, f, f), (F.avg_pool2d(wafer, f, f) > 0.5).to(x.dtype)
                a_s = self._anom(d_s, w_s)
                if self.mode == "pyr_avg_learn":
                    p = torch.exp(self.log_p9_pyr[idx])
                    k_s = self._kernel_from_params(9, p[0], p[1], p[2], p[3]).to(x.dtype)
                else:
                    k_s = k9
                mx, an = self._resp(a_s, w_s, k_s)
                if f > 1:
                    mx = F.interpolate(mx, size=(H, W), mode="bilinear", align_corners=False) * wafer
                    an = F.interpolate(an, size=(H, W), mode="bilinear", align_corners=False) * wafer
                mx_all.append(mx); ans.append(an)
            max_all = torch.stack(mx_all, 0).max(0).values
            feats = torch.cat((defect, anom, ans[0], ans[1], ans[2], max_all), dim=1)
        elif self.mode == "scale":
            s = self.estimate_die_scale(x).to(x.dtype)
            mxs, ans = [], []
            for d in (1, 2, 3):
                mx, an = self._resp(anom, wafer, k9, dilation=d)
                mxs.append(mx); ans.append(an)
            w1 = (2.0 - s).clamp(0.0, 1.0)
            w2 = (1.0 - (s - 2.0).abs()).clamp(0.0, 1.0)
            w3 = (s - 2.0).clamp(0.0, 1.0)
            max_a = w1 * mxs[0] + w2 * mxs[1] + w3 * mxs[2]
            aniso_a = w1 * ans[0] + w2 * ans[1] + w3 * ans[2]
            feats = torch.cat((defect, anom, max_a, aniso_a, ans[0], ans[2]), dim=1)
        else:
            raise ValueError(self.mode)
        return self.net(feats)


class SpatialEvidencePooling(nn.Module):
    """ver9.7: 균등 AvgPool 대신 결함 신호가 켜진 공간 칸에 가중치를 집중하는 어텐션 풀링.

    mode:
      base      : 기존 0-초기화 gated pooling
      multitemp : ver9.8PA - 3온도(0.25, 1.0, 4.0) 병렬 게이트 + 채널별 가중 결합
      radial    : ver9.8PB - 3x3 중심/모서리/꼭짓점 위치 임베딩(0-초기화) 추가
      contrast  : ver9.8PC - 전경(fg) - 배경(bg) 대비 residual 추가 (0-초기화)
      peak      : ver9.8PD - (max - mean) 피크 잔차 추가 (0-초기화)
    """

    def __init__(self, in_dim=512, hidden_dim=128, dropout=0.1, mode="base"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(in_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        if mode == "multitemp":
            self.multi_gate = nn.Linear(hidden_dim, 3)
            nn.init.zeros_(self.multi_gate.weight)
            nn.init.zeros_(self.multi_gate.bias)
            self.head_mix = nn.Parameter(torch.zeros(3, in_dim))
        elif mode == "radial":
            self.register_buffer("zone_idx", torch.tensor([2, 1, 2, 1, 0, 1, 2, 1, 2], dtype=torch.long), persistent=False)
            self.zone_emb = nn.Parameter(torch.zeros(3, in_dim))
        elif mode == "contrast":
            self.contrast_scale = nn.Parameter(torch.tensor(0.0))
        elif mode == "peak":
            self.peak_scale = nn.Parameter(torch.zeros(1, in_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        if self.mode == "radial" and H == 3 and W == 3:
            tokens = tokens + self.zone_emb[self.zone_idx].unsqueeze(0).to(tokens.dtype)
        norm_t = self.norm(tokens)
        if self.mode == "multitemp":
            h_g = self.gate[:-1](norm_t)
            scores3 = self.multi_gate(h_g).float()  # (B, N, 3)
            ps = []
            for idx, tau in enumerate((0.25, 1.0, 4.0)):
                wk = F.softmax(scores3[..., idx:idx + 1] / tau, dim=1).to(x.dtype)
                ps.append((tokens * wk).sum(dim=1))
            mix = F.softmax(self.head_mix, dim=0).unsqueeze(1).to(x.dtype)
            pooled = (torch.stack(ps, dim=0) * mix).sum(dim=0)
            return pooled.unsqueeze(-1).unsqueeze(-1)
        scores = self.gate(norm_t).float()  # (B, H*W, 1)
        weights = F.softmax(scores, dim=1).to(x.dtype)
        pooled = (tokens * weights).sum(dim=1)  # (B, C)
        if self.mode == "contrast":
            bg_w = F.softmax(-scores, dim=1).to(x.dtype)
            bg = (tokens * bg_w).sum(dim=1)
            pooled = pooled + self.contrast_scale.to(x.dtype) * (pooled - bg)
        elif self.mode == "peak":
            peak = tokens.max(dim=1).values - tokens.mean(dim=1)
            pooled = pooled + self.peak_scale.to(x.dtype) * peak
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


def build_model(arch, num_classes, in_ch=1):
    """WBM 은 단일 채널 → 첫 conv 를 1채널로 교체 (ImageNet 사전학습 없음, WaPIRL 과 동일)."""
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
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif USE_SA == "layer3":
            dim = _layer_out_channels(m.layer3)
            m.layer3.add_module("sa", SpatialSelfAttention(in_dim=dim, h=6, w=6, num_heads=4, dropout=0.1, use_gate=True))
        elif USE_SA == "layer2":
            dim = _layer_out_channels(m.layer2)
            m.layer2.add_module("sa", SpatialSelfAttention(in_dim=dim, h=12, w=12, num_heads=4, dropout=0.1, use_gate=True))
        elif USE_SA in ("ver9.7", "ver9.7_1step"):
            dim3 = _layer_out_channels(m.layer3)
            m.layer3.add_module("sa", IterativeSpatialBeliefAttention(
                in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4,
                num_steps=1 if USE_SA == "ver9.7_1step" else 3, dropout=0.1
            ))
            m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1)
        elif USE_SA == "ver9.7_belief":
            dim3 = _layer_out_channels(m.layer3)
            m.layer3.add_module("sa", IterativeSpatialBeliefAttention(
                in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1
            ))
        elif USE_SA == "ver9.7_pool":
            m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1)
        elif USE_SA == "ver9.8":
            dim2 = _layer_out_channels(m.layer2)
            dim3 = _layer_out_channels(m.layer3)
            m.layer2.add_module("sa", RelSpatialAttention(in_dim=dim2, h=12, w=12, num_heads=4, dropout=0.1, init_scale=0.05))
            m.layer3.add_module("sa", IterativeSpatialBeliefAttention(
                in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1
            ))
            m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1)
        elif USE_SA in ("ver9.9", "ver9.9A", "ver9.9B", "ver9.9C", "ver9.9_pyr_max", "ver9.9_pyr_avg", "ver9.9_learn", "ver9.9_scale",
                    "ver9.9A_pyr_max", "ver9.9A_pyr_avg", "ver9.9A_learn", "ver9.9A_scale"):
            V97 = {"ver9.9_pyr_max": "pyr_max", "ver9.9_pyr_avg": "pyr_avg", "ver9.9_learn": "learn", "ver9.9_scale": "scale",
                   "ver9.9A_pyr_max": "pyr_max", "ver9.9A_pyr_avg": "pyr_avg", "ver9.9A_learn": "learn", "ver9.9A_scale": "scale"}
            dim3 = _layer_out_channels(m.layer3)
            m.line_branch = OrientedLineBranch(out_dim=dim3, dropout=0.1, mode=V97.get(USE_SA, "base"))
            m.line_scale = nn.Parameter(torch.tensor(0.05))
            if USE_SA in ("ver9.9", "ver9.9B", "ver9.9_pyr_max", "ver9.9_pyr_avg", "ver9.9_learn", "ver9.9_scale"):
                m.layer3_sa = IterativeSpatialBeliefAttention(
                    in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1
                )
            if USE_SA in ("ver9.9", "ver9.9C", "ver9.9_pyr_max", "ver9.9_pyr_avg", "ver9.9_learn", "ver9.9_scale"):
                m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1)

            def _forward_features_v99(x):
                f_line = m.line_branch(x)
                h = m.conv1(x)
                h = m.bn1(h)
                h = m.relu(h)
                h = m.maxpool(h)
                h = m.layer1(h)
                h = m.layer2(h)
                h = m.layer3(h)
                if f_line.shape[-2:] != h.shape[-2:]:
                    f_line = F.adaptive_max_pool2d(f_line, h.shape[-2:])
                h = h + m.line_scale * f_line
                if hasattr(m, "layer3_sa"):
                    h = m.layer3_sa(h)
                h = m.layer4(h)
                h = m.avgpool(h)
                return torch.flatten(h, 1)

            def _forward_impl_v99(x):
                return m.fc(_forward_features_v99(x))

            m.forward_features = _forward_features_v99
            m._forward_impl = _forward_impl_v99
        elif isinstance(USE_SA, tuple) and len(USE_SA) == 4 and USE_SA[0] == "combo":
            _, l_m, b_m, p_m = USE_SA
            dim3 = _layer_out_channels(m.layer3)
            if l_m is not None:
                m.line_branch = OrientedLineBranch(out_dim=dim3, dropout=0.1, mode=l_m)
                m.line_scale = nn.Parameter(torch.tensor(0.05))
            if b_m is not None:
                m.layer3_sa = IterativeSpatialBeliefAttention(
                    in_dim=dim3, num_classes=num_classes, h=6, w=6, num_heads=4, num_steps=3, dropout=0.1, mode=b_m
                )
            if p_m is not None:
                m.avgpool = SpatialEvidencePooling(in_dim=m.fc.in_features, hidden_dim=128, dropout=0.1, mode=p_m)

            def _forward_features_combo(x):
                f_line = m.line_branch(x) if hasattr(m, "line_branch") else None
                h = m.conv1(x)
                h = m.bn1(h)
                h = m.relu(h)
                h = m.maxpool(h)
                h = m.layer1(h)
                h = m.layer2(h)
                h = m.layer3(h)
                if f_line is not None:
                    if f_line.shape[-2:] != h.shape[-2:]:
                        f_line = F.adaptive_max_pool2d(f_line, h.shape[-2:])
                    h = h + m.line_scale * f_line
                if hasattr(m, "layer3_sa"):
                    h = m.layer3_sa(h)
                h = m.layer4(h)
                h = m.avgpool(h)
                return torch.flatten(h, 1)

            def _forward_impl_combo(x):
                return m.fc(_forward_features_combo(x))

            m.forward_features = _forward_features_combo
            m._forward_impl = _forward_impl_combo
        else:
            raise ValueError(f"Unsupported USE_SA={USE_SA!r}")
    return m


def to_input(x):
    """{0,1,2} 맵 → 모델 입력 [0,1]."""
    return x * 0.5
