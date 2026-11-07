"""fw640 architectures (fw1 LiftGD, fw2 AsymRes, fw3 DenseQuery) designed for 640px across scales n/s/m."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.conv import Conv as _Conv, DWConv as _DWConv, Index as _Index
    from ultralytics.nn.modules.block import C3k2 as _C3k2
    from ultralytics.nn.modules.head import Detect as _Detect
    from ultralytics.nn.tasks import DetectionModel as _DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss as _V8Loss, BboxLoss as _BboxLoss
    from ultralytics.utils.tal import TaskAlignedAssigner as _TALAssigner
except Exception:  # pragma: no cover
    _Conv = _DWConv = _Index = _C3k2 = _Detect = _DetectionModel = _V8Loss = _BboxLoss = _TALAssigner = object


def _zero_bn(conv) -> None:
    nn.init.zeros_(conv.bn.weight)
    nn.init.zeros_(conv.bn.bias)


# ==============================================================================
# fw1: LiftGD — Dual-Hub Gather-and-Inject Neck (Low-GD s8 + High-GD PPA s32)
# ==============================================================================

class PolyKernelBlock(nn.Module):
    """Multi-branch depthwise large-kernel block (3x3, 5x5, 1x11, 11x1) + spatial gate + PW residual."""

    def __init__(self, c: int):
        super().__init__()
        self.pw1 = _Conv(c, c, 1, 1)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw1x11 = nn.Conv2d(c, c, (1, 11), 1, (0, 5), groups=c, bias=False)
        self.dw11x1 = nn.Conv2d(c, c, (11, 1), 1, (5, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c, eps=1e-3, momentum=0.03)
        self.spatial_gate = nn.Conv2d(2, 1, 7, 1, 3)
        self.pw2 = _Conv(c, c, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw1(x)
        d = F.silu(self.bn(self.dw3(y) + self.dw5(y) + self.dw1x11(y) + self.dw11x1(y)))
        g = torch.sigmoid(self.spatial_gate(torch.cat([d.mean(1, keepdim=True), d.amax(1, keepdim=True)], 1)))
        return x + self.pw2(d * g)


class PPABlock(nn.Module):
    """Pyramid-Pooling Attention block for the stride-32 High-GD hub (HCF-Net PPA + strip context)."""

    def __init__(self, c: int):
        super().__init__()
        cp = max(16, c // 4)
        self.p1 = _Conv(c, cp, 1, 1)
        self.p2 = _Conv(c, cp, 1, 1)
        self.p4 = _Conv(c, cp, 1, 1)
        self.loc = _Conv(c, c, 3, 1, g=c, act=False)
        self.s1 = nn.Conv2d(c, c, (1, 11), 1, (0, 5), groups=c, bias=False)
        self.s2 = nn.Conv2d(c, c, (11, 1), 1, (5, 0), groups=c, bias=False)
        self.sbn = nn.BatchNorm2d(c, eps=1e-3, momentum=0.03)
        self.fuse = _Conv(c + 3 * cp, c, 1, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(16, c // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(16, c // 4), c, 1),
            nn.Sigmoid(),
        )
        self.out = _Conv(c, c, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sz = x.shape[2:]
        p1 = F.interpolate(self.p1(F.adaptive_avg_pool2d(x, 1)), size=sz, mode="nearest")
        p2 = F.interpolate(self.p2(F.adaptive_avg_pool2d(x, 2)), size=sz, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.p4(F.adaptive_avg_pool2d(x, 4)), size=sz, mode="bilinear", align_corners=False)
        loc = F.silu(self.loc(x) + self.sbn(self.s1(x) + self.s2(x)))
        y = self.fuse(torch.cat([loc, p1, p2, p4], 1))
        return x + self.out(y * self.ca(y))


class FGInject(nn.Module):
    """Foreground-gated zero-init injection: x * (1 + m * tanh(G)) + m * V.
    At step 0: G=0, V=0 so output == x exactly, while d(out)/dV = m = 0.5 != 0."""

    def __init__(self, c_hub: int, c_lvl: int, mode: str):
        super().__init__()
        self.mode = mode
        self.fg = nn.Conv2d(c_lvl, 1, 1)
        nn.init.zeros_(self.fg.weight)
        nn.init.zeros_(self.fg.bias)
        if mode == "down":
            self.pre = _Conv(c_hub, c_hub, 3, 2, g=c_hub)
        else:
            self.pre = None
        self.proj = _Conv(c_hub, 2 * c_lvl, 1, 1, act=False)
        _zero_bn(self.proj)

    def forward(self, x: torch.Tensor, hub: torch.Tensor) -> torch.Tensor:
        h = self.pre(hub) if self.pre is not None else hub
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        g, v = self.proj(h).chunk(2, 1)
        m = torch.sigmoid(self.fg(x))
        return x * (1.0 + m * torch.tanh(g)) + m * v


class FW1DualHubNeck(nn.Module):
    """Dual-Hub Gather-and-Inject Neck on top of COCO PANet [P2, P3, P4, P5] and backbone [L2, L4, L6, L10].
    Preserves exact PANet outputs at step 0 while providing non-zero step-1 gradients to both hubs."""

    def __init__(self, c_pan: list[int], c_bk: list[int], nhub: int = 2):
        super().__init__()
        c2, c3, c4, c5 = c_pan
        b2, b4, b6, b10 = c_bk
        ch_low = c3
        q_low = max(16, ch_low // 4)
        self.l_b2 = _Conv(b2, q_low, 1, 1)
        self.l_b4 = _Conv(b4, q_low, 1, 1)
        self.l_b6 = _Conv(b6, q_low, 1, 1)
        self.l_b10 = _Conv(b10, q_low, 1, 1)
        self.low_fuse = _Conv(4 * q_low + c3, ch_low, 1, 1)
        self.low_hub = nn.Sequential(*[PolyKernelBlock(ch_low) for _ in range(nhub)])

        ch_high = c4
        q_high = max(16, ch_high // 4)
        self.h_p3 = _Conv(c3, q_high, 1, 1)
        self.h_p4 = _Conv(c4, q_high, 1, 1)
        self.h_p5 = _Conv(c5, 2 * q_high, 1, 1)
        self.high_fuse = _Conv(4 * q_high, ch_high, 1, 1)
        self.high_hub = PPABlock(ch_high)

        self.inj_p2 = FGInject(ch_low, c2, "up")
        self.inj_p3 = FGInject(ch_low, c3, "same")
        self.inj_p4_low = FGInject(ch_low, c4, "down")
        self.inj_p4_high = FGInject(ch_high, c4, "up")
        self.inj_p5 = FGInject(ch_high, c5, "same")

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        p2, p3, p4, p5, l2, l4, l6, l10 = xs
        s8 = p3.shape[2:]
        g_low = torch.cat([
            F.avg_pool2d(self.l_b2(l2), 2, 2),
            self.l_b4(l4),
            F.interpolate(self.l_b6(l6), size=s8, mode="bilinear", align_corners=False),
            F.interpolate(self.l_b10(l10), size=s8, mode="bilinear", align_corners=False),
            p3,
        ], 1)
        h_low = self.low_hub(self.low_fuse(g_low))

        s32 = p5.shape[2:]
        g_high = torch.cat([
            F.adaptive_avg_pool2d(self.h_p3(p3), s32),
            F.avg_pool2d(self.h_p4(p4), 2, 2),
            self.h_p5(p5),
        ], 1)
        h_high = self.high_hub(self.high_fuse(g_high))

        o2 = self.inj_p2(p2, h_low)
        o3 = self.inj_p3(p3, h_low)
        o4 = self.inj_p4_high(self.inj_p4_low(p4, h_low), h_high)
        o5 = self.inj_p5(p5, h_high)
        return [o2, o3, o4, o5]


# ==============================================================================
# fw2: AsymRes — Shallow x2 Lift + Context-Guided Bridge + DySample + CSP-Attn
# ==============================================================================

class CGDownBridge(nn.Module):
    """Context-Guided Downsample Bridge (2x downsample from lifted shallow trunk back to native grid).
    Shortcut is AvgPool2d(2, 2); residual combines local DW 3x3 s2 + surround dilated DW 3x3 s2 d=2 + global context gate."""

    def __init__(self, c: int):
        super().__init__()
        self.loc = _Conv(c, c, 3, 2, g=c)
        self.sur_conv = nn.Conv2d(c, c, 3, stride=2, padding=2, dilation=2, groups=c, bias=False)
        self.sur_bn = nn.BatchNorm2d(c, eps=1e-3, momentum=0.03)
        self.pw = _Conv(2 * c, c, 1, 1, act=False)
        _zero_bn(self.pw)
        cg = max(16, c // 4)
        self.glo = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, cg, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(cg, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.avg_pool2d(x, 2, 2)
        loc = self.loc(x)
        sur = F.silu(self.sur_bn(self.sur_conv(x)))
        joint = self.pw(torch.cat([loc, sur], 1))
        return base + joint * self.glo(base)


class BridgedConv(_Conv):
    """Pretrained Ultralytics Conv preceded by a Context-Guided Downsample Bridge (`cg_bridge`).
    Parameter names of Conv (`conv`, `bn`) stay identical so `inherit()` copies COCO weights directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.cg_bridge(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward_fuse(self.cg_bridge(x))


class CGConcat(nn.Module):
    """Concatenate top-down 80x80 feature with Context-Guided downsampled L4 (160x160 -> 80x80)."""

    def __init__(self, c_l4: int):
        super().__init__()
        self.bridge = CGDownBridge(c_l4)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        top, l4 = xs
        b4 = self.bridge(l4)
        if b4.shape[2:] != top.shape[2:]:
            b4 = F.interpolate(b4, size=top.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([top, b4], 1)


class DySampleUp(nn.Module):
    """DySample point-sampling upsampler (arXiv:2308.15085), safe across FP16/FP32."""

    def __init__(self, c: int, s: int = 2):
        super().__init__()
        self.s = s
        self.off = nn.Conv2d(c, 2 * s * s, 1)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        s = self.s
        off = F.pixel_shuffle(self.off(x), s).float() * 0.25
        ys = (torch.arange(s * h, device=x.device, dtype=torch.float32) + 0.5) / s - 0.5
        xs = (torch.arange(s * w, device=x.device, dtype=torch.float32) + 0.5) / s - 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        px = (gx[None] + off[:, 0] + 0.5) * (2.0 / w) - 1.0
        py = (gy[None] + off[:, 1] + 0.5) * (2.0 / h) - 1.0
        grid = torch.stack([px, py], -1).to(x.dtype)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class AsymP2Fuse(nn.Module):
    """Fuse DySample-upsampled P3 (160^2), pooled L2 (320^2 -> 160^2), and high-res L4 (160^2)."""

    def __init__(self, c_l2: int, c_l4: int):
        super().__init__()
        self.l4_proj = _Conv(c_l4, c_l2, 1, 1, act=False)
        _zero_bn(self.l4_proj)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        up_p3, l2, l4 = xs
        sz = l4.shape[2:]
        if up_p3.shape[2:] != sz:
            up_p3 = F.interpolate(up_p3, size=sz, mode="bilinear", align_corners=False)
        p_l2 = F.avg_pool2d(l2, 2, 2)
        if p_l2.shape[2:] != sz:
            p_l2 = F.interpolate(p_l2, size=sz, mode="bilinear", align_corners=False)
        return torch.cat([up_p3, p_l2 + self.l4_proj(l4)], 1)


class CSPAttnBlock(nn.Module):
    """Cross-Stage Partial Attention block: half channels bypass, half go through DW 7x7 + channel-spatial attention."""

    def __init__(self, c: int):
        super().__init__()
        ch = c // 2
        self.dw7 = _Conv(ch, ch, 7, 1, g=ch)
        self.pw = _Conv(ch, ch, 1, 1, act=False)
        _zero_bn(self.pw)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(8, ch // 2), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, ch // 2), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, 1)
        y = self.dw7(x2)
        x2 = x2 + self.pw(y * self.ca(y))
        return torch.cat([x1, x2], 1)


class C3k2CSPAttn(_C3k2):
    """C3k2 followed by CSPAttnBlock."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csp_attn(super().forward(x))


class DownP2ToP3(nn.Module):
    """Downsample 160x160 P2 to 80x80 P3 and add as zero-init residual into P3 inside Detect."""

    def __init__(self, c0: int, c1: int):
        super().__init__()
        self.net = nn.Sequential(_Conv(c0, c1, 3, 2), _Conv(c1, c1, 1, 1, act=False))
        _zero_bn(self.net[-1])

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        d = self.net(p2)
        if d.shape[2:] != p3.shape[2:]:
            d = F.interpolate(d, size=p3.shape[2:], mode="bilinear", align_corners=False)
        return p3 + d


class AsymDetect(_Detect):
    """Detect where level 0 (P2) is at stride 4 (160x160) and levels 1..3 are at native strides 8, 16, 32."""

    def forward_head(self, x: list[torch.Tensor], box_head=None, cls_head=None):
        if hasattr(self, "p2_bu"):
            x = [x[0], self.p2_bu(x[0], x[1]), *x[2:]]
        return super().forward_head(x, box_head, cls_head)


# ==============================================================================
# fw3: DenseQuery — Gaussian Query Re-Ranking + Interaction Decoder + NWD
# ==============================================================================

def bbox_nwd(b1: torch.Tensor, b2: torch.Tensor, c: float = 12.0, eps: float = 1e-6) -> torch.Tensor:
    """Normalized Gaussian Wasserstein Distance between xyxy boxes in pixel coordinates."""
    b1, b2 = b1.float(), b2.float()
    cx1, cy1 = (b1[..., 0] + b1[..., 2]) * 0.5, (b1[..., 1] + b1[..., 3]) * 0.5
    w1, h1 = (b1[..., 2] - b1[..., 0]).clamp(min=eps), (b1[..., 3] - b1[..., 1]).clamp(min=eps)
    cx2, cy2 = (b2[..., 0] + b2[..., 2]) * 0.5, (b2[..., 1] + b2[..., 3]) * 0.5
    w2, h2 = (b2[..., 2] - b2[..., 0]).clamp(min=eps), (b2[..., 3] - b2[..., 1]).clamp(min=eps)
    w2_sq = (cx1 - cx2).square() + (cy1 - cy2).square() + 0.25 * ((w1 - w2).square() + (h1 - h2).square())
    return torch.exp(-torch.sqrt(w2_sq + eps) / c)


class QueryInteractionLayer(nn.Module):
    """One layer of the Efficient Interaction Decoder:
    1) 4-point Deformable Cross-Scale Sampling on [P2, P3, P4] around query centers;
    2) Multi-head Self-Attention across the K queries;
    3) FFN."""

    def __init__(self, d: int, nhead: int = 4, n_levels: int = 3, n_points: int = 4):
        super().__init__()
        self.d = d
        self.nhead = nhead
        self.n_levels = n_levels
        self.n_points = n_points
        self.offsets = nn.Linear(d, n_levels * n_points * 2)
        self.weights = nn.Linear(d, n_levels * n_points)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        self.cross_proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)

        self.qkv = nn.Linear(d, 3 * d)
        self.attn_proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(inplace=True),
            nn.Linear(2 * d, d),
        )
        self.norm3 = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor, pos: torch.Tensor, feats: list[torch.Tensor], ref_xy: torch.Tensor) -> torch.Tensor:
        # q, pos: (B, K, D); ref_xy: (B, K, 2) in [-1, 1]; feats: list of (B, D, H_l, W_l)
        b, k, d = q.shape
        qp = q + pos
        # 1. Cross-scale deformable sampling
        off = self.offsets(qp).tanh().view(b, k, self.n_levels, self.n_points, 2)  # [-1, 1] scaled below
        w = self.weights(qp).view(b, k, self.n_levels * self.n_points).softmax(-1)
        sampled = []
        for l_idx, f in enumerate(feats):
            _, _, hl, wl = f.shape
            scale = torch.tensor([4.0 / wl, 4.0 / hl], device=q.device, dtype=q.dtype).view(1, 1, 1, 2)
            grid = (ref_xy.unsqueeze(2) + off[:, :, l_idx] * scale).clamp(-1.0, 1.0).to(f.dtype)  # (B, K, P, 2)
            s = F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False)  # (B, D, K, P)
            sampled.append(s.permute(0, 2, 3, 1))  # (B, K, P, D)
        all_s = torch.cat(sampled, dim=2).to(q.dtype)  # (B, K, L*P, D)
        agg = (all_s * w.unsqueeze(-1)).sum(dim=2)  # (B, K, D)
        q = self.norm1(q + self.cross_proj(agg))

        # 2. Multi-head self-attention across K queries
        qp = q + pos
        qkv = self.qkv(qp).view(b, k, 3, self.nhead, d // self.nhead).permute(2, 0, 3, 1, 4)
        qi, ki, vi = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(qi, ki, vi)
        attn = attn.transpose(1, 2).reshape(b, k, d)
        q = self.norm2(q + self.attn_proj(attn))

        # 3. FFN
        q = self.norm3(q + self.ffn(q))
        return q


class DenseQueryDetect(_Detect):
    """Detect head with Gaussian Center Heatmap Query Selection, 2-layer Efficient Interaction Decoder,
    and zero-init gated write-back into P2 & P3 dense predictions."""

    def setup_query_head(self, num_queries: int = 300, n_layers: int = 2):
        ch = [s[0].conv.in_channels for s in self.cv2]
        c0, c1, c2 = ch[0], ch[1], ch[2]
        d = max(64, 2 * c0)
        self.num_queries = num_queries
        self.d_model = d
        self.bu = nn.Sequential(_Conv(c0, c1, 3, 2), _Conv(c1, c1, 1, 1, act=False))
        _zero_bn(self.bu[1])

        ch_hm = max(16, c0 // 2)
        self.hm_head = nn.Sequential(
            _DWConv(c0, c0, 3),
            _Conv(c0, ch_hm, 1, 1),
            nn.Conv2d(ch_hm, 1, 1),
        )
        nn.init.constant_(self.hm_head[-1].bias, -4.60)

        self.in_projs = nn.ModuleList([_Conv(c0, d, 1, 1), _Conv(c1, d, 1, 1), _Conv(c2, d, 1, 1)])
        self.pos_mlp = nn.Sequential(nn.Linear(6, d), nn.SiLU(inplace=True), nn.Linear(d, d))
        self.decoder = nn.ModuleList([QueryInteractionLayer(d, nhead=4, n_levels=3, n_points=4) for _ in range(n_layers)])

        nb = 4 * self.reg_max
        self.cls_delta_p2 = nn.Linear(d, self.nc)
        self.box_delta_p2 = nn.Linear(d, nb)
        self.cls_delta_p3 = nn.Linear(d, self.nc)
        self.box_delta_p3 = nn.Linear(d, nb)
        for lin in (self.cls_delta_p2, self.box_delta_p2, self.cls_delta_p3, self.box_delta_p3):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.gate_cls = nn.Parameter(torch.full((2,), 0.1))
        self.gate_box = nn.Parameter(torch.full((2,), 0.1))

    def forward_head(self, x: list[torch.Tensor], box_head=None, cls_head=None) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return {}
        if hasattr(self, "bu"):
            x = [x[0], x[1] + self.bu(x[0]), *x[2:]]
        bs = x[0].shape[0]
        boxes_list = [box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        scores_list = [cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)]

        # Gaussian center heatmap on P2
        p2 = x[0]
        _, _, h2, w2 = p2.shape
        hm_logit = self.hm_head(p2)  # (B, 1, H2, W2)
        hm_prob = torch.sigmoid(hm_logit)
        peak = (hm_prob == F.max_pool2d(hm_prob, 3, 1, 1)).to(hm_prob.dtype)
        sal = (hm_prob * (1.0 + 0.5 * peak)).view(bs, -1)  # (B, H2*W2)

        k = min(self.num_queries, h2 * w2)
        topk_val, idx_p2 = torch.topk(sal, k, dim=-1)  # (B, K)
        qy = (idx_p2 // w2).to(p2.dtype)
        qx = (idx_p2 % w2).to(p2.dtype)
        ref_x = (qx + 0.5) * (2.0 / w2) - 1.0
        ref_y = (qy + 0.5) * (2.0 / h2) - 1.0
        ref_xy = torch.stack([ref_x, ref_y], dim=-1)  # (B, K, 2) in [-1, 1]

        # Multi-scale projections
        feats = [proj(x[i]) for i, proj in enumerate(self.in_projs)]  # P2, P3, P4
        f0_flat = feats[0].view(bs, self.d_model, -1)
        q = torch.gather(f0_flat, 2, idx_p2.unsqueeze(1).expand(-1, self.d_model, -1)).permute(0, 2, 1)  # (B, K, D)

        # Pos encoding from (ref_x, ref_y) + detached dense box (ltrb) at idx_p2
        b0 = torch.gather(boxes_list[0].detach(), 2, idx_p2.unsqueeze(1).expand(-1, 4 * self.reg_max, -1)).permute(0, 2, 1)
        pos_in = torch.cat([ref_xy, b0[..., :4].clamp(-10.0, 10.0) * 0.1], dim=-1)
        pos = self.pos_mlp(pos_in)

        for layer in self.decoder:
            q = layer(q, pos, feats, ref_xy)

        # Write-back into P2 (level 0) and parent cell on P3 (level 1)
        dt = scores_list[0].dtype
        dc2 = (self.gate_cls[0] * self.cls_delta_p2(q)).permute(0, 2, 1).to(dt)  # (B, nc, K)
        db2 = (self.gate_box[0] * self.box_delta_p2(q)).permute(0, 2, 1).to(boxes_list[0].dtype)
        scores_list[0] = scores_list[0].scatter_add(2, idx_p2.unsqueeze(1).expand(-1, self.nc, -1), dc2)
        boxes_list[0] = boxes_list[0].scatter_add(2, idx_p2.unsqueeze(1).expand(-1, 4 * self.reg_max, -1), db2)

        _, _, h3, w3 = x[1].shape
        idx_p3 = ((idx_p2 // w2) // 2).clamp(0, h3 - 1) * w3 + ((idx_p2 % w2) // 2).clamp(0, w3 - 1)
        dc3 = (self.gate_cls[1] * self.cls_delta_p3(q)).permute(0, 2, 1).to(dt)
        db3 = (self.gate_box[1] * self.box_delta_p3(q)).permute(0, 2, 1).to(boxes_list[1].dtype)
        scores_list[1] = scores_list[1].scatter_add(2, idx_p3.unsqueeze(1).expand(-1, self.nc, -1), dc3)
        boxes_list[1] = boxes_list[1].scatter_add(2, idx_p3.unsqueeze(1).expand(-1, 4 * self.reg_max, -1), db3)

        return {
            "boxes": torch.cat(boxes_list, dim=-1),
            "scores": torch.cat(scores_list, dim=-1),
            "feats": x,
            "hm_logit": hm_logit,
        }


class NWDTALAssigner(_TALAssigner):
    """TaskAlignedAssigner whose alignment metric blends CIoU and Gaussian Wasserstein similarity (NWD)."""

    def __init__(self, *args, nwd_c: float = 12.0, nwd_w: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.nwd_c = nwd_c
        self.nwd_w = nwd_w

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        ciou = super().iou_calculation(gt_bboxes, pd_bboxes)
        nwd = bbox_nwd(gt_bboxes, pd_bboxes, c=self.nwd_c).to(ciou.dtype)
        return (1.0 - self.nwd_w) * ciou + self.nwd_w * nwd


class NWDBboxLoss(_BboxLoss):
    """BboxLoss adding 0.5 * (1 - NWD) to the CIoU box loss for tiny-object gradients."""

    def __init__(self, reg_max: int = 1, nwd_c: float = 12.0, nwd_w: float = 0.5):
        super().__init__(reg_max)
        self.nwd_c = nwd_c
        self.nwd_w = nwd_w

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride):
        loss_iou, loss_dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride
        )
        idx = fg_mask.nonzero(as_tuple=True)
        if idx[0].numel() > 0:
            weight = target_scores[idx].sum(-1, keepdim=True)
            st = stride[idx[1]]
            nwd = bbox_nwd(pred_bboxes[idx] * st, target_bboxes[idx] * st, c=self.nwd_c).unsqueeze(-1).to(loss_iou.dtype)
            loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = loss_iou + self.nwd_w * loss_nwd
        return loss_iou, loss_dfl


class NWDQueryDetLoss(_V8Loss):
    """v8DetectionLoss with NWDTALAssigner, NWDBboxLoss, and CenterNet Gaussian focal loss on `hm_logit`."""

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict, **assign_kwargs):
        info, loss, items = super().get_assigned_targets_and_loss(preds, batch, **assign_kwargs)
        if "hm_logit" in preds:
            hm = preds["hm_logit"].float()  # (B, 1, H2, W2)
            bs, _, h2, w2 = hm.shape
            t_hm = torch.zeros(bs * h2 * w2, device=hm.device, dtype=torch.float32)
            b_idx = batch["batch_idx"].view(-1).long().to(hm.device)
            if b_idx.numel() > 0:
                boxes = batch["bboxes"].to(hm.device).float()
                cx = boxes[:, 0] * w2
                cy = boxes[:, 1] * h2
                bw = boxes[:, 2] * w2
                bh = boxes[:, 3] * h2
                sigma = (torch.sqrt((bw * bh).clamp(min=1.0)) / 6.0).clamp(min=1.0, max=3.0).unsqueeze(1)
                offs = torch.arange(-3, 4, device=hm.device, dtype=torch.long)
                dy, dx = torch.meshgrid(offs, offs, indexing="ij")
                dx, dy = dx.reshape(1, -1), dy.reshape(1, -1)
                ix = (cx.long().unsqueeze(1) + dx).clamp(0, w2 - 1)
                iy = (cy.long().unsqueeze(1) + dy).clamp(0, h2 - 1)
                val = torch.exp(-((ix.float() + 0.5 - cx.unsqueeze(1)).square() + (iy.float() + 0.5 - cy.unsqueeze(1)).square()) / (2.0 * sigma.square()))
                flat = (b_idx.unsqueeze(1).clamp(0, bs - 1)) * (h2 * w2) + iy * w2 + ix
                t_hm.scatter_reduce_(0, flat.reshape(-1), val.reshape(-1), reduce="amax", include_self=True)
            t_hm = t_hm.view(bs, 1, h2, w2)
            p = torch.sigmoid(hm).clamp(1e-4, 1.0 - 1e-4)
            pos = t_hm.ge(0.99)
            neg_w = (1.0 - t_hm).pow(4)
            n_pos = pos.float().sum().clamp(min=1.0)
            l_hm = -(pos.float() * (1.0 - p).pow(2) * torch.log(p) + (~pos).float() * neg_w * p.pow(2) * torch.log(1.0 - p)).sum() / n_pos
            loss[1] = loss[1] + 0.5 * l_hm.to(loss.dtype)
        return info, loss, items


class NWDQueryLossModel(_DetectionModel):
    """DetectionModel using NWDQueryDetLoss."""

    def init_criterion(self):
        c = super().init_criterion()
        nwd_c = getattr(self, "nwd_c", 12.0)
        nwd_w = getattr(self, "nwd_w", 0.5)
        c.assigner = NWDTALAssigner(
            topk=10, num_classes=c.nc, alpha=0.5, beta=6.0, stride=c.stride.tolist(), nwd_c=nwd_c, nwd_w=nwd_w
        )
        c.bbox_loss = NWDBboxLoss(c.reg_max, nwd_c=nwd_c, nwd_w=nwd_w).to(c.device)
        c.__class__ = NWDQueryDetLoss
        return c


# ==============================================================================
# Surgery dispatcher for fw1, fw2, fw3
# ==============================================================================

def _out_ch(m: nn.Module) -> int:
    if hasattr(m, "cv2") and hasattr(m.cv2, "conv"):
        return m.cv2.conv.out_channels
    if hasattr(m, "conv"):
        return m.conv.out_channels
    raise ValueError(f"Cannot infer out_channels of {type(m)}")


def apply_coco_head_c3(det: nn.Module) -> None:
    """Set Detect cv3 hidden widths to [ch[0], ch[1], ch[1], ...] so levels 1..3 match stock YOLO26 cv3 shapes."""
    ch = [s[0].conv.in_channels for s in det.cv2]
    if len(ch) < 2:
        return
    c3s = [ch[0]] + [ch[1]] * (len(ch) - 1)
    det.cv3 = nn.ModuleList(
        nn.Sequential(
            nn.Sequential(_DWConv(x, x, 3), _Conv(x, c3, 1, 1)),
            nn.Sequential(_DWConv(c3, c3, 3), _Conv(c3, c3, 1, 1)),
            nn.Conv2d(c3, det.nc, 1),
        )
        for x, c3 in zip(ch, c3s)
    )


def fw640_surgery(model: nn.Module, arch: dict) -> None:
    fw = arch.get("fw640")
    if not fw:
        return
    seq = model.model
    det = seq[-1]
    kind = fw["type"]

    if kind == "fw1_liftgd":
        # Replace placeholder at `at` with FW1DualHubNeck and next 4 layers with Index(0..3)
        at = fw["at"]
        src_pan = fw["src_pan"]  # [25, 16, 19, 22]
        src_bk = fw["src_bk"]    # [2, 4, 6, 10]
        c_pan = [_out_ch(seq[i]) for i in src_pan]
        c_bk = [_out_ch(seq[i]) for i in src_bk]
        neck = FW1DualHubNeck(c_pan, c_bk, nhub=fw.get("nhub", 2))
        src_all = list(src_pan) + list(src_bk)
        neck.i, neck.f, neck.type = at, src_all, "bench.fw640.FW1DualHubNeck"
        neck.np = sum(p.numel() for p in neck.parameters())
        seq[at] = neck
        for k in range(4):
            old = seq[at + 1 + k]
            idx = _Index(k)
            idx.i, idx.f, idx.type, idx.np = old.i, at, "ultralytics.nn.modules.conv.Index", 0
            seq[at + 1 + k] = idx
        model.save = sorted(set(model.save) | set(src_all) | {at})
        if fw.get("coco_head_c3", True):
            apply_coco_head_c3(det)

    elif kind == "fw2_asymres":
        # 1) L1 gets LiftConv x2.0 (handled via arch['lift'] or here)
        # 2) L5 gets BridgedConv (CGDownBridge 160->80 before L5 Conv s2)
        m5 = seq[5]
        c4 = _out_ch(seq[4])
        m5.__class__ = BridgedConv
        m5.cg_bridge = CGDownBridge(c4)
        m5.np = sum(p.numel() for p in m5.parameters())
        # 3) L15 Concat([-1, 4]) becomes CGConcat(c4)
        old15 = seq[15]
        cg_cat = CGConcat(c4)
        cg_cat.i, cg_cat.f, cg_cat.type = old15.i, old15.f, "bench.fw640.CGConcat"
        cg_cat.np = sum(p.numel() for p in cg_cat.parameters())
        seq[15] = cg_cat
        # 4) L23 Upsample becomes DySampleUp(c_p3, 2)
        old23 = seq[23]
        c_p3 = _out_ch(seq[16])
        dys = DySampleUp(c_p3, 2)
        dys.i, dys.f, dys.type = old23.i, old23.f, "bench.fw640.DySampleUp"
        dys.np = sum(p.numel() for p in dys.parameters())
        seq[23] = dys
        # 5) L24 Concat([-1, 2]) becomes AsymP2Fuse taking [23, 2, 4]
        old24 = seq[24]
        c_l2 = _out_ch(seq[2])
        fuse24 = AsymP2Fuse(c_l2, c4)
        fuse24.i, fuse24.f, fuse24.type = old24.i, [23, 2, 4], "bench.fw640.AsymP2Fuse"
        fuse24.np = sum(p.numel() for p in fuse24.parameters())
        seq[24] = fuse24
        model.save = sorted(set(model.save) | {2, 4, 23})
        # 6) L25 C3k2 gets CSPAttnBlock
        m25 = seq[25]
        c_p2 = _out_ch(m25)
        m25.__class__ = C3k2CSPAttn
        m25.csp_attn = CSPAttnBlock(c_p2)
        m25.np = sum(p.numel() for p in m25.parameters())
        # 7) Detect: COCO head_c3 + AsymDetect with DownP2ToP3 and true strides [4, 8, 16, 32]
        if fw.get("coco_head_c3", True):
            apply_coco_head_c3(det)
        det.__class__ = AsymDetect
        det.p2_bu = DownP2ToP3(c_p2, c_p3)
        det.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])

    elif kind == "fw3_densequery":
        if fw.get("coco_head_c3", True):
            apply_coco_head_c3(det)
        det.__class__ = DenseQueryDetect
        det.setup_query_head(num_queries=fw.get("num_queries", 300), n_layers=fw.get("n_layers", 2))
        det.np = sum(p.numel() for p in det.parameters())

    else:
        raise ValueError(f"Unknown fw640 type: {kind}")

    for mm in model.modules():
        if isinstance(mm, nn.BatchNorm2d):
            mm.eps, mm.momentum = 1e-3, 0.03
