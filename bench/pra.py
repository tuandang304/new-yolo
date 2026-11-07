"""Proposed-model builders (YOLO26n-derived), with explicit COCO weight inheritance.

An architecture is configured in configs/node.yaml as `train.arch`:
  yaml:          repo-relative model YAML (models/*.yaml)
  pooled_first:  index of the first trunk conv that gets a 2x average-pool in front (RDD: trunk runs at 1/2 res)
  head_c3:       width of the Detect classification branch, an int or one per level (default: Ultralytics rule)
  head_k:        kernel of the DWConvs in the classification branch, one per level (default 3)
  spd_layers:    shallow 3x3 stride-2 convs replaced by space-to-depth + 1x1 conv (lose COCO weights)
  deep_plus:     [{layer, n}] C3k2 layers given an extra zero-init residual C3k2 (keeps COCO weights; adds deep capacity)
  adown_layers:  stride-2 convs replaced by Ultralytics ADown (avg-pool + split 3x3-s2 / maxpool-1x1; lose COCO weights)
  stride_mult:   letterbox/grid multiple for the whole model (64 when the trunk is pooled)
  detail_inject: {src, dst} adds zero-init residual space-to-depth(src) to the output of C3k2 layer dst (keeps COCO tensors)
  inherit:       {offset: k, max_src: L, detect_levels: [src level for each new Detect level, -1 = none]}
                 maps yolo26n.pt tensors model.{i} -> model.{i+k} for i <= L, and Detect level l' <- level src.
                 `partial: true` also copies the overlapping slice of backbone/neck tensors whose channels were widened.
  lift:          {layer, scale} bilinearly upsamples the input of trunk conv `layer` by `scale` (keeps COCO tensors);
                 the trunk then runs on a finer grid, Detect strides are divided by `scale` (stride_mult must stay integral)
  detect_bu:     true adds a zero-init bottom-up residual P2->P3 inside Detect (level 0 -> level 1; no-op at init)
  pair_loss:     {pairs: [[a, b], ...], mode: relax | sharpen, value} changes the class BCE for confusable pairs (training only)
  detail_inject also takes ch (default 16) and dw (true: 3x3 depthwise after the space-to-depth).
  Round 11 modules; every one is an exact identity at step 0 (last BN gain = 0):
  win_attn:      {layer, dim, heads, win} window self-attention residual after C3k2 `layer` (Swin-style, one block)
  ctx_xattn:     {layer, src, dim, heads} cross-attention residual after C3k2 `layer`: queries from it, keys/values from `src`
  head_ctx:      {src, dim} stride-64 context skip into each Detect classification branch (src is an extra Detect input)
  head_fb:       true adds P4 -> P3 feedback inside Detect: P3 += sigmoid(a(P3)) * up(f(P4))
  head_b2c:      true adds a box -> class edge per level: cls input += g(first box-branch conv output)
  Round 12 modules (r12_neck, r12_head, density_loss): see the block above `surgery`.
Every inherited tensor must match in shape unless `partial`; the count of transferred tensors is printed.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

try:
    from ultralytics.nn.modules.conv import Conv as _Conv
except Exception:  # pragma: no cover - ultralytics always present in runs
    _Conv = nn.Module
try:
    from ultralytics.nn.tasks import DetectionModel as _DetectionModel
except Exception:  # pragma: no cover - ultralytics always present in runs
    _DetectionModel = nn.Module


class PooledConv(_Conv):
    """Ultralytics Conv with a 2x average-pool in front; parameter names identical to Conv (COCO transfer)."""

    def forward(self, x):
        return super().forward(F.avg_pool2d(x, 2))

    def forward_fuse(self, x):
        return super().forward_fuse(F.avg_pool2d(x, 2))


class LiftConv(_Conv):
    """Ultralytics Conv whose input is bilinearly upsampled by self.lift first; parameter names identical to Conv."""

    def forward(self, x):
        return super().forward(F.interpolate(x, scale_factor=self.lift, mode="bilinear", align_corners=False))

    def forward_fuse(self, x):
        return super().forward_fuse(F.interpolate(x, scale_factor=self.lift, mode="bilinear", align_corners=False))


try:
    from ultralytics.nn.modules.head import Detect as _Detect
except Exception:  # pragma: no cover
    _Detect = nn.Module


class DetectBU(_Detect):
    """Detect with a zero-init bottom-up residual: level1 += bu(level0) (stride-2 conv, BN gain 0 -> no-op at init)."""

    def forward(self, x):
        x = list(x)
        x[1] = x[1] + self.bu(x[0])
        return super().forward(x)


def _p3_split(det, x: list) -> list:
    """Stack P twin P3 (round 14): the P3 input is cat(plain, detail); the head reads only the detail half."""
    k = getattr(det, "p3_split", 0)
    if k:
        x[0] = x[0][:, k:]
    return x


class R11Detect(_Detect):
    """Detect with optional round-11 additions, each a zero contribution at step 0:
    ctx  - stride-64 context (extra last input) projected once, upsampled and added to every classification input;
    fb   - P4 -> P3 feedback: P3 += sigmoid(fba(P3)) * up(fb(P4));
    b2c  - box -> class edge: classification input += b2c(first box-branch conv output)."""

    def forward(self, x):
        x = _p3_split(self, list(x))
        self._ctxf = self.ctx(x.pop()) if getattr(self, "ctx", None) is not None else None
        if getattr(self, "fb", None) is not None:
            f = F.interpolate(self.fb(x[1]), size=x[0].shape[2:], mode="nearest")
            x[0] = x[0] + torch.sigmoid(self.fba(x[0])) * f
        try:
            return super().forward(x)
        finally:
            self._ctxf = None

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        b2c, ctxf = getattr(self, "b2c", None), getattr(self, "_ctxf", None)
        boxes, scores = [], []
        for i in range(self.nl):
            xi, ci = x[i], x[i]
            if b2c is not None:
                h = box_head[i][0](xi)
                b = box_head[i][2](box_head[i][1](h))
                ci = ci + b2c[i](h)
            else:
                b = box_head[i](xi)
            if ctxf is not None:
                ci = ci + self.ctxo[i](F.interpolate(ctxf, size=xi.shape[2:], mode="nearest"))
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            scores.append(cls_head[i](ci).view(bs, self.nc, -1))
        return {"boxes": torch.cat(boxes, dim=-1), "scores": torch.cat(scores, dim=-1), "feats": x}


class SPDConv(nn.Module):
    """Space-to-depth (PixelUnshuffle x2) + Conv-BN-SiLU 1x1: stride-2 without discarding pixels."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = _Conv(4 * c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.unshuffle(x))


try:
    from ultralytics.nn.modules.block import C3k2 as _C3k2
except Exception:  # pragma: no cover
    _C3k2 = nn.Module


class C3k2Plus(_C3k2):
    """C3k2 with an extra zero-initialised residual block: y = C3k2(x) + gamma * extra(C3k2(x)), gamma = 0.

    Keeps every COCO tensor of the original block (same parameter names) and its function at init,
    while adding capacity (params) cheaply at deep, low-resolution stages."""

    def forward(self, x):
        y = super().forward(x)
        return y + self.zgamma * self.zx(y)


class C3k2Detail(_C3k2):
    """C3k2 taking [x, detail]: y = C3k2(x) + dz(detail), dz = 1x1 + space-to-depth + 1x1 Conv-BN with BN gain 0 (no-op at init)."""

    def forward(self, inputs):
        x, d = inputs
        return super().forward(x) + self.dz(d)


def _zero_bn(conv) -> None:
    nn.init.zeros_(conv.bn.weight)


class WinAttn(nn.Module):
    """One Swin-style window self-attention block as a zero-init residual: x + proj(attn(qkv(x)) + pe(v))."""

    def __init__(self, c: int, dim: int = 32, heads: int = 2, win: int = 8):
        super().__init__()
        self.dim, self.heads, self.win = dim, heads, win
        self.qkv = _Conv(c, 3 * dim, 1, act=False)
        self.pe = _Conv(dim, dim, 3, g=dim, act=False)
        self.proj = _Conv(dim, c, 1, act=False)
        _zero_bn(self.proj)

    def forward(self, x):
        B, _, H, W = x.shape
        w, h, d = self.win, self.heads, self.dim
        dh = d // h
        qkv = self.qkv(x)
        ph, pw = (-H) % w, (-W) % w
        if ph or pw:
            qkv = F.pad(qkv, (0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        q, k, v = qkv.split(d, 1)

        def part(t):  # B, d, Hp, Wp -> B*nW, h, w*w, dh
            t = t.reshape(B, h, dh, Hp // w, w, Wp // w, w).permute(0, 3, 5, 1, 4, 6, 2)
            return t.reshape(-1, h, w * w, dh)

        o = F.scaled_dot_product_attention(part(q), part(k), part(v))
        o = o.reshape(B, Hp // w, Wp // w, h, w, w, dh).permute(0, 3, 6, 1, 4, 2, 5).reshape(B, d, Hp, Wp)
        o = o[:, :, :H, :W] + self.pe(v[:, :, :H, :W])
        return x + self.proj(o)


class C3k2Win(_C3k2):
    """C3k2 followed by a zero-init window-attention residual."""

    def forward(self, x):
        return self.wa(super().forward(x))


class XAttn(nn.Module):
    """Cross-attention residual term: queries from y (every location), keys/values from a coarse context map c."""

    def __init__(self, cq: int, ck: int, dim: int = 32, heads: int = 2):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.q = _Conv(cq, dim, 1, act=False)
        self.kv = _Conv(ck, 2 * dim, 1, act=False)
        self.proj = _Conv(dim, cq, 1, act=False)
        _zero_bn(self.proj)

    def forward(self, y, c):
        B, _, H, W = y.shape
        h, d = self.heads, self.dim
        q = self.q(y).reshape(B, h, d // h, H * W).transpose(-1, -2)
        k, v = (t.reshape(B, h, d // h, -1).transpose(-1, -2) for t in self.kv(c).split(d, 1))
        o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, d, H, W)
        return self.proj(o)


class C3k2XAttn(_C3k2):
    """C3k2 taking [x, context]: y = C3k2(x) + xa(C3k2(x), context), zero at init."""

    def forward(self, inputs):
        x, c = inputs
        y = super().forward(x)
        return y + self.xa(y, c)


class ZeroResBottleneck(nn.Module):
    """x + gamma * Bottleneck(x) with gamma = 0: an identity at init (used to replace C2PSA cheaply)."""

    def __init__(self, c: int, e: float = 0.25):
        super().__init__()
        from ultralytics.nn.modules.block import Bottleneck
        self.b = Bottleneck(c, c, shortcut=False, e=e)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.b(x)


class LowRankConv(nn.Module):
    """3x3 stride-2 conv factorised as 3x3-s2 (c1->r, no BN) then Conv-BN-SiLU 1x1 (r->c2); init from SVD."""

    def __init__(self, c1: int, c2: int, r: int):
        super().__init__()
        self.a = nn.Conv2d(c1, r, 3, 2, 1, bias=False)
        self.b = _Conv(r, c2, 1, 1)

    def forward(self, x):
        return self.b(self.a(x))

    @torch.no_grad()
    def init_from(self, conv) -> float:
        """Copy the best rank-r approximation of conv.conv.weight; returns retained energy fraction."""
        W = conv.conv.weight.detach().float()
        c2, c1 = W.shape[:2]
        U, S, Vh = torch.linalg.svd(W.reshape(c2, -1), full_matrices=False)
        r = min(self.a.out_channels, S.numel())
        sq = S[:r].sqrt()
        a = (sq[:, None] * Vh[:r]).reshape(r, c1, 3, 3)
        b = (U[:, :r] * sq[None]).reshape(c2, r, 1, 1)
        if self.a.weight.shape == a.shape and self.b.conv.weight.shape == b.shape:
            self.a.weight.copy_(a)
            self.b.conv.weight.copy_(b)
            self.b.bn.load_state_dict(conv.bn.state_dict())
        else:  # widened layer: copy the COCO factors into the overlapping slice, keep the fresh init elsewhere
            self.a.weight[:r, :c1].copy_(a[:, :self.a.weight.shape[1]])
            self.b.conv.weight[:c2, :r].copy_(b[:self.b.conv.weight.shape[0]])
            for k, v in conv.bn.state_dict().items():
                t = getattr(self.b.bn, k)
                if t.ndim:
                    t[:v.shape[0]].copy_(v[:t.shape[0]])
        return float((S[:r] ** 2).sum() / (S ** 2).sum())


# ----------------------------------------------------------------------------------------------------------------------
# Round 12 modules. Every addition is an exact identity at step 0 (zero BN gain or zero scale) and keeps every COCO tensor.
#   r12_neck: {carafe: [12, 15], hpconcat: [13, 16], updw: 12, upmix: 12, dysample: 15, gated_concat: 16, conv_dw: 18}
#   r12_head: {br, edge_img, density, scr, route, mona: {pos, split, ctx, edge}, freq, edge_l3, edge_m3, l11gate, pconv_mf, dquery, route_ctx}
#   density_loss: weight of the normalised auxiliary density MSE (needs r12_head.density)
#   r12_head.scene: {dim, film, bias, aux} scene vector from GAP(L11) (Stack N); scene_loss: weight of its aux count MSE
# ----------------------------------------------------------------------------------------------------------------------
_TAP = {}


class TapConv(_Conv):
    """L0 Conv that keeps its input image for the head (image edge prior); parameter names identical to Conv."""

    def forward(self, x):
        _TAP["img"] = x
        return super().forward(x)

    def forward_fuse(self, x):
        _TAP["img"] = x
        return super().forward_fuse(x)


def _ckpt(fn, *args):
    if torch.is_grad_enabled() and any(a.requires_grad for a in args if torch.is_tensor(a)):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _local_mix(x, w):
    """Per-pixel 3x3 weighted sum: x (B,C,H,W), w (B,9,H,W) -> (B,C,H,W). Loops over the 9 taps (no 9x unfold tensor)."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    H, W = x.shape[2:]
    out = 0
    for k in range(9):
        dy, dx = divmod(k, 3)
        out = out + xp[:, :, dy:dy + H, dx:dx + W] * w[:, k:k + 1]
    return out


class CarafeUp(nn.Module):
    """FreqFusion-lite adaptive low-pass upsampler (arXiv:2408.12879, ALPF part):
    y = nearest(x) + g * (content_aware_reassembly(x) - nearest(x)), per-channel g = 0 at init."""

    def __init__(self, c: int, s: int, cm: int = 32):
        super().__init__()
        self.s = s
        self.comp = _Conv(c, cm, 1, 1)
        self.gen = nn.Conv2d(cm, s * s * 9, 3, 1, 1)
        self.g = nn.Parameter(torch.zeros(1, c, 1, 1))

    def _body(self, x):
        s = self.s
        w = F.pixel_shuffle(self.gen(self.comp(x)), s).softmax(1)  # B, 9, sH, sW
        up = F.interpolate(x, scale_factor=s, mode="nearest")
        # reassembly on the coarse grid = 3x3 mix of nearest-upsampled neighbours at stride s
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        H, W = x.shape[2:]
        out = 0
        for k in range(9):
            dy, dx = divmod(k, 3)
            out = out + F.interpolate(xp[:, :, dy:dy + H, dx:dx + W], scale_factor=s, mode="nearest") * w[:, k:k + 1]
        return up + self.g * (out - up)

    def forward(self, x):
        return _ckpt(self._body, x)


class HPConcat(nn.Module):
    """Concat whose fine input (index 1) gets an adaptive high-pass boost (FreqFusion AHPF part): f + g * (f - lowpass(f))."""

    def __init__(self, c: int, cm: int = 16, d: int = 1):
        super().__init__()
        self.d = d
        self.comp = _Conv(c, cm, 1, 1)
        self.gen = nn.Conv2d(cm, 9, 3, 1, 1)
        self.g = nn.Parameter(torch.zeros(1, c, 1, 1))

    def _hp(self, f):
        return f + self.g * (f - _local_mix(f, self.gen(self.comp(f)).softmax(1)))

    def forward(self, x):
        x = list(x)
        x[1] = _ckpt(self._hp, x[1])
        return torch.cat(x, self.d)


class UpDW(nn.Module):
    """Nearest upsample followed by a zero-init depthwise k x k residual (Up-ConvMixer-lite)."""

    def __init__(self, c: int, s: int, k: int = 5):
        super().__init__()
        self.s = s
        self.dw = _Conv(c, c, k, 1, g=c, act=False)
        _zero_bn(self.dw)

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.s, mode="nearest")
        return y + self.dw(y)


class UpMix(nn.Module):
    """Up-ConvMixer with a partial-conv token stage (FasterNet PConv, arXiv:2303.03667) and a MetaFormer channel MLP:
    y = up(x); t = [conv3x3(y[:p]), y[p:]]; out = y + BN0(W2 GELU(W1 t))."""

    def __init__(self, c: int, s: int, p: int = 32, hid: int = 32):
        super().__init__()
        self.s, self.p = s, p
        self.pc = nn.Conv2d(p, p, 3, 1, 1, bias=False)
        self.w1 = nn.Conv2d(c, hid, 1)
        self.w2 = _Conv(hid, c, 1, 1, act=False)
        _zero_bn(self.w2)

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.s, mode="nearest")
        t = torch.cat([self.pc(y[:, :self.p]), y[:, self.p:]], 1)
        return y + self.w2(F.gelu(self.w1(t)))


class DySampleUp(nn.Module):
    """DySample-style learned point sampling (arXiv:2308.15085). Zero offsets sample the source pixel centres, i.e. nearest."""

    def __init__(self, c: int, s: int = 2):
        super().__init__()
        self.s = s
        self.off = nn.Conv2d(c, 2 * s * s, 1)
        nn.init.zeros_(self.off.weight)
        nn.init.zeros_(self.off.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        s = self.s
        off = F.pixel_shuffle(self.off(x), s) * 0.25  # B, 2, sH, sW in source pixels
        ys = (torch.arange(s * H, device=x.device, dtype=torch.float32) // s) + 0.5
        xs = (torch.arange(s * W, device=x.device, dtype=torch.float32) // s) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        px = gx[None] + off[:, 0].float()
        py = gy[None] + off[:, 1].float()
        grid = torch.stack([px / W * 2 - 1, py / H * 2 - 1], -1)
        return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=False)


class GatedConcat(nn.Module):
    """Concat([top-down t, fine f]) with a cross-layer gate from f: t * (1 + tanh(g * mlp(avgpool2(f)))), g = 0 at init."""

    def __init__(self, cf: int, hid: int = 8, d: int = 1):
        super().__init__()
        self.d = d
        self.m1, self.m2 = nn.Conv2d(cf, hid, 1), nn.Conv2d(hid, 1, 1)
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        t, f = x
        q = self.m2(F.silu(self.m1(F.avg_pool2d(f, 2))))
        q = F.interpolate(q, size=t.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([t * (1 + torch.tanh(self.g * q)), f], self.d)


class ConvDW(_Conv):
    """Conv followed by a zero-init depthwise 3x3 residual; Conv parameter names unchanged (COCO transfer)."""

    def forward(self, x):
        y = super().forward(x)
        return y + self.zdw(y)

    def forward_fuse(self, x):
        y = super().forward_fuse(x)
        return y + self.zdw(y)


_SOBEL = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                       [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                       [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]])[:, None]


def _edges(m, n=3):
    """m (B,1,H,W) -> |Sobel x|, |Sobel y|, |Laplacian| (first n), fixed filters."""
    return F.conv2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), _SOBEL[:n].to(m.device, m.dtype)).abs()


class EdgePrior(nn.Module):
    """Parameter-free Sobel/Laplacian prior of a head feature, encoded to `co` channels: [|Gx|,|Gy|,|Lap|](mean_c x), max_c x."""

    def __init__(self, co: int = 16, k: int = 3):
        super().__init__()
        self.enc = _Conv(4, co, k, 1)

    def forward(self, x):
        return self.enc(torch.cat([_edges(x.mean(1, keepdim=True)), x.amax(1, keepdim=True)], 1))


class Mona(nn.Module):
    """Mona multi-cognitive adapter (arXiv:2311.15010) in MetaFormer form, as a zero-init residual term:
    v = down(scaled-LN(u)); m = DW3/DW5/DW7 pyramid(v) (split channels or averaged); h = GELU(v + m*alpha + e); out = BN0(up(h))."""

    def __init__(self, c: int, mid: int = 16, split: bool = False, ctx_ch: int = 0, edge: bool = False):
        super().__init__()
        self.split, self.edge = split, edge
        self.ln = nn.GroupNorm(1, c)
        self.s1 = nn.Parameter(torch.ones(1, c, 1, 1))
        self.s2 = nn.Parameter(torch.ones(1, c, 1, 1))
        self.down = nn.Conv2d(c, mid, 1)
        chs = [mid - 2 * (mid // 3), mid // 3, mid // 3] if split else [mid] * 3
        self.chs = chs
        self.dw = nn.ModuleList(nn.Conv2d(ch, ch, k, 1, k // 2, groups=ch) for ch, k in zip(chs, (3, 5, 7)))
        self.alpha = nn.Conv2d(ctx_ch, mid, 1) if ctx_ch else None
        self.up = _Conv(mid, c, 1, 1, act=False)
        _zero_bn(self.up)

    def forward(self, u, ctx=None, e=None):
        v = self.down(self.s1 * self.ln(u) + self.s2 * u)
        if self.split:
            m = torch.cat([dw(t) for dw, t in zip(self.dw, v.split(self.chs, 1))], 1)
        else:
            m = sum(dw(v) for dw in self.dw) / 3
        if self.alpha is not None and ctx is not None:
            m = m * F.interpolate(torch.sigmoid(self.alpha(ctx)), size=v.shape[2:], mode="nearest")
        h = v + m
        if e is not None:
            h = h + e
        return self.up(F.gelu(h))


class PConvMF(nn.Module):
    """FasterNet partial conv (dense 3x3 on the first p channels) + MetaFormer channel MLP, zero-init residual term."""

    def __init__(self, c: int, p: int = 8, hid: int = 16):
        super().__init__()
        self.p = p
        self.pc = nn.Conv2d(p, p, 3, 1, 1, bias=False)
        self.w1 = nn.Conv2d(c, hid, 1)
        self.w2 = _Conv(hid, c, 1, 1, act=False)
        _zero_bn(self.w2)

    def forward(self, u):
        t = torch.cat([self.pc(u[:, :self.p]), u[:, self.p:]], 1)
        return self.w2(F.gelu(self.w1(t)))


def routed_attn(q, k, v, topk: int = 8, heads: int = 2):
    """Bi-level routing attention (BiFormer, arXiv:2303.08810) from dense queries to a coarse token map.
    q (B,d,H,W) with H = s*Hk; k, v (B,d,Hk,Wk). Each s x s query region routes to its top-k tokens, then every query
    cell attends to the gathered tokens. Returns (B,d,H,W)."""
    B, d, H, W = q.shape
    Hk, Wk = k.shape[2:]
    s = H // Hk
    R = Hk * Wk
    kt, vt = k.flatten(2).transpose(1, 2), v.flatten(2).transpose(1, 2)  # B, R, d
    qr = F.avg_pool2d(q, s).flatten(2).transpose(1, 2)  # B, R, d
    idx = (qr @ kt.transpose(1, 2)).topk(min(topk, R), -1).indices  # B, R, k
    kk = min(topk, R)
    gi = idx.reshape(B, R * kk, 1).expand(-1, -1, d)
    kg = kt.gather(1, gi).reshape(B, R, kk, d)
    vg = vt.gather(1, gi).reshape(B, R, kk, d)
    qc = q.reshape(B, d, Hk, s, Wk, s).permute(0, 2, 4, 3, 5, 1).reshape(B, R, s * s, d)
    dh = d // heads

    def mh(t):
        return t.reshape(B * R, t.shape[2], heads, dh).transpose(1, 2)

    o = F.scaled_dot_product_attention(mh(qc), mh(kg), mh(vg))  # B*R, h, s*s, dh
    o = o.transpose(1, 2).reshape(B, Hk, Wk, s, s, d).permute(0, 5, 1, 3, 2, 4).reshape(B, d, H, W)
    return o


class R12Detect(R11Detect):
    """R11Detect plus the optional round-12 head modules (see `_r12_head`). Every addition is zero at step 0."""

    def forward(self, x):
        x = _p3_split(self, list(x))
        raw = x.pop() if getattr(self, "ctx", None) is not None else None
        self._ctxraw = raw
        ctxf = self.ctx(raw) if raw is not None else None
        o = getattr(self, "r12", {})
        if ctxf is not None and o.get("route_ctx"):  # M2: stride-32 routed attention inside the context skip
            q = self.m2q(F.avg_pool2d(x[1], 2))
            kv = self.m2kv(ctxf)
            att = routed_attn(q, *kv.split(q.shape[1], 1), topk=8, heads=2)
            ctxf = F.interpolate(ctxf, scale_factor=2, mode="nearest") + 0.1 * self.m2p(att)
        if ctxf is not None and o.get("freq"):  # L2: depthwise low-pass on the upsampled context
            c2 = F.interpolate(ctxf, scale_factor=2, mode="nearest")
            ctxf = c2 + self.l2c(c2)
        self._ctxf = ctxf
        if getattr(self, "fb", None) is not None:
            f = F.interpolate(self.fb(x[1]), size=x[0].shape[2:], mode="nearest")
            if o.get("freq"):
                f = f + self.l2f(f)
            x0 = x[0] + torch.sigmoid(self.fba(x[0])) * f
            if o.get("freq"):  # P3 high-pass detail, gated by [P3, feedback]
                hp = x[0] - F.avg_pool2d(x[0], 3, 1, 1, count_include_pad=False)
                x0 = x0 + self.l2h(hp * torch.sigmoid(self.l2g(torch.cat([x[0], f], 1))))
            x[0] = x0
        self._edge_img = None
        if o.get("edge_img"):
            img = _TAP.pop("img", None)
            if img is not None:
                g = F.avg_pool2d(img.mean(1, keepdim=True), 2)
                e = _edges(g, 2)
                self._edge_img = (e[:, :1] ** 2 + e[:, 1:] ** 2 + 1e-6).sqrt()
        try:
            return _Detect.forward(self, x)
        finally:
            self._ctxf = self._ctxraw = self._edge_img = None

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        o = getattr(self, "r12", {})
        bs = x[0].shape[0]
        b2c, ctxf = getattr(self, "b2c", None), getattr(self, "_ctxf", None)
        boxes, scores = [], []
        d0 = None
        sv = None
        if o.get("scene") and self._ctxraw is not None:
            sv = self.sc_mlp(self._ctxraw.float().mean((2, 3))).to(x[0].dtype)
            if self.training and getattr(self, "sc_aux", None) is not None:
                self._sc_pred = self.sc_aux(sv)
        for i in range(self.nl):
            xi = x[i]
            ci, bi = xi, xi
            size = xi.shape[2:]
            if o.get("edge_img") and self._edge_img is not None:  # H2
                e_img = F.adaptive_max_pool2d(self._edge_img, size)
                bi = xi * (1 + torch.tanh(self.h2[i](torch.cat([e_img, xi.amax(1, keepdim=True)], 1))))
            e = self.ef[i](xi) if (o.get("edge_l3") or o.get("edge_m3")) else None
            h0 = box_head[i][0](bi)
            h1 = box_head[i][1](h0)
            hl3 = self.l3d[i](h1 * (1 + torch.tanh(e))) if o.get("edge_l3") else None
            if o.get("edge_m3"):
                h1 = h1 + self.m3b[i](h1 + e)
            b = box_head[i][2](h1)
            if hl3 is not None:
                b = b + self.l3r[i](hl3)
            if o.get("br"):
                b = b + self.br[i](h1)
            if b2c is not None:
                ci = ci + b2c[i](h0)
            if ctxf is not None:
                ci = ci + self.ctxo[i](F.interpolate(ctxf, size=size, mode="nearest"))
            if sv is not None and getattr(self, "sc_film", None) is not None:  # N: scene FiLM
                g, bt = self.sc_film[i](sv)[:, :, None, None].chunk(2, 1)
                ci = ci * (1 + torch.tanh(g)) + bt
            if o.get("l11gate") and self._ctxraw is not None:  # J3
                g = F.interpolate(self.j3[i](self._ctxraw), size=size, mode="bilinear", align_corners=False)
                ci = ci * (1 + torch.tanh(g))
            if o.get("route") and self._ctxraw is not None:  # I1
                kv = self.i1kv(self._ctxraw)
                q = self.i1q[i](ci)
                ci = ci + self.i1o[i](routed_attn(q, *kv.split(q.shape[1], 1), topk=8, heads=2))
            if o.get("density"):  # H1
                if i == 0:
                    d0 = self.hd(ci)  # linear density estimate; last 1x1 starts at zero
                    self._dens = d0
                    g = self.hg[0](d0)
                else:
                    g = self.hg[i](F.adaptive_avg_pool2d(d0, size))
                ci = ci * (1 + torch.tanh(g))
            u = cls_head[i][0](ci)
            if o.get("edge_l3"):
                u = u + self.l3c[i](e)
            mo = o.get("mona")
            if mo and mo.get("pos") == "pre":
                u = u + self.mona[i](u, ctx=ctxf, e=e if mo.get("edge") else None)
            u2 = cls_head[i][1](u)
            if mo and mo.get("pos", "par") == "par":
                u2 = u2 + self.mona[i](u, ctx=ctxf, e=e if mo.get("edge") else None)
            if o.get("pconv_mf"):
                u2 = u2 + self.mf[i](u)
            z = cls_head[i][2](u2)
            if sv is not None and getattr(self, "sc_bias", None) is not None:  # N: scene prior on logits
                z = z + self.sc_bias(sv)[:, :, None, None]
            if o.get("scr") and i == 0:  # H3
                z = z + F.max_pool2d(self.scr(F.interpolate(u, scale_factor=2, mode="nearest")), 2)
            if o.get("dquery") and i == 0:  # M1
                p = torch.sigmoid(z.detach()).amax(1, keepdim=True)
                dl = F.avg_pool2d(p, 5, 1, 2, count_include_pad=False)
                q = self.m1q(torch.cat([p, dl, p - dl], 1))
                ut = self.m1u(u2)
                h = (self.m1a(ut) + self.m1b(ut)) * (1 + torch.tanh(q))
                z = z + self.m1z(h)
                b = b + self.m1x(h)
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            scores.append(z.view(bs, self.nc, -1))
        return {"boxes": torch.cat(boxes, dim=-1), "scores": torch.cat(scores, dim=-1), "feats": x}


def _set_layer(seq, i, new, typ):
    old = seq[i]
    new.i, new.f, new.type = old.i, old.f, typ
    new.np = sum(p.numel() for p in new.parameters())
    seq[i] = new


def _r12_surgery(model, arch: dict) -> None:
    from ultralytics.nn.modules.conv import Conv
    seq, det = model.model, model.model[-1]
    nk = arch.get("r12_neck") or {}
    ch_out = lambda j: (seq[j].cv2.conv.out_channels if hasattr(seq[j], "cv2") else seq[j].conv.out_channels)
    for j in nk.get("carafe", []):  # replaces nn.Upsample at j; input channels = output of the layer feeding it
        src = seq[j].f if seq[j].f != -1 else j - 1
        _set_layer(seq, j, CarafeUp(ch_out(src), int(seq[j].scale_factor)), "bench.pra.CarafeUp")
    for j in nk.get("hpconcat", []):
        _set_layer(seq, j, HPConcat(ch_out(seq[j].f[1])), "bench.pra.HPConcat")
    if nk.get("updw") is not None:
        j = nk["updw"]
        _set_layer(seq, j, UpDW(ch_out(j - 1), int(seq[j].scale_factor)), "bench.pra.UpDW")
    if nk.get("upmix") is not None:
        j = nk["upmix"]
        _set_layer(seq, j, UpMix(ch_out(j - 1), int(seq[j].scale_factor)), "bench.pra.UpMix")
    if nk.get("dysample") is not None:
        j = nk["dysample"]
        _set_layer(seq, j, DySampleUp(ch_out(j - 1), int(seq[j].scale_factor)), "bench.pra.DySampleUp")
    if nk.get("gated_concat") is not None:
        j = nk["gated_concat"]
        _set_layer(seq, j, GatedConcat(ch_out(seq[j].f[1])), "bench.pra.GatedConcat")
    if nk.get("conv_dw") is not None:
        m = seq[nk["conv_dw"]]
        assert type(m) is Conv, type(m)
        m.__class__ = ConvDW
        c = m.conv.out_channels
        m.zdw = _Conv(c, c, 3, 1, g=c, act=False)
        _zero_bn(m.zdw)
        m.np = sum(p.numel() for p in m.parameters())
    o = arch.get("r12_head") or {}
    if not o:
        return
    assert isinstance(det, R11Detect), type(det)
    det.__class__ = R12Detect
    det.r12 = dict(o)
    chx = [s[0].conv.in_channels for s in det.cv2]
    cb = [s[0].conv.out_channels for s in det.cv2]
    c3 = [s[2].in_channels for s in det.cv3]
    nl, nc = len(chx), det.nc
    zc = lambda a, b, k=1, g=1: (lambda m: (_zero_bn(m), m)[1])(_Conv(a, b, k, 1, g=g, act=False))
    dctx = det.ctx.conv.out_channels if getattr(det, "ctx", None) is not None else 0
    craw = det.ctx.conv.in_channels if dctx else 0
    if o.get("br"):
        det.br = nn.ModuleList(nn.Sequential(_Conv(b, b, 3, 1, g=b), zc(b, 4 * det.reg_max)) for b in cb)
    if o.get("edge_img"):
        seq[0].__class__ = TapConv
        det.h2 = nn.ModuleList(zc(2, 1, 3) for _ in chx)
    if o.get("edge_l3") or o.get("edge_m3"):
        det.ef = nn.ModuleList(EdgePrior(16, 3 if o.get("edge_l3") else 1) for _ in chx)
    if o.get("edge_l3"):
        det.l3d = nn.ModuleList(_Conv(b, b, 3, 1, g=b) for b in cb)
        det.l3r = nn.ModuleList(zc(b, 4 * det.reg_max) for b in cb)
        det.l3c = nn.ModuleList(zc(16, c) for c in c3)
    if o.get("edge_m3"):
        det.m3b = nn.ModuleList(zc(b, b, 3, g=b) for b in cb)
    if o.get("freq"):
        assert dctx and getattr(det, "fb", None) is not None
        det.l2c = zc(dctx, dctx, 3, g=dctx)
        det.l2f = zc(chx[0], chx[0], 3, g=chx[0])
        det.l2g = nn.Conv2d(2 * chx[0], chx[0], 1, groups=4)
        det.l2h = zc(chx[0], chx[0], 3, g=chx[0])
    if o.get("route_ctx"):
        assert dctx
        det.m2q = nn.Conv2d(chx[1], dctx, 1)
        det.m2kv = nn.Conv2d(dctx, 2 * dctx, 1)
        det.m2p = _Conv(dctx, dctx, 1, 1, act=False)
    if o.get("l11gate"):
        det.j3 = nn.ModuleList(zc(craw, c) for c in chx)
    if o.get("route"):
        det.i1kv = nn.Conv2d(craw, 64, 1)
        det.i1q = nn.ModuleList(nn.Conv2d(c, 32, 1) for c in chx)
        det.i1o = nn.ModuleList(zc(32, c) for c in chx)
    if o.get("density"):
        det.hd = nn.Sequential(_Conv(chx[0], chx[0], 3, 1, g=chx[0]), nn.Conv2d(chx[0], 1, 1))
        nn.init.zeros_(det.hd[1].weight)
        nn.init.zeros_(det.hd[1].bias)
        det.hg = nn.ModuleList(zc(1, 1, 3) for _ in chx)
        det._dens = None
    if o.get("mona"):
        mo = o["mona"]
        det.mona = nn.ModuleList(Mona(c, 16, mo.get("split", False), dctx if mo.get("ctx") else 0, mo.get("edge", False)) for c in c3)
    if o.get("pconv_mf"):
        det.mf = nn.ModuleList(PConvMF(c) for c in c3)
    if o.get("scene"):  # N: one scene vector per image from global-pooled L11 (stride-64 C2PSA output)
        sc = o["scene"]
        assert craw, "scene needs head_ctx (L11 raw features)"
        d = int(sc.get("dim", 64))
        det.sc_mlp = nn.Sequential(nn.Linear(craw, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        if sc.get("film", True):  # per-image, per-channel scale/shift of the cls input; zero init = identity
            det.sc_film = nn.ModuleList(nn.Linear(d, 2 * c) for c in chx)
            for m in det.sc_film:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if sc.get("bias", True):  # per-image prior on the class logits; zero init
            det.sc_bias = nn.Linear(d, nc)
            nn.init.zeros_(det.sc_bias.weight); nn.init.zeros_(det.sc_bias.bias)
        if sc.get("aux", False):  # training-only head: log(1 + objects per class) of the image
            det.sc_aux = nn.Linear(d, nc)
        det._sc_pred = None
    if o.get("scr"):
        det.scr = nn.Sequential(_Conv(c3[0], c3[0], 3, 1, g=c3[0]), zc(c3[0], nc))
    if o.get("dquery"):
        det.m1q = _Conv(3, 16, 3, 1)
        det.m1u = nn.Conv2d(c3[0], 16, 1)
        det.m1a = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
        det.m1b = nn.Conv2d(16, 16, 3, 1, 2, dilation=2, groups=16)
        det.m1z = zc(16, nc)
        det.m1x = zc(16, 4 * det.reg_max)


class DensityLoss:
    """Wraps the detection criterion: + w * normalised MSE between the P3 density head and a Gaussian map of GT centres."""

    def __init__(self, crit, det, w: float):
        self.crit, self.det, self.w, self.n = crit, det, w, 0
        k = torch.arange(5, dtype=torch.float32) - 2
        g = torch.exp(-(k[:, None] ** 2 + k[None] ** 2) / 2)
        self.kern = (g / g.sum())[None, None]

    def __getattr__(self, name):
        if name.startswith("__") or name in ("crit", "det", "w", "n", "kern"):
            raise AttributeError(name)
        return getattr(self.crit, name)

    def __call__(self, preds, batch):
        loss, items = self.crit(preds, batch)
        d = getattr(self.det, "_dens", None)
        if d is None or not self.w:
            return loss, items
        B, _, H, W = d.shape
        t = torch.zeros(B, 1, H, W, device=d.device)
        bb = batch["bboxes"].to(d.device)
        if bb.numel():
            bi = batch["batch_idx"].to(d.device).long().view(-1)
            xs = (bb[:, 0] * W).long().clamp(0, W - 1)
            ys = (bb[:, 1] * H).long().clamp(0, H - 1)
            t.index_put_((bi, torch.zeros_like(bi), ys, xs), torch.ones_like(xs, dtype=t.dtype), accumulate=True)
            t = F.conv2d(t, self.kern.to(d.device), padding=2)
        mse = ((d.float() - t) ** 2).mean() / ((t ** 2).mean() + 1e-4)
        self.n += 1
        if self.n % 500 == 1:
            print(f"[pra] density aux: nmse {mse.item():.4f} det_loss/img {loss.sum().item() / B:.3f}", flush=True)
        self.det._dens = None
        return loss + self.w * mse * B / loss.numel(), items  # the trainer sums the loss vector


class SceneLoss:
    """Wraps the detection criterion: + w * MSE between the scene head and log(1 + objects per class) of each image."""

    def __init__(self, crit, det, w: float):
        self.crit, self.det, self.w, self.n = crit, det, w, 0

    def __getattr__(self, name):
        if name.startswith("__") or name in ("crit", "det", "w", "n"):
            raise AttributeError(name)
        return getattr(self.crit, name)

    def __call__(self, preds, batch):
        loss, items = self.crit(preds, batch)
        p = getattr(self.det, "_sc_pred", None)
        if p is None or not self.w:
            return loss, items
        B, nc = p.shape
        t = torch.zeros(B, nc, device=p.device)
        cls = batch["cls"].to(p.device).long().view(-1)
        if cls.numel():
            bi = batch["batch_idx"].to(p.device).long().view(-1)
            t.index_put_((bi, cls), torch.ones_like(cls, dtype=t.dtype), accumulate=True)
        mse = F.mse_loss(p.float(), torch.log1p(t))
        self.n += 1
        if self.n % 500 == 1:
            print(f"[pra] scene aux: mse {mse.item():.4f} det_loss/img {loss.sum().item() / B:.3f}", flush=True)
        self.det._sc_pred = None
        return loss + self.w * mse * B / loss.numel(), items  # the trainer sums the loss vector


class SceneLossModel(_DetectionModel):
    """DetectionModel whose criterion adds the auxiliary scene-count loss; `scene_loss` is set by the trainer."""

    def init_criterion(self):
        return SceneLoss(super().init_criterion(), self.model[-1], self.scene_loss)


class DensityLossModel(_DetectionModel):
    """DetectionModel whose criterion adds the auxiliary density loss; `density_loss` is set by the trainer."""

    def init_criterion(self):
        return DensityLoss(super().init_criterion(), self.model[-1], self.density_loss)


# ---- Round 13: Stack O (stack-models/proposals/r13/synthesis.md) ----
#   stack_o.s4:        {at: 16, src: 2, guided: true, mid: 32, ch: 64}
#                      O-1: Concat `at` ([top-down stride 8, backbone stride 8]) becomes S4Concat, which also appends
#                      stem(s4) = 1x1(DW3x3(PixelUnshuffle(s4))) with `ch` channels; the next C3k2's cv1 gets a zero-init
#                      input slice for it, so the network is exactly the base at step 0.
#                      O-2 (guided: true): s4 = C3k2-light(cat(up2(1x1(top-down stride 8)), L2)); guided: false uses raw L2.
#   stack_o.aux:       {w: 0.25, max_px: 32} O-3: training-only centre-heatmap focal loss on s4 for objects whose longer
#                      side is < max_px input pixels (EGSS-style fixed assignment, no TAL); never evaluated at inference.
#   stack_o.p3_expand: [4, 17] O-4: one extra C3k(c, c, n=2, e=1.0) appended inside each C3k2; cv2 gets a zero-init slice.

class S4Concat(nn.Module):
    """[top, b8, s4] -> cat(top, b8, stem(s4')) with s4' = guided(top, s4) or s4. Keeps a training-only aux heatmap of s4'."""

    def __init__(self, c_top: int, c_s4: int, guided: bool, mid: int, ch: int, nc: int, aux: bool):
        super().__init__()
        from ultralytics.nn.modules.block import C3k2
        from ultralytics.nn.modules.conv import DWConv
        self.d = 1
        self.guided = guided
        if guided:
            self.t = _Conv(c_top, mid, 1, 1)
            self.g = C3k2(mid + c_s4, c_s4, n=1, c3k=False, e=0.25)
        self.stem = nn.Sequential(nn.PixelUnshuffle(2), DWConv(4 * c_s4, 4 * c_s4, 3), _Conv(4 * c_s4, ch, 1, 1))
        self.aux = nn.Conv2d(c_s4, nc, 1) if aux else None
        if aux:
            nn.init.constant_(self.aux.bias, -4.6)  # prior 0.01 (RetinaNet): keeps the 8M-cell negative term small at step 0
        self._aux = None

    def forward(self, x):
        top, b8, s4 = x
        if self.guided:
            s4 = self.g(torch.cat([F.interpolate(self.t(top), scale_factor=2.0, mode="nearest"), s4], 1))
        if self.aux is not None and self.training:
            self._aux = self.aux(s4)
        return torch.cat([top, b8, self.stem(s4)], 1)


class TwinC3k2(nn.Module):
    """Stack P (round 14) twin P3 block: `plain` (= the base P3 C3k2) sees only the first c0 input channels and feeds the
    bottom-up path to P4; `det` (a COCO-initialised copy, cv1 widened for the stride-4 stem) sees everything and feeds
    only the P3 head. Output cat(plain, det); TakeFirst at the next layer and Detect.p3_split pick their halves."""

    def __init__(self, plain: nn.Module, det: nn.Module, c0: int):
        super().__init__()
        self.plain, self.det, self.c0 = plain, det, c0

    def forward(self, x):
        return torch.cat([self.plain(x[:, :self.c0]), self.det(x)], 1)


class TakeFirst(nn.Module):
    """Runs `m` on the first `c` channels of its input (the plain half of TwinC3k2)."""

    def __init__(self, m: nn.Module, c: int):
        super().__init__()
        self.m, self.c = m, c

    def forward(self, x):
        return self.m(x[:, :self.c])


def _widen_in(conv, extra: int) -> int:
    """Ultralytics Conv: add `extra` zero-weight input channels at the end. Returns the old input width."""
    old = conv.conv
    new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
    conv.conv = new
    return old.in_channels


def _stack_o_surgery(model, arch: dict) -> None:
    so = arch.get("stack_o") or {}
    if not so:
        return
    from ultralytics.nn.modules.block import C3k
    seq, det = model.model, model.model[-1]
    model._o_widen = []  # (layer, attr, old_in): slices whose COCO part is re-copied after inherit()
    import copy as _cp
    twin_at = so.get("twin_p3")  # layer index of the P3 C3k2 (17) or None
    plain = _cp.deepcopy(seq[twin_at]) if twin_at is not None else None
    if so.get("s4"):
        s = so["s4"]
        at, src = s.get("at", 16), s.get("src", 2)
        old = seq[at]
        assert type(old).__name__ == "Concat" and len(old.f) == 2, (at, old)
        top_src = old.f[0] if old.f[0] != -1 else at - 1
        ch_of = lambda j: (seq[j].cv2.conv.out_channels if hasattr(seq[j], "cv2") else None)
        c_top = ch_of(top_src - 1) if ch_of(top_src) is None else ch_of(top_src)  # Upsample/DySample keep channels
        c_s4, ch = ch_of(src), s.get("ch", 64)
        new = S4Concat(c_top, c_s4, s.get("guided", True), s.get("mid", 32), ch, det.nc, bool(so.get("aux")))
        new.i, new.f, new.type = old.i, [*old.f, src], "bench.pra.S4Concat"
        new.np = sum(p.numel() for p in new.parameters())
        seq[at] = new
        model.save = sorted(set(model.save) | {src})
        nxt = seq[at + 1]
        model._o_widen.append((at + 1, "det.cv1" if twin_at == at + 1 else "cv1", _widen_in(nxt.cv1, ch)))
        nxt.np = sum(p.numel() for p in nxt.parameters())
    if twin_at is not None:
        det_m = seq[twin_at]
        c0 = plain.cv1.conv.in_channels
        cout = plain.cv2.conv.out_channels
        tw = TwinC3k2(plain, det_m, c0)
        _set_layer(seq, twin_at, tw, "bench.pra.TwinC3k2")
        nb = seq[twin_at + 1]
        assert nb.f == -1, nb.f
        _set_layer(seq, twin_at + 1, TakeFirst(nb, cout), "bench.pra.TakeFirst")
        assert det.f[0] == twin_at, det.f
        det.p3_split = cout
        model._o_twin = twin_at
    for i in so.get("p3_expand", []):
        m = seq[i]
        assert type(m).__name__ == "C3k2", (i, type(m))
        m.m.append(C3k(m.c, m.c, n=2, shortcut=True, e=1.0))
        model._o_widen.append((i, "cv2", _widen_in(m.cv2, m.c)))
        m.np = sum(p.numel() for p in m.parameters())


def stack_o_fix(model, arch: dict, src: str = "yolo26n.pt") -> None:
    """After inherit(): widened convs missed the shape match; copy their COCO slice and keep the new slice at zero.
    Twin P3 (Stack P): both halves of TwinC3k2 and the wrapped next layer get the COCO tensors of their source layers."""
    if not getattr(model, "_o_widen", None) and getattr(model, "_o_twin", None) is None:
        return
    import functools
    from ultralytics.nn.tasks import load_checkpoint
    w, _ = load_checkpoint(src)
    sd_src = w.float().state_dict()
    ranges = (arch.get("inherit") or {}).get("ranges") or [[0, 10 ** 6, 0]]
    src_of = lambda i: next(i - o for lo, hi, o in ranges if lo <= i - o <= hi)
    t = getattr(model, "_o_twin", None)
    if t is not None:
        n = 0
        for j, parts in ((t, ("plain", "det")), (t + 1, ("m",))):
            sj = src_of(j)
            for part in parts:
                for k, v in getattr(model.model[j], part).state_dict().items():
                    sv = sd_src[f"model.{sj}.{k}"]
                    if v.shape == sv.shape:
                        v.copy_(sv)
                        n += 1
        print(f"[pra] stack_p: twin P3 layer {t} (plain + det) and wrapped layer {t + 1}: {n} COCO tensors", flush=True)
    for i, attr, old_in in getattr(model, "_o_widen", []):
        si = src_of(i)
        conv = functools.reduce(getattr, attr.split("."), model.model[i])
        attr = attr.split(".")[-1]
        with torch.no_grad():
            conv.conv.weight.zero_()
            conv.conv.weight[:, :old_in] = sd_src[f"model.{si}.{attr}.conv.weight"]
            for k in ("weight", "bias", "running_mean", "running_var"):
                getattr(conv.bn, k).copy_(sd_src[f"model.{si}.{attr}.bn.{k}"])
        print(f"[pra] stack_o: layer {i}.{attr} COCO slice [:, :{old_in}] from model.{si}, "
              f"new slice [:, {old_in}:{conv.conv.in_channels}] = 0", flush=True)


class S4AuxLoss:
    """Wraps the detection criterion: + w * CenterNet focal loss between the S4Concat aux heatmap and Gaussian centres
    of GT boxes with longer side < max_px (ellipse sigma = side / 4 in cells, 7x7 support)."""

    def __init__(self, crit, mod, w: float, max_px: float):
        self.crit, self.mod, self.w, self.max_px, self.n = crit, mod, w, max_px, 0

    def __getattr__(self, name):
        if name.startswith("__") or name in ("crit", "mod", "w", "max_px", "n"):
            raise AttributeError(name)
        return getattr(self.crit, name)

    def __call__(self, preds, batch):
        loss, items = self.crit(preds, batch)
        p = getattr(self.mod, "_aux", None)
        if p is None or not self.w:
            return loss, items
        self.mod._aux = None
        B, nc, H, W = p.shape
        dev = p.device
        stride = batch["img"].shape[-1] / W
        t = torch.zeros(B * nc * H * W, device=dev)
        bb = batch["bboxes"].to(dev).float()
        if bb.numel():
            cls = batch["cls"].to(dev).long().view(-1)
            bi = batch["batch_idx"].to(dev).long().view(-1)
            wg, hg = bb[:, 2] * W, bb[:, 3] * H
            k = torch.maximum(wg, hg) * stride < self.max_px
            if k.any():
                cx = (bb[k, 0] * W).floor().long().clamp(0, W - 1)
                cy = (bb[k, 1] * H).floor().long().clamp(0, H - 1)
                sx, sy = (wg[k] / 4).clamp(min=0.5), (hg[k] / 4).clamp(min=0.5)
                o = torch.arange(-3, 4, device=dev)
                dy, dx = torch.meshgrid(o, o, indexing="ij")
                dx, dy = dx.reshape(1, -1), dy.reshape(1, -1)
                xs, ys = cx[:, None] + dx, cy[:, None] + dy
                val = torch.exp(-(dx ** 2 / (2 * sx[:, None] ** 2) + dy ** 2 / (2 * sy[:, None] ** 2)))
                ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                idx = ((bi[k][:, None] * nc + cls[k][:, None]) * H + ys) * W + xs
                t.scatter_reduce_(0, idx[ok], val[ok], reduce="amax")
        t = t.view(B, nc, H, W)
        q = p.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = t.eq(1.0)
        pos_l = (torch.log(q) * (1 - q) ** 2)[pos].sum()
        neg_l = (torch.log(1 - q) * q ** 2 * (1 - t) ** 4)[~pos].sum()
        fl = -(pos_l + neg_l) / pos.sum().clamp(min=1)
        self.n += 1
        if self.n % 500 == 1:
            print(f"[pra] s4 aux: focal {fl.item():.4f} pos {int(pos.sum())} det_loss/img {loss.sum().item() / B:.3f}", flush=True)
        return loss + self.w * fl * B / loss.numel(), items  # the trainer sums the loss vector


class S4AuxLossModel(_DetectionModel):
    """DetectionModel whose criterion adds the Stack O stride-4 aux loss; `s4_aux` = (w, max_px, layer) set by the trainer."""

    def init_criterion(self):
        w, max_px, at = self.s4_aux
        return S4AuxLoss(super().init_criterion(), self.model[at], w, max_px)


def surgery(model, arch: dict) -> None:
    from ultralytics.nn.modules.conv import Conv, DWConv
    seq = model.model
    if arch.get("pooled_first") is not None:
        m = seq[arch["pooled_first"]]
        assert type(m) is Conv, m
        m.__class__ = PooledConv
    for i in arch.get("spd_layers", []):
        old = seq[i]
        assert isinstance(old, Conv) and old.conv.stride == (2, 2), (i, old)
        new = SPDConv(old.conv.in_channels, old.conv.out_channels)
        new.i, new.f, new.type = old.i, old.f, "bench.pra.SPDConv"
        new.np = sum(p.numel() for p in new.parameters())
        seq[i] = new
    for i in arch.get("adown_layers", []):
        from ultralytics.nn.modules.block import ADown
        old = seq[i]
        assert isinstance(old, Conv) and old.conv.stride == (2, 2), (i, old)
        new = ADown(old.conv.in_channels, old.conv.out_channels)
        new.i, new.f, new.type = old.i, old.f, "ultralytics.nn.modules.block.ADown"
        new.np = sum(p.numel() for p in new.parameters())
        seq[i] = new
    for spec in arch.get("deep_plus", []):  # [{layer: i, n: repeats}] zero-init extra capacity
        from ultralytics.nn.modules.block import C3k2
        m = seq[spec["layer"]]
        assert type(m) is C3k2, (spec, type(m))
        c = m.cv2.conv.out_channels
        m.__class__ = C3k2Plus
        m.zx = C3k2(c, c, n=spec.get("n", 1), c3k=True)
        m.zgamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        m.np = sum(p.numel() for p in m.parameters())
    import copy as _copy
    for spec in arch.get("append_bottlenecks", []):  # [{layer, n}] zero-init Bottlenecks appended inside C3k
        inner = seq[spec["layer"]].m[0].m  # C3k2(c3k=True).m[0] is a C3k; .m is a Sequential of Bottlenecks
        for _ in range(spec.get("n", 1)):
            b = _copy.deepcopy(inner[0])
            nn.init.zeros_(b.cv2.bn.weight)
            inner.append(b)
    for spec in arch.get("append_psa", []):  # [{layer, n}] zero-init PSABlocks appended to C2PSA
        blocks = seq[spec["layer"]].m
        for _ in range(spec.get("n", 1)):
            b = _copy.deepcopy(blocks[0])
            nn.init.zeros_(b.attn.proj.bn.weight)
            nn.init.zeros_(b.ffn[1].bn.weight)
            blocks.append(b)
    if arch.get("replace_c2psa") is not None:
        i = arch["replace_c2psa"]
        old = seq[i]
        c = old.cv2.conv.out_channels
        new = ZeroResBottleneck(c)
        new.i, new.f, new.type = old.i, old.f, "bench.pra.ZeroResBottleneck"
        new.np = sum(p.numel() for p in new.parameters())
        seq[i] = new
    if arch.get("lowrank"):  # random init here; PRATrainer re-inits from COCO via SVD (svd_init)
        for i in arch["lowrank"]["layers"]:
            old = seq[i]
            assert isinstance(old, Conv) and old.conv.stride == (2, 2), (i, old)
            c1, c2 = old.conv.in_channels, old.conv.out_channels
            new = LowRankConv(c1, c2, max(8, int(round(arch["lowrank"]["ratio"] * c2 / 8)) * 8))
            new.i, new.f, new.type = old.i, old.f, "bench.pra.LowRankConv"
            new.np = sum(p.numel() for p in new.parameters())
            seq[i] = new
    if arch.get("detail_inject"):
        src, dst = arch["detail_inject"]["src"], arch["detail_inject"]["dst"]
        m = seq[dst]
        assert type(m).__name__ == "C3k2", (dst, type(m))
        cs, c = seq[src].cv2.conv.out_channels, m.cv2.conv.out_channels
        m.__class__ = C3k2Detail
        cd = arch["detail_inject"].get("ch", 16)  # thin 1x1 first: the stride-4 map is the most expensive place to spend FLOPs
        parts = [_Conv(cs, cd, 1, 1), nn.PixelUnshuffle(2)]
        if arch["detail_inject"].get("dw"):  # 3x3 depthwise after space-to-depth: spatial mixing on the detail path
            parts.append(DWConv(4 * cd, 4 * cd, 3))
        m.dz = nn.Sequential(*parts, _Conv(4 * cd, c, 1, 1, act=False))
        nn.init.zeros_(m.dz[-1].bn.weight)
        m.f = [m.f, src]
        model.save = sorted(set(model.save) | {src})
        m.np = sum(p.numel() for p in m.parameters())
    det = seq[-1]
    if arch.get("lift"):
        m = seq[arch["lift"]["layer"]]
        assert type(m) is Conv, (arch["lift"], type(m))
        m.__class__ = LiftConv
        m.lift = float(arch["lift"]["scale"])
        det.stride = det.stride / m.lift
        print(f"[pra] lift x{m.lift} at layer {arch['lift']['layer']}: Detect strides {det.stride.tolist()}", flush=True)
    if arch.get("detect_bu"):
        assert type(det) is _Detect, type(det)
        c0, c1 = (det.cv2[i][0].conv.in_channels for i in (0, 1))
        det.__class__ = DetectBU
        det.bu = nn.Sequential(Conv(c0, c1, 3, 2), Conv(c1, c1, 1, 1, act=False))
        nn.init.zeros_(det.bu[1].bn.weight)
    if arch.get("head_c3"):
        ch = [s[0][0].conv.in_channels for s in det.cv3]
        c3s = arch["head_c3"] if isinstance(arch["head_c3"], list) else [arch["head_c3"]] * len(ch)  # int or per level
        lite = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, k), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, k), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, det.nc, 1)) for x, c3, k in zip(ch, c3s, arch.get("head_k", [3] * len(ch))))
        det.cv3 = lite
        if getattr(det, "one2one_cv3", None) is not None:
            import copy
            det.one2one_cv3 = copy.deepcopy(lite)
    if arch.get("win_attn"):
        wa = arch["win_attn"]
        m = seq[wa["layer"]]
        assert type(m).__name__ == "C3k2", (wa, type(m))
        m.__class__ = C3k2Win
        m.wa = WinAttn(m.cv2.conv.out_channels, wa.get("dim", 32), wa.get("heads", 2), wa.get("win", 8))
        m.np = sum(p.numel() for p in m.parameters())
    if arch.get("ctx_xattn"):
        xa = arch["ctx_xattn"]
        m = seq[xa["layer"]]
        assert type(m).__name__ == "C3k2", (xa, type(m))
        m.__class__ = C3k2XAttn
        ck = seq[xa["src"]].cv2.conv.out_channels
        m.xa = XAttn(m.cv2.conv.out_channels, ck, xa.get("dim", 32), xa.get("heads", 2))
        m.f = [m.f, xa["src"]]
        model.save = sorted(set(model.save) | {xa["src"]})
        m.np = sum(p.numel() for p in m.parameters())
    if arch.get("head_ctx") or arch.get("head_fb") or arch.get("head_b2c"):
        assert type(det) is _Detect, type(det)
        chx = [s[0].conv.in_channels for s in det.cv2]
        det.__class__ = R11Detect
        if arch.get("head_ctx"):
            hc = arch["head_ctx"]
            ck, d = seq[hc["src"]].cv2.conv.out_channels, hc.get("dim", 32)
            det.ctx = Conv(ck, d, 1, 1)  # normal init (with SiLU): only the per-level outputs start at zero
            det.ctxo = nn.ModuleList(Conv(d, c, 1, 1, act=False) for c in chx)
            for o in det.ctxo:
                _zero_bn(o)
            det.f = [*det.f, hc["src"]]
            model.save = sorted(set(model.save) | {hc["src"]})
        if arch.get("head_fb"):
            det.fb = Conv(chx[1], chx[0], 1, 1, act=False)
            _zero_bn(det.fb)
            det.fba = nn.Conv2d(chx[0], 1, 1)
        if arch.get("head_b2c"):
            cb = [s[0].conv.out_channels for s in det.cv2]
            det.b2c = nn.ModuleList(Conv(b, c, 1, 1, act=False) for b, c in zip(cb, chx))
            for o in det.b2c:
                _zero_bn(o)
    _r12_surgery(model, arch)
    _stack_o_surgery(model, arch)
    if hasattr(det, "bias_init"):
        det.bias_init()
    if arch.get("stride_mult"):
        # model.stride drives letterbox / rect-batch padding; Detect keeps its own per-level strides.
        model.stride = torch.tensor([float(arch["stride_mult"])])


def inherit(model, arch: dict, src: str = "yolo26n.pt") -> tuple[int, int]:
    """Copy COCO tensors from yolo26n.pt into `model` using the arch's index map. Returns (copied, total)."""
    from ultralytics.nn.tasks import load_checkpoint
    w, _ = load_checkpoint(src)
    sd_src = w.float().state_dict()
    inh = arch.get("inherit", {})
    off, max_src = inh.get("offset", 0), inh.get("max_src", 22)
    det_idx = len(model.model) - 1
    src_det = len(w.model) - 1
    levels = inh.get("detect_levels")
    mapped = {}
    for k, v in sd_src.items():
        mm = re.match(r"model\.(\d+)\.(.*)", k)
        if not mm:
            continue
        i, rest = int(mm.group(1)), mm.group(2)
        ranges = inh.get("ranges")  # [[lo, hi, offset], ...] for architectures with inserted layers
        if ranges:
            hit = [o for lo, hi, o in ranges if lo <= i <= hi]
            if hit:
                mapped[f"model.{i + hit[0]}.{rest}"] = v
        elif i <= max_src:
            mapped[f"model.{i + off}.{rest}"] = v
        if i == src_det and levels is not None:
            lm = re.match(r"((?:one2one_)?cv[23])\.(\d+)\.(.*)", rest)
            if lm:
                for new_l, src_l in enumerate(levels):
                    if src_l == int(lm.group(2)):
                        mapped[f"model.{det_idx}.{lm.group(1)}.{new_l}.{lm.group(3)}"] = v
    sd = model.state_dict()
    ok = {k: v for k, v in mapped.items() if k in sd and sd[k].shape == v.shape}
    n_partial = 0
    if inh.get("partial"):  # widened layers: copy the overlapping channels, keep the fresh init for the rest
        for k, v in mapped.items():
            if k in sd and k not in ok and sd[k].ndim == v.ndim and not k.startswith(f"model.{det_idx}."):
                t = sd[k].clone()
                cut = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
                t[cut] = v[cut]
                ok[k] = t
                n_partial += 1
        print(f"[pra] partial copy of {n_partial} widened tensors", flush=True)
    model.load_state_dict(ok, strict=False)
    print(f"[pra] inherited {len(ok)}/{len(sd)} tensors from {src} "
          f"(mapped {len(mapped)}, shape-mismatch {sum(1 for k in mapped if k in sd and sd[k].shape != mapped[k].shape)})",
          flush=True)
    return len(ok), len(sd)


def svd_init(model, arch: dict) -> None:
    """Initialise every LowRankConv from the SVD of the corresponding COCO stride-2 conv."""
    if not arch.get("lowrank"):
        return
    from ultralytics.nn.tasks import load_checkpoint
    w, _ = load_checkpoint(arch.get("init", "yolo26n.pt"))
    src = w.float()
    inh = arch.get("inherit", {})
    for i in arch["lowrank"]["layers"]:
        lr = model.model[i]
        si = i
        for lo, hi, o in inh.get("ranges", []) or []:
            if lo <= i - o <= hi:
                si = i - o
        if not inh.get("ranges"):
            si = i - inh.get("offset", 0)
        energy = lr.init_from(src.model[si])
        print(f"[pra] layer {i}: SVD rank {lr.a.out_channels}/{lr.b.conv.out_channels} retains {energy:.3f} energy", flush=True)


class PairBCE(nn.Module):
    """BCE for confusable class pairs. Applied to the class channels only, box weights keep the original targets.
    relax: a positive of class a also targets its partner b at eps times its score.
    sharpen: the partner channel of a positive gets its negative term weighted by w."""

    def __init__(self, pairs: list, mode: str, value: float):
        super().__init__()
        self.pairs, self.mode, self.value = pairs, mode, value
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        if self.mode == "relax":
            t = target.clone()
            for a, b in self.pairs:
                t[..., a] = torch.maximum(t[..., a], self.value * target[..., b])
                t[..., b] = torch.maximum(t[..., b], self.value * target[..., a])
            return self.bce(pred, t)
        w = torch.ones_like(target)
        for a, b in self.pairs:
            w[..., a] = torch.where((target[..., b] > 0) & (target[..., a] == 0), self.value, w[..., a])
            w[..., b] = torch.where((target[..., a] > 0) & (target[..., b] == 0), self.value, w[..., b])
        return self.bce(pred, target) * w


class PairLossModel(_DetectionModel):
    """DetectionModel whose criterion uses PairBCE; `pair_loss` is set by the trainer (architecture unchanged)."""

    def init_criterion(self):
        c = super().init_criterion()
        cfg = self.pair_loss
        for part in (c.one2many, c.one2one) if hasattr(c, "one2many") else (c,):
            part.bce = PairBCE(cfg["pairs"], cfg["mode"], cfg["value"])
        return c


def build(arch: dict, nc: int = 10, verbose: bool = False):
    from ultralytics.nn.tasks import DetectionModel
    m = DetectionModel(str(ROOT / arch["yaml"]), nc=nc, verbose=verbose)
    surgery(m, arch)
    return m


def make_trainer(arch: dict):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class PRATrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build(arch, nc=self.data["nc"])
            if arch.get("init", "yolo26n.pt"):
                inherit(model, arch, arch.get("init", "yolo26n.pt"))
                svd_init(model, arch)
                stack_o_fix(model, arch, arch.get("init", "yolo26n.pt"))
            so = arch.get("stack_o") or {}
            if so.get("aux"):
                assert so.get("s4"), "stack_o.aux needs stack_o.s4"
                model.__class__ = S4AuxLossModel
                model.s4_aux = (float(so["aux"].get("w", 0.25)), float(so["aux"].get("max_px", 32)), so["s4"].get("at", 16))
            if arch.get("pair_loss"):
                model.__class__, model.pair_loss = PairLossModel, arch["pair_loss"]
            if arch.get("density_loss"):
                model.__class__, model.density_loss = DensityLossModel, float(arch["density_loss"])
            if arch.get("scene_loss"):
                model.__class__, model.scene_loss = SceneLossModel, float(arch["scene_loss"])
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[pra] arch={arch} params={n_par / 1e6:.3f}M", flush=True)
            return model

    return PRATrainer
