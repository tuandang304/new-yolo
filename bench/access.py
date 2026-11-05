"""Access branch (IEEE Access successors of R9b): block-level redesigns applied AFTER COCO inheritance.

All transforms run on a model whose YOLO26n tensors are already loaded (pra.inherit + pra.svd_init), so a replaced block can
be built from the pretrained weights it replaces. Config: arch["access"] = list of {op: ..., layers: [...], ...}.

ops
- linattn_psa   (Access-1): every PSABlock attention in the C2PSA at `layers` uses ReLU-kernel linear attention with the
                 inherited qkv / pe / proj weights (softmax replaced, weights kept).
- hybrid_c3k    (Access-1 L9, Access-2): in the C3k2 at `layers`, the inner C3k's bottleneck `idx` becomes a hybrid
                 block: the pretrained bottleneck (local conv mixer) + zero-init partial-channel single-head linear
                 attention + zero-init ConvFFN. Exact at step 0. `replace: true` drops the pretrained bottleneck instead
                 (local mixer = zero-init DW 5x5), used where the whole inner C3k is replaced.
- uib_c3k       (Access-2 L20): inner bottleneck `idx` becomes a MobileNetV4 ExtraDW UIB (DW3x3 -> 1x1 expand ->
                 DW3x3 -> 1x1 project, zero-init) added to the pretrained bottleneck. Exact at step 0.
- repconv       (Access-1 L14, Access-2 L18 / inner 3x3s): 3x3 Conv -> RepConv (3x3 inherited + zero-init 1x1 branch),
                 folded to one 3x3 by model.fuse() at export. Exact at step 0, zero inference cost.
- dwsep_c3k     (Access-3): every 3x3 conv inside the inner bottlenecks of the C3k2 at `layers` becomes a depthwise 3x3 +
                 pointwise 1x1 pair, initialised by a per-input-channel rank-1 SVD of the pretrained 3x3 kernel.
- sra_psa       (Access-3): PSABlock attention keys/values come from a 2x average-pooled map (spatial-reduction
                 attention, PVT-style) with the inherited qkv / pe / proj weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import Attention, Bottleneck, C2PSA, C3k, C3k2
from ultralytics.nn.modules.conv import Conv, RepConv


def _zero_bn(conv: Conv) -> Conv:
    nn.init.zeros_(conv.bn.weight)
    return conv


def _linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q,k: (B, h, d, N); v: (B, h, e, N) -> (B, h, e, N). ReLU kernel, computed in fp32."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = F.relu(q.float()) + 1e-6, F.relu(k.float()) + 1e-6, v.float()
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        z = 1.0 / (torch.einsum("bhdn,bhd->bhn", q, k.sum(-1)) + 1e-6)
        return torch.einsum("bhdn,bhde->bhen", q, kv) * z.unsqueeze(2)


class LinearAttention(Attention):
    """Ultralytics Attention with softmax replaced by ReLU-kernel linear attention; keeps qkv/pe/proj (inherited)."""

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        out = _linear_attn(q, k, v).to(v.dtype).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(out)


class SRAttention(Attention):
    """Ultralytics Attention whose keys/values come from a 2x average-pooled map (spatial reduction); weights inherited."""

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, _, v_full = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        p = F.avg_pool2d(qkv, 2, ceil_mode=True)
        M = p.shape[-2] * p.shape[-1]
        _, k, v = p.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, M).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v_full.reshape(B, C, H, W))
        return self.proj(out)


class PartialLinAttn(nn.Module):
    """Single-head linear attention on the first c/r channels (SHViT-style partial channels); zero-init output."""

    def __init__(self, c: int, ratio: int = 4, d: int = 16):
        super().__init__()
        self.cp = max(8, c // ratio)
        self.d = d
        self.qkv = Conv(self.cp, 2 * d + self.cp, 1, act=False)
        self.proj = _zero_bn(Conv(c, c, 1, act=False))

    def forward(self, x):
        B, C, H, W = x.shape
        xa, xr = x[:, : self.cp], x[:, self.cp:]
        q, k, v = self.qkv(xa).flatten(2).split([self.d, self.d, self.cp], dim=1)
        a = _linear_attn(q[:, None], k[:, None], v[:, None])[:, 0].to(x.dtype).reshape(B, self.cp, H, W)
        return self.proj(torch.cat([a, xr], 1))


class ConvFFN(nn.Module):
    """1x1 expand -> DW 3x3 -> 1x1 project (zero-init)."""

    def __init__(self, c: int, e: float = 2.0):
        super().__init__()
        h = int(c * e)
        self.fc1 = Conv(c, h, 1)
        self.dw = Conv(h, h, 3, g=h)
        self.fc2 = _zero_bn(Conv(h, c, 1, act=False))

    def forward(self, x):
        return self.fc2(self.dw(self.fc1(x)))


class HybridBottleneck(nn.Module):
    """MetaFormer-style hybrid: x + local(x); x + attn(x); x + ffn(x). local = pretrained bottleneck or zero-init DW 5x5."""

    def __init__(self, c: int, local: nn.Module | None, ratio: int = 4, d: int = 16, e: float = 2.0):
        super().__init__()
        self.local = local if local is not None else _zero_bn(Conv(c, c, 5, g=c, act=False))
        self.keep_bottleneck = local is not None
        self.attn = PartialLinAttn(c, ratio, d)
        self.ffn = ConvFFN(c, e)

    def forward(self, x):
        x = self.local(x) if self.keep_bottleneck else x + self.local(x)  # a pretrained Bottleneck already adds its shortcut
        x = x + self.attn(x)
        return x + self.ffn(x)


class UIB(nn.Module):
    """MobileNetV4 ExtraDW universal inverted bottleneck, zero-init output, added to the pretrained bottleneck."""

    def __init__(self, base: nn.Module, c: int, e: float = 2.0):
        super().__init__()
        h = int(c * e)
        self.base = base
        self.uib = nn.Sequential(Conv(c, c, 3, g=c, act=False), Conv(c, h, 1), Conv(h, h, 3, g=h), _zero_bn(Conv(h, c, 1, act=False)))

    def forward(self, x):
        return self.base(x) + self.uib(x)


def to_repconv(conv: Conv) -> RepConv:
    """3x3 Conv -> RepConv with the conv as branch 1 (inherited) and a zero-init 1x1 branch; act kept."""
    c1, c2 = conv.conv.in_channels, conv.conv.out_channels
    s, g = conv.conv.stride[0], conv.conv.groups
    r = RepConv(c1, c2, 3, s, 1, g=g, act=conv.act)
    r.conv1.load_state_dict(conv.state_dict())
    nn.init.zeros_(r.conv2.bn.weight)
    nn.init.zeros_(r.conv2.bn.bias)
    return r.to(conv.conv.weight.device)


@torch.no_grad()
def dwsep_from(conv: Conv) -> nn.Sequential:
    """3x3 Conv(c1->c2) -> DW 3x3 (c1) + pointwise 1x1 (c1->c2, BN and act of the original), rank-1 SVD per input channel."""
    W = conv.conv.weight.detach().float()  # c2, c1, 3, 3
    c2, c1 = W.shape[:2]
    dw = nn.Conv2d(c1, c1, 3, conv.conv.stride, 1, groups=c1, bias=False)
    pw = Conv(c1, c2, 1, act=conv.act)
    for i in range(c1):
        U, S, Vh = torch.linalg.svd(W[:, i].reshape(c2, 9), full_matrices=False)
        s = S[0].sqrt()
        dw.weight[i, 0] = (Vh[0] * s).reshape(3, 3)
        pw.conv.weight[:, i, 0, 0] = U[:, 0] * s
    pw.bn.load_state_dict(conv.bn.state_dict())
    return nn.Sequential(dw, pw).to(W.device)


def _inner_c3k(m: nn.Module) -> C3k:
    assert isinstance(m, C3k2) and isinstance(m.m[0], C3k), type(m)
    return m.m[0]


def apply(model, specs: list) -> None:
    seq = model.model
    for sp in specs:
        op = sp["op"]
        for li in sp["layers"]:
            m = seq[li]
            if op == "linattn_psa":
                assert isinstance(m, C2PSA), (li, type(m))
                for blk in m.m:
                    blk.attn.__class__ = LinearAttention
            elif op == "sra_psa":
                assert isinstance(m, C2PSA), (li, type(m))
                for blk in m.m:
                    blk.attn.__class__ = SRAttention
            elif op == "hybrid_c3k":
                ck = _inner_c3k(m)
                idx = sp.get("idx", 1)
                b = ck.m[idx]
                c = b.cv2.conv.out_channels
                local = None if sp.get("replace") else b
                ck.m[idx] = HybridBottleneck(c, local, sp.get("ratio", 4), sp.get("d", 16), sp.get("e", 2.0)).to(b.cv1.conv.weight.device)
            elif op == "uib_c3k":
                ck = _inner_c3k(m)
                idx = sp.get("idx", 1)
                b = ck.m[idx]
                ck.m[idx] = UIB(b, b.cv2.conv.out_channels, sp.get("e", 2.0)).to(b.cv1.conv.weight.device)
            elif op == "repconv":
                if isinstance(m, Conv):  # a stand-alone 3x3 conv layer (e.g. the bottom-up downsampler)
                    r = to_repconv(m)
                    r.i, r.f, r.type, r.np = m.i, m.f, "ultralytics.nn.modules.conv.RepConv", sum(p.numel() for p in r.parameters())
                    seq[li] = r
                else:  # every 3x3 conv inside the inner bottlenecks of a C3k2
                    for b in _inner_c3k(m).m:
                        for name in ("cv1", "cv2"):
                            cv = getattr(b, name)
                            if isinstance(cv, Conv) and cv.conv.kernel_size == (3, 3):
                                setattr(b, name, to_repconv(cv))
            elif op == "lowrank_conv":  # 3x3 Conv layer -> LowRankConv of rank r, SVD of the inherited kernel
                from .pra import LowRankConv
                new = LowRankConv(m.conv.in_channels, m.conv.out_channels, sp["rank"]).to(m.conv.weight.device)
                if m.conv.stride != (1, 1):
                    new.a.stride = m.conv.stride
                e = new.init_from(m)
                new.i, new.f, new.type = m.i, m.f, "bench.pra.LowRankConv"
                seq[li] = new
                print(f"[access] layer {li}: SVD rank {sp['rank']} retains {e:.3f} energy", flush=True)
            elif op == "lowrank_pw":  # a 1x1 Conv attribute (e.g. SPPF.cv2) -> rank-r pair, SVD of the inherited weight
                cv = getattr(m, sp["attr"])
                W = cv.conv.weight.detach().float()[:, :, 0, 0]
                U, S_, Vh = torch.linalg.svd(W, full_matrices=False)
                r = sp["rank"]; s = S_[:r].sqrt()
                a1 = nn.Conv2d(W.shape[1], r, 1, bias=False); b1 = Conv(r, W.shape[0], 1, act=cv.act)
                with torch.no_grad():
                    a1.weight.copy_((s[:, None] * Vh[:r])[:, :, None, None]); b1.conv.weight.copy_((U[:, :r] * s[None])[:, :, None, None])
                b1.bn.load_state_dict(cv.bn.state_dict())
                setattr(m, sp["attr"], nn.Sequential(a1, b1).to(W.device))
                print(f"[access] layer {li}.{sp['attr']}: SVD rank {r} retains {float((S_[:r]**2).sum()/(S_**2).sum()):.3f} energy", flush=True)
            elif op == "dwsep_c3k":
                for b in _inner_c3k(m).m:
                    for name in ("cv1", "cv2"):
                        cv = getattr(b, name)
                        if isinstance(cv, Conv) and cv.conv.kernel_size == (3, 3):
                            setattr(b, name, dwsep_from(cv))
            else:
                raise ValueError(op)
            if hasattr(seq[li], "np"):
                seq[li].np = sum(p.numel() for p in seq[li].parameters())
        print(f"[access] {op} at layers {sp['layers']}", flush=True)
    for mod in model.modules():  # new BNs must match Ultralytics' BN settings (eps 1e-3, momentum 0.03), or step 0 is not exact
        if isinstance(mod, nn.BatchNorm2d):
            mod.eps, mod.momentum = 1e-3, 0.03
