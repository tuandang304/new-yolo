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
  hier:          {mode: product|sum, weight, groups} superclass x fine class head + auxiliary superclass BCE
  head_cls_depth: number of DW+1x1 pairs in the lite class tower (default 2)
  framework:     {type: hr|hex|gd, at, src: [s4, s8, s16, ctx], cs: [c4, c8, c16], units, k, e} G40 new neck replacing the PAN
                 (placeholders in the YAML; see framework_surgery)
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


class R11Detect(_Detect):
    """Detect with optional round-11 additions, each a zero contribution at step 0:
    ctx  - stride-64 context (extra last input) projected once, upsampled and added to every classification input;
    fb   - P4 -> P3 feedback: P3 += sigmoid(fba(P3)) * up(fb(P4));
    b2c  - box -> class edge: classification input += b2c(first box-branch conv output)."""

    def forward(self, x):
        x = list(x)
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


def surgery(model, arch: dict) -> None:
    from ultralytics.nn.modules.conv import Conv, DWConv
    seq = model.model
    if arch.get("framework"):
        framework_surgery(model, arch)
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
        depth, n_out = arch.get("head_cls_depth", 2), det.nc + (4 if arch.get("hier") else 0)
        lite = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, k), Conv(x, c3, 1)),
                          *[nn.Sequential(DWConv(c3, c3, k), Conv(c3, c3, 1)) for _ in range(depth - 1)],
                          nn.Conv2d(c3, n_out, 1)) for x, c3, k in zip(ch, c3s, arch.get("head_k", [3] * len(ch))))
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
    if arch.get("hier"):
        hc = arch["hier"]
        if type(det) is _Detect:
            det.__class__ = R11Detect
        det.__class__ = HierDetect
        det.ns, det.hier_mode = 4, hc.get("mode", "product")
        det.register_buffer("sup_index", torch.tensor(hc.get("groups", VISDRONE_SUPER)), persistent=False)
        det.gamma = nn.Parameter(torch.zeros(det.nc))  # used by mode 'sum' only
    if arch.get("fw640"):
        from .fw640 import fw640_surgery
        fw640_surgery(model, arch)
    if hasattr(det, "bias_init"):
        det.bias_init()
    if arch.get("hier"):
        for s_ in det.cv3:  # superclass logits start confident (sigmoid ~0.98) so the product equals the fine score
            nn.init.constant_(s_[-1].bias[det.nc:], 4.0)
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
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    d = yaml_model_load(str(ROOT / arch["yaml"]))
    if arch.get("scale"):
        d["scale"] = arch["scale"]
    m = DetectionModel(d, nc=nc, verbose=verbose)
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
                fw = arch.get("framework") or {}
                if hasattr(model.model[fw.get("at", 0)], "load_coco"):
                    from ultralytics.nn.tasks import load_checkpoint
                    n = model.model[fw["at"]].load_coco(load_checkpoint(arch.get("init", "yolo26n.pt"))[0].float().state_dict())
                    print(f"[pra] framework neck: {n} COCO tensors copied into its stems", flush=True)
            if arch.get("pair_loss"):
                model.__class__, model.pair_loss = PairLossModel, arch["pair_loss"]
            if arch.get("hier"):
                model.__class__, model.hier_weight = HierLossModel, float(arch["hier"].get("weight", 0.35))
            if (arch.get("fw640") or {}).get("nwd_loss"):
                from .fw640 import NWDQueryLossModel
                model.__class__ = NWDQueryLossModel
                model.nwd_c = float(arch["fw640"].get("nwd_c", 12.0))
                model.nwd_w = float(arch["fw640"].get("nwd_w", 0.5))
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[pra] arch={arch} params={n_par / 1e6:.3f}M", flush=True)
            return model

    return PRATrainer


# ---- G40 frameworks: a new neck + head on the Stack B backbone ----
class RepLKBlock(nn.Module):
    """Re-parameterisable large-kernel block: x + PW(PW_e(DW_k(x) + DW_3(x))). The DW 3x3 branch can be folded into
    the DW kxk kernel after training (RepLKNet-style); both are BN-ed convs here."""

    def __init__(self, c: int, k: int = 7, e: int = 2):
        super().__init__()
        self.dwk = _Conv(c, c, k, 1, g=c, act=False)
        self.dw3 = _Conv(c, c, 3, 1, g=c, act=False)
        self.pw1 = _Conv(c, c * e, 1, 1)
        self.pw2 = _Conv(c * e, c, 1, 1, act=False)

    def forward(self, x):
        return x + self.pw2(self.pw1(F.silu(self.dwk(x) + self.dw3(x))))


class ParallelDown(nn.Module):
    """Stride-2 parallel downsampling: cat(DWConv 3x3 s2, MaxPool 2) -> 1x1 to c2."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.dw = _Conv(c1, c1, 3, 2, g=c1, act=False)
        self.pw = _Conv(2 * c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(torch.cat([self.dw(x), F.max_pool2d(x, 2)], 1))


class Exchange(nn.Module):
    """All-to-all exchange between parallel streams; each target fuses its inputs with per-pixel softmax weights."""

    def __init__(self, cs: list):
        super().__init__()
        self.n = n = len(cs)
        self.t = nn.ModuleDict()
        for i in range(n):
            for j in range(n):
                if i < j:  # finer -> coarser: parallel downsampling, chained per octave
                    steps = [ParallelDown(cs[i], cs[i]) for _ in range(j - i - 1)] + [ParallelDown(cs[i], cs[j])]
                    self.t[f"{i}_{j}"] = nn.Sequential(*steps)
                elif i > j:  # coarser -> finer: 1x1 then nearest upsample
                    self.t[f"{i}_{j}"] = _Conv(cs[i], cs[j], 1, 1, act=False)
        self.w = nn.ModuleList(nn.Conv2d(cs[j], n, 1) for j in range(n))
        for w in self.w:  # start as "keep own stream": weight on self dominates the softmax
            nn.init.zeros_(w.weight)
            with torch.no_grad():
                w.bias.fill_(0.0)
                w.bias[list(self.w).index(w)] = 2.0

    def forward(self, xs):
        out = []
        for j in range(self.n):
            a = self.w[j](xs[j]).softmax(1)
            y = a[:, j:j + 1] * xs[j]
            for i in range(self.n):
                if i == j:
                    continue
                z = self.t[f"{i}_{j}"](xs[i])
                if i > j:
                    z = F.interpolate(z, size=xs[j].shape[2:], mode="nearest")
                y = y + a[:, i:i + 1] * z
            out.append(y)
        return out


class HRNeck(nn.Module):
    """Parallel multi-resolution neck (Framework A). Inputs [stride-4, stride-8, stride-16, context]; returns the list of
    stream outputs (finest first). Context (any coarser stride) is projected and added to the coarsest stream."""

    def __init__(self, cin: list, cs: list, units: int = 3, k: int = 7, e: int = 2, cctx: int = 0):
        super().__init__()
        self.inp = nn.ModuleList(_Conv(a, b, 1, 1) for a, b in zip(cin, cs))
        self.ctx = _Conv(cctx, cs[-1], 1, 1) if cctx else None
        self.blocks = nn.ModuleList(nn.ModuleList(RepLKBlock(c, k, e) for c in cs) for _ in range(units))
        self.ex = nn.ModuleList(Exchange(cs) for _ in range(units))

    def forward(self, x):
        feats, ctx = (x[:-1], x[-1]) if self.ctx is not None else (x, None)
        xs = [m(f) for m, f in zip(self.inp, feats)]
        if ctx is not None:
            xs[-1] = xs[-1] + F.interpolate(self.ctx(ctx), size=xs[-1].shape[2:], mode="nearest")
        for blk, ex in zip(self.blocks, self.ex):
            xs = ex([b(t) for b, t in zip(blk, xs)])
        return xs


def framework_surgery(model, arch: dict) -> None:
    """Replace placeholder layers with a framework neck. The YAML puts a placeholder at index `at` followed by one
    placeholder per Detect level (1x1 convs with the right output strides/channels, so Ultralytics computes strides);
    here `at` becomes the neck (fed by `src`) and the per-level placeholders become Index(k) of its output list."""
    from ultralytics.nn.modules.conv import Index
    fw = arch["framework"]
    seq = model.model
    at, src = fw["at"], fw["src"]
    if fw["type"] == "hr":
        cin = [seq[s].cv2.conv.out_channels if hasattr(seq[s], "cv2") else seq[s].conv.out_channels for s in src[:-1]]
        cctx = seq[src[-1]].cv2.conv.out_channels if fw.get("ctx", True) else 0
        neck = HRNeck(cin, fw["cs"], fw.get("units", 3), fw.get("k", 7), fw.get("e", 2), cctx)
    elif fw["type"] == "hex":
        oc = lambda s_: seq[s_].cv2.conv.out_channels
        neck = HexNeck(oc(src[0]), oc(src[1]), oc(src[2]), oc(src[3]), fw["cs"][0], fw["cs"][1], fw.get("cap", 5), fw.get("k", 7))
    elif fw["type"] == "gd":
        oc = lambda s_: seq[s_].cv2.conv.out_channels
        c4, c8, c16 = fw["cs"]
        neck = GDNeck(oc(src[0]), oc(src[1]), oc(src[2]), oc(src[3]), c4, c8, c16, fw.get("hub", 96), fw.get("nhub", 2))
    else:
        raise ValueError(fw["type"])
    for mm in neck.modules():  # Ultralytics BN settings (initialize_weights runs before surgery)
        if isinstance(mm, nn.BatchNorm2d):
            mm.eps, mm.momentum = 1e-3, 0.03
    neck.i, neck.f, neck.type = at, list(src), f"bench.pra.{type(neck).__name__}"
    neck.np = sum(p.numel() for p in neck.parameters())
    seq[at] = neck
    for k in range(len(fw["cs"])):
        old = seq[at + 1 + k]
        idx = Index(k)
        idx.i, idx.f, idx.type, idx.np = old.i, at, "ultralytics.nn.modules.conv.Index", 0
        seq[at + 1 + k] = idx
    model.save = sorted(set(model.save) | set(src) | {at})


class RKLOpen(nn.Module):
    """Stream opening block (no residual): 1x1 (cin->c) + re-param DW 7x7 (+DW 3x3 branch) + 1x1 (c->c)."""

    def __init__(self, c1: int, c: int, k: int = 7):
        super().__init__()
        self.pw0 = _Conv(c1, c, 1, 1)
        self.dwk = _Conv(c, c, k, 1, g=c, act=False)
        self.dw3 = _Conv(c, c, 3, 1, g=c, act=False)
        self.pw1 = _Conv(c, c, 1, 1)

    def forward(self, x):
        y = self.pw0(x)
        return self.pw1(F.silu(self.dwk(y) + self.dw3(y)))


class RKLRes(nn.Module):
    """Residual large-kernel block: x + 1x1(2c->c)(DW7(1x1(c->2c)(x))); zero=True starts as identity (last BN gain 0)."""

    def __init__(self, c: int, e: int = 2, k: int = 7, zero: bool = False):
        super().__init__()
        self.pw1 = _Conv(c, c * e, 1, 1)
        self.dwk = _Conv(c * e, c * e, k, 1, g=c * e, act=False)
        self.dw3 = _Conv(c * e, c * e, 3, 1, g=c * e, act=False)
        self.pw2 = _Conv(c * e, c, 1, 1, act=False)
        if zero:
            _zero_bn(self.pw2)

    def forward(self, x):
        y = self.pw1(x)
        return x + self.pw2(F.silu(self.dwk(y) + self.dw3(y)))


class HexNeck(nn.Module):
    """Framework B (HexNet, proposed by the OpenCode harness): two live streams (stride 8, stride 16) with a stride-4
    detail feed and a stride-64 context gate, two HRNet exchange rounds, and parallel zero-init capacity on stride 8.
    Inputs [L2 (s4), L4 (s8), L6 (s16), L11 (ctx)]; returns [P3 (c8), P4 (c16)]."""

    def __init__(self, c4: int, c8in: int, c16in: int, cctx: int, c8: int = 64, c16: int = 128, cap: int = 5, k: int = 7):
        super().__init__()
        self.detail = nn.Sequential(_Conv(c4, 16, 1, 1), nn.PixelUnshuffle(2), _Conv(64, 64, 3, 1, g=64), _Conv(64, 32, 1, 1))
        self.ctx = _Conv(cctx, 32, 1, 1, act=False)
        self.open16, self.open8 = RKLOpen(c16in + 32, c16, k), RKLOpen(c8in + 32, c8, k)
        self.deep16a, self.deep16b = RKLRes(c16, 2, k), RKLRes(c16, 2, k, zero=True)
        self.deep8a, self.deep8b = RKLRes(c8, 2, k), RKLRes(c8, 2, k)
        self.up = nn.ModuleList(_Conv(c16, c8, 1, 1, act=False) for _ in range(2))
        for m in self.up:
            _zero_bn(m)
        self.down = nn.ModuleList(nn.Sequential(_Conv(c8, c8, 3, 2, g=c8, act=False), _Conv(c8, c16, 1, 1)) for _ in range(2))
        self.cap = nn.ModuleList(RKLRes(c8, 2, k) for _ in range(cap))
        self.cap_proj = _Conv(c8, c8, 1, 1, act=False)
        _zero_bn(self.cap_proj)

    def _exchange(self, i, s8, s16):
        u = F.interpolate(self.up[i](s16), size=s8.shape[2:], mode="nearest")
        return s8 + u, s16 + self.down[i](s8)

    def forward(self, x):
        l2, l4, l6, l11 = x
        g = F.interpolate(self.ctx(l11), size=l6.shape[2:], mode="nearest")
        s16 = self.deep16a(self.open16(torch.cat([l6, g], 1)))
        s8 = self.deep8a(self.open8(torch.cat([l4, self.detail(l2)], 1)))
        s8, s16 = self._exchange(0, s8, s16)
        s8, s16 = self.deep8b(s8), self.deep16b(s16)
        s8, s16 = self._exchange(1, s8, s16)
        s8 = s8 + self.cap_proj(sum(c(s8) - s8 for c in self.cap))  # parallel capacity: branch residuals summed, gated 0
        return [s8, s16]


# ---- hierarchical (superclass x fine) classification ----
VISDRONE_SUPER = [0, 0, 1, 3, 3, 2, 3, 3, 1, 2]  # pedestrian, people | bicycle, motor | tricycle, awning | van car truck bus


class HierDetect(R11Detect):
    """R11Detect whose class tower ends in nc fine + ns superclass logits. The returned class logit combines both:
    mode 'product': logit(sigmoid(fine) * sigmoid(super[g(c)]));  mode 'sum': fine + gamma * super[g(c)].
    Superclass logits are also returned as 'sup' for the auxiliary loss."""

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        b2c, ctxf = getattr(self, "b2c", None), getattr(self, "_ctxf", None)
        boxes, scores, sups = [], [], []
        g = self.sup_index
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
            raw = cls_head[i](ci).view(bs, self.nc + self.ns, -1)
            fine, sup = raw[:, :self.nc], raw[:, self.nc:]
            sg = sup[:, g]
            if self.hier_mode == "product":
                lp = F.logsigmoid(fine) + F.logsigmoid(sg)  # log p, p = sigmoid(fine) * sigmoid(super)
                z = lp - torch.log1p(-lp.exp().clamp(max=1 - 1e-6))
            else:
                z = fine + self.gamma.view(1, -1, 1) * sg
            boxes.append(b.view(bs, 4 * self.reg_max, -1))
            scores.append(z)
            sups.append(sup)
        return {"boxes": torch.cat(boxes, -1), "scores": torch.cat(scores, -1), "sup": torch.cat(sups, -1), "feats": x}


class CaptureBCE(nn.Module):
    """BCEWithLogits(reduction='none') that keeps the last target tensor (the assigner's soft class targets)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.last_target = None

    def forward(self, pred, target):
        self.last_target = target.detach()
        return self.bce(pred, target)


try:
    from ultralytics.utils.loss import v8DetectionLoss as _V8Loss
except Exception:  # pragma: no cover
    _V8Loss = object


class HierDetLoss(_V8Loss):
    """v8DetectionLoss + auxiliary BCE on superclass logits over assigned foreground anchors (mean per anchor);
    the target is 1 for the group of the anchor's assigned class, 0 for the other groups."""

    def get_assigned_targets_and_loss(self, preds, batch, **kw):
        info, loss, items = super().get_assigned_targets_and_loss(preds, batch, **kw)
        t = self.bce.last_target  # (b, a, nc)
        if "sup" in preds and t is not None:
            sup = preds["sup"].permute(0, 2, 1).contiguous()  # (b, a, ns)
            ts = torch.stack([t[..., self.sup_members[s]].amax(-1) for s in range(sup.shape[-1])], -1)
            fg = ts.amax(-1) > 0  # superclass = P(group | object): trained on assigned foreground anchors only
            ts = ts / ts.amax(-1, keepdim=True).clamp(min=1e-9)  # hard group label on each foreground anchor
            ls = F.binary_cross_entropy_with_logits(sup[fg], ts[fg].to(sup.dtype), reduction="sum") / fg.sum().clamp(min=1)
            loss[1] = loss[1] + self.sup_weight * self.hyp.cls * ls
        return info, loss, items


class HierLossModel(_DetectionModel):
    """DetectionModel whose criterion adds the superclass auxiliary loss (weight `hier_weight`)."""

    def init_criterion(self):
        c = super().init_criterion()
        c.bce = CaptureBCE()
        c.__class__ = HierDetLoss
        sup = self.model[-1].sup_index.tolist()
        c.sup_members = [[k for k, s in enumerate(sup) if s == j] for j in range(max(sup) + 1)]
        c.sup_weight = self.hier_weight
        return c


# ---- Framework C (HR-GD-HHD, proposed by the Antigravity harness; simplified) ----
class PDDown(nn.Module):
    """Parallel downsampling (stride 2): cat(1x1(DW3 s2 x), 1x1(AvgPool2 x)) -> c2 (half the channels from each path)."""

    def __init__(self, c1: int, c2: int, zero: bool = False):
        super().__init__()
        self.a = nn.Sequential(_Conv(c1, c1, 3, 2, g=c1, act=False), _Conv(c1, c2 // 2, 1, 1, act=zero is False))
        self.b = _Conv(c1, c2 - c2 // 2, 1, 1, act=zero is False)
        if zero:
            _zero_bn(self.a[-1]), _zero_bn(self.b)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(F.avg_pool2d(x, 2))], 1)


class PolyKernelLSK(nn.Module):
    """Re-parameterisable poly-kernel block with a selective spatial gate:
    y = 1x1(x); d = DW3(y)+DW5(y)+DW9(y)+DW1x11(y)+DW11x1(y) (all fold into one DW 11x11 at deploy);
    a = sigmoid(conv7x7([mean_c d, max_c d])); out = x + 1x1(d * a)."""

    def __init__(self, c: int):
        super().__init__()
        self.pw1 = _Conv(c, c, 1, 1)
        self.dws = nn.ModuleList([nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False) for k in (3, 5, 9)] +
                                 [nn.Conv2d(c, c, (1, 11), 1, (0, 5), groups=c, bias=False),
                                  nn.Conv2d(c, c, (11, 1), 1, (5, 0), groups=c, bias=False)])
        self.bn = nn.BatchNorm2d(c)
        self.gate = nn.Conv2d(2, 1, 7, 1, 3)
        self.pw2 = _Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        y = self.pw1(x)
        d = F.silu(self.bn(sum(m(y) for m in self.dws)))
        a = torch.sigmoid(self.gate(torch.cat([d.mean(1, keepdim=True), d.amax(1, keepdim=True)], 1)))
        return x + self.pw2(d * a)


class GatedInject(nn.Module):
    """Distribute: s * (1 + tanh(G)) + V with [G, V] from the hub; zero-init (identity at step 0)."""

    def __init__(self, ch: int, c: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "up":  # hub at stride 8 -> stride 4
            self.f = nn.Sequential(_Conv(ch, 2 * c, 1, 1, act=False))
        elif mode == "same":
            self.f = _Conv(ch, 2 * c, 1, 1, g=2 if ch % 2 == 0 and (2 * c) % 2 == 0 else 1, act=False)
        else:  # down: hub -> stride 16
            self.f = nn.Sequential(PDDown(ch, ch), _Conv(ch, 2 * c, 1, 1, act=False))
        _zero_bn(self.f[-1] if isinstance(self.f, nn.Sequential) else self.f)

    def forward(self, s, hub):
        gv = self.f(hub)
        if self.mode == "up":
            gv = F.interpolate(gv, size=s.shape[2:], mode="nearest")
        g, v = gv.chunk(2, 1)
        return s * (1 + torch.tanh(g)) + v


class GDNeck(nn.Module):
    """Framework C neck. Inputs [L2 (s4), L4 (s8), L6 (s16), L11 (ctx)]; returns [P2, P3, P4].
    Stems s16/s8 have the exact shapes of yolo26n L13/L16 (C3k2 384->128, 256->64) so they take COCO weights."""

    def __init__(self, c2in: int, c4in: int, c6in: int, cctx: int, c4: int = 32, c8: int = 64, c16: int = 128,
                 hub: int = 96, nhub: int = 2):
        super().__init__()
        from ultralytics.nn.modules.block import C3k2
        self.s16_stem = C3k2(cctx + c6in, c16, 1, True)      # = yolo26n model.13 (C3k2 384 -> 128, c3k)
        self.freq_up = _Conv(c16, c16, 3, 1, g=c16, act=False)  # low-pass DW before x2 upsample
        with torch.no_grad():  # starts as identity so the s8 stem reproduces the COCO top-down path exactly
            self.freq_up.conv.weight.zero_()
            self.freq_up.conv.weight[:, :, 1, 1] = 1.0
            self.freq_up.bn.eps = 1e-3
            self.freq_up.bn.weight.fill_((1 + 1e-3) ** 0.5)  # BN(x) = x exactly with running var 1
        self.s8_stem = C3k2(c16 + c4in, c8, 1, True)          # = yolo26n model.16 (C3k2 256 -> 64, c3k)
        self.hp = nn.Parameter(torch.zeros(1, c2in, 1, 1))     # adaptive high-pass on L2, zero-init
        self.s4_stem = nn.Sequential(_Conv(c2in, c4, 1, 1), _Conv(c4, c4, 3, 1, g=c4), _Conv(c4, c4, 1, 1))
        self.g4, self.g16, self.gctx = PDDown(c4, c4), _Conv(c16, 64, 1, 1), _Conv(cctx, 32, 1, 1)
        self.g16dw = _Conv(64, 64, 3, 1, g=64, act=False)
        self.gfuse = _Conv(c4 + c8 + 64 + 32, hub, 1, 1)
        self.hub = nn.Sequential(*[PolyKernelLSK(hub) for _ in range(nhub)])
        self.d4, self.d8, self.d16 = GatedInject(hub, c4, "up"), GatedInject(hub, c8, "same"), GatedInject(hub, c16, "down")
        self.p2_in = nn.Sequential(_Conv(c8, c4, 1, 1, act=False), _Conv(c4, c4, 3, 1, g=c4, act=False))
        _zero_bn(self.p2_in[0])
        self.p2_dw5, self.p2_dw3, self.p2_pw = _Conv(c4, c4, 5, 1, g=c4, act=False), _Conv(c4, c4, 3, 1, g=c4, act=False), _Conv(c4, c4, 1, 1)
        self.p3_from2, self.p3_from16 = PDDown(c4, c8, zero=True), _Conv(c16, c8, 1, 1, act=False)
        _zero_bn(self.p3_from16)
        self.p3_blk = C3k2(c8, c8, 1, True)
        self.p4_down = PDDown(c8, c8)
        self.p4_blk = C3k2(c8 + c16, c16, 1, True)

    def forward(self, x):
        l2, l4, l6, l11 = x
        s16 = self.s16_stem(torch.cat([F.interpolate(l11, size=l6.shape[2:], mode="nearest"), l6], 1))
        s8 = self.s8_stem(torch.cat([F.interpolate(self.freq_up(s16), size=l4.shape[2:], mode="nearest"), l4], 1))
        s4 = self.s4_stem(l2 + self.hp * (l2 - F.avg_pool2d(l2, 3, 1, 1)))
        hz = s8.shape[2:]
        g = torch.cat([self.g4(s4), s8, F.interpolate(self.g16dw(self.g16(s16)), size=hz, mode="nearest"),
                       F.interpolate(self.gctx(l11), size=hz, mode="nearest")], 1)
        hub = self.hub(self.gfuse(g))
        s4, s8, s16 = self.d4(s4, hub), self.d8(s8, hub), self.d16(s16, hub)
        y = s4 + F.interpolate(self.p2_in(s8), size=s4.shape[2:], mode="nearest")
        p2 = y + self.p2_pw(F.silu(self.p2_dw5(y) + self.p2_dw3(y)))
        p3 = self.p3_blk(s8 + self.p3_from2(p2) + F.interpolate(self.p3_from16(s16), size=hz, mode="nearest"))
        p4 = self.p4_blk(torch.cat([self.p4_down(p3), s16], 1))
        return [p2, p3, p4]

    @torch.no_grad()
    def load_coco(self, sd: dict) -> int:
        """Copy yolo26n model.13 -> s16_stem and model.16 -> s8_stem (exact shapes). Returns tensors copied."""
        n = 0
        for name, idx in (("s16_stem", 13), ("s8_stem", 16)):
            mod = getattr(self, name)
            for k, v in mod.state_dict().items():
                src = sd.get(f"model.{idx}.{k}")
                if src is not None and src.shape == v.shape:
                    v.copy_(src)
                    n += 1
        return n
