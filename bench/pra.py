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
        m.dz = nn.Sequential(_Conv(cs, cd, 1, 1), nn.PixelUnshuffle(2), _Conv(4 * cd, c, 1, 1, act=False))
        nn.init.zeros_(m.dz[2].bn.weight)
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
            nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, det.nc, 1)) for x, c3 in zip(ch, c3s))
        det.cv3 = lite
        if getattr(det, "one2one_cv3", None) is not None:
            import copy
            det.one2one_cv3 = copy.deepcopy(lite)
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
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[pra] arch={arch} params={n_par / 1e6:.3f}M", flush=True)
            return model

    return PRATrainer
