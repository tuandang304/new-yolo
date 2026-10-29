"""Proposed-model builders (YOLO26n-derived), with explicit COCO weight inheritance.

An architecture is configured in configs/node.yaml as `train.arch`:
  yaml:          repo-relative model YAML (models/*.yaml)
  pooled_first:  index of the first trunk conv that gets a 2x average-pool in front (RDD: trunk runs at 1/2 res)
  head_c3:       width of the Detect classification branch (default: Ultralytics rule)
  spd_layers:    shallow 3x3 stride-2 convs replaced by space-to-depth + 1x1 conv (lose COCO weights)
  adown_layers:  stride-2 convs replaced by Ultralytics ADown (avg-pool + split 3x3-s2 / maxpool-1x1; lose COCO weights)
  stride_mult:   letterbox/grid multiple for the whole model (64 when the trunk is pooled)
  inherit:       {offset: k, max_src: L, detect_levels: [src level for each new Detect level]}
                 maps yolo26n.pt tensors model.{i} -> model.{i+k} for i <= L, and Detect level l' <- level src.
Every inherited tensor must match in shape; the count of transferred tensors is printed.
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


class SPDConv(nn.Module):
    """Space-to-depth (PixelUnshuffle x2) + Conv-BN-SiLU 1x1: stride-2 without discarding pixels."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = _Conv(4 * c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.unshuffle(x))


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
    det = seq[-1]
    if arch.get("head_c3"):
        c3 = arch["head_c3"]
        ch = [s[0][0].conv.in_channels for s in det.cv3]
        lite = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, det.nc, 1)) for x in ch)
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
        if i <= max_src:
            mapped[f"model.{i + off}.{rest}"] = v
        elif i == src_det and levels is not None:
            lm = re.match(r"((?:one2one_)?cv[23])\.(\d+)\.(.*)", rest)
            if lm:
                for new_l, src_l in enumerate(levels):
                    if src_l == int(lm.group(2)):
                        mapped[f"model.{det_idx}.{lm.group(1)}.{new_l}.{lm.group(3)}"] = v
    sd = model.state_dict()
    ok = {k: v for k, v in mapped.items() if k in sd and sd[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    print(f"[pra] inherited {len(ok)}/{len(sd)} tensors from {src} "
          f"(mapped {len(mapped)}, shape-mismatch {sum(1 for k in mapped if k in sd and sd[k].shape != mapped[k].shape)})",
          flush=True)
    return len(ok), len(sd)


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
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[pra] arch={arch} params={n_par / 1e6:.3f}M", flush=True)
            return model

    return PRATrainer
