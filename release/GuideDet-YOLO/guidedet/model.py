"""Build GuideDet-YOLO / GuideDet-YOLO-F as Ultralytics DetectionModels and initialize them from YOLO26n (COCO).

Layer map (index: module):
  0-6   YOLO26n backbone (L3, L5: LR-Down in GuideDet-YOLO, plain 3x3 s2 Conv in GuideDet-YOLO-F)
  7     2x2 average pool (inserted), so L8-L11 run at stride 64
  8-11  deep stage (L8: LR-Down / Conv, C3k2, SPPF, C2PSA)
  12-14 x4 upsample, concat L6, C3k2 -> stride-16 top-down feature
  15    nearest x2 upsample -> t (stride 8)
  16    guided detail concat [t, L4, s(L2)]
  17    C3k2 (cv1 widened by 64 zero-initialized input channels) -> x3
  18-20 Conv s2, concat L14, C3k2 -> x4
  21    Detect on (x3, x4): one-to-many head, two levels (strides 8 and 16), NMS at inference
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .modules import GuidedDetailConcat, LowRankDown

YAML = Path(__file__).with_name("guidedet_yolo.yaml")
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
VARIANTS = {"guidedet-yolo": True, "guidedet-yolo-f": False}  # variant -> uses LR-Down at L3/L5/L8
LOWRANK_LAYERS, LOWRANK_RATIO = (3, 5, 8), 0.5
HEAD_C3 = 64
DETAIL_AT, DETAIL_SRC, DETAIL_MID, DETAIL_CH = 16, 2, 32, 64
STRIDE_MULT = 64  # input sizes must be multiples of 64 (stride-64 deep stage)


def _src_layer(i: int) -> int:
    """Index of the YOLO26n layer that target layer i inherits from (L7 is inserted; no source for it)."""
    return i if i <= 6 else i - 1


def _replace(seq, i: int, new: nn.Module, f=None) -> None:
    old = seq[i]
    new.i, new.f, new.type = old.i, old.f if f is None else f, f"guidedet.{type(new).__name__}"
    new.np = sum(p.numel() for p in new.parameters())
    seq[i] = new


def build_model(variant: str = "guidedet-yolo", nc: int = 10, verbose: bool = False):
    """Return a randomly initialized GuideDet-YOLO ('guidedet-yolo') or GuideDet-YOLO-F ('guidedet-yolo-f')."""
    from ultralytics.nn.modules.conv import Conv, DWConv
    from ultralytics.nn.tasks import DetectionModel

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
    model = DetectionModel(yaml.safe_load(YAML.read_text()), nc=nc, verbose=verbose)
    seq, det = model.model, model.model[-1]
    if VARIANTS[variant]:  # LR-Down: rank r = c2 / 2 (32, 64, 128 at L3, L5, L8)
        for i in LOWRANK_LAYERS:
            c1, c2 = seq[i].conv.in_channels, seq[i].conv.out_channels
            _replace(seq, i, LowRankDown(c1, c2, int(LOWRANK_RATIO * c2)))
    # light classification branch: two (DWConv 3x3, 1x1 Conv to 64) stages and a 1x1 to nc
    chx = [s[0].conv.in_channels for s in det.cv2]
    det.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, HEAD_C3, 1)),
                                          nn.Sequential(DWConv(HEAD_C3, HEAD_C3, 3), Conv(HEAD_C3, HEAD_C3, 1)),
                                          nn.Conv2d(HEAD_C3, det.nc, 1)) for x in chx)
    # guided stride-4 detail path; the next C3k2 gets DETAIL_CH extra zero-weight input channels
    c_top, c_s4 = seq[DETAIL_AT - 2].cv2.conv.out_channels, seq[DETAIL_SRC].cv2.conv.out_channels
    _replace(seq, DETAIL_AT, GuidedDetailConcat(c_top, c_s4, DETAIL_MID, DETAIL_CH), f=[*seq[DETAIL_AT].f, DETAIL_SRC])
    nxt = seq[DETAIL_AT + 1]
    old = nxt.cv1.conv
    new = nn.Conv2d(old.in_channels + DETAIL_CH, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
    nxt.cv1.conv = new
    nxt.np = sum(p.numel() for p in nxt.parameters())
    model.save = sorted(set(model.save) | {DETAIL_SRC})
    det.bias_init()
    model.stride = torch.tensor([float(STRIDE_MULT)])  # letterbox multiple; Detect keeps its strides (8, 16)
    model.names = dict(enumerate(NAMES)) if nc == len(NAMES) else {i: str(i) for i in range(nc)}
    return model


@torch.no_grad()
def inherit_coco(model, src: str = "yolo26n.pt", verbose: bool = True) -> int:
    """Copy YOLO26n COCO tensors into `model`: source layers 0-6 -> 0-6, 7-19 -> 8-20, Detect levels P3/P4 -> 0/1.
    LR-Down layers are initialized from the truncated SVD of the source kernel; the widened cv1 of L17 keeps the
    pretrained slice and zero weights for the detail-path channels. Returns the number of tensors copied unchanged."""
    from ultralytics.nn.tasks import load_checkpoint

    w, _ = load_checkpoint(src)
    w = w.float()
    sd_src = w.state_dict()
    det_src, det_dst = len(w.model) - 1, len(model.model) - 1
    mapped = {}
    for k, v in sd_src.items():
        m = re.match(r"model\.(\d+)\.(.*)", k)
        i, rest = int(m.group(1)), m.group(2)
        if i <= 19:
            mapped[f"model.{i if i <= 6 else i + 1}.{rest}"] = v
        elif i == det_src and re.match(r"cv[23]\.[01]\.", rest):  # one-to-many head, stride-8 and stride-16 levels
            mapped[f"model.{det_dst}.{rest}"] = v
    sd = model.state_dict()
    ok = {k: v for k, v in mapped.items() if k in sd and sd[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    if verbose:
        print(f"[guidedet] inherited {len(ok)}/{len(sd)} tensors from {src}", flush=True)
    for i, layer in enumerate(model.model):
        if isinstance(layer, LowRankDown):
            e = layer.init_from(w.model[_src_layer(i)])
            if verbose:
                print(f"[guidedet] L{i} LR-Down rank {layer.a.out_channels}/{layer.b.conv.out_channels}: "
                      f"SVD retains {e:.3f} of the kernel energy", flush=True)
    i = DETAIL_AT + 1
    conv, s = model.model[i].cv1, _src_layer(i)
    old_in = sd_src[f"model.{s}.cv1.conv.weight"].shape[1]
    conv.conv.weight.zero_()
    conv.conv.weight[:, :old_in] = sd_src[f"model.{s}.cv1.conv.weight"]
    for k in ("weight", "bias", "running_mean", "running_var"):
        getattr(conv.bn, k).copy_(sd_src[f"model.{s}.cv1.bn.{k}"])
    return len(ok)


def variant_of(state_dict: dict) -> str:
    return "guidedet-yolo" if "model.3.a.weight" in state_dict else "guidedet-yolo-f"


def load_model(weights: str, variant: str | None = None):
    """Build the model and load a released state_dict (strict). The variant is inferred from the keys if not given."""
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_model(variant or variant_of(sd))
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_yolo(weights: str, variant: str | None = None):
    """Wrap a released state_dict as an Ultralytics YOLO object (for predict / val)."""
    from ultralytics import YOLO, __version__

    model = load_model(weights, variant)
    tmp = Path(tempfile.mkdtemp()) / f"{Path(weights).stem}.pt"
    torch.save({"model": model.half(), "train_args": {"task": "detect", "imgsz": 1280}, "version": __version__}, tmp)
    return YOLO(str(tmp), task="detect")
