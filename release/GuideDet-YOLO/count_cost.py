"""Parameters and FLOPs on the fused inference graph (BN folded into convolutions), PyTorch FlopCounterMode,
one multiply-accumulate = 2 FLOPs (the Ultralytics GFLOPs convention).

    python count_cost.py --imgsz 1280
"""
import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

from guidedet import VARIANTS, build_model


def cost(model: torch.nn.Module, imgsz: int) -> tuple[float, float]:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.zeros(1, 3, imgsz, imgsz))
    return sum(p.numel() for p in model.parameters()) / 1e6, fc.get_total_flops() / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    for v in VARIANTS:
        m = build_model(v).eval()
        p0, g0 = cost(m, a.imgsz)
        p1, g1 = cost(m.fuse(verbose=False), a.imgsz)
        print(f"{v:16s} fused: {p1:.3f} M params, {g1:.2f} GFLOPs @{a.imgsz} | training graph: {p0:.3f} M, {g0:.2f} G")
