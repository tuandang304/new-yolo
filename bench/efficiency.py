"""Params, FLOPs (torch FlopCounterMode, FLOPs = 2*MACs, same convention as Ultralytics GFLOPs) and latency."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def params_flops(module: torch.nn.Module, imgsz: int) -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    m = copy.deepcopy(module).float().cpu().eval()
    out = {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 3)}
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 3, imgsz, imgsz))
        out[f"GFLOPs@{imgsz}"] = round(fc.get_total_flops() / 1e9, 2)
    except Exception as e:  # noqa: BLE001 - report, do not fail the run
        out[f"GFLOPs@{imgsz}"] = None
        out["flops_error"] = repr(e)[:300]
    return out


def latency(pred, image_path: str, imgsz: int, n_warm: int = 30, n_iter: int = 200) -> dict:
    """End-to-end single-image latency (pre + forward + post) at conf 0.25, FP16 on CUDA."""
    if not torch.cuda.is_available():
        return {"latency_note": "no CUDA; skipped"}
    from PIL import Image
    im = np.asarray(Image.open(image_path).convert("RGB"))
    if pred.family == "ultralytics":
        call = lambda: pred.model.predict(im[..., ::-1], imgsz=imgsz, conf=0.25, half=True, device=0, verbose=False)
    else:
        try:
            pred.model.inference(compile=False, dtype=torch.float16)
        except Exception as e:  # noqa: BLE001
            print(f"[latency] inference(fp16) failed: {e!r}", flush=True)
        call = lambda: pred.model.predict(im, threshold=0.25, shape=(imgsz, imgsz), include_source_image=False)
    for _ in range(n_warm):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        t = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return {f"e2e_ms@{imgsz}_median": round(float(np.median(ts)), 2), f"e2e_ms@{imgsz}_p90": round(float(np.percentile(ts, 90)), 2),
            "gpu": torch.cuda.get_device_name(0), "precision": "fp16", "batch": 1}
