"""GPU latency at batch 1 in half precision (the paper used an NVIDIA T4).

    python benchmark_latency.py --weights weights/guidedet_yolo_s0.pt --images datasets/visdrone/valid/images

end-to-end: pre-processing + forward + NMS per image (Ultralytics predict, conf 0.25), cycling over the first
             --n-images images of --images, after --warm warm-up calls; median of --iters timed calls.
forward:     the fused FP16 network on a fixed 1x3xSxS tensor, CUDA events, no NMS.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from guidedet import load_yolo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="directory of images (the paper used the first 200 val images)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the latency benchmark")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)  # silence a per-call deprecation warning for `half`
    names = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:a.n_images]
    images = [np.asarray(Image.open(p).convert("RGB"))[..., ::-1].copy() for p in names]  # BGR, as cv2 would load
    y = load_yolo(a.weights)
    call = lambda im: y.predict(im, imgsz=a.imgsz, conf=0.25, half=True, device=0, verbose=False)
    for i in range(a.warm):
        call(images[i % len(images)])
    torch.cuda.synchronize()
    ts = []
    for i in range(a.iters):
        t = time.perf_counter()
        call(images[i % len(images)])
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    net = y.predictor.model  # AutoBackend holding the fused FP16 network
    x = torch.zeros(1, 3, a.imgsz, a.imgsz, device="cuda", dtype=torch.half if net.fp16 else torch.float)
    fw = []
    with torch.inference_mode():
        for _ in range(a.warm):
            net(x)
        torch.cuda.synchronize()
        for _ in range(a.iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            net(x)
            e.record()
            torch.cuda.synchronize()
            fw.append(s.elapsed_time(e))
    print(f"{torch.cuda.get_device_name(0)} FP16 batch 1 @{a.imgsz}: end-to-end median {np.median(ts):.2f} ms "
          f"(p90 {np.percentile(ts, 90):.2f}), forward-only median {np.median(fw):.2f} ms")
