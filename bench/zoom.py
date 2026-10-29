"""Glance-and-Zoom inference (round-2 departure kill-test; no training).

One detector, two kinds of passes under a fixed FLOPs budget:
  glance: the whole image letterboxed to `glance` px (long side);
  zoom:   K windows of `win` native pixels, each resized to `crop` px, chosen from the glance output
          (score mode: windows maximising the summed confidence of small glance detections), merged with
          the glance detections by class-wise NMS.
Selection modes: score | oracle (GT tiny-object density, upper bound) | grid (fixed 2x2 layout) | none.
FLOPs (square convention) = F(glance) + K * F(crop).
"""
from __future__ import annotations

import numpy as np
import torch


def _windows(points: np.ndarray, weights: np.ndarray, W: int, H: int, win: int, K: int) -> list[tuple[int, int]]:
    """Greedy K windows (top-left corners) maximising summed weight of points inside; points = (N, 2) centers."""
    win_w, win_h = min(win, W), min(win, H)
    out, live = [], np.ones(len(points), bool)
    if len(points) == 0:
        return out
    step = max(win // 8, 8)
    xs = np.arange(0, W - win_w + 1, step)
    ys = np.arange(0, H - win_h + 1, step)
    for _ in range(K):
        if not live.any():
            break
        p, w = points[live], weights[live]
        best, arg = 0.0, None
        for y in ys:
            iny = (p[:, 1] >= y) & (p[:, 1] < y + win_h)
            if not iny.any():
                continue
            px = p[iny, 0]
            ww = w[iny]
            # sliding sum over x via sorted cumulative weights
            order = np.argsort(px)
            px, cw = px[order], np.concatenate([[0], np.cumsum(ww[order])])
            lo = np.searchsorted(px, xs, "left")
            hi = np.searchsorted(px, xs + win_w, "left")
            sums = cw[hi] - cw[lo]
            j = int(sums.argmax())
            if sums[j] > best:
                best, arg = float(sums[j]), (int(xs[j]), int(y))
        if arg is None:
            break
        out.append(arg)
        x0, y0 = arg
        live &= ~((points[:, 0] >= x0) & (points[:, 0] < x0 + win_w) & (points[:, 1] >= y0) & (points[:, 1] < y0 + win_h))
    return out


def _grid(W: int, H: int, win: int, K: int) -> list[tuple[int, int]]:
    xs = np.linspace(0, max(W - win, 0), 2).astype(int)
    ys = np.linspace(0, max(H - win, 0), 2).astype(int)
    return [(int(x), int(y)) for y in ys for x in xs][:K]


def run(model, paths: list[str], ids: list[int], glance: int, K: int, win: int, crop: int, mode: str,
        gt_by_img: dict | None = None, small_px: float = 32.0, device="cpu") -> list[dict]:
    import torchvision
    from PIL import Image
    dets = []
    for p, iid in zip(paths, ids):
        im = np.asarray(Image.open(p).convert("RGB"))[..., ::-1]  # BGR for Ultralytics
        H, W = im.shape[:2]
        r = model.predict(im, imgsz=glance, conf=0.001, iou=0.7, max_det=500, device=device, verbose=False)[0]
        boxes, scores, cls = r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()
        allb, alls, allc = [boxes], [scores], [cls]
        if K > 0 and mode != "none":
            if mode == "score":
                wh = (boxes[:, 2:] - boxes[:, :2]).numpy()
                small = np.sqrt(wh.prod(1)) < small_px * max(W, H) / glance  # small at glance scale
                ctr = ((boxes[:, :2] + boxes[:, 2:]) / 2).numpy()[small]
                wins = _windows(ctr, scores.numpy()[small], W, H, win, K)
            elif mode == "oracle":
                g = np.array(gt_by_img.get(iid, []), float).reshape(-1, 4)
                tiny = np.sqrt(g[:, 2] * g[:, 3]) * glance / max(W, H) < 16
                ctr = g[tiny, :2] + g[tiny, 2:] / 2
                wins = _windows(ctr, np.ones(len(ctr)), W, H, win, K)
            else:
                wins = _grid(W, H, win, K)
            for x0, y0 in wins:
                tile = im[y0:y0 + win, x0:x0 + win]
                rr = model.predict(np.ascontiguousarray(tile), imgsz=crop, conf=0.001, iou=0.7, max_det=500,
                                   device=device, verbose=False)[0]
                b = rr.boxes.xyxy.cpu().clone()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                allb.append(b), alls.append(rr.boxes.conf.cpu()), allc.append(rr.boxes.cls.cpu())
        b, s, c = torch.cat(allb), torch.cat(alls), torch.cat(allc)
        if len(allb) > 1 and len(b):
            keep = torchvision.ops.batched_nms(b.float(), s.float(), c.long(), 0.6)[:500]
            b, s, c = b[keep], s[keep], c[keep]
        for (x0, y0, x1, y1), sc, cl in zip(b.tolist(), s.tolist(), c.tolist()):
            dets.append({"image_id": iid, "category_id": int(cl), "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(sc)})
    return dets
