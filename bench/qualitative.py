"""Qualitative dump on the fixed test-dev sample set: TP (green), FP (red), missed GT (yellow), ignore regions (grey).

Matching: class-aware greedy by score, IoU >= 0.5, detections with score >= conf_thr.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import NAMES


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0, a1, b0, b1 = a[:, None, :2], a[:, None, :2] + a[:, None, 2:], b[None, :, :2], b[None, :, :2] + b[None, :, 2:]
    inter = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None).prod(-1)
    return inter / (a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter + 1e-9)


def dump(data_root: Path, gt: dict, ignore: dict, dets: list[dict], samples: list[str], out_dir: Path,
         tag: str, conf_thr: float = 0.3) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {im["file_name"]: im for im in gt["images"]}
    stats = {}
    for name in samples:
        im = by_name[name]
        g = [a for a in gt["annotations"] if a["image_id"] == im["id"]]
        d = sorted([x for x in dets if x["image_id"] == im["id"] and x["score"] >= conf_thr], key=lambda x: -x["score"])
        gb = np.array([a["bbox"] for a in g], float).reshape(-1, 4)
        db = np.array([x["bbox"] for x in d], float).reshape(-1, 4)
        iou = _iou(db, gb) if len(db) and len(gb) else np.zeros((len(db), len(gb)))
        used, tp = set(), []
        for i, x in enumerate(d):
            cand = [(iou[i, j], j) for j, a in enumerate(g) if j not in used and a["category_id"] == x["category_id"]]
            best = max(cand, default=(0, -1))
            if best[0] >= 0.5:
                used.add(best[1])
                tp.append(True)
            else:
                tp.append(False)
        img = Image.open(data_root / "test" / "images" / name).convert("RGB")
        dr = ImageDraw.Draw(img)
        for x, y, w, h in ignore.get(str(im["id"]), []):
            dr.rectangle([x, y, x + w, y + h], outline=(150, 150, 150), width=1)
        for j, a in enumerate(g):
            if j not in used:
                x, y, w, h = a["bbox"]
                dr.rectangle([x, y, x + w, y + h], outline=(255, 215, 0), width=2)
        for x, t in zip(d, tp):
            bx, by, bw, bh = x["bbox"]
            dr.rectangle([bx, by, bx + bw, by + bh], outline=(0, 200, 0) if t else (230, 30, 30), width=2)
        img.save(out_dir / f"{Path(name).stem}__{tag}.png")
        stats[name] = {"gt": len(g), "tp": int(sum(tp)), "fp": int(len(tp) - sum(tp)), "fn": len(g) - len(used)}
    meta = {"tag": tag, "conf_thr": conf_thr, "iou_thr": 0.5, "legend": "green=TP red=FP yellow=missed GT grey=ignore",
            "classes": NAMES, "per_image": stats}
    (out_dir / f"qualitative_{tag}.json").write_text(json.dumps(meta, indent=2))
    return meta
