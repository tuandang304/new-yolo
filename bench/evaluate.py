"""Unified COCO-style evaluator for VisDrone (10 classes).

Primary metric (VisDrone policy): AP@[.5:.95] with maxDets=500 (the VisDrone toolkit limit) over all areas,
computed after the ignore-region rule below (`ign_*` keys). Also AP50, AP75, AP at maxDets=100 (COCO default,
comparable to COCO-style VisDrone papers),
AP for tiny (<16^2 px), small (<32^2), medium, large, and per-class AP / AP50.
`ignore=...` additionally drops detections AND ground-truth boxes whose area lies >50% inside a VisDrone
ignore region (category 0 'ignored regions' and category 11 'others'), mirroring the official toolkit's
dropObjectsInIgr rule.
"""
from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .data import NAMES

AREAS = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [0, 32 ** 2], "medium": [32 ** 2, 96 ** 2],
         "large": [96 ** 2, 1e10]}
MAXDETS = [1, 10, 100, 500]


def _ioa(det: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Fraction of each det box (xywh) covered by the union-free max over ignore regions (xywh)."""
    if len(regions) == 0 or len(det) == 0:
        return np.zeros(len(det))
    d0, d1 = det[:, None, :2], det[:, None, :2] + det[:, None, 2:]
    r0, r1 = regions[None, :, :2], regions[None, :, :2] + regions[None, :, 2:]
    inter = np.clip(np.minimum(d1, r1) - np.maximum(d0, r0), 0, None).prod(-1)
    return (inter / np.maximum(det[:, 2:].prod(-1), 1e-9)[:, None]).max(1)


def drop_ignored(dets: list[dict], ignore: dict) -> list[dict]:
    by_img: dict[int, list[int]] = {}
    for k, d in enumerate(dets):
        by_img.setdefault(d["image_id"], []).append(k)
    keep = []
    for iid, idx in by_img.items():
        reg = np.array(ignore.get(str(iid), ignore.get(iid, [])), dtype=float).reshape(-1, 4)
        boxes = np.array([dets[k]["bbox"] for k in idx], dtype=float)
        keep += [k for k, f in zip(idx, _ioa(boxes, reg)) if f <= 0.5]
    return [dets[k] for k in sorted(keep)]


def _ap(prec: np.ndarray) -> float:
    v = prec[prec > -1]
    return float(v.mean()) if v.size else float("nan")


def coco_eval(gt: dict, dets: list[dict], ignore: dict | None = None) -> dict:
    if ignore is not None:
        dets = drop_ignored(dets, ignore)
        gt = {**gt, "annotations": drop_ignored(gt["annotations"], ignore)}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = copy.deepcopy(gt)
        cg.createIndex()
        cd = cg.loadRes(dets) if dets else COCO()
        E = COCOeval(cg, cd, "bbox")
        E.params.maxDets = MAXDETS
        E.params.areaRng = list(AREAS.values())
        E.params.areaRngLbl = list(AREAS.keys())
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]  # [T, R, K, A, M]
    R = E.eval["recall"]     # [T, K, A, M]
    m300, m100 = MAXDETS.index(500), MAXDETS.index(100)  # m300 holds the maxDets=500 index
    t50, t75 = 0, 5
    out = {
        "mAP50_95": _ap(P[:, :, :, 0, m300]), "mAP50": _ap(P[t50, :, :, 0, m300]), "mAP75": _ap(P[t75, :, :, 0, m300]),
        "mAP50_95_md100": _ap(P[:, :, :, 0, m100]), "mAP50_md100": _ap(P[t50, :, :, 0, m100]),
        "AR500": _ap(R[:, :, 0, m300]),
    }
    for a, name in enumerate(AREAS):
        if name != "all":
            out[f"AP_{name}"] = _ap(P[:, :, :, a, m300])
            out[f"AP50_{name}"] = _ap(P[t50, :, :, a, m300])
    out["per_class"] = {NAMES[k]: {"AP": _ap(P[:, :, k, 0, m300]), "AP50": _ap(P[t50, :, k, 0, m300])}
                        for k in range(len(NAMES))}
    out["n_dets"] = len(dets)
    return {k: (round(v * 100, 3) if isinstance(v, float) else v) if k != "per_class" else
            {c: {m: round(x * 100, 3) for m, x in d.items()} for c, d in v.items()} for k, v in out.items()}


def fmt(m: dict) -> str:
    keys = ["mAP50_95", "mAP50", "mAP75", "AP_tiny", "AP_small", "AP_medium", "AP_large", "mAP50_95_md100", "AR500"]
    return " ".join(f"{k}={m[k]:.2f}" for k in keys if k in m)
