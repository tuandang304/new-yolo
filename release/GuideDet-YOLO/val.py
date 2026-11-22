"""Ignore-aware COCO-API evaluation on VisDrone2019-DET (the evaluator used for every number in the paper).

    python val.py --weights weights/guidedet_yolo_s0.pt --root datasets/visdrone --split valid   # or --split test

Detections: 1280x1280 letterbox, confidence 0.001, NMS IoU 0.7 with multi-label assignment, at most 1000 per image.
Scoring: detections and annotated boxes lying more than 50% inside an ignore region (VisDrone "ignored regions"
or class "others") are dropped, as in the official toolkit; AP over IoU 0.50:0.95 and the ten classes with
maxDets = 500. Size buckets in original image pixels: tiny < 16^2, small < 32^2, medium < 96^2, large >= 96^2.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import io
import json
from pathlib import Path

import numpy as np

from guidedet import NAMES, load_yolo

AREAS = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [0, 32 ** 2], "medium": [32 ** 2, 96 ** 2],
         "large": [96 ** 2, 1e10]}
MAXDETS = [1, 10, 100, 500]


def multi_label_nms() -> None:
    """Ultralytics predict() keeps one class per box; score every box/class pair, as Ultralytics val does."""
    import ultralytics.utils.nms as nms
    if not getattr(nms.non_max_suppression, "_multi_label", False):
        orig = nms.non_max_suppression
        nms.non_max_suppression = lambda *a, **k: orig(*a, **{**k, "multi_label": True})
        nms.non_max_suppression._multi_label = True


def predict_split(yolo, root: Path, split: str, coco: dict, imgsz: int, device: str, batch: int = 8,
                  conf: float = 0.001, iou: float = 0.7, max_det: int = 1000, half: bool = False) -> list[dict]:
    """COCO-results list (xywh in original pixels) for every image of one split."""
    multi_label_nms()
    ids = {im["file_name"]: im["id"] for im in coco["images"]}
    paths = [str(root / split / "images" / im["file_name"]) for im in coco["images"]]
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        res = yolo.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, half=half,
                           verbose=False)
        for p, r in zip(chunk, res):
            b = r.boxes
            for (x0, y0, x1, y1), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.int().tolist()):
                out.append({"image_id": ids[Path(p).name], "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "score": s})
        if (i // batch + 1) % 25 == 0:
            print(f"{split}: {i + len(chunk)}/{len(paths)} images", flush=True)
    return out


def _ioa(det: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each box (xywh), the largest fraction of its area covered by a single ignore region (xywh)."""
    if len(regions) == 0 or len(det) == 0:
        return np.zeros(len(det))
    d0, d1 = det[:, None, :2], det[:, None, :2] + det[:, None, 2:]
    r0, r1 = regions[None, :, :2], regions[None, :, :2] + regions[None, :, 2:]
    inter = np.clip(np.minimum(d1, r1) - np.maximum(d0, r0), 0, None).prod(-1)
    return (inter / np.maximum(det[:, 2:].prod(-1), 1e-9)[:, None]).max(1)


def drop_ignored(boxes: list[dict], ignore: dict) -> list[dict]:
    by_img: dict[int, list[int]] = {}
    for k, d in enumerate(boxes):
        by_img.setdefault(d["image_id"], []).append(k)
    keep = []
    for iid, idx in by_img.items():
        reg = np.array(ignore.get(str(iid), ignore.get(iid, [])), dtype=float).reshape(-1, 4)
        xywh = np.array([boxes[k]["bbox"] for k in idx], dtype=float)
        keep += [k for k, f in zip(idx, _ioa(xywh, reg)) if f <= 0.5]
    return [boxes[k] for k in sorted(keep)]


def _ap(prec: np.ndarray) -> float:
    v = prec[prec > -1]
    return float(v.mean()) if v.size else float("nan")


def coco_eval(gt: dict, dets: list[dict], ignore: dict | None = None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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
    P, R = E.eval["precision"], E.eval["recall"]  # [T, R, K, A, M], [T, K, A, M]
    m500, m100 = MAXDETS.index(500), MAXDETS.index(100)
    out = {"AP": _ap(P[:, :, :, 0, m500]), "AP50": _ap(P[0, :, :, 0, m500]), "AP75": _ap(P[5, :, :, 0, m500]),
           "AP_maxdet100": _ap(P[:, :, :, 0, m100]), "AR500": _ap(R[:, :, 0, m500])}
    for a, name in enumerate(AREAS):
        if name != "all":
            out[f"AP_{name}"] = _ap(P[:, :, :, a, m500])
    out = {k: float(f"{100 * v:.2f}") for k, v in out.items()}
    out["per_class_AP"] = {NAMES[k]: float(f"{100 * _ap(P[:, :, k, 0, m500]):.2f}") for k in range(len(NAMES))}
    out["n_dets"] = len(dets)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="released state_dict, e.g. weights/guidedet_yolo_s0.pt")
    ap.add_argument("--root", default="datasets/visdrone", help="directory made by prepare_visdrone.py")
    ap.add_argument("--split", default="valid", choices=["valid", "test"], help="valid = val, test = test-dev")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-dets", default=None, help="optional .json.gz path for the raw detections")
    a = ap.parse_args()
    root = Path(a.root)
    gt = json.loads((root / f"coco_{a.split}.json").read_text())
    ign = json.loads((root / f"ignore_{a.split}.json").read_text())
    dets = predict_split(load_yolo(a.weights), root, a.split, gt, a.imgsz, a.device, a.batch)
    if a.save_dets:
        with gzip.open(a.save_dets, "wt") as f:
            json.dump(dets, f)
    res = {"ignore_aware": coco_eval(gt, dets, ignore=ign), "standard": coco_eval(gt, dets)}
    m = res["ignore_aware"]
    print(json.dumps(res, indent=1))
    print(f"{a.split}@{a.imgsz} ignore-aware AP={m['AP']:.2f} AP50={m['AP50']:.2f} AP75={m['AP75']:.2f} "
          f"tiny={m['AP_tiny']:.2f} small={m['AP_small']:.2f} medium={m['AP_medium']:.2f} large={m['AP_large']:.2f}")
