"""Unified predictors: every model returns, per image, xyxy boxes (original pixels), scores, class idx in NAMES.

Class ids are mapped *by name* so checkpoints with a different label order (e.g. an extra 'others'
class) are evaluated on the same 10 VisDrone classes; unmapped classes are dropped and reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import NAMES


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _name_map(names) -> dict[int, int]:
    target = {_norm(n): i for i, n in enumerate(NAMES)}
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(k): target[_norm(v)] for k, v in items if _norm(v) in target}


class YoloPredictor:
    family = "ultralytics"

    def __init__(self, ckpt: str, device: str = "cpu", half: bool = False):
        from ultralytics import YOLO
        self.model = YOLO(ckpt)
        self.device, self.half = device, half
        self.map = _name_map(self.model.names)
        print(f"[predict] {ckpt}: names={self.model.names} -> map={self.map}", flush=True)

    def __call__(self, paths: list[str], imgsz: int, conf: float = 0.001, max_det: int = 500, batch: int = 8):
        # default Ultralytics predict path (nms=None): one-to-many head + NMS, also for YOLO26/YOLOv10
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            res = self.model.predict(chunk, imgsz=imgsz, conf=conf, iou=0.7, max_det=max_det, device=self.device,
                                     half=self.half, verbose=False)
            for p, r in zip(chunk, res):
                b = r.boxes
                cls = b.cls.cpu().numpy().astype(int)
                keep = np.array([c in self.map for c in cls], dtype=bool)
                yield p, b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep], \
                    np.array([self.map[c] for c in cls[keep]], dtype=int)

    def torch_module(self):
        return self.model.model


class RFDETRPredictor:
    family = "rfdetr"

    def __init__(self, ckpt: str, device: str = "cpu", half: bool = False):
        import rfdetr
        self.model = rfdetr.RFDETR.from_checkpoint(ckpt, trust_checkpoint=True, device=device) \
            if hasattr(rfdetr, "RFDETR") else rfdetr.RFDETRNano(pretrain_weights=ckpt, device=device)
        self.device, self.half = device, half
        self.map = _name_map(self.model.class_names)
        print(f"[predict] {ckpt}: class_names={self.model.class_names} -> map={self.map} "
              f"res={self.model.model_config.resolution}", flush=True)

    def __call__(self, paths: list[str], imgsz: int, conf: float = 0.001, max_det: int = 500, batch: int = 8):
        # RF-DETR returns at most num_select (=300 for Nano) boxes per image; recorded as a known cap
        from PIL import Image
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            ims = [Image.open(p).convert("RGB") for p in chunk]
            dets = self.model.predict(ims, threshold=conf, shape=(imgsz, imgsz), include_source_image=False)
            if not isinstance(dets, list):
                dets = [dets]
            for p, d in zip(chunk, dets):
                cls = d.class_id.astype(int) if d.class_id is not None else np.zeros(0, int)
                order = np.argsort(-d.confidence)[:max_det]
                keep = np.array([cls[j] in self.map for j in order], dtype=bool)
                order = order[keep]
                yield p, d.xyxy[order], d.confidence[order], np.array([self.map[c] for c in cls[order]], dtype=int)

    def torch_module(self):
        return self.model.model.model


def load_predictor(family: str, ckpt: str, device: str = "cpu", half: bool = False):
    return {"ultralytics": YoloPredictor, "rfdetr": RFDETRPredictor}[family](ckpt, device, half)


def predict_split(pred, data_root: Path, split: str, imgsz: int, coco: dict, max_det: int = 500) -> list[dict]:
    """COCO-results list for one split."""
    ids = {im["file_name"]: im["id"] for im in coco["images"]}
    paths = [str(data_root / split / "images" / im["file_name"]) for im in coco["images"]]
    out = []
    for n, (p, xyxy, sc, cl) in enumerate(pred(paths, imgsz=imgsz, max_det=max_det)):
        iid = ids[Path(p).name]
        for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), sc.tolist(), cl.tolist()):
            out.append({"image_id": iid, "category_id": int(c), "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
        if (n + 1) % 200 == 0:
            print(f"[predict] {split}@{imgsz}: {n + 1}/{len(paths)}", flush=True)
    return out
