"""Train GuideDet-YOLO / GuideDet-YOLO-F on VisDrone2019-DET with the paper recipe (Ultralytics 8.4.192):
100 epochs at 1280x1280, MuSGD lr0 0.01, momentum 0.9, weight decay 5e-4, 3 warm-up epochs, linear decay to 1%,
batch 8 with gradient accumulation to a nominal batch of 64, mosaic until the last epoch (close_mosaic 0),
scale 0.5, no mixup, mixed precision. The model starts from the YOLO26n COCO checkpoint (see guidedet.inherit_coco).

    python train.py --data datasets/visdrone/data.yaml --variant guidedet-yolo --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from guidedet import build_model, inherit_coco

RECIPE = dict(epochs=100, imgsz=1280, batch=8, nbs=64, optimizer="MuSGD", lr0=0.01, lrf=0.01, momentum=0.9,
              weight_decay=5e-4, warmup_epochs=3.0, close_mosaic=0, scale=0.5, mixup=0.0, cos_lr=False, amp=True,
              deterministic=True, cache=False, workers=8, val=True, plots=True)


def make_trainer(variant: str, init: str):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class GuideDetTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = build_model(variant, nc=self.data["nc"])
            inherit_coco(model, init)
            return model

    return GuideDetTrainer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml written by prepare_visdrone.py")
    ap.add_argument("--variant", default="guidedet-yolo", choices=["guidedet-yolo", "guidedet-yolo-f"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="yolo26n.pt", help="COCO checkpoint to inherit from (downloaded if missing)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a recipe value, e.g. --set epochs=1 imgsz=640 batch=2 workers=0")
    a = ap.parse_args()
    import yaml
    from ultralytics import YOLO

    kw = {**RECIPE, **{k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}}
    name = a.name or f"{a.variant}_s{a.seed}"
    y = YOLO(a.init)  # handle only; the trainer builds the GuideDet model in get_model
    y.train(data=a.data, seed=a.seed, device=a.device, project=str(Path(a.project).resolve()), name=name,
            exist_ok=True, trainer=make_trainer(a.variant, a.init), **kw)
    best = Path(a.project).resolve() / name / "weights" / "best.pt"
    ck = torch.load(best, map_location="cpu", weights_only=False)
    sd = (ck.get("ema") or ck["model"]).state_dict()
    torch.save(sd, best.with_name("best_state_dict.pt"))  # same format as weights/*.pt, loads with guidedet.load_model
    print(f"saved {best.with_name('best_state_dict.pt')}")
