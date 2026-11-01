"""Fine-tune from official COCO checkpoints with each framework's default recipe; only the shared knobs
(dataset, epochs, image size, seed, effective batch 16) are set explicitly."""
from __future__ import annotations

import time
from pathlib import Path

FAMILY = {"yolo26n": "ultralytics", "yolo11n": "ultralytics", "yolov10n": "ultralytics", "rfdetr-nano": "rfdetr"}


def train(model: str, data_root: Path, out: Path, epochs: int, imgsz: int, seed: int, device: str,
          extra: dict | None = None, arch: dict | None = None) -> tuple[str, dict]:
    extra = extra or {}
    t0 = time.time()
    if arch:  # proposed architecture: custom trainer builds the model and inherits YOLO26n COCO tensors
        from ultralytics import YOLO
        from .pra import make_trainer
        y = YOLO(arch.get("init", "yolo26n.pt"))  # only a handle; PRATrainer.get_model builds `arch`
        kw = dict(data=str(data_root / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=16, seed=seed,
                  deterministic=True, device=device, project=str(out), name="train", exist_ok=True, plots=True,
                  workers=4, cache="ram", val=True, trainer=make_trainer(arch))
        kw.update(extra)
        print(f"[train] proposed {model} arch={arch} {kw}", flush=True)
        y.train(**kw)
        ckpt = out / "train" / "weights" / "best.pt"
    elif FAMILY[model] == "ultralytics":
        from ultralytics import YOLO
        y = YOLO(f"{model}.pt")
        kw = dict(data=str(data_root / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=16, seed=seed,
                  deterministic=True, device=device, project=str(out), name="train", exist_ok=True, plots=True,
                  workers=4, cache="ram", val=True)
        kw.update(extra)
        print(f"[train] ultralytics {model} {kw}", flush=True)
        y.train(**kw)
        ckpt = out / "train" / "weights" / "best.pt"
    else:
        import rfdetr
        m = rfdetr.RFDETRNano()
        kw = dict(dataset_dir=str(data_root), epochs=epochs, resolution=imgsz, batch_size=4, grad_accum_steps=4,
                  seed=seed, output_dir=str(out / "train"), device=device, num_workers=4)
        kw.update(extra)
        print(f"[train] rfdetr {model} {kw}", flush=True)
        m.train(**kw)
        ckpt = out / "train" / "checkpoint_best_total.pth"
    return str(ckpt), {"train_hours": round((time.time() - t0) / 3600, 3), "train_kwargs": {k: str(v) for k, v in kw.items()}}
