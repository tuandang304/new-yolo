"""Fine-tune from official COCO checkpoints with each framework's default recipe; only the shared knobs
(dataset, epochs, image size, seed, effective batch 16) are set explicitly."""
from __future__ import annotations

import json
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
        _attach_log(y)
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
        _attach_log(y)
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


def _attach_log(y) -> None:
    """Compute logging rules: one [config]/[model] line at train start (resolved optimizer, not 'auto') and one line per epoch."""
    import torch

    def on_train_start(tr):
        opt = tr.optimizer
        g = opt.param_groups[0] if opt is not None else {}
        a = tr.args
        print("[config] " + json.dumps({"optimizer": type(opt).__name__, "lr0": a.lr0, "lr_group0": g.get("lr"), "momentum": a.momentum,
                                        "weight_decay": a.weight_decay, "batch": tr.batch_size, "nbs": a.nbs, "accumulate": tr.accumulate,
                                        "epochs": tr.epochs, "imgsz": a.imgsz, "close_mosaic": a.close_mosaic, "mosaic": a.mosaic,
                                        "scale": a.scale, "mixup": a.mixup, "cos_lr": a.cos_lr, "lrf": a.lrf, "warmup_epochs": a.warmup_epochs,
                                        "amp": bool(tr.amp), "seed": a.seed, "deterministic": a.deterministic, "workers": a.workers,
                                        "cache": a.cache, "data": a.data}), flush=True)
        m = tr.model.module if hasattr(tr.model, "module") else tr.model
        print("[model] " + json.dumps({"class": type(m).__name__, "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
                                       "detect_strides": getattr(m.model[-1], "stride", torch.zeros(0)).tolist()}), flush=True)

    def on_fit_epoch_end(tr):
        loss = {n.replace("_loss", ""): round(float(v), 4) for n, v in zip(tr.loss_names, tr.tloss)} if tr.tloss is not None else {}
        mt = tr.metrics or {}
        lr = round(float(next(iter(tr.lr.values()))), 6) if getattr(tr, "lr", None) else None
        et = float(getattr(tr, "epoch_time", 0) or 0)
        left = (tr.epochs - tr.epoch - 1) * et
        mem = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        print(f"[epoch {tr.epoch + 1}/{tr.epochs}] loss " + " ".join(f"{k}={v}" for k, v in loss.items())
              + f" | val map={100 * mt.get('metrics/mAP50-95(B)', 0):.2f} map50={100 * mt.get('metrics/mAP50(B)', 0):.2f}"
              + f" | lr={lr} | {et / 60:.1f}m eta={left / 3600:.2f}h mem={mem:.1f}G", flush=True)

    y.add_callback("on_train_start", on_train_start)
    y.add_callback("on_fit_epoch_end", on_fit_epoch_end)
