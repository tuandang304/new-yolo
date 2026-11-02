"""Score public VisDrone checkpoints on one GPU with the headline evaluator. No training, val only.

Each checkpoint runs at its training input (from the checkpoint's own train_args) and, when `cost_match` is set, at the
input whose FLOPs are closest to the target. Third-party pickles are loaded only inside this run, and a checkpoint that
fails to load or lacks the ten VisDrone classes is recorded and skipped. Writes public_measurements.json after every model.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import measure as M
from . import predict as P


def count(pred, s: int) -> dict:
    """Params and FLOPs on the inference graph (one-to-one branch removed), FLOPs = 2 x MACs."""
    from torch.utils.flop_counter import FlopCounterMode
    if pred.family == "ultralytics":
        e = copy.deepcopy(pred.model.model).float().cpu().eval()
        d = e.model[-1]
        if getattr(d, "one2one_cv2", None) is not None:
            d.one2one_cv2 = None
            d.one2one_cv3 = None
        with torch.no_grad(), FlopCounterMode(display=False) as f:
            e(torch.zeros(1, 3, s, s))
        return {"params_M": round(sum(p.numel() for p in e.parameters()) / 1e6, 2), "GFLOPs": round(f.get_total_flops() / 1e9, 2), "counter": "FlopCounterMode"}
    net = copy.deepcopy(pred.torch_module()).float().cpu().eval()
    out = {"params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2)}
    try:  # FlopCounterMode's module hooks fail on this model; fvcore counts MACs
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fa = FlopCountAnalysis(net, torch.zeros(1, 3, s, s)).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            out.update(GFLOPs=round(fa.total() * 2 / 1e9, 2), counter="fvcore x2")
    except Exception as e:  # noqa: BLE001
        out.update(GFLOPs=None, flops_error=repr(e)[:200])
    return out


def closest_size(pred, target: float, stride: int = 32) -> int:
    f640 = count(pred, 640)["GFLOPs"]
    est = 640 * math.sqrt(target / f640)
    base = round(est / stride) * stride
    return min((base - stride, base, base + stride), key=lambda s: abs(count(pred, s)["GFLOPs"] - target))


def score(pred, tag: str, s: int, root: Path, gt: dict, ign: dict) -> dict:
    t0 = time.time()
    dets = P.predict_split(pred, root, "valid", s, gt, max_det=1000)
    infer = round(time.time() - t0, 1)
    m = E.coco_eval(gt, dets, ignore=ign)
    row = {"imgsz": s, "infer_sec": infer, "n_dets": len(dets), "multi_md1000": m, **count(pred, s)}
    print("RESULT " + json.dumps({"kind": "public", "tag": tag, "imgsz": s, "AP": m["mAP50_95"], "AP50": m["mAP50"], "AP_tiny": m["AP_tiny"], "AP_small": m["AP_small"], "AP_medium": m["AP_medium"],
                                  "AP_large": m["AP_large"], "params_M": row["params_M"], "GFLOPs": row["GFLOPs"], "sec": round(time.time() - t0, 1)}), flush=True)
    return row


def run(cfg: dict, root: Path, out: Path, device: str = "0", limit: int | None = None) -> dict:
    from huggingface_hub import hf_hub_download
    from PIL import Image
    gt = json.loads((root / "coco_valid.json").read_text())
    ign = json.loads((root / "ignore_valid.json").read_text())
    if limit:  # local smoke test only
        gt = {**gt, "images": gt["images"][:limit]}
        keep = {im["id"] for im in gt["images"]}
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "split": "valid", "limit": limit, "target_GFLOPs": cfg["target_GFLOPs"], "models": {}, "errors": {}}
    path = out / "public_measurements.json"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cfg["models"]:
        tag = c["repo"]
        try:
            ck = hf_hub_download(c["repo"], c["file"])
            if c.get("family", "ultralytics") == "rfdetr":
                m = {"family": "rfdetr", "repo": c["repo"], "file": c["file"], "imgsz": c["imgsz"], "pos_enc": c["pos_enc"], "tag": tag}
                pred = M.load(m, "cuda" if device != "cpu" else "cpu")
                res["models"][f"{tag}@{c['imgsz']}"] = {"repo": c["repo"], "family": "rfdetr", "role": "training", **score(pred, tag, c["imgsz"], root, gt, ign)}
                rows.append({**m, "tag": f"{tag}@{c['imgsz']}"})
            else:
                m = {"family": "ultralytics", "ckpt": ck}
                pred = M.load(m, device)
                missing = [n for n in range(10) if n not in pred.map.values()]
                if missing:
                    raise ValueError(f"lacks VisDrone classes {missing}: {pred.model.names}")
                args = (getattr(pred.model, "ckpt", None) or {}).get("train_args", {}) or {}
                s = int(args["imgsz"]) if isinstance(args.get("imgsz"), int) else c.get("imgsz", 640)
                sizes = [("training", s)]
                if c.get("cost_match"):
                    t = closest_size(pred, cfg["target_GFLOPs"])
                    if t != s:
                        sizes.append(("cost_match", t))
                for role, z in sizes:
                    key = f"{tag}@{z}"
                    res["models"][key] = {"repo": c["repo"], "family": "ultralytics", "role": role, "epochs": args.get("epochs"), "base": args.get("model"), **score(pred, key, z, root, gt, ign)}
                    rows.append({"family": "ultralytics", "ckpt": ck, "imgsz": z, "tag": key})
            M.free(pred)
        except Exception as e:  # noqa: BLE001 - one bad third-party checkpoint must not lose the others
            res["errors"][tag] = repr(e)[:400]
            print(f"[public] {tag} failed: {e!r}", flush=True)
        gc.collect()
        path.write_text(json.dumps(res, indent=1))
    if torch.cuda.is_available():
        names = sorted(im["file_name"] for im in gt["images"])[: cfg.get("latency_images", 200)]
        images = [np.asarray(Image.open(root / "valid" / "images" / n).convert("RGB")) for n in names]
        for m in rows:
            try:
                ts = []
                for _ in range(cfg.get("latency_passes", 2)):
                    t, note = M.latency(m, images, cfg.get("n_warm", 30), cfg.get("n_iter", 200))
                    ts += t
                r = res["models"][m["tag"]]
                r.update(latency_median_ms=round(float(np.median(ts)), 2), latency_p90_ms=round(float(np.percentile(ts, 90)), 2), latency_precision=note)
                print("RESULT " + json.dumps({"kind": "public_latency", "tag": m["tag"], "gpu": res["gpu"], "median_ms": r["latency_median_ms"], "p90_ms": r["latency_p90_ms"], "precision": note}), flush=True)
            except Exception as e:  # noqa: BLE001
                res["models"][m["tag"]]["latency_error"] = repr(e)[:300]
                print(f"[public] latency {m['tag']} failed: {e!r}", flush=True)
            path.write_text(json.dumps(res, indent=1))
    return res
