"""Re-measure every reported number on one GPU: val detections, the headline evaluator and FP16 latency. No training.

Headline setting: multi-label NMS, at most 1000 detections per image, ignore-aware scoring, COCOeval maxDets 500.
Writes $ORX_OUTPUT_DIR/measurements.json after every step, so a partial run keeps what it finished.
Revision options (IEEE Access, 2026-11-05): cfg["split"] picks valid or test; a model's `grid` adds the
single-/multi-label cap grid (300/500/1000); cfg["latency_square"] pads each image to a square before the
Ultralytics letterbox so the network sees exactly imgsz x imgsz; cfg["rfdetr_complexity"] recounts RF-DETR-Nano
parameters and FLOPs at its published configuration.
"""
from __future__ import annotations

import gc
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import evaluate as E
from . import predict as P

MULTI_LABEL = {"on": True}
RF_REPO, RF_FILE = "dronefreak/visdrone-rfdetr-nano", "checkpoint_best_total.pth"


def patch_nms() -> None:
    """One switchable wrapper; the `_multi_label` mark stops YoloPredictor from wrapping it again with multi-label forced on."""
    import ultralytics.utils.nms as nms
    if getattr(nms.non_max_suppression, "_multi_label", False):
        return
    orig = nms.non_max_suppression
    nms.non_max_suppression = lambda *a, **k: orig(*a, **{**k, "multi_label": MULTI_LABEL["on"]})
    nms.non_max_suppression._multi_label = True


def top_k(dets: list[dict], k: int) -> list[dict]:
    """Top k per image by score. Equals running NMS or top-k selection with that cap, since both keep the best-scored results."""
    by = defaultdict(list)
    for d in dets:
        by[d["image_id"]].append(d)
    out = []
    for v in by.values():
        out += sorted(v, key=lambda d: -d["score"])[:k]
    return out


def load(m: dict, device: str, top: int = 1000):
    if m["family"] == "ultralytics":
        from . import pra  # noqa: F401  registers the custom modules that the proposed checkpoints pickle
        patch_nms()
        return P.load_predictor("ultralytics", m["ckpt"], device=device)
    import rfdetr
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download(RF_REPO, RF_FILE)
    pred = P.RFDETRPredictor.__new__(P.RFDETRPredictor)
    # The checkpoint is stripped of its config, so the published resolution and positional-encoding size are passed explicitly.
    pred.model = rfdetr.RFDETR.from_checkpoint(ck, trust_checkpoint=True, device=device, resolution=m["imgsz"], positional_encoding_size=m["pos_enc"])
    pred.model.model.postprocess.num_select = top
    pred.device, pred.half = device, False
    pred.map = P._name_map(pred.model.class_names)
    return pred


def free(pred) -> None:
    del pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def latency(m: dict, images: list[np.ndarray], n_warm: int, n_iter: int) -> tuple[list[float], str]:
    """End-to-end ms per image (pre + forward + post) at conf 0.25, FP16, batch 1, cycling over distinct val images."""
    pred = load(m, "0" if m["family"] == "ultralytics" else "cuda", top=300)
    note = "fp16"
    if m["family"] == "ultralytics":
        call = lambda im: pred.model.predict(im[..., ::-1].copy(), imgsz=m["imgsz"], conf=0.25, half=True, device=0, verbose=False)
    else:
        try:
            pred.model.inference(compile=False, dtype=torch.float16)
        except Exception as e:  # noqa: BLE001 - recorded, so the row is not silently reported as fp16
            note = f"fp32 ({e!r})"[:200]
        call = lambda im: pred.model.predict(im, threshold=0.25, shape=(m["imgsz"], m["imgsz"]), include_source_image=False)
    for i in range(n_warm):
        call(images[i % len(images)])
    torch.cuda.synchronize()
    ts = []
    for i in range(n_iter):
        t = time.perf_counter()
        call(images[i % len(images)])
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    free(pred)
    return ts, note


def latency_rounds(cfg: dict, rows: list[dict], images: list[np.ndarray], res: dict, path: Path) -> None:
    """Interleaved timing: `rounds` rounds, model order shuffled per round, each model reloaded and warmed up per round.
    Per round and model: end-to-end ms (pre + FP16 forward + NMS at conf 0.25) over the val images, and forward-only ms
    (fixed 1x3xSxS FP16 tensor through the fused network, CUDA events, no NMS). Reports the median of round medians."""
    import random
    rng = random.Random(cfg.get("order_seed", 0))
    n_warm, n_iter, n_rounds = cfg.get("n_warm", 30), cfg.get("n_iter", 200), cfg["rounds"]
    per = defaultdict(lambda: {"e2e": [], "fwd": [], "e2e_p90": []})
    for rd in range(n_rounds):
        order = rows[:]
        rng.shuffle(order)
        print(f"[latency] round {rd + 1}/{n_rounds} order={[m['tag'] for m in order]}", flush=True)
        for m in order:
            tag = m["tag"]
            try:
                ts, note = latency(m, images, n_warm, n_iter)
                per[tag]["e2e"].append(float(np.median(ts)))
                per[tag]["e2e_p90"].append(float(np.percentile(ts, 90)))
                if m["family"] == "ultralytics":
                    per[tag]["fwd"].append(forward_ms(m, n_warm, n_iter))
            except Exception as e:  # noqa: BLE001 - one model must not lose the others
                print(f"[latency] {tag} round {rd + 1} failed: {e!r}", flush=True)
                res["models"].setdefault(tag, {})["latency_error"] = repr(e)[:300]
                continue
            print(f"[latency] r{rd + 1} {tag} e2e={per[tag]['e2e'][-1]:.2f} fwd={per[tag]['fwd'][-1] if per[tag]['fwd'] else float('nan'):.2f} ms", flush=True)
        for tag, v in per.items():
            row = res["models"].setdefault(tag, {})
            for k in ("e2e", "fwd"):
                if v[k]:
                    row[f"{k}_round_ms"] = [round(x, 2) for x in v[k]]
                    row[f"{k}_median_ms"] = round(float(np.median(v[k])), 2)
                    row[f"{k}_min_ms"], row[f"{k}_max_ms"] = round(min(v[k]), 2), round(max(v[k]), 2)
            row["e2e_p90_ms"] = round(float(np.median(v["e2e_p90"])), 2)
        path.write_text(json.dumps(res, indent=1))
    for tag, row in res["models"].items():
        print("RESULT " + json.dumps({"kind": "latency_rounds", "tag": tag, "gpu": res["gpu"], **{k: row.get(k) for k in
              ("e2e_median_ms", "e2e_min_ms", "e2e_max_ms", "e2e_p90_ms", "fwd_median_ms", "fwd_min_ms", "fwd_max_ms")}}), flush=True)


def forward_ms(m: dict, n_warm: int, n_iter: int) -> float:
    """Median forward-only ms of the fused FP16 network that Ultralytics predict() builds, on a fixed square tensor."""
    pred = load(m, "0", top=300)
    pred.model.predict(np.zeros((m["imgsz"], m["imgsz"], 3), np.uint8), imgsz=m["imgsz"], conf=0.25, half=True, device=0, verbose=False)
    net = pred.model.predictor.model  # AutoBackend holding the fused FP16 model
    x = torch.zeros(1, 3, m["imgsz"], m["imgsz"], device="cuda", dtype=torch.half if net.fp16 else torch.float)
    ts = []
    with torch.inference_mode():
        for _ in range(n_warm):
            net(x)
        torch.cuda.synchronize()
        for _ in range(n_iter):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            net(x)
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    free(pred)
    return float(np.median(ts))


def square(im: np.ndarray) -> np.ndarray:
    """Pad an HxWx3 image to a square with gray 114 (the Ultralytics letterbox colour), keeping it top-left."""
    h, w = im.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), 114, dtype=im.dtype)
    out[:h, :w] = im
    return out


def rfdetr_complexity(imgsz: int = 640, pos_enc: int = 40) -> dict:
    """Params and FLOPs (2 x fvcore MACs) of the public RF-DETR-Nano, loaded with and without its published config."""
    import rfdetr
    from fvcore.nn import FlopCountAnalysis
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download(RF_REPO, RF_FILE)
    res = {}
    for name, kw in (("published", {"resolution": imgsz, "positional_encoding_size": pos_enc}), ("fallback_default", {})):
        m = rfdetr.RFDETR.from_checkpoint(ck, trust_checkpoint=True, device="cpu", **kw)
        net = m.model.model.eval()
        r = {"params": int(sum(p.numel() for p in net.parameters()))}
        with torch.no_grad():
            fa = FlopCountAnalysis(net, torch.zeros(1, 3, imgsz, imgsz)).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            r["GFLOPs"] = round(fa.total() * 2 / 1e9, 2)
        r["pos_embed"] = [list(v.shape) for k, v in net.state_dict().items() if k.endswith("position_embeddings")]
        res[name] = r
        print("RESULT " + json.dumps({"kind": "rfdetr_complexity", "config": name, "imgsz": imgsz, **r}), flush=True)
        del m, net
        gc.collect()
    return res


def run(cfg: dict, root: Path, out: Path, device: str = "0", limit: int | None = None) -> dict:
    from PIL import Image
    split = cfg.get("split", "valid")
    gt = json.loads((root / f"coco_{split}.json").read_text())
    ign = json.loads((root / f"ignore_{split}.json").read_text())
    if limit:  # local smoke test only
        gt = {**gt, "images": gt["images"][:limit]}
        keep = {im["id"] for im in gt["images"]}
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "split": split, "limit": limit, "models": {}}
    path = out / "measurements.json"
    out.mkdir(parents=True, exist_ok=True)
    if cfg.get("rfdetr_complexity"):
        try:
            res["rfdetr_complexity"] = rfdetr_complexity()
        except Exception as e:  # noqa: BLE001 - must not lose the AP measurements
            res["rfdetr_complexity"] = {"error": repr(e)[:300]}
            print(f"[rfdetr_complexity] failed: {e!r}", flush=True)
        path.write_text(json.dumps(res, indent=1))

    for m in (cfg["models"] if cfg.get("ap", True) else []):
        tag, sz = m["tag"], m["imgsz"]
        t0 = time.time()
        MULTI_LABEL["on"] = True
        pred = load(m, device if m["family"] == "ultralytics" else ("cuda" if device != "cpu" else "cpu"))
        dets = P.predict_split(pred, root, split, sz, gt, max_det=1000)
        row = {"imgsz": sz, "family": m["family"], "infer_sec": round(time.time() - t0, 1), "n_dets": len(dets)}
        row["multi_md1000"] = E.coco_eval(gt, dets, ignore=ign)
        row["multi_md500"] = {k: v for k, v in E.coco_eval(gt, top_k(dets, 300 if m["family"] == "rfdetr" else 500), ignore=ign).items() if k != "per_class"}
        if m.get("grid") or m.get("single"):
            if m["family"] != "rfdetr":
                MULTI_LABEL["on"] = False
                d1 = P.predict_split(pred, root, split, sz, gt, max_det=500)
                row["single_md500"] = {k: v for k, v in E.coco_eval(gt, d1, ignore=ign).items() if k != "per_class"}
                MULTI_LABEL["on"] = True
                if m.get("grid"):
                    row["single_md300"] = {k: v for k, v in E.coco_eval(gt, top_k(d1, 300), ignore=ign).items() if k != "per_class"}
            if m.get("grid"):
                row["multi_md300"] = {k: v for k, v in E.coco_eval(gt, top_k(dets, 300), ignore=ign).items() if k != "per_class"}
                row["multi_md500_true"] = {k: v for k, v in E.coco_eval(gt, top_k(dets, 500), ignore=ign).items() if k != "per_class"}
        if m.get("save_dets"):
            with gzip.open(out / f"dets_{tag}_{split}_{sz}.json.gz", "wt") as f:
                json.dump(dets, f)
        free(pred)
        res["models"][tag] = row
        path.write_text(json.dumps(res, indent=1))
        h = row["multi_md1000"]
        print("RESULT " + json.dumps({"kind": "measure", "tag": tag, "split": split, "imgsz": sz, "gpu": res["gpu"], "AP": h["mAP50_95"], "AP50": h["mAP50"], "AP_tiny": h["AP_tiny"],
                                      "AP_small": h["AP_small"], "AP_medium": h["AP_medium"], "AP_large": h["AP_large"], "sec": round(time.time() - t0, 1)}), flush=True)

    if not torch.cuda.is_available() or not cfg.get("latency", True):
        return res
    vgt = gt if split == "valid" else json.loads((root / "coco_valid.json").read_text())
    names = sorted(im["file_name"] for im in vgt["images"])[: cfg.get("latency_images", 200)]
    images = [np.asarray(Image.open(root / "valid" / "images" / n).convert("RGB")) for n in names]
    if cfg.get("latency_square"):
        images = [square(im) for im in images]
    res["latency_input"] = "square (padded before letterbox)" if cfg.get("latency_square") else "rectangular letterbox"
    rows = [m for m in cfg["models"] if m.get("latency")]
    if cfg.get("rounds"):
        latency_rounds(cfg, rows, images, res, path)
        return res
    pooled = defaultdict(list)
    for p in range(cfg.get("latency_passes", 2)):  # two passes in opposite order, to expose drift over the session
        for m in (rows if p % 2 == 0 else rows[::-1]):
            try:
                ts, note = latency(m, images, cfg.get("n_warm", 30), cfg.get("n_iter", 200))
            except Exception as e:  # noqa: BLE001 - one model's latency must not lose the finished AP measurements
                print(f"[latency] {m['tag']} failed: {e!r}", flush=True)
                res["models"][m["tag"]]["latency_error"] = repr(e)[:300]
                continue
            pooled[m["tag"]] += ts
            res["models"][m["tag"]].setdefault("latency_pass_median_ms", []).append(round(float(np.median(ts)), 2))
            res["models"][m["tag"]]["latency_precision"] = note
    for tag, ts in pooled.items():
        r = res["models"][tag]
        r["latency_median_ms"], r["latency_p90_ms"] = round(float(np.median(ts)), 2), round(float(np.percentile(ts, 90)), 2)
        print("RESULT " + json.dumps({"kind": "latency", "input": res["latency_input"], "tag": tag, "gpu": res["gpu"], "median_ms": r["latency_median_ms"], "p90_ms": r["latency_p90_ms"],
                                      "passes": r["latency_pass_median_ms"], "precision": r["latency_precision"]}), flush=True)
    path.write_text(json.dumps(res, indent=1))
    return res
