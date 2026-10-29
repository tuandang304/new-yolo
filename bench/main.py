"""Entry point. Reads configs/node.yaml (the per-node knobs) and runs its task list.

Tasks: data | tpu_probe | parity | train (fine-tune + eval + efficiency + qualitative)
All outputs go to $ORX_OUTPUT_DIR (default ./outputs); metrics are also printed as `RESULT {json}` lines.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import data as D

OUT = Path(os.environ.get("ORX_OUTPUT_DIR", "outputs"))


def log_result(kind: str, payload: dict) -> None:
    print("RESULT " + json.dumps({"kind": kind, **payload}), flush=True)


def env_info() -> dict:
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "nproc": os.cpu_count(), "platform": platform.platform(), "commit": os.environ.get("ORX_COMMIT")}
    for pkg in ("ultralytics", "rfdetr", "pycocotools", "torch_xla", "transformers"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception as e:  # noqa: BLE001
            info[pkg] = f"n/a ({type(e).__name__})"
    try:
        info["mem_GB"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError):
        pass
    return info


def data_root(cfg: dict) -> Path:
    return D.prepare(cfg.get("data_root", "/tmp/visdrone/data"))


def eval_root(cfg: dict, root: Path) -> Path:
    """Smoke runs evaluate on a 48-image subset of val/test; full runs on the complete splits."""
    if cfg.get("smoke"):
        return D.make_subset(root, root.parent / "data_smoke", {"train": 64, "valid": 48, "test": 48})
    return root


def evaluate_ckpt(tag: str, family: str, ckpt: str, root: Path, imgszs: list[int], splits: list[str],
                  device: str, out: Path, efficiency_imgsz: int = 640, qual: bool = True) -> dict:
    from . import efficiency, evaluate, predict, qualitative
    pred = predict.load_predictor(family, ckpt, device=device)
    res: dict = {"tag": tag, "family": family, "ckpt": ckpt, "metrics": {}}
    out.mkdir(parents=True, exist_ok=True)
    test_dets_640 = None
    for split in splits:
        gt = json.loads((root / f"coco_{split}.json").read_text())
        ign = json.loads((root / f"ignore_{split}.json").read_text())
        for s in imgszs:
            t0 = time.time()
            dets = predict.predict_split(pred, root, split, s, gt)
            with gzip.open(out / f"dets_{tag}_{split}_{s}.json.gz", "wt") as f:
                json.dump(dets, f)
            m = evaluate.coco_eval(gt, dets)
            mi = evaluate.coco_eval(gt, dets, ignore=ign)
            key = f"{split}@{s}"
            res["metrics"][key] = {"std": m, "ignore_aware": {k: v for k, v in mi.items() if k != "per_class"},
                                   "infer_sec": round(time.time() - t0, 1)}
            print(f"[eval] {tag} {key}: {evaluate.fmt(m)} | ignore-aware mAP50_95={mi['mAP50_95']:.2f}", flush=True)
            log_result("eval", {"tag": tag, "split": split, "imgsz": s, "mAP50_95": m["mAP50_95"], "mAP50": m["mAP50"],
                                "AP_tiny": m["AP_tiny"], "AP_small": m["AP_small"], "ign_mAP50_95": mi["mAP50_95"]})
            if split == "test" and s == efficiency_imgsz:
                test_dets_640 = (gt, ign, dets)
    if qual and test_dets_640 is not None:
        gt, ign, dets = test_dets_640
        samples = D.figure_samples(root) if len(gt["images"]) >= 64 else sorted(im["file_name"] for im in gt["images"])[:4]
        res["qualitative"] = qualitative.dump(root, gt, ign, dets, samples, out / "qualitative", tag)["per_image"]
    res["efficiency"] = efficiency.params_flops(pred.torch_module(), efficiency_imgsz)
    if device.startswith("cuda") or device.isdigit():
        sample = str(next((root / "test" / "images").iterdir()))
        res["efficiency"].update(efficiency.latency(pred, sample, efficiency_imgsz))
    log_result("efficiency", {"tag": tag, **res["efficiency"]})
    (out / f"result_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def task_parity(cfg: dict, root: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 4)
    outs = []
    for p in cfg["parity"]:
        ckpt = hf_hub_download(p["repo"], p["file"])
        dev = ("0" if device == "cuda" else "cpu") if p["family"] == "ultralytics" else device
        outs.append(evaluate_ckpt(p["name"], p["family"], ckpt, eval_root(cfg, root), p["imgsz"], cfg.get("splits", ["valid", "test"]),
                                  dev, OUT / "parity", efficiency_imgsz=640))
    return outs


def task_tpu_probe(cfg: dict, root: Path) -> dict:
    rep: dict = {}
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        rep["xla_device"] = str(xm.xla_device())
        try:
            import torch_xla.runtime as xr
            rep["xla_world"] = xr.global_runtime_device_count()
        except Exception as e:  # noqa: BLE001
            rep["xla_world_err"] = repr(e)[:200]
        rep["torch_xla"] = torch_xla.__version__
    except Exception as e:  # noqa: BLE001
        rep["torch_xla_error"] = repr(e)[:300]
    if cfg.get("probe_rfdetr_xla", True) and "xla_device" in rep:
        sub = D.make_subset(D.prepare(cfg.get("data_root", "/tmp/visdrone/data")), Path("/tmp/visdrone/data_probe"),
                            {"train": 64, "valid": 16, "test": 16})
        cmd = [sys.executable, "-m", "bench.main", "--probe-rfdetr-xla", str(sub)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, timeout=int(cfg.get("probe_timeout_s", 1800)), capture_output=True, text=True)
            rep["rfdetr_xla_rc"] = p.returncode
            rep["rfdetr_xla_tail"] = (p.stdout + p.stderr)[-3000:]
        except subprocess.TimeoutExpired as e:
            rep["rfdetr_xla_rc"] = "timeout"
            rep["rfdetr_xla_tail"] = str(e.stdout or "")[-2000:]
        rep["rfdetr_xla_sec"] = round(time.time() - t0, 1)
        print(rep["rfdetr_xla_tail"], flush=True)
    log_result("tpu_probe", {k: v for k, v in rep.items() if k != "rfdetr_xla_tail"})
    (OUT / "tpu_probe.json").write_text(json.dumps(rep, indent=2))
    return rep


def probe_rfdetr_xla(sub: str) -> None:
    import rfdetr
    t0 = time.time()
    m = rfdetr.RFDETRNano()
    m.train(dataset_dir=sub, epochs=1, resolution=640, batch_size=4, grad_accum_steps=1, device="xla",
            output_dir="/tmp/rfdetr_xla_probe", num_workers=2)
    print(f"PROBE_OK rfdetr xla 1 epoch on 64 imgs: {time.time() - t0:.1f}s", flush=True)


def task_train(cfg: dict, root: Path, seed: int, gpu: int) -> dict:
    from .train import FAMILY, train
    t = cfg["train"]
    model = t["model"]
    out = OUT / f"{model}_s{seed}"
    epochs = 1 if cfg.get("smoke") else t.get("epochs", 100)  # smoke: 1 full epoch -> real epoch time
    dev = str(gpu) if FAMILY[model] == "ultralytics" else f"cuda:{gpu}"
    ckpt, info = train(model, root, out, epochs, t.get("imgsz", 640), seed, dev, t.get("extra"))
    log_result("train", {"model": model, "seed": seed, **info})
    ev = cfg.get("eval", {})
    res = evaluate_ckpt(f"{model}_s{seed}", FAMILY[model], ckpt, eval_root(cfg, root), ev.get("imgsz", [640, 1280]),
                        ev.get("splits", ["valid", "test"]), dev if FAMILY[model] == "ultralytics" else f"cuda:{gpu}",
                        out, efficiency_imgsz=t.get("imgsz", 640))
    res["train"] = info
    (out / f"result_{model}_s{seed}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/node.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--probe-rfdetr-xla")
    a = ap.parse_args()
    if a.probe_rfdetr_xla:
        return probe_rfdetr_xla(a.probe_rfdetr_xla)
    cfg = yaml.safe_load(Path(a.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    print("[cfg] " + json.dumps(cfg), flush=True)
    info = env_info()
    print("[env] " + json.dumps(info), flush=True)
    (OUT / "env.json").write_text(json.dumps(info, indent=2))
    t0 = time.time()
    root = data_root(cfg)
    log_result("data", {"root": str(root), "manifest": json.loads((root / "manifest.json").read_text())["splits"],
                        "prep_sec": round(time.time() - t0, 1)})
    summary: dict = {"config": cfg, "env": info}
    for task in cfg["tasks"]:
        t1 = time.time()
        if task == "data":
            continue
        if task == "tpu_probe":
            summary["tpu_probe"] = task_tpu_probe(cfg, root)
        elif task == "parity":
            summary["parity"] = task_parity(cfg, root)
        elif task == "train":
            seeds = [a.seed] if a.seed is not None else cfg["train"].get("seeds", [0])
            if len(seeds) == 1:
                summary["train"] = [task_train(cfg, root, seeds[0], a.gpu)]
            else:  # one process per seed, one GPU each (no DDP)
                procs = [subprocess.Popen([sys.executable, "-m", "bench.main", "--config", a.config, "--seed", str(s),
                                           "--gpu", str(i)]) for i, s in enumerate(seeds)]
                summary["train_rc"] = [p.wait() for p in procs]
                if any(summary["train_rc"]):
                    raise SystemExit(f"seed subprocess failed: {summary['train_rc']}")
        print(f"[task] {task} done in {(time.time() - t1) / 60:.1f} min", flush=True)
    (OUT / ("summary.json" if a.seed is None else f"summary_s{a.seed}.json")).write_text(json.dumps(summary, indent=2, default=str))
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
