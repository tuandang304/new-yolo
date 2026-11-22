"""Jetson edge study, inference only: TensorRT FP16 engines (batch 1/4/8) and INT8 engines (batch 1), timed in every
power mode with board power from tegrastats (VDD_IN), plus AP of the FP16 and INT8 engines on val.

Per power mode: set the mode, fix clocks (jetson_clocks), read idle board power, then `rounds` rounds with the model
order shuffled. Per model and round: end-to-end latency per precision (letterbox + engine + NMS at conf 0.25, batch 1,
the first 200 sorted val images after 30 warm-up) and network-only throughput of every engine on a fixed tensor.
Energy per image = mean VDD_IN power over the timed window x window time / images. Engines are built once, each in a
fresh process (equal free memory), and cached by weights hash, settings and TensorRT version. INT8 is calibrated on
`calib_images` random train images (seed 0), never on the val images it is scored on.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODES = {"15W": 0, "25W": 1, "MAXN_SUPER": 2}  # nvpmodel ids on Orin Nano Super
CACHE = Path.home() / "engine-cache"
REPO = Path(__file__).resolve().parent.parent
BUILD = ("import sys; from bench import pra; from ultralytics import YOLO; a = sys.argv; "
         "kw = dict(data=a[5]) if a[3] == 'int8' else {}; "
         "YOLO(a[1]).export(format='engine', imgsz=int(a[2]), quantize=8 if a[3] == 'int8' else 16, batch=int(a[4]), "
         "device=0, simplify=True, verbose=False, **kw)")


def result(kind: str, **kw) -> None:
    print("RESULT " + json.dumps({"kind": kind, **kw}), flush=True)


def progress(phase: str, step: int, of: int) -> None:
    print("::orx:: " + json.dumps({"phase": phase, "step": step, "of": of}), flush=True)


def engine_path(m: dict, prec: str, batch: int) -> Path:
    import tensorrt
    sha = hashlib.sha256(Path(m["ckpt"]).read_bytes()).hexdigest()[:10]
    return CACHE / f"{m['tag']}_{prec}_b{batch}_{m['imgsz']}_{sha}_trt{tensorrt.__version__}.engine"


def build(m: dict, prec: str, batch: int, calib: Path | None) -> Path:
    eng = engine_path(m, prec, batch)
    if eng.exists():
        result("engine", tag=m["tag"], prec=prec, batch=batch, cached=True, mb=round(eng.stat().st_size / 2**20, 2))
        return eng
    CACHE.mkdir(parents=True, exist_ok=True)
    pt = eng.with_suffix(".pt")  # export writes <stem>.engine next to its source
    pt.write_bytes(Path(m["ckpt"]).read_bytes())
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", BUILD, str(pt), str(m["imgsz"]), prec, str(batch), str(calib)],
                       cwd=REPO, capture_output=True, text=True)
    text = r.stdout + r.stderr
    ok = r.returncode == 0 and eng.exists()
    if not ok:
        print("\n".join(text.strip().splitlines()[-15:]), flush=True)
        raise RuntimeError(f"engine build failed: {m['tag']} {prec} b{batch} rc={r.returncode}")
    result("engine", tag=m["tag"], prec=prec, batch=batch, cached=False, sec=round(time.time() - t0),
           mb=round(eng.stat().st_size / 2**20, 2), skipped_tactics=text.count("insufficient memory"))
    return eng


class Power:
    """Board input power (VDD_IN, mW) sampled by tegrastats every `ms` while the block runs."""

    def __init__(self, ms: int = 100):
        self.ms = ms

    def __enter__(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.ms)], stdout=subprocess.PIPE, text=True)
        time.sleep(0.3)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        out, _ = self.proc.communicate()
        mw = [float(line.split("VDD_IN ")[1].split("mW")[0]) for line in out.splitlines() if "VDD_IN " in line]
        self.watts = float(np.mean(mw)) / 1000 if mw else float("nan")
        self.samples = len(mw)
        return False


def set_mode(name: str) -> str:
    subprocess.run(["sudo", "-n", "nvpmodel", "-m", str(MODES[name])], input="no\n", text=True, capture_output=True, check=True)
    time.sleep(2)
    subprocess.run(["sudo", "-n", "jetson_clocks"], check=True)
    time.sleep(5)
    q = subprocess.run(["sudo", "-n", "nvpmodel", "-q"], capture_output=True, text=True).stdout.splitlines()[0]
    assert q.strip().endswith(name), f"power mode is '{q}', wanted {name}"
    gpu = [l for l in subprocess.run(["sudo", "-n", "jetson_clocks", "--show"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU ")]
    return gpu[0] if gpu else "?"


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def e2e(eng: Path, imgsz: int, images: list[np.ndarray], n_warm: int, n_iter: int) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(eng), task="detect")
    call = lambda im: model.predict(im, imgsz=imgsz, conf=0.25, device=0, verbose=False)
    for i in range(n_warm):
        call(images[i % len(images)])
    torch.cuda.synchronize()
    ts = []
    with Power() as pw:
        t_all = time.perf_counter()
        for i in range(n_iter):
            t = time.perf_counter()
            call(images[i % len(images)])
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        total = time.perf_counter() - t_all
    del model
    free()
    return {"ms": round(float(np.median(ts)), 2), "p90": round(float(np.percentile(ts, 90)), 2), "W": round(pw.watts, 3),
            "mJ": round(pw.watts * total / n_iter * 1000, 1), "samples": pw.samples}


def net(eng: Path, batch: int, imgsz: int, min_s: float = 5.0, n_warm: int = 20) -> dict:
    from ultralytics.nn.autobackend import AutoBackend
    be = AutoBackend(str(eng), device=torch.device("cuda:0"), fp16=True, verbose=False)
    x = torch.zeros(batch, 3, imgsz, imgsz, device="cuda:0", dtype=torch.half if be.fp16 else torch.float)
    for _ in range(n_warm):
        be(x)
    torch.cuda.synchronize()
    n = 0
    with Power() as pw:
        t0 = time.perf_counter()
        while n < 50 or time.perf_counter() - t0 < min_s:
            be(x)
            n += 1
            if n % 10 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    del be, x
    free()
    return {"ms_per_batch": round(total / n * 1000, 2), "img_s": round(batch * n / total, 1), "W": round(pw.watts, 3),
            "mJ": round(pw.watts * total / (batch * n) * 1000, 1), "samples": pw.samples}


def ap(eng: Path, imgsz: int, root: Path, gt: dict, ign: dict) -> dict:
    from . import evaluate as E
    from . import predict as P
    pred = P.load_predictor("ultralytics", str(eng), device="0")
    ids = {im["file_name"]: im["id"] for im in gt["images"]}
    paths = [str(root / "valid" / "images" / im["file_name"]) for im in gt["images"]]
    dets = []
    for p, xyxy, sc, cl in pred(paths, imgsz=imgsz, max_det=1000, batch=1):  # static batch-1 engine
        for (x0, y0, x1, y1), s, c in zip(xyxy.tolist(), sc.tolist(), cl.tolist()):
            dets.append({"image_id": ids[Path(p).name], "category_id": int(c), "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
    del pred
    free()
    r = E.coco_eval(gt, dets, ignore=ign)
    return {k: r[k] for k in ("mAP50_95", "mAP50", "AP_tiny", "AP_small", "AP_medium", "AP_large")}


def calib_yaml(root: Path, n: int) -> Path:
    from . import data as D
    sub = D.make_subset(root, root.parent / f"data_calib{n}", {"train": n})
    y = sub / "calib.yaml"  # val points at the calibration images: Ultralytics calibrates on the 'val' key
    y.write_text(f"path: {sub.resolve()}\ntrain: train/images\nval: train/images\nnc: {len(D.NAMES)}\nnames:\n"
                 + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(D.NAMES)))
    return y


def run(cfg: dict, root: Path, out: Path) -> dict:
    from PIL import Image
    from . import measure  # noqa: F401  registers the Ultralytics warning filter
    out.mkdir(parents=True, exist_ok=True)
    path = out / "jetson.json"
    models, batches, precs = cfg["models"], cfg.get("batches", [1]), cfg.get("precisions", ["fp16"])
    res: dict = {"config": cfg, "builds": [], "modes": {}, "ap": {}}

    calib = calib_yaml(root, cfg.get("calib_images", 512)) if "int8" in precs else None
    # INT8 first: its calibration is the least tested path, so a failure shows before the FP16 builds
    plan = [(m, "int8", 1) for m in models if "int8" in precs] + [(m, "fp16", b) for m in models for b in batches]
    engines: dict = {}
    for i, (m, prec, b) in enumerate(plan):
        progress("build", i, len(plan))
        engines[(m["tag"], prec, b)] = build(m, prec, b, calib)
    gt = json.loads((root / "coco_valid.json").read_text())
    names = sorted(im["file_name"] for im in gt["images"])[: cfg.get("latency_images", 200)]
    images = [np.asarray(Image.open(root / "valid" / "images" / n).convert("RGB"))[..., ::-1].copy() for n in names]

    rng = random.Random(cfg.get("order_seed", 0))
    n_warm, n_iter, rounds = cfg.get("n_warm", 30), cfg.get("n_iter", 200), cfg.get("rounds", 3)
    steps, step = len(cfg["modes"]) * rounds, 0
    try:
        for mode in cfg["modes"]:
            gpu = set_mode(mode)
            with Power() as idle:
                time.sleep(cfg.get("idle_s", 10))
            res["modes"][mode] = {"gpu": gpu, "idle_W": round(idle.watts, 3), "rows": []}
            result("mode", mode=mode, gpu=gpu, idle_W=round(idle.watts, 3))
            for rd in range(rounds):
                progress(f"time {mode}", step, steps)
                step += 1
                order = models[:]
                rng.shuffle(order)
                for m in order:
                    for prec in precs:
                        row = {"mode": mode, "round": rd, "tag": m["tag"], "prec": prec, "measure": "e2e", "batch": 1,
                               **e2e(engines[(m["tag"], prec, 1)], m["imgsz"], images, n_warm, n_iter)}
                        res["modes"][mode]["rows"].append(row)
                        result("jetson", **row)
                    for (tag, prec, b), eng in engines.items():
                        if tag == m["tag"]:
                            row = {"mode": mode, "round": rd, "tag": tag, "prec": prec, "measure": "net", "batch": b, **net(eng, b, m["imgsz"])}
                            res["modes"][mode]["rows"].append(row)
                            result("jetson", **row)
                    path.write_text(json.dumps(res, indent=1))
    finally:
        set_mode(cfg.get("final_mode", "MAXN_SUPER"))

    if cfg.get("ap", True):
        ign = json.loads((root / "ignore_valid.json").read_text())
        jobs = [(m, p) for m in models for p in precs]
        for i, (m, prec) in enumerate(jobs):
            progress("ap", i, len(jobs))
            r = ap(engines[(m["tag"], prec, 1)], m["imgsz"], root, gt, ign)
            res["ap"][f"{m['tag']}/{prec}"] = r
            result("jetson_ap", tag=m["tag"], prec=prec, mode=cfg.get("final_mode", "MAXN_SUPER"), **r)
            path.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":  # self-check of the VDD_IN parser on a recorded tegrastats line
    line = "RAM 2352/7485MB GR3D_FREQ 0% VDD_IN 6894mW/6894mW/6894mW VDD_CPU_GPU_CV 1569mW/1569mW/1569mW"
    assert float(line.split("VDD_IN ")[1].split("mW")[0]) == 6894.0
    print("ok")
