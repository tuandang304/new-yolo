"""TensorRT latency, board energy and accuracy on an NVIDIA Jetson (the paper used an Orin Nano 8 GB Super, JetPack 7.2).

    sudo nvpmodel -m 2 && sudo jetson_clocks          # power mode first (Orin Nano Super: 0 = 15 W, 1 = 25 W, 2 = MAXN SUPER)
    python benchmark_jetson.py --weights weights/guidedet_yolo_s0.pt --root datasets/visdrone --int8 --ap

Per weights file and precision, a batch-1 TensorRT engine is built in a fresh process (so every engine is built with
the same free memory; a loaded process can starve TensorRT of tactics) and cached next to --cache. Then:
  end-to-end  pre-processing + engine + NMS per image (Ultralytics predict, conf 0.25) on the first --n-images val
              images after --warm warm-up calls; median of --iters calls, over --rounds rounds with shuffled order
  energy      mean board input power (tegrastats VDD_IN, every 100 ms) during timing x wall time per image,
              idle draw included
  AP          (--ap) the engine scored on the val split with the evaluator of val.py at batch 1
INT8 engines (--int8) are calibrated on --calib-images random training images (seed 0), never on val.
TensorRT 10.x is required by ultralytics 8.4.192 (on JetPack 7: pip install tensorrt-cu13==10.16.1.11).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BUILD = ("import sys; from guidedet import load_yolo; a = sys.argv; "
         "kw = dict(int8=True, data=a[4]) if a[3] == 'int8' else dict(half=True); "
         "load_yolo(a[1]).export(format='engine', imgsz=int(a[2]), batch=1, device=0, simplify=True, verbose=False, **kw)")


def vdd_in_mw(line: str) -> float | None:
    return float(line.split("VDD_IN ")[1].split("mW")[0]) if "VDD_IN " in line else None


class Power:
    """Board input power (W) averaged over tegrastats samples taken while the block runs."""

    def __enter__(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)
        time.sleep(0.3)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        mw = [v for v in map(vdd_in_mw, self.proc.communicate()[0].splitlines()) if v is not None]
        self.watts = float(np.mean(mw)) / 1000 if mw else float("nan")
        return False


def calib_yaml(root: Path, n: int) -> Path:
    """`n` random training images (seed 0) as the calibration set; Ultralytics calibrates on the 'val' key."""
    out = root / f"calib{n}"
    y = out / "calib.yaml"
    if y.exists():
        return y
    names = sorted(p.name for p in (root / "train" / "images").iterdir())
    for sub in ("images", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for name in sorted(random.Random(0).sample(names, min(n, len(names)))):
        os.symlink((root / "train" / "images" / name).resolve(), out / "images" / name)
        lab = root / "train" / "labels" / Path(name).with_suffix(".txt").name
        if lab.exists():
            (out / "labels" / lab.name).write_text(lab.read_text())
    from guidedet import NAMES
    y.write_text(f"path: {out.resolve()}\ntrain: images\nval: images\nnames: {json.dumps(dict(enumerate(NAMES)))}\n")
    return y


def build(weights: Path, prec: str, imgsz: int, cache: Path, calib: Path | None) -> Path:
    import tensorrt
    eng = cache / f"{weights.stem}_{prec}_{imgsz}_trt{tensorrt.__version__}.engine"
    if eng.exists():
        return eng
    cache.mkdir(parents=True, exist_ok=True)
    src = eng.with_suffix(".pt")  # export writes <stem>.engine next to its source
    src.write_bytes(weights.read_bytes())
    r = subprocess.run([sys.executable, "-c", BUILD, str(src), str(imgsz), prec, str(calib)],
                       cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
    if r.returncode or not eng.exists():
        print("\n".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
        raise SystemExit(f"engine build failed: {weights.name} {prec}")
    print(f"built {eng.name} ({eng.stat().st_size / 2**20:.1f} MB)", flush=True)
    return eng


def end_to_end(eng: Path, imgsz: int, images: list, warm: int, iters: int) -> dict:
    import torch
    from ultralytics import YOLO
    model = YOLO(str(eng), task="detect")
    call = lambda im: model.predict(im, imgsz=imgsz, conf=0.25, device=0, verbose=False)
    for i in range(warm):
        call(images[i % len(images)])
    torch.cuda.synchronize()
    ts = []
    with Power() as pw:
        t0 = time.perf_counter()
        for i in range(iters):
            t = time.perf_counter()
            call(images[i % len(images)])
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        total = time.perf_counter() - t0
    return {"ms": float(np.median(ts)), "W": pw.watts, "mJ": pw.watts * total / iters * 1e3}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="released state_dicts, e.g. weights/guidedet_yolo_s0.pt")
    ap.add_argument("--root", default="datasets/visdrone", help="directory made by prepare_visdrone.py")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--int8", action="store_true", help="also build and measure INT8 engines")
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--ap", action="store_true", help="score each engine on the val split")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cache", default=str(Path.home() / "engine-cache"))
    ap.add_argument("--out", default=None, help="optional JSON file for all results")
    a = ap.parse_args()
    assert vdd_in_mw("RAM 2352/7485MB VDD_IN 6894mW/6894mW/6894mW") == 6894.0  # tegrastats format check
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    root, precs = Path(a.root), ["fp16"] + (["int8"] if a.int8 else [])
    calib = calib_yaml(root, a.calib_images) if a.int8 else None
    weights = [Path(w) for w in a.weights]
    engines = {(w.stem, p): build(w, p, a.imgsz, Path(a.cache), calib) for w in weights for p in precs}

    from PIL import Image
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.splitlines()[:1]
    print(f"power mode: {mode[0] if mode else 'unknown'}", flush=True)
    gt = json.loads((root / "coco_valid.json").read_text())
    names = sorted(im["file_name"] for im in gt["images"])[:a.n_images]
    images = [np.asarray(Image.open(root / "valid" / "images" / n).convert("RGB"))[..., ::-1].copy() for n in names]
    rows, rng, keys = [], random.Random(0), list(engines)
    for rd in range(a.rounds):
        rng.shuffle(keys)
        for k in keys:
            rows.append({"round": rd, "weights": k[0], "prec": k[1], **end_to_end(engines[k], a.imgsz, images, a.warm, a.iters)})
    res = {"mode": mode, "latency": rows, "ap": {}}
    for k in sorted(engines):
        r = [x for x in rows if (x["weights"], x["prec"]) == k]
        print(f"{k[0]} {k[1]}: end-to-end {np.median([x['ms'] for x in r]):.2f} ms, "
              f"{np.median([x['mJ'] for x in r]):.0f} mJ/image, engine {engines[k].stat().st_size / 2**20:.1f} MB", flush=True)
    if a.ap:
        from ultralytics import YOLO
        from val import coco_eval, predict_split
        ign = json.loads((root / "ignore_valid.json").read_text())
        for k in sorted(engines):
            dets = predict_split(YOLO(str(engines[k]), task="detect"), root, "valid", gt, a.imgsz, "0", batch=1)
            m = coco_eval(gt, dets, ignore=ign)
            res["ap"][f"{k[0]}/{k[1]}"] = m
            print(f"{k[0]} {k[1]}: val AP {m['AP']:.2f} AP50 {m['AP50']:.2f} tiny {m['AP_tiny']:.2f} "
                  f"small {m['AP_small']:.2f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
