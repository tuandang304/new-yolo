"""End-to-end check of the public code release (release/GuideDet-YOLO) with its own scripts, as a reader would run them.
Each step is a subprocess in the release directory; a step fails the run if it exits non-zero or misses its check."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def step(name: str, args: list[str], cwd: Path, log: Path) -> str:
    print(f"::orx:: {json.dumps({'phase': name})}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    text = r.stdout + r.stderr
    (log / f"{name}.log").write_text(text)
    tail = "\n".join(text.strip().splitlines()[-6:])
    print(f"[{name}] rc={r.returncode} {time.time() - t0:.0f}s\n{tail}", flush=True)
    if r.returncode:
        raise RuntimeError(f"release step '{name}' failed (rc={r.returncode})")
    return text


def ap_line(text: str, key: str = "AP") -> float:
    return float(re.findall(rf"ignore-aware {key}=([0-9.]+)", text)[-1])


def run(cfg: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rel = REPO / cfg["repo"]
    root = Path(cfg["data_root"]).expanduser()
    w = "weights/guidedet_yolo_s0.pt"
    res: dict = {}
    step("prepare", ["prepare_visdrone.py", "--root", str(root), "--cache", str(Path(cfg["zip_cache"]).expanduser())], rel, out)
    for split, want in cfg["expect_ap"].items():
        a = ap_line(step(f"val_{split}", ["val.py", "--weights", w, "--root", str(root), "--split", split], rel, out))
        res[f"val_{split}"] = {"AP": a, "paper": want, "ok": abs(a - want) <= cfg["tol"]}
        print("RESULT " + json.dumps({"kind": "release_ap", "split": split, **res[f"val_{split}"]}), flush=True)
    step("train", ["train.py", "--data", str(root / "data.yaml"), "--seed", "0", "--project", str(out / "train"),
                   "--set", *cfg["train_set"]], rel, out)
    sd = out / "train" / "guidedet-yolo_s0" / "weights" / "best_state_dict.pt"
    step("load_trained", ["-c", f"from guidedet import load_model; load_model('{sd}'); print('loaded')"], rel, out)
    res["train"] = {"ok": sd.exists()}
    res["cost"] = step("count_cost", ["count_cost.py", "--imgsz", "1280"], rel, out).strip().splitlines()[-2:]
    res["latency"] = step("latency", ["benchmark_latency.py", "--weights", w, "--images", str(root / "valid" / "images"),
                                      "--n-images", "50", "--warm", "10", "--iters", "50"], rel, out).strip().splitlines()[-1]
    jet = step("jetson", ["benchmark_jetson.py", "--weights", w, "--root", str(root), "--int8", "--ap", "--rounds", "1",
                          "--iters", "100", "--cache", str(out / "engines"), "--out", str(out / "jetson.json")], rel, out)
    j = json.loads((out / "jetson.json").read_text())
    fp16 = j["ap"]["guidedet_yolo_s0/fp16"]["AP"]
    res["jetson"] = {"fp16_AP": fp16, "int8_AP": j["ap"]["guidedet_yolo_s0/int8"]["AP"],
                     "fp16_vs_paper_S3": cfg["expect_trt_fp16"], "ok": abs(fp16 - cfg["expect_trt_fp16"]) <= cfg["tol_trt"],
                     "summary": [l for l in jet.splitlines() if "end-to-end" in l]}
    print("RESULT " + json.dumps({"kind": "release_jetson", **res["jetson"]}), flush=True)
    (out / "release.json").write_text(json.dumps(res, indent=1))
    bad = [k for k, v in res.items() if isinstance(v, dict) and not v.get("ok", True)]
    if bad:
        raise AssertionError(f"release checks failed: {bad}")
    print("release check: all steps ran and all checks hold", flush=True)
    return res
