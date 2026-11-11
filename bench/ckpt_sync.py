"""Durable training checkpoints on a Modal Volume, so a run cut off with its Colab runtime can resume.

Key = <ORX_EXPERIMENT>/<tag>: a relaunch of the same experiment node finds the files of the lost run.
Uploads weights/last.pt, weights/best.pt and results.csv every `every` epochs and at the end of training.
Disabled (no-op) when MODAL_TOKEN_ID is not set, so local and other backends behave as before.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

VOLUME = "orx-ckpt"
FILES = ("weights/last.pt", "weights/best.pt", "results.csv")


def enabled() -> bool:
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("ORX_EXPERIMENT"))


def _vol():
    try:
        import modal  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "modal"], check=True)
    import modal
    return modal.Volume.from_name(VOLUME, create_if_missing=True)


def key(tag: str) -> str:
    return f"/{os.environ['ORX_EXPERIMENT']}/{tag}"


def restore(tag: str, save_dir: Path) -> Path | None:
    """Download the saved files into save_dir. Returns the local last.pt, or None when nothing was saved."""
    if not enabled():
        return None
    try:
        vol, k = _vol(), key(tag)
        names = {e.path.split(k.lstrip("/") + "/", 1)[-1] for e in vol.listdir(k, recursive=True)}
    except Exception as e:  # noqa: BLE001 - no checkpoint folder yet (first run) or Modal unreachable
        print(f"[ckpt] no saved checkpoint for {tag}: {type(e).__name__}", flush=True)
        return None
    if "weights/last.pt" not in names:
        return None
    for rel in FILES:
        if rel in names:
            dst = save_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in vol.read_file(f"{k}/{rel}"):
                    f.write(chunk)
    print(f"[ckpt] restored {sorted(names & set(FILES))} for {tag} into {save_dir}", flush=True)
    return save_dir / "weights" / "last.pt"


def push(tag: str, save_dir: Path) -> None:
    t0 = time.time()
    try:
        vol, k = _vol(), key(tag)
        with vol.batch_upload(force=True) as b:
            for rel in FILES:
                if (save_dir / rel).exists():
                    b.put_file(str(save_dir / rel), f"{k}/{rel}")
        print(f"[ckpt] pushed {tag} in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - a failed upload must not stop training
        print(f"[ckpt] push failed for {tag}: {e!r}"[:300], flush=True)


def attach(y, tag: str, every: int = 2) -> None:
    """Register Ultralytics callbacks that push the checkpoint every `every` epochs and at train end."""
    if not enabled():
        return

    def on_model_save(trainer):
        if (trainer.epoch + 1) % every == 0:
            push(tag, Path(trainer.save_dir))

    y.add_callback("on_model_save", on_model_save)
    y.add_callback("on_train_end", lambda trainer: push(tag, Path(trainer.save_dir)))


def finished(last: Path) -> bool:
    """True when last.pt was stripped at the end of training (Ultralytics sets epoch to -1)."""
    import torch
    ck = torch.load(last, map_location="cpu", weights_only=False)
    return ck.get("epoch", -1) == -1
