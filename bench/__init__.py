"""Benchmark package. On images without the X11/GL libraries that opencv-python links against (Modal's default
CUDA image), install them once before anything imports cv2; elsewhere this is a no-op."""
import os
import subprocess

try:
    import cv2  # noqa: F401
except ImportError as e:  # pragma: no cover - environment repair only
    print(f"[env] cv2 import failed ({e}); installing system libraries", flush=True)
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    subprocess.run("apt-get update -qq && apt-get install -y -qq libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1",
                   shell=True, check=False, env=env)
    import cv2  # noqa: F401,F811
