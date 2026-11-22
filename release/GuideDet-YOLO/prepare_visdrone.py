"""Download VisDrone2019-DET (train / val / test-dev) and convert it for training and evaluation.

    python prepare_visdrone.py --root datasets/visdrone

Layout produced under --root:
  data.yaml                          Ultralytics dataset file (10 classes)
  {train,valid,test}/images/*.jpg    symlinks into the extracted archives
  {train,valid,test}/labels/*.txt    YOLO format (cls cx cy w h, normalized)
  coco_{valid,test}.json             COCO-format annotations for val.py
  ignore_{valid,test}.json           per-image ignore boxes (VisDrone "ignored regions", class "others", score 0)

Annotation rows are x,y,w,h,score,category,truncation,occlusion. A box is a training/evaluation target when
score == 1 and category is 1..10; category 0, category 11 and score-0 boxes become ignore regions.
Boxes are clipped to the image; degenerate boxes are dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from guidedet import NAMES

ASSETS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # mirror of the official VisDrone archives
SPLITS = {"train": "VisDrone2019-DET-train", "valid": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dst)


def convert_split(src: Path, out: Path, split: str) -> dict:
    img_dir, lab_dir = out / split / "images", out / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    images, anns, ignores = [], [], {}
    n_ign = n_bad = 0
    for img_id, img_path in enumerate(sorted((src / "images").glob("*.jpg"))):
        link = img_dir / img_path.name
        if not link.exists():
            os.symlink(img_path.resolve(), link)
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        rows = (src / "annotations" / f"{img_path.stem}.txt").read_text().strip().splitlines()
        lines, ign = [], []
        for row in rows:
            x, y, bw, bh, score, cat = [int(v) for v in row.strip().rstrip(",").split(",")[:6]]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                n_bad += 1
                continue
            if cat in (0, 11) or score == 0:
                ign.append([x0, y0, x1 - x0, y1 - y0])
                n_ign += 1
                continue
            cls = cat - 1
            assert 0 <= cls < 10, row
            cw, ch = x1 - x0, y1 - y0
            lines.append(f"{cls} {(x0 + cw / 2) / w:.6f} {(y0 + ch / 2) / h:.6f} {cw / w:.6f} {ch / h:.6f}")
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cls, "bbox": [x0, y0, cw, ch],
                         "area": cw * ch, "iscrowd": 0})
        (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        ignores[img_id] = ign
    if split != "train":
        coco = {"images": images, "annotations": anns,
                "categories": [{"id": i, "name": n, "supercategory": "visdrone"} for i, n in enumerate(NAMES)]}
        (out / f"coco_{split}.json").write_text(json.dumps(coco))
        (out / f"ignore_{split}.json").write_text(json.dumps(ignores))
    return {"images": len(images), "gt_boxes": len(anns), "ignore_boxes": n_ign, "degenerate_dropped": n_bad}


def prepare(root: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    raw = root.parent / f"{root.name}_raw"
    for split, name in SPLITS.items():
        z = cache / f"{name}.zip"
        download(f"{ASSETS}/{name}.zip", z)
        if not (raw / name).exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
        print(split, convert_split(raw / name, root, split), flush=True)
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/visdrone", help="output directory")
    ap.add_argument("--cache", default=None, help="where the zip archives are kept (default: <root>_zips)")
    a = ap.parse_args()
    root = Path(a.root)
    prepare(root, Path(a.cache) if a.cache else root.parent / f"{root.name}_zips")
