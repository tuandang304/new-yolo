"""VisDrone2019-DET download, conversion and audit.

Layout produced under <root> (readable by both Ultralytics and RF-DETR's YOLO reader):
  data.yaml
  {train,valid,test}/images/*.jpg      (symlinks into the extracted zips)
  {train,valid,test}/labels/*.txt      (YOLO cls cx cy w h, normalised)
  coco_{valid,test}.json               (COCO GT for the unified evaluator)
  ignore_{valid,test}.json             (per-image ignore boxes: VisDrone cat 0 'ignored regions' + cat 11 'others')
  manifest.json                        (zip SHA256, per-split counts, file-list SHA256)

Annotation rows are x,y,w,h,score,category,truncation,occlusion. A box is kept as GT when
score == 1 and category in 1..10; category 0 and 11 become ignore regions.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
ASSETS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
SPLITS = {"train": "VisDrone2019-DET-train", "valid": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(".part")
    print(f"[data] downloading {url}", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dst)


def _convert_split(src: Path, out: Path, split: str) -> dict:
    img_dir, lab_dir = out / split / "images", out / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    images, anns, ignores = [], [], {}
    n_gt = n_ign = n_bad = 0
    for img_id, img_path in enumerate(sorted((src / "images").glob("*.jpg"))):
        link = img_dir / img_path.name
        if not link.exists():
            os.symlink(img_path.resolve(), link)
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        rows = (src / "annotations" / f"{img_path.stem}.txt").read_text().strip().splitlines()
        lines, ign = [], []
        for row in rows:
            v = [int(x) for x in row.strip().rstrip(",").split(",")[:8]]
            x, y, bw, bh, score, cat = v[:6]
            # clip to image
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
            n_gt += 1
        (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        ignores[img_id] = ign
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n, "supercategory": "visdrone"} for i, n in enumerate(NAMES)]}
    if split != "train":
        (out / f"coco_{split}.json").write_text(json.dumps(coco))
        (out / f"ignore_{split}.json").write_text(json.dumps(ignores))
    listing = hashlib.sha256("\n".join(i["file_name"] for i in images).encode()).hexdigest()
    return {"images": len(images), "gt_boxes": n_gt, "ignore_boxes": n_ign, "degenerate_dropped": n_bad,
            "filelist_sha256": listing}


def prepare(root: str | Path, cache: str | Path | None = None) -> Path:
    """Download + convert VisDrone into `root`. Idempotent (skips when manifest exists)."""
    root = Path(root)
    if (root / "manifest.json").exists():
        return root
    cache = Path(cache or root.parent / "visdrone_zips")
    cache.mkdir(parents=True, exist_ok=True)
    raw = root.parent / "visdrone_raw"
    manifest = {"source": ASSETS, "splits": {}}
    for split, name in SPLITS.items():
        z = cache / f"{name}.zip"
        _download(f"{ASSETS}/{name}.zip", z)
        if not (raw / name).exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
        stats = _convert_split(raw / name, root, split)
        stats["zip_sha256"] = sha256(z)
        manifest["splits"][split] = stats
        print(f"[data] {split}: {stats}", flush=True)
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def make_subset(root: Path, out: Path, n: dict[str, int], seed: int = 0) -> Path:
    """Smoke-test subset: first-n random images per split, symlinked, with re-indexed COCO GT."""
    if (out / "data.yaml").exists():
        return out
    rng = random.Random(seed)
    for split, k in n.items():
        names = sorted(p.name for p in (root / split / "images").iterdir())
        keep = sorted(rng.sample(names, min(k, len(names))))
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        for name in keep:
            os.symlink((root / split / "images" / name).resolve(), out / split / "images" / name)
            lab = Path(name).with_suffix(".txt").name
            shutil.copy(root / split / "labels" / lab, out / split / "labels" / lab)
        if split != "train":
            coco = json.loads((root / f"coco_{split}.json").read_text())
            ign = json.loads((root / f"ignore_{split}.json").read_text())
            ids = {im["id"] for im in coco["images"] if im["file_name"] in set(keep)}
            coco["images"] = [im for im in coco["images"] if im["id"] in ids]
            coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in ids]
            (out / f"coco_{split}.json").write_text(json.dumps(coco))
            (out / f"ignore_{split}.json").write_text(json.dumps({k2: v for k2, v in ign.items() if int(k2) in ids}))
    text = (root / "data.yaml").read_text().replace(f"path: {root.resolve()}", f"path: {out.resolve()}")
    (out / "data.yaml").write_text(text)
    shutil.copy(root / "manifest.json", out / "manifest.json")
    return out


def figure_samples(root: Path, k: int = 16, seed: int = 2026) -> list[str]:
    """Fixed qualitative sample set on test-dev: stratified by GT-count quartile, deterministic."""
    coco = json.loads((root / "coco_test.json").read_text())
    counts = {im["id"]: 0 for im in coco["images"]}
    for a in coco["annotations"]:
        counts[a["image_id"]] += 1
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    order = sorted(counts, key=lambda i: (counts[i], by_id[i]))
    q = len(order) // 4
    rng = random.Random(seed)
    picked = []
    for b in range(4):
        bucket = order[b * q:(b + 1) * q if b < 3 else len(order)]
        picked += rng.sample(bucket, k // 4)
    return sorted(by_id[i] for i in picked)
