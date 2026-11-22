# GuideDet-YOLO

Code and trained weights for **GuideDet-YOLO: semantic guidance of high-resolution detail for lightweight object
detection in UAV imagery** (accompanying manuscript).

GuideDet-YOLO is a 1.54 M-parameter detector derived from YOLO26n. It keeps prediction at strides 8 and 16 and brings
stride-4 detail into the stride-8 neck through a guided path instead of adding a stride-4 prediction level:

- **Guided stride-4 detail path.** The top-down stride-8 feature `t` is compressed by a 1x1 convolution to 32 channels,
  upsampled to stride 4, concatenated with the stride-4 backbone feature (L2) and refined by one light C3k2 block.
  A pixel-unshuffle stem (depthwise 3x3, then 1x1 to 64 channels) moves the result to stride 8, and it joins the
  concatenation that feeds the finest neck block (L17). The new input channels of L17 start with zero weights, so
  training starts exactly at the function of the pretrained network.
- **LR-Down.** The stride-2 3x3 convolutions at L3, L5 and L8 are factorized as a rank-`c2/2` 3x3 stride-2
  convolution and a 1x1 convolution, initialized from the truncated SVD of the pretrained kernel.
- **Stride-64 deep stage.** A 2x2 average pool (L7) makes the deepest stage (L8-L11) run at stride 64.
- **Two-level head:** predictions at strides 8 and 16 only (no stride-32 level), with the one-to-many head of
  YOLO26n and NMS at inference.

**GuideDet-YOLO-F** is the same detector with plain 3x3 stride-2 convolutions instead of LR-Down.

## Results (VisDrone2019-DET, 1280x1280, 100 epochs, mean ± s.d. over seeds 0, 1, 2)

| Model | val AP | test-dev AP | Params (M) | GFLOPs @1280 |
|---|---|---|---|---|
| YOLO26n (same recipe) | 30.84 ± 0.24 | 23.85 ± 0.44 | 2.38 | 22.75 |
| **GuideDet-YOLO** | 30.94 ± 0.38 | 23.73 ± 0.12 | 1.54 | 17.52 |
| **GuideDet-YOLO-F** | 31.09 ± 0.15 | 24.02 ± 0.08 | 1.74 | 19.29 |

AP is averaged over IoU 0.50:0.95 with the ignored-region rule of the VisDrone toolkit (see `val.py`). Parameters and
FLOPs are counted on the inference graph with batch normalization folded (for YOLO26n, without its training-only
one-to-many branch); one multiply-accumulate counts as two FLOPs.

With three seeds, neither variant differs significantly from YOLO26n in AP on either split (95% Welch intervals of
the differences include zero); the variants match its accuracy with fewer parameters and FLOPs. See the paper for
size-wise results, the ablation and limitations (in particular, both variants are less accurate than YOLO26n on large
objects).

### Size, memory and latency (seed-0 weights, 1280x1280, batch 1; NVIDIA T4 in one session, Jetson Orin Nano 8 GB Super)

| Model | ONNX file (MB) | Peak GPU memory (MB) | T4 PyTorch FP16 forward / end-to-end (ms) | T4 TensorRT FP16 end-to-end (ms) | Jetson Orin Nano TensorRT FP16 (ms) |
|---|---|---|---|---|---|
| YOLO26n | 9.9 | 64 | 9.8 / 19.0 | 15.6 | 21.2 |
| GuideDet-YOLO | 6.6 | 129 | 9.5 / 18.2 | 15.6 | 23.4 |
| GuideDet-YOLO-F | 7.4 | 129 | 9.4 / 17.9 | 15.3 | 23.4 |

Fewer parameters and FLOPs give a smaller model file but no consistent speed gain: on the T4 the variants are up to
0.8 ms faster than YOLO26n in PyTorch FP16 and as fast with TensorRT, while on the Jetson Orin Nano they are 2.2 ms
(about 10%) slower with TensorRT. They also use more GPU memory (twice as much on the T4; the stride-4 path works on 320x320 feature
maps). End-to-end times include NMS for all three models: YOLO26n is run with its one-to-many branch and NMS, as in all
our measurements. PyTorch timings are medians of five rounds with shuffled model order (200 validation images after
30 warm-up images each). Jetson: JetPack 7.2, MAXN SUPER power mode with `jetson_clocks`, TensorRT 10.16 FP16, each
engine built in a fresh process, median of three rounds with shuffled model order.

### Jetson Orin Nano: power modes, energy and INT8 (TensorRT, seed-0 weights, 1280x1280, batch 1)

| Power mode | Model | FP16 end-to-end (ms) | INT8 end-to-end (ms) | FP16 energy per image (mJ) |
|---|---|---|---|---|
| MAXN SUPER | YOLO26n | 21.2 | 18.7 | 337 |
| | GuideDet-YOLO | 23.4 | 20.0 | 371 |
| | GuideDet-YOLO-F | 23.4 | 20.1 | 372 |
| 25 W | YOLO26n | 23.7 | 21.0 | 339 |
| | GuideDet-YOLO | 25.8 | 22.4 | 374 |
| | GuideDet-YOLO-F | 25.8 | 22.5 | 376 |
| 15 W | YOLO26n | 31.1 | 27.0 | 326 |
| | GuideDet-YOLO | 34.3 | 29.3 | 362 |
| | GuideDet-YOLO-F | 34.1 | 29.1 | 363 |

The relative cost is the same in every power mode: about 10% slower and 10% more energy per image (board input
power from `tegrastats`, idle draw included). The FP16 engine of GuideDet-YOLO is 23% smaller (5.9 vs 7.7 MB).
INT8 engines are calibrated on 512 random training images. Accuracy of the engines on val: see the paper,
Supplementary Table S3. Reproduce with `benchmark_jetson.py`.

## Installation

```bash
pip install -r requirements.txt   # ultralytics==8.4.192, pycocotools; plus a PyTorch build of your choice
```

The COCO checkpoint `yolo26n.pt` is downloaded by Ultralytics on first use (training only).

## Usage

```bash
# 1. Data: download VisDrone2019-DET (train, val, test-dev) and convert it (~2 GB)
python prepare_visdrone.py --root datasets/visdrone

# 2. Evaluate a released checkpoint (val = --split valid, test-dev = --split test)
python val.py --weights weights/guidedet_yolo_s0.pt --root datasets/visdrone --split valid
python val.py --weights weights/guidedet_yolo_f_s0.pt --root datasets/visdrone --split test

# 3. Train with the paper recipe (one GPU; the paper used an NVIDIA T4)
python train.py --data datasets/visdrone/data.yaml --variant guidedet-yolo --seed 0
python train.py --data datasets/visdrone/data.yaml --variant guidedet-yolo-f --seed 0

# 4. Parameters / FLOPs on the fused graph, and GPU latency (FP16, batch 1)
python count_cost.py --imgsz 1280
python benchmark_latency.py --weights weights/guidedet_yolo_s0.pt --images datasets/visdrone/valid/images

# 5. On an NVIDIA Jetson: TensorRT FP16/INT8 engines, latency, board energy and val AP (set the power mode first)
python benchmark_jetson.py --weights weights/guidedet_yolo_s0.pt --root datasets/visdrone --int8 --ap
```

In Python:

```python
from guidedet import build_model, load_model, load_yolo

model = load_model("weights/guidedet_yolo_s0.pt")      # Ultralytics DetectionModel, eval mode
yolo = load_yolo("weights/guidedet_yolo_s0.pt")         # Ultralytics YOLO wrapper
results = yolo.predict("image.jpg", imgsz=1280, conf=0.25)
```

Input sizes must be multiples of 64.

## Training recipe

Ultralytics 8.4.192, 100 epochs at 1280x1280, MuSGD with learning rate 0.01, momentum 0.9 and weight decay 5e-4,
3 warm-up epochs and linear decay to 1% of the initial rate, batch 8 with gradient accumulation to a nominal batch of 64,
mixed precision, mosaic until the last epoch (`close_mosaic=0`), scale jitter 0.5, no mixup. The checkpoint with the best
in-training validation fitness is kept. All weights start from the YOLO26n COCO checkpoint: source layers 0-6 map to
0-6 and 7-19 to 8-20, the two prediction levels inherit the stride-8 and stride-16 levels of the source head, and
LR-Down layers are initialized by SVD (435 of 533 state tensors of GuideDet-YOLO, about 77% of its parameters, are
copied unchanged).
`train.py` writes `best_state_dict.pt` next to Ultralytics' `best.pt`, in the same format as the released weights.

## Evaluation protocol

`val.py` predicts at 1280x1280 with confidence threshold 0.001, NMS IoU 0.7 with multi-label assignment and at most
1000 detections per image, then scores with the COCO API: detections and annotated boxes lying more than 50% inside
an ignored region or a box of class "others" are dropped, as in the official toolkit, and AP is computed with at most
500 detections per image. Size buckets (original image pixels): tiny < 16², small < 32², medium 32²-96², large ≥ 96².

## Files

```
guidedet/
  guidedet_yolo.yaml   base graph (YOLO26n layers, stride-64 deep stage, two-level neck and head)
  modules.py           LowRankDown (LR-Down), GuidedDetailConcat (guided stride-4 detail path)
  model.py             build_model, inherit_coco (COCO weight inheritance + SVD init), load_model, load_yolo
weights/               guidedet_yolo_s{0,1,2}.pt, guidedet_yolo_f_s{0,1,2}.pt (state_dicts, FP16)
train.py  val.py  prepare_visdrone.py  count_cost.py  benchmark_latency.py  benchmark_jetson.py
```

## Citation

Citation details will be added on publication.

## License

AGPL-3.0 (see `LICENSE`). This code builds on [Ultralytics](https://github.com/ultralytics/ultralytics), which is
licensed under AGPL-3.0. VisDrone2019-DET is not redistributed here; see its authors' terms of use.
