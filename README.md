# PP-LCNet-PyTorch

PyTorch implementation of **PP-LCNet: A Lightweight CPU Convolutional Neural Network**
([arXiv:2109.15099](https://arxiv.org/abs/2109.15099)), ported from the official
[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) implementation
(`ppcls/arch/backbone/legendary_models/pp_lcnet.py`, commit `202ac79`).

## Architecture

DepthSepConv (MobileNetV1) blocks without shortcuts, plus the four techniques from the paper:
H-Swish, SE in the last two blocks only, 5x5 depthwise kernels in the tail, and a
1280-d 1x1 conv (no BN) + dropout after global average pooling.

```
Conv 3x3 s2 (16)
DSConv 3x3: s1 32 | s2 64 | s1 64 | s2 128 | s1 128 | s2 256
5 x DSConv 5x5 s1 256
DSConv 5x5 s2 512 (SE) | DSConv 5x5 s1 512 (SE)
GAP -> 1x1 conv 1280 (no BN) -> Hardswish -> Dropout -> FC
```

Parameter and FLOPs counts reproduce Table 2 of the paper for all eight scales
(`python tools/count_params_flops.py`, Paddle counting conventions, 224x224):

| Scale | 0.25x | 0.35x | 0.5x | 0.75x | 1x | 1.5x | 2x | 2.5x |
|---|---|---|---|---|---|---|---|---|
| Params (M) | 1.52 | 1.65 | 1.89 | 2.37 | 2.96 | 4.52 | 6.54 | 9.04 |
| FLOPs (M) | 18 | 29 | 47 | 99 | 161 | 342 | 590 | 906 |

Paddle counts BatchNorm running statistics as parameters and BN/bias ops as FLOPs;
plain PyTorch counting gives slightly smaller numbers (e.g. 2.95M / 156M MACs for 1x).

## Usage

```python
import torch
import pplcnet

model = pplcnet.PPLCNet_x1_0().eval()
logits = model(torch.randn(1, 3, 224, 224))          # (1, 1000)
features = model.forward_stages(torch.randn(1, 3, 224, 224))  # strides 2..32, for detection/segmentation

# Official PaddleClas ImageNet weights (downloaded and converted on the fly)
model = pplcnet.PPLCNet_x1_0(pretrained=True)
model = pplcnet.PPLCNet_x1_0(pretrained=True, use_ssld=True)  # SSLD: 0.25x, 0.5x, 1x, 2.5x
model = pplcnet.PPLCNet_x1_0(pretrained="PPLCNet_x1_0.pth")
```

Preprocessing matches PaddleClas: RGB, resize short side to 256, center crop 224,
`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

### Ablation switches

| Paper item | Argument |
|---|---|
| H-Swish vs ReLU | `act="hardswish" / "relu" / "relu6"` |
| 1x1 conv after GAP | `use_last_conv=False`, `class_expand=1280` |
| Dropout | `dropout_prob=0.2` |
| SE / 5x5 placement | `net_config=...` (copy of `pplcnet.NET_CONFIG`, entries `[k, in_c, out_c, s, use_se]`) |
| Output stride | `stride_list=[2, 2, 2, 2, 2]` |

Unlike PaddleClas, `stride_list` and `net_config` never mutate the module-level `NET_CONFIG`.

### Training recipe

`pplcnet.param_groups(model, lr, weight_decay)` reproduces PaddleClas parameter attributes:
no weight decay on BatchNorm, and per-stage `lr_mult_list`. The PaddleClas ImageNet recipe:

```python
opt = torch.optim.SGD(pplcnet.param_groups(model, lr=0.8, weight_decay=3e-5), momentum=0.9)
# cosine schedule, 5 warmup epochs, 360 epochs, total batch 2048 (scale lr linearly)
# weight_decay=4e-5 for 1.5x and larger; label smoothing 0.1 (not stated in the paper)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Differences from `torch.nn` defaults kept for fidelity

- Dropout uses Paddle's `downscale_in_infer` mode (scale by `1 - p` at inference),
  which the official checkpoints expect.
- Weight init follows Paddle: convs `Normal(0, sqrt(2 / fan_in))`, Linear Xavier-uniform, zero biases.

## Converting weights

```bash
python tools/convert_weights.py --scale 1.0 \
    --src PPLCNet_x1_0_pretrained.pdparams --dst PPLCNet_x1_0.pth
```

`.pdparams` files are read with a restricted unpickler that only reconstructs numpy
arrays; PaddlePaddle is not required.

### Verified against PaddleClas

`tools/compare_with_paddle.py` runs the original PaddleClas `pp_lcnet.py` and this port
on the same inputs (CPU, PaddlePaddle 3.3.1, random batch + `whl_demo.jpg`):

| Checkpoint | Max logit diff | Top-5 agree |
|---|---|---|
| `PPLCNet_x0_25_pretrained` | 2.9e-05 | yes |
| `PPLCNet_x1_0_pretrained` | 1.0e-05 | yes |
| `PPLCNet_x1_0_ssld_pretrained` | 1.6e-05 | yes |
| `PPLCNet_x2_5_ssld_pretrained` | 1.7e-05 | yes |

```bash
# needs a PaddleClas checkout and an image with paddlepaddle installed
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/workspace \
    -v /path/to/PaddleClas:/pc:ro -v /path/to/weights:/w:ro <image-with-paddle> \
    python tools/compare_with_paddle.py --paddleclas /pc --scale 1.0 \
    --weights /w/PPLCNet_x1_0_pretrained.pdparams \
    --image /pc/docs/images/inference_deployment/whl_demo.jpg
```

### Imagenette evaluation

`tools/eval_imagenette.py` evaluates the official 1000-class weights on the
[Imagenette](https://github.com/fastai/imagenette) `val` split (3,925 images taken from the
ImageNet validation set; labels mapped to ImageNet-1k indices). `top-1(10)` restricts the
prediction to the 10 Imagenette classes. With `--paddleclas`, the original PaddleClas model
runs on the same input tensors.

| Model | Top-1 | Top-5 | Top-1 (10 cls) | PaddleClas Top-1 | Agreement | Max logit diff |
|---|---|---|---|---|---|---|
| 0.25x | 60.99 | 83.13 | 95.77 | 60.99 | 100% | 4.0e-05 |
| 0.25x SSLD | 57.86 | 81.94 | 95.34 | 57.86 | 100% | 9.8e-05 |
| 0.35x | 68.20 | 88.79 | 97.48 | 68.20 | 100% | 5.5e-05 |
| 0.5x | 73.89 | 91.95 | 98.42 | 73.89 | 100% | 3.5e-05 |
| 0.5x SSLD | 72.61 | 91.01 | 98.11 | 72.61 | 100% | 5.9e-05 |
| 0.75x | 79.92 | 95.21 | 99.41 | 79.92 | 100% | 3.0e-05 |
| 1x | 83.82 | 97.12 | 99.62 | 83.82 | 100% | 2.5e-05 |
| 1x SSLD | 82.55 | 96.15 | 99.29 | 82.55 | 100% | 3.9e-05 |
| 1.5x | 88.69 | 98.68 | 99.77 | 88.69 | 100% | 6.2e-05 |
| 2x | 93.43 | 99.75 | 99.97 | 93.43 | 100% | 2.8e-05 |
| 2.5x | 94.52 | 99.64 | 99.95 | 94.52 | 100% | 2.8e-05 |
| 2.5x SSLD | 88.89 | 98.60 | 99.77 | 88.89 | 100% | 4.6e-05 |

These numbers cover only 10 classes and are not comparable to the paper's ImageNet top-1
(e.g. 71.32% for 1x). SSLD checkpoints score lower than the plain ones on this subset even
though they are better on full ImageNet.

TF32 is disabled by default for this comparison. With `--allow-tf32`, cuDNN TF32 convolutions
raise the logit difference to ~1e-1 and flip ~0.3% of predictions.

```bash
docker run --rm --gpus all --shm-size=2g -u $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD:/workspace -v ~/data/imagenette2:/data:ro -v /path/to/weights:/w:ro \
    pplcnet-torch python tools/eval_imagenette.py --root /data --weights-dir /w
```

### Ablation on Imagenette

`tools/ablation.py` retrains the paper's cumulative ablation (Table 3, 0.5x) from scratch on
Imagenette `train` (9,469 images, 10 classes) and evaluates on `val` (3,925 images). Recipe
from `tools/train.py`: 100 epochs, batch 128, lr 0.05 (0.8 scaled to batch 128), cosine with
5 warmup epochs, label smoothing 0.1, weight decay 3e-5, bf16 autocast. Five seeds per step;
Δ is the change from the previous step, ± its standard error.

| Step | Top-1 (mean ± std) | Runs (seeds 0–4) | Δ | Params (M) | Paper top-1 (ImageNet) | Paper Δ |
|---|---|---|---|---|---|---|
| BaseNet | 81.66 ± 0.51 | 81.10, 81.20, 81.89, 81.81, 82.32 | | 0.22 | 55.58 | |
| + H-Swish | 83.46 ± 0.45 | 83.62, 84.15, 83.21, 83.01, 83.31 | **+1.80 ± 0.30** | 0.22 | 58.18 | +2.60 |
| + 5x5 | 83.92 ± 0.31 | 83.46, 83.92, 84.00, 83.90, 84.33 | +0.46 ± 0.24 | 0.23 | 59.09 | +0.91 |
| + SE | 83.62 ± 0.54 | 83.31, 84.18, 83.16, 83.21, 84.23 | −0.31 ± 0.28 | 0.27 | 59.91 | +0.82 |
| + 1x1 conv | 84.12 ± 0.58 | 84.00, 83.64, 83.97, 85.12, 83.85 | +0.50 ± 0.35 | 0.61 | 62.50 | +2.59 |
| + dropout | 84.59 ± 0.37 | 84.15, 84.66, 84.61, 85.15, 84.36 | +0.47 ± 0.31 | 0.61 | 63.14 | +0.64 |

- H-Swish is the only step that is clearly significant on its own (t ≈ 6).
- 5x5, 1x1 conv and dropout are each positive with roughly the paper's sign and order of
  magnitude, but none is significant alone (t ≈ 1.4–1.9 with 5 runs).
- The head changes taken together are significant: 1x1 conv + dropout over +SE is
  +0.97 ± 0.29 (t ≈ 3.3). So are all tail techniques together: the full model over
  +H-Swish is +1.13 ± 0.26.
- SE shows no gain here (−0.31 ± 0.28), unlike the paper (+0.82). The 1x1 head's gain is also
  much smaller than the paper's +2.59, likely because Imagenette has 10 classes, not 1,000.
- Small samples mislead at this dataset size. A single 50-epoch run showed +1.37 for 5x5.
  With 3 seeds, dropout looked significant (+0.60, t ≈ 3). Neither held up with 5 seeds.

CPU latency is not reported: at 0.5x, PyTorch eager overhead in a WSL container dominates
(3.0–3.5 ms for every step), so it does not reproduce the paper's per-step latency costs.

```bash
docker run --rm --gpus all --shm-size=4g -u $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD:/workspace -v ~/data/imagenette2:/data:ro \
    pplcnet-torch python tools/ablation.py --root /data --epochs 100 --seeds 0 1 2 3 4 \
    --out-dir runs/ablation_e100
```

## Deployment: ONNX, ONNX Runtime, TensorRT

```bash
docker build --target deploy -t pplcnet-deploy .
python tools/export_onnx.py --scale 1.0 --weights official --out PPLCNet_x1_0.onnx
trtexec --onnx=PPLCNet_x1_0.onnx --fp16 --saveEngine=PPLCNet_x1_0.engine  # optional, standalone TensorRT
```

The exported graph contains only Conv, HardSwish, HardSigmoid, Relu, Mul, ReduceMean, Reshape
and Gemm. BatchNorm is folded and inference dropout becomes a constant scale. The batch
dimension is dynamic. Against PyTorch, ONNX Runtime differs by 6.9e-06 max logit for the
official 1x weights, and Imagenette accuracy is identical (83.82%).

### Latency and accuracy vs. other mobile models

`tools/benchmark_deploy.py`: batch 1 median latency in ms (300 runs), plus TensorRT FP16
throughput at batch 64. CPU: AMD Ryzen 9 7945HX, 10 threads (the paper used an Intel Xeon
Gold 6148 with MKLDNN). GPU: RTX 4070 Laptop. Software: ONNX Runtime 1.31, TensorRT 10.16,
PyTorch 2.14, run in WSL2 and Docker. torchvision models use `IMAGENET1K_V1` weights. ImageNet
top-1 values are the published numbers (PaddleClas docs / torchvision docs). Imagenette top-1
is measured here (1000-way, val split).

| Model | ImageNet top-1 (reported) | Imagenette top-1 FP32 / TRT FP16 | CPU ORT | CPU PyTorch | GPU ORT-CUDA | GPU TRT FP32 | GPU TRT FP16 | TRT FP16 bs64 (img/s) |
|---|---|---|---|---|---|---|---|---|
| PP-LCNet-0.25x | 51.86 | 60.99 / 60.94 | 0.36 | 2.39 | 1.10 | 0.82 | 0.86 | 52,934 |
| PP-LCNet-0.35x | 58.09 | 68.20 / 68.36 | 0.41 | 2.63 | 1.36 | 0.80 | 0.80 | 40,056 |
| PP-LCNet-0.5x | 63.14 | 73.89 / 73.83 | 0.47 | 3.15 | 1.57 | 0.82 | 0.73 | 38,144 |
| PP-LCNet-0.75x | 68.18 | 79.92 / 80.08 | 0.62 | 3.91 | 1.15 | 0.80 | 0.73 | 20,874 |
| PP-LCNet-1x | 71.32 | 83.82 / 83.87 | 0.84 | 4.12 | 1.55 | 0.74 | 0.57 | 17,135 |
| PP-LCNet-1.5x | 73.71 | 88.69 / 88.69 | 1.47 | 6.18 | 1.66 | 0.86 | 0.73 | 10,825 |
| PP-LCNet-2x | 75.18 | 93.43 / 93.48 | 2.37 | 8.00 | 1.57 | 0.83 | 0.70 | 7,723 |
| PP-LCNet-2.5x | 76.60 | 94.52 / 94.52 | 3.45 | 9.79 | 1.55 | 0.92 | 0.64 | 6,068 |
| MobileNetV2-1.0 | 71.88 | 82.65 / 82.65 | 1.49 | 9.61 | 1.65 | 0.63 | 0.73 | 9,226 |
| MobileNetV3-small | 67.67 | 81.12 / 81.10 | 1.01 | 4.37 | 2.07 | 1.23 | 1.01 | 27,082 |
| MobileNetV3-large | 74.04 | 92.61 / 92.59 | 1.70 | 9.64 | 2.28 | 1.25 | 1.18 | 11,039 |
| ShuffleNetV2-1.0 | 69.36 | 82.73 / 82.73 | 1.63 | 7.42 | 2.06 | 1.04 | 1.07 | 14,097 |
| ShuffleNetV2-2.0 | 76.23 | 87.11 / 87.11 | 3.76 | 11.25 | 2.48 | 1.24 | 1.20 | 7,379 |

Comparing models at similar ImageNet accuracy:

- **CPU (ONNX Runtime): the paper's claim holds on this AMD CPU.** At each accuracy level,
  PP-LCNet is faster than the alternative:

  | Accuracy level | PP-LCNet | Alternative | PP-LCNet speedup |
  |---|---|---|---|
  | ~68% | 0.75x, 0.62 ms | MobileNetV3-small, 1.01 ms | 1.6x |
  | ~71.5% | 1x, 0.84 ms | MobileNetV2, 1.49 ms | 1.8x |
  | ~74% | 1.5x, 1.47 ms | MobileNetV3-large, 1.70 ms | 1.16x (0.3 points lower accuracy) |
  | ~76.5% | 2.5x, 3.45 ms | ShuffleNetV2-2.0, 3.76 ms | 1.09x |

- **GPU batch 1:** every model takes 0.6–1.3 ms whatever its size, because kernel-launch
  overhead dominates. Latency here says little about the architecture.
- **GPU throughput (TensorRT FP16, batch 64): the advantage mostly disappears.** PP-LCNet-1x
  still beats MobileNetV2 by 1.9x. PP-LCNet-1.5x is on par with MobileNetV3-large.
  MobileNetV3-small and ShuffleNetV2-2.0 are faster than PP-LCNet at matching accuracy.
  This fits the paper's own caveat that its design targets Intel CPU latency.
- **FP16 TensorRT is safe:** Imagenette top-1 changes by at most 0.16 points from FP32.
- **ONNX Runtime vs. PyTorch eager on CPU:** ONNX Runtime is 3–7x faster for these small models.
  This is why the eager-mode latencies in the ablation above showed no per-step differences.
- **Imagenette ranks some models differently from ImageNet.** For example, MobileNetV3-large
  scores 92.6 vs 88.7 for PP-LCNet-1.5x despite similar ImageNet accuracy. Imagenette has
  only 10 classes, so these accuracy columns are for checking the runtimes, not for ranking models.
- All numbers come from a single benchmark pass on a laptop. Expect roughly ±10% run-to-run
  variation on the small batch-1 timings.

## Development (Docker)

```bash
docker build -t pplcnet-torch .
docker run --rm --gpus all -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/workspace \
    pplcnet-torch python -m pytest -q -p no:cacheprovider
docker run --rm -u $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/workspace \
    pplcnet-torch python tools/count_params_flops.py
```

## License

Apache 2.0, same as PaddleClas.
