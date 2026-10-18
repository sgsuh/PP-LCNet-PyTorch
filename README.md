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
Imagenette `train` (9,469 images, 10 classes) and evaluates on `val`. Recipe from
`tools/train.py`: 50 epochs, batch 128, lr 0.05 (0.8 scaled to batch 128), cosine with
5 warmup epochs, label smoothing 0.1, weight decay 3e-5, bf16 autocast. Single seed.

| Step | Imagenette top-1 | Δ | Params (M) | Paper top-1 (ImageNet) | Paper Δ |
|---|---|---|---|---|---|
| BaseNet | 76.08 | | 0.22 | 55.58 | |
| + H-Swish | 77.48 | +1.40 | 0.22 | 58.18 | +2.60 |
| + 5x5 | 78.85 | +1.37 | 0.23 | 59.09 | +0.91 |
| + SE | 78.42 | −0.43 | 0.27 | 59.91 | +0.82 |
| + 1x1 conv | 78.68 | +0.26 | 0.61 | 62.50 | +2.59 |
| + dropout | 78.24 | −0.44 | 0.61 | 63.14 | +0.64 |

H-Swish and 5x5 kernels reproduce clearly. The last three steps are within the noise of a
single run on 3,925 validation images (~±0.7%p). Two of them likely depend on the dataset:
the 1280-d head mainly adds capacity for 1,000 classes, and dropout does not help a model
that is still underfitting (train loss is still falling at epoch 50).

```bash
docker run --rm --gpus all --shm-size=4g -u $(id -u):$(id -g) -e HOME=/tmp \
    -v $PWD:/workspace -v ~/data/imagenette2:/data:ro \
    pplcnet-torch python tools/ablation.py --root /data --epochs 50 --seeds 0
```

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
