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
