# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# PyTorch port based on PaddleClas `ppcls/arch/backbone/legendary_models/pp_lcnet.py`
# (commit 202ac79). Licensed under the Apache License, Version 2.0.
"""PP-LCNet: A Lightweight CPU Convolutional Neural Network (arXiv:2109.15099).

The module and parameter names mirror PaddleClas so that official `.pdparams`
checkpoints map onto this implementation one-to-one (see `pplcnet.convert`).
"""

import copy
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "NET_CONFIG",
    "PPLCNet",
    "PPLCNet_x0_25",
    "PPLCNet_x0_35",
    "PPLCNet_x0_5",
    "PPLCNet_x0_75",
    "PPLCNet_x1_0",
    "PPLCNet_x1_5",
    "PPLCNet_x2_0",
    "PPLCNet_x2_5",
]

# Table 1 of the paper. Each entry is one depthwise separable block:
# [kernel_size, in_channels, out_channels, stride, use_se]
# 5x5 kernels are used only in the tail of the network, and SE only in the
# last two blocks, which the paper found to be the best accuracy/latency trade-off.
NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}

STAGE_NAMES = ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6")


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act: str) -> nn.Module:
    if act == "hardswish":
        return nn.Hardswish()
    if act == "relu":
        return nn.ReLU()
    if act == "relu6":
        return nn.ReLU6()
    raise ValueError(f"The activation function is not supported: {act}")


class DownscaleInInferDropout(nn.Module):
    """Dropout matching Paddle's `mode="downscale_in_infer"`.

    Training keeps surviving activations unscaled; inference multiplies by
    (1 - p). PyTorch's `nn.Dropout` instead scales by 1 / (1 - p) during
    training. Both agree in expectation, but the official Paddle checkpoints
    were trained with this variant, so the head expects inputs scaled by
    (1 - p) at inference time.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return x
        if self.training:
            mask = torch.empty_like(x).bernoulli_(1.0 - self.p)
            return x * mask
        return x * (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ConvBNLayer(nn.Module):
    def __init__(self, num_channels: int, filter_size: int, num_filters: int,
                 stride: int, num_groups: int = 1, act: str = "hardswish"):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.act = _create_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEModule(nn.Module):
    """Squeeze-and-excitation with ReLU and H-Sigmoid, as in MobileNetV3."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.relu(self.conv1(scale))
        scale = self.hardsigmoid(self.conv2(scale))
        return x * scale


class DepthwiseSeparable(nn.Module):
    """MobileNetV1-style block without a shortcut: dw conv -> (SE) -> pw conv."""

    def __init__(self, num_channels: int, num_filters: int, stride: int,
                 dw_size: int = 3, use_se: bool = False, act: str = "hardswish"):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            act=act)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        return self.pw_conv(x)


class PPLCNet(nn.Module):
    """PP-LCNet backbone with classification head.

    Args:
        scale: Width multiplier applied to every stage.
        class_num: Number of output classes.
        dropout_prob: Dropout probability after the 1x1 head conv.
        class_expand: Output channels of the 1x1 conv after global pooling.
        stride_list: Strides of conv1 and the first block of blocks3..blocks6.
        use_last_conv: Whether to use the 1x1 conv (+ act + dropout) after GAP.
        act: Activation used throughout the network.
        net_config: Optional override of `NET_CONFIG` (e.g. for ablations on
            SE / 5x5 placement). It is deep-copied, never mutated.
        lr_mult_list: Per-stage learning-rate multipliers for
            [conv1, blocks2, ..., blocks6]. Stored on the model and consumed by
            `pplcnet.param_groups`; it does not change the forward pass.
    """

    def __init__(self,
                 scale: float = 1.0,
                 class_num: int = 1000,
                 dropout_prob: float = 0.2,
                 class_expand: int = 1280,
                 stride_list: Sequence[int] = (2, 2, 2, 2, 2),
                 use_last_conv: bool = True,
                 act: str = "hardswish",
                 net_config: Optional[dict] = None,
                 lr_mult_list: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        if len(stride_list) != 5:
            raise ValueError(f"stride_list length should be 5 but got {len(stride_list)}")
        if len(lr_mult_list) != 6:
            raise ValueError(f"lr_mult_list length should be 6 but got {len(lr_mult_list)}")

        self.scale = scale
        self.class_expand = class_expand
        self.use_last_conv = use_last_conv
        self.stride_list = list(stride_list)
        self.lr_mult_list = list(lr_mult_list)
        # PaddleClas writes stride_list into the module-level NET_CONFIG, which
        # leaks into every model built afterwards. Work on a private copy.
        self.net_config = copy.deepcopy(NET_CONFIG if net_config is None else net_config)
        for name, stride in zip(STAGE_NAMES[1:], self.stride_list[1:]):
            self.net_config[name][0][3] = stride

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=self.stride_list[0],
            act=act)

        for name in STAGE_NAMES:
            blocks = [
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * scale),
                    num_filters=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    act=act)
                for k, in_c, out_c, s, se in self.net_config[name]
            ]
            setattr(self, name, nn.Sequential(*blocks))

        self.out_channels = [
            make_divisible(self.net_config[name][-1][2] * scale) for name in STAGE_NAMES
        ]
        self.num_features = self.out_channels[-1]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_last_conv:
            self.last_conv = nn.Conv2d(
                self.num_features, class_expand, kernel_size=1, bias=False)
            self.act = _create_act(act)
            self.dropout = DownscaleInInferDropout(p=dropout_prob)
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(class_expand if use_last_conv else self.num_features, class_num)

        self._init_weights()

    def _init_weights(self) -> None:
        # Matches Paddle defaults: convs use Normal(0, sqrt(2 / fan_in)) (explicit
        # KaimingNormal for ConvBNLayer, Paddle's conv default elsewhere), Linear
        # uses XavierUniform, biases start at zero.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the outputs of blocks2..blocks6 (strides 2, 4, 8, 16, 32 by default)."""
        x = self.conv1(x)
        outs = []
        for name in STAGE_NAMES:
            x = getattr(self, name)(x)
            outs.append(x)
        return outs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for name in STAGE_NAMES:
            x = getattr(self, name)(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.dropout(self.act(self.last_conv(x)))
        return self.fc(self.flatten(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def _build(scale: float, pretrained, use_ssld: bool, **kwargs) -> PPLCNet:
    model = PPLCNet(scale=scale, **kwargs)
    if pretrained:
        from .convert import load_pretrained
        load_pretrained(model, pretrained, use_ssld=use_ssld)
    return model


def PPLCNet_x0_25(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(0.25, pretrained, use_ssld, **kwargs)


def PPLCNet_x0_35(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(0.35, pretrained, use_ssld, **kwargs)


def PPLCNet_x0_5(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(0.5, pretrained, use_ssld, **kwargs)


def PPLCNet_x0_75(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(0.75, pretrained, use_ssld, **kwargs)


def PPLCNet_x1_0(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(1.0, pretrained, use_ssld, **kwargs)


def PPLCNet_x1_5(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(1.5, pretrained, use_ssld, **kwargs)


def PPLCNet_x2_0(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(2.0, pretrained, use_ssld, **kwargs)


def PPLCNet_x2_5(pretrained=False, use_ssld=False, **kwargs) -> PPLCNet:
    return _build(2.5, pretrained, use_ssld, **kwargs)
