"""Optimizer helpers reproducing PaddleClas parameter attributes."""

from typing import Dict, List

import torch.nn as nn

from .model import STAGE_NAMES, PPLCNet


def param_groups(model: PPLCNet, lr: float, weight_decay: float) -> List[Dict]:
    """Build optimizer parameter groups that follow PaddleClas.

    - BatchNorm weights and biases get no weight decay (`L2Decay(0.0)`).
    - conv1 and blocks2..blocks6 use `lr * model.lr_mult_list[i]`; the head
      (last_conv, fc) always uses the base learning rate.

    The per-group `lr` is set directly, so standard LR schedulers scale each
    group from its own starting value.
    """
    stage_mult = dict(zip(("conv1",) + STAGE_NAMES, model.lr_mult_list))
    bn_params = {id(p) for m in model.modules() if isinstance(m, nn.BatchNorm2d)
                 for p in m.parameters(recurse=False)}

    groups: Dict[tuple, List] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult = stage_mult.get(name.split(".", 1)[0], 1.0)
        decay = 0.0 if id(param) in bn_params else weight_decay
        groups.setdefault((mult, decay), []).append(param)

    return [{"params": params, "lr": lr * mult, "weight_decay": decay}
            for (mult, decay), params in groups.items()]
