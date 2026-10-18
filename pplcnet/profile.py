"""Parameter and FLOPs counting that reproduces `paddle.flops` / Paddle's `parameters()`.

The paper's Table 2 numbers come from Paddle, which differs from common
PyTorch counters in two ways:
- BatchNorm running mean/variance are (non-trainable) parameters in Paddle,
  so they are included in the parameter count.
- `paddle.flops` counts multiply-accumulates of convs and linear layers, plus
  conv bias adds, 2 ops per BatchNorm element and adaptive average pooling.
  Activations and the SE elementwise multiply count as zero.
"""

import math

import torch
import torch.nn as nn


def count_params(model: nn.Module, include_bn_stats: bool = True) -> int:
    total = sum(p.numel() for p in model.parameters())
    if include_bn_stats:
        total += sum(b.numel() for name, b in model.named_buffers()
                     if name.endswith(("running_mean", "running_var")))
    return total


def count_flops(model: nn.Module, image_size: int = 224) -> int:
    """FLOPs for a single image, counted the way `paddle.flops` does."""
    total = 0

    def conv_hook(m, inputs, output):
        nonlocal total
        kernel_ops = math.prod(m.kernel_size) * m.in_channels // m.groups
        total += output.numel() * (kernel_ops + (1 if m.bias is not None else 0))

    def bn_hook(m, inputs, output):
        nonlocal total
        total += 2 * inputs[0].numel()

    def linear_hook(m, inputs, output):
        nonlocal total
        total += m.in_features * output.numel()

    def pool_hook(m, inputs, output):
        nonlocal total
        kernel = math.prod(inputs[0].shape[2:]) // math.prod(output.shape[2:])
        total += (kernel + 1) * output.numel()

    hooks = {nn.Conv2d: conv_hook, nn.BatchNorm2d: bn_hook,
             nn.Linear: linear_hook, nn.AdaptiveAvgPool2d: pool_hook}
    handles = [m.register_forward_hook(hooks[type(m)])
               for m in model.modules() if type(m) in hooks]
    was_training = model.training
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total
