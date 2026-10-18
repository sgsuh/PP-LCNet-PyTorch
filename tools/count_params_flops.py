"""Print parameter count and FLOPs for every PP-LCNet scale.

The "paddle" columns follow Paddle's conventions and reproduce Table 2 of the
paper; the "torch" columns are trainable parameters only and conv/linear
multiply-accumulates from `torch.utils.flop_counter`.
"""

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode

import pplcnet
from pplcnet.profile import count_flops, count_params

SCALES = ["x0_25", "x0_35", "x0_5", "x0_75", "x1_0", "x1_5", "x2_0", "x2_5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    x = torch.randn(1, 3, args.image_size, args.image_size)
    print(f"{'model':<16}{'params(M)':>11}{'FLOPs(M)':>10}  |{'params(M)':>11}{'MACs(M)':>9}")
    print(f"{'':<16}{'paddle':>11}{'paddle':>10}  |{'torch':>11}{'torch':>9}")
    for s in SCALES:
        model = getattr(pplcnet, f"PPLCNet_{s}")().eval()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        print(f"{'PPLCNet_' + s:<16}"
              f"{count_params(model) / 1e6:>11.2f}"
              f"{count_flops(model, args.image_size) / 1e6:>10.0f}  |"
              f"{count_params(model, include_bn_stats=False) / 1e6:>11.2f}"
              f"{counter.get_total_flops() / 2 / 1e6:>9.0f}")


if __name__ == "__main__":
    main()
