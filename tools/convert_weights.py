"""Convert a PaddleClas PP-LCNet `.pdparams` checkpoint into a PyTorch state dict.

Example:
    python tools/convert_weights.py --scale 1.0 \
        --src PPLCNet_x1_0_pretrained.pdparams --dst PPLCNet_x1_0.pth
"""

import argparse

import torch

from pplcnet import PPLCNet
from pplcnet.convert import load_pdparams, paddle_to_torch_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--src", required=True, help="input .pdparams file")
    parser.add_argument("--dst", required=True, help="output .pth file")
    parser.add_argument("--class-num", type=int, default=1000)
    args = parser.parse_args()

    model = PPLCNet(scale=args.scale, class_num=args.class_num)
    state = paddle_to_torch_state_dict(load_pdparams(args.src), model)
    model.load_state_dict(state)
    torch.save(model.state_dict(), args.dst)
    print(f"Saved {len(state)} tensors to {args.dst}")


if __name__ == "__main__":
    main()
