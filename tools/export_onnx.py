"""Export PP-LCNet to ONNX and check it against PyTorch with ONNX Runtime.

The graph has input "input" (N, 3, 224, 224, dynamic N) and output "logits".
BatchNorm is folded into the convs and inference-mode dropout into a constant scale.

Example:
    python tools/export_onnx.py --scale 1.0 --weights PPLCNet_x1_0_pretrained.pdparams \
        --out PPLCNet_x1_0.onnx
"""

import argparse

from pplcnet import PPLCNet
from pplcnet.convert import load_pretrained
from pplcnet.export import check_onnx, export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--weights", help=".pdparams/.pth file, or 'official' to download")
    parser.add_argument("--ssld", action="store_true", help="with --weights official")
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    model = PPLCNet(scale=args.scale)
    if args.weights:
        load_pretrained(model, True if args.weights == "official" else args.weights,
                        use_ssld=args.ssld)
    export_onnx(model, args.out, opset=args.opset, dynamic_batch=not args.static_batch)
    print(f"Saved {args.out}")
    if not args.no_check:
        print(f"max |logit diff| vs PyTorch (ONNX Runtime CPU): {check_onnx(model, args.out):.2e}")


if __name__ == "__main__":
    main()
