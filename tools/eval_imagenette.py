"""Evaluate official PP-LCNet ImageNet weights on the Imagenette validation split.

The 1000-class heads are used unchanged; Imagenette labels are mapped to their
ImageNet-1k indices. Reported metrics:
- top-1 / top-5 over all 1000 ImageNet classes,
- top-1 restricted to the 10 Imagenette classes,
- with --paddleclas, agreement with the original PaddleClas model on the
  exact same input tensors.

Example:
    python tools/eval_imagenette.py --root ~/data/imagenette2 --weights-dir weights/
"""

import argparse
import glob
import os
import re

import numpy as np
import torch

from pplcnet import PPLCNet
from pplcnet.convert import load_pretrained
from pplcnet.data import IMAGENETTE_TO_IMAGENET, ImagenetteEval

_CKPT_RE = re.compile(r"PPLCNet_x(\d+)_(\d+)_(ssld_)?pretrained\.pdparams$")


def find_checkpoints(weights_dir: str):
    found = []
    for path in glob.glob(os.path.join(weights_dir, "PPLCNet_*.pdparams")):
        m = _CKPT_RE.search(os.path.basename(path))
        if m:
            scale = float(f"{m.group(1)}.{m.group(2)}")
            found.append((scale, bool(m.group(3)), path))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="imagenette2 directory")
    parser.add_argument("--split", default="val")
    parser.add_argument("--weights-dir", required=True, help="directory with .pdparams files")
    parser.add_argument("--paddleclas", help="PaddleClas checkout; also run the Paddle model")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="allow TF32 on CUDA (faster, ~1e-2 logit error vs FP32)")
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    dataset = ImagenetteEval(args.root, args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=args.workers)
    subset = torch.tensor(sorted(IMAGENETTE_TO_IMAGENET.values()))
    checkpoints = find_checkpoints(args.weights_dir)
    if not checkpoints:
        raise SystemExit(f"No PPLCNet_*_pretrained.pdparams found in {args.weights_dir}")

    paddle_module = None
    if args.paddleclas:
        import paddle
        from compare_with_paddle import load_paddle_module
        paddle.set_device("cpu")
        paddle_module = load_paddle_module(args.paddleclas)

    print(f"Imagenette/{args.split}: {len(dataset)} images, device={args.device}")
    header = f"{'model':<20}{'top-1':>8}{'top-5':>8}{'top-1(10)':>11}"
    if paddle_module:
        header += f"{'paddle top-1':>14}{'agree':>8}{'max diff':>10}"
    print(header)

    for scale, ssld, path in checkpoints:
        model = PPLCNet(scale=scale)
        load_pretrained(model, path)
        model.to(args.device).eval()
        pd_model = None
        if paddle_module:
            pd_model = paddle_module.PPLCNet(scale=scale)
            pd_model.set_state_dict(paddle.load(path))
            pd_model.eval()

        top1 = top5 = top1_sub = pd_top1 = agree = 0
        max_diff = 0.0
        for x, y in loader:
            with torch.no_grad():
                logits = model(x.to(args.device)).float().cpu()
            pred5 = logits.topk(5, dim=1).indices
            top1 += (pred5[:, 0] == y).sum().item()
            top5 += (pred5 == y[:, None]).any(1).sum().item()
            top1_sub += (subset[logits[:, subset].argmax(1)] == y).sum().item()
            if pd_model is not None:
                pd_logits = torch.from_numpy(pd_model(paddle.to_tensor(x.numpy())).numpy())
                pd_top1 += (pd_logits.argmax(1) == y).sum().item()
                agree += (pd_logits.argmax(1) == pred5[:, 0]).sum().item()
                max_diff = max(max_diff, (pd_logits - logits).abs().max().item())

        n = len(dataset)
        name = f"PPLCNet_x{scale:g}" + ("_ssld" if ssld else "")
        row = f"{name:<20}{100 * top1 / n:>8.2f}{100 * top5 / n:>8.2f}{100 * top1_sub / n:>11.2f}"
        if pd_model is not None:
            row += f"{100 * pd_top1 / n:>14.2f}{100 * agree / n:>8.2f}{max_diff:>10.1e}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
