"""Check that converted weights give the same outputs as the PaddleClas model.

Runs the original PaddleClas `pp_lcnet.py` (PaddlePaddle required) and this
PyTorch port on the same inputs and reports the max logit difference plus
top-5 predictions for a sample image.

Example:
    python tools/compare_with_paddle.py --paddleclas /path/to/PaddleClas \
        --scale 1.0 --weights PPLCNet_x1_0_pretrained.pdparams \
        --image /path/to/PaddleClas/docs/images/inference_deployment/whl_demo.jpg
"""

import argparse
import os
import types

import numpy as np
import paddle
import torch
from PIL import Image

from pplcnet import PPLCNet
from pplcnet.data import preprocess

# Imports in pp_lcnet.py that pull in the rest of ppcls; replaced with stand-ins.
_IMPORT_STUBS = {
    "from .custom_devices_layers import AdaptiveAvgPool2D":
        "from paddle.nn import AdaptiveAvgPool2D",
    "from ..base.theseus_layer import TheseusLayer":
        "TheseusLayer = paddle.nn.Layer",
    "from ....utils.save_load import load_dygraph_pretrain":
        "load_dygraph_pretrain = None",
}


def load_paddle_module(paddleclas_root: str) -> types.ModuleType:
    path = os.path.join(paddleclas_root, "ppcls/arch/backbone/legendary_models/pp_lcnet.py")
    with open(path) as f:
        source = f.read()
    for old, new in _IMPORT_STUBS.items():
        if old not in source:
            raise RuntimeError(f"Expected import not found in {path}: {old}")
        source = source.replace(old, new)
    module = types.ModuleType("paddle_pp_lcnet")
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def load_labels(path: str):
    if not path:
        return None
    with open(path) as f:
        return [line.strip().split(" ", 1)[1] for line in f if line.strip()]


def describe_top5(logits: np.ndarray, labels) -> str:
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(-probs)[:5]
    return ", ".join(f"{i}:{labels[i] if labels else ''}({probs[i]:.3f})" for i in top)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paddleclas", required=True, help="PaddleClas checkout root")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--weights", required=True, help=".pdparams checkpoint")
    parser.add_argument("--image", help="sample image for a top-5 comparison")
    parser.add_argument("--labels", help="imagenet1k_label_list.txt (default: from PaddleClas)")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    paddle.set_device("cpu")
    pd_model = load_paddle_module(args.paddleclas).PPLCNet(scale=args.scale)
    pd_model.set_state_dict(paddle.load(args.weights))
    pd_model.eval()

    torch_model = PPLCNet(scale=args.scale)
    from pplcnet.convert import load_pretrained
    load_pretrained(torch_model, args.weights)
    torch_model.eval()

    inputs = {"random": np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)}
    if args.image:
        with Image.open(args.image) as img:
            inputs["image"] = preprocess(img)[None]
    labels = load_labels(args.labels or os.path.join(
        args.paddleclas, "ppcls/utils/imagenet1k_label_list.txt"))

    ok = True
    for name, x in inputs.items():
        with torch.no_grad():
            out_t = torch_model(torch.from_numpy(x)).numpy()
        out_p = pd_model(paddle.to_tensor(x)).numpy()
        diff = float(np.abs(out_t - out_p).max())
        same_top1 = bool((out_t.argmax(1) == out_p.argmax(1)).all())
        ok &= diff <= args.atol and same_top1
        print(f"[{name}] max |logit diff| = {diff:.2e}, top-1 agree = {same_top1}")
        if name == "image":
            print(f"  paddle: {describe_top5(out_p[0], labels)}")
            print(f"  torch : {describe_top5(out_t[0], labels)}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
