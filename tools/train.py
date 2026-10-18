"""Train PP-LCNet from scratch on Imagenette.

The recipe follows the paper / PaddleClas ImageNet config, with the learning
rate scaled linearly from 0.8 @ batch 2048: SGD momentum 0.9, cosine decay
with linear warmup (per iteration), label smoothing 0.1, weight decay 3e-5
(no decay on BatchNorm), RandomResizedCrop + horizontal flip.

Architecture switches mirror the paper's ablation (Table 3):
    --act relu|hardswish, --no-5x5, --no-se, --no-last-conv, --dropout

Example:
    python tools/train.py --root ~/data/imagenette2 --scale 0.5 --epochs 100 --out run.json
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T

from pplcnet import NET_CONFIG, PPLCNet, param_groups
from pplcnet.data import MEAN, STD, Imagenette


def build_net_config(use_5x5: bool = True, use_se: bool = True) -> dict:
    """NET_CONFIG with the paper's 5x5 kernels and/or SE blocks switched off."""
    config = copy.deepcopy(NET_CONFIG)
    for blocks in config.values():
        for block in blocks:
            if not use_5x5:
                block[0] = 3
            if not use_se:
                block[4] = False
    return config


def build_model(args, num_classes: int) -> PPLCNet:
    return PPLCNet(
        scale=args.scale,
        class_num=num_classes,
        act=args.act,
        net_config=build_net_config(use_5x5=not args.no_5x5, use_se=not args.no_se),
        use_last_conv=not args.no_last_conv,
        dropout_prob=args.dropout)


def build_loaders(args):
    normalize = [T.ToImage(), T.ToDtype(torch.float32, scale=True),
                 T.Normalize(MEAN.tolist(), STD.tolist())]
    train_tf = T.Compose([T.RandomResizedCrop(args.image_size, antialias=True),
                          T.RandomHorizontalFlip(), *normalize])
    eval_tf = T.Compose([T.Resize(int(args.image_size * 256 / 224), antialias=True),
                         T.CenterCrop(args.image_size), *normalize])
    train_set = Imagenette(args.root, "train", train_tf)
    val_set = Imagenette(args.root, "val", eval_tf)
    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, **common)
    return train_loader, val_loader, len(train_set.classes)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, num_classes = build_loaders(args)
    model = build_model(args, num_classes).to(device).to(memory_format=torch.channels_last)
    lr = args.lr if args.lr is not None else 0.8 * args.batch_size / 2048
    optimizer = torch.optim.SGD(param_groups(model, lr, args.weight_decay),
                                lr=lr, momentum=0.9)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        last = epoch == args.epochs - 1
        record = {"epoch": epoch + 1, "loss": loss_sum / steps_per_epoch}
        if last or (epoch + 1) % args.eval_interval == 0:
            record["val_top1"] = evaluate(model, val_loader, device)
        history.append(record)
        print(json.dumps(record), flush=True)

    result = {
        "config": vars(args),
        "lr": lr,
        "params": sum(p.numel() for p in model.parameters()),
        "final_top1": history[-1]["val_top1"],
        "best_top1": max(r["val_top1"] for r in history if "val_top1" in r),
        "train_minutes": (time.time() - start) / 60,
        "history": history,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="imagenette2 directory")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--act", default="hardswish", choices=["hardswish", "relu", "relu6"])
    parser.add_argument("--no-5x5", action="store_true", help="use 3x3 depthwise everywhere")
    parser.add_argument("--no-se", action="store_true", help="remove SE from the last blocks")
    parser.add_argument("--no-last-conv", action="store_true", help="remove 1x1 conv after GAP")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, help="default: 0.8 * batch_size / 2048")
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", help="write result JSON here")
    return parser


if __name__ == "__main__":
    result = train(get_parser().parse_args())
    print(f"final top-1 {result['final_top1']:.2f}  best {result['best_top1']:.2f}  "
          f"({result['train_minutes']:.1f} min)")
