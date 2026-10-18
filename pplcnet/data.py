"""Evaluation preprocessing and the Imagenette subset of ImageNet."""

import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Imagenette class folders (WordNet IDs) -> ImageNet-1k class index.
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def preprocess(img: Image.Image, resize_short: int = 256, crop: int = 224) -> np.ndarray:
    """PaddleClas eval transform: resize short side, center crop, normalize. Returns CHW."""
    img = img.convert("RGB")
    w, h = img.size
    ratio = resize_short / min(w, h)
    img = img.resize((round(w * ratio), round(h * ratio)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class ImagenetteEval(torch.utils.data.Dataset):
    """Imagenette split with labels as ImageNet-1k indices.

    Use the `val` split for pretrained ImageNet models: its images come from
    the ImageNet validation set, while `train` overlaps ImageNet training data.
    """

    def __init__(self, root: str, split: str = "val"):
        self.samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(root, split)
        for wnid in sorted(os.listdir(split_dir)):
            if wnid not in IMAGENETTE_TO_IMAGENET:
                raise ValueError(f"Unknown Imagenette class folder: {wnid}")
            class_dir = os.path.join(split_dir, wnid)
            for name in sorted(os.listdir(class_dir)):
                self.samples.append((os.path.join(class_dir, name), IMAGENETTE_TO_IMAGENET[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            return torch.from_numpy(preprocess(img)), label


class Imagenette(torch.utils.data.Dataset):
    """Imagenette split with labels 0..9 (sorted WordNet IDs), for training from scratch.

    `transform` maps a PIL image to a tensor. Corrupt files are not expected;
    images are converted to RGB because a few Imagenette files are grayscale/CMYK.
    """

    classes = sorted(IMAGENETTE_TO_IMAGENET)

    def __init__(self, root: str, split: str, transform):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(root, split)
        for label, wnid in enumerate(self.classes):
            class_dir = os.path.join(split_dir, wnid)
            for name in sorted(os.listdir(class_dir)):
                self.samples.append((os.path.join(class_dir, name), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label
