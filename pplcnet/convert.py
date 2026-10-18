"""Convert official PaddleClas PP-LCNet checkpoints (`.pdparams`) to PyTorch.

A `.pdparams` file is a pickled dict of numpy arrays. It is read with a
restricted unpickler that only reconstructs numpy arrays, so neither
PaddlePaddle nor arbitrary pickle globals are needed.
"""

import os
import pickle
from typing import Dict, Optional, Union

import numpy as np
import torch

from .model import PPLCNet

_URL_TEMPLATE = (
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/"
    "PPLCNet_{scale}_{kind}.pdparams")

SCALE_NAMES = {
    0.25: "x0_25", 0.35: "x0_35", 0.5: "x0_5", 0.75: "x0_75",
    1.0: "x1_0", 1.5: "x1_5", 2.0: "x2_0", 2.5: "x2_5",
}
# Scales with an SSLD-distilled checkpoint published by PaddleClas.
SSLD_SCALES = (0.25, 0.5, 1.0, 2.5)

# Metadata key written by paddle.save; not a tensor.
_PADDLE_META_KEYS = ("StructuredToParameterName@@",)

_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
    # Protocol-2 pickles (e.g. the SSLD checkpoints) encode raw bytes this way.
    ("_codecs", "encode"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Refusing to load global {module}.{name}")
        return super().find_class(module, name)


def pretrained_url(scale: float, use_ssld: bool = False) -> str:
    if scale not in SCALE_NAMES:
        raise ValueError(f"No official checkpoint for scale {scale}")
    if use_ssld and scale not in SSLD_SCALES:
        raise ValueError(f"No SSLD checkpoint for scale {scale}; available: {SSLD_SCALES}")
    kind = "ssld_pretrained" if use_ssld else "pretrained"
    return _URL_TEMPLATE.format(scale=SCALE_NAMES[scale], kind=kind)


def load_pdparams(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        state = _NumpyOnlyUnpickler(f, encoding="latin1").load()
    return {k: v for k, v in state.items() if k not in _PADDLE_META_KEYS}


def paddle_to_torch_state_dict(paddle_state: Dict[str, np.ndarray],
                               model: PPLCNet) -> Dict[str, torch.Tensor]:
    """Rename and reshape Paddle parameters to match `model.state_dict()`."""
    target = model.state_dict()
    converted = {}
    for key, value in paddle_state.items():
        name = key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
        if name not in target:
            raise KeyError(f"Unexpected parameter in checkpoint: {key}")
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        # Paddle Linear stores weights as [in_features, out_features].
        if isinstance(getattr(model, "fc", None), torch.nn.Linear) and name == "fc.weight":
            tensor = tensor.t()
        if tensor.shape != target[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                f"vs model {tuple(target[name].shape)}")
        converted[name] = tensor.to(target[name].dtype)

    missing = [k for k in target if k not in converted and not k.endswith("num_batches_tracked")]
    if missing:
        raise KeyError(f"Checkpoint is missing parameters: {missing}")
    for k in target:
        if k.endswith("num_batches_tracked"):
            converted[k] = target[k].clone()
    return converted


def _download(url: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        torch.hub.download_url_to_file(url, path, progress=True)
    return path


def load_pretrained(model: PPLCNet, pretrained: Union[bool, str], use_ssld: bool = False) -> None:
    """Load weights into `model`.

    `pretrained` may be True (download the official checkpoint for
    `model.scale`), a path to a `.pdparams` file, or a path to a PyTorch
    state dict (`.pth` / `.pt`).
    """
    if pretrained is True:
        path = _download(pretrained_url(model.scale, use_ssld))
    elif isinstance(pretrained, str):
        path = pretrained
    else:
        raise TypeError("pretrained must be True or a path")

    if path.endswith(".pdparams"):
        state = paddle_to_torch_state_dict(load_pdparams(path), model)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
