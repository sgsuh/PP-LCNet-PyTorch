"""ONNX export with an ONNX Runtime numerical check."""

from typing import Optional

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, path: str, image_size: int = 224,
                opset: int = 18, dynamic_batch: bool = True) -> str:
    """Export `model` to ONNX with input "input" (N, 3, H, W) and output "logits"."""
    model = model.cpu().eval()
    x = torch.randn(2 if dynamic_batch else 1, 3, image_size, image_size)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=1024)}} if dynamic_batch else None
    # Wrap so the exported input is always named `x`, whatever forward() calls it.
    wrapper = _Wrapper(model)
    program = torch.onnx.export(
        wrapper, (x,), dynamo=True, opset_version=opset, dynamic_shapes=dynamic_shapes,
        input_names=["input"], output_names=["logits"])
    program.optimize()
    program.save(path)
    return path


class _Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def check_onnx(model: torch.nn.Module, path: str, image_size: int = 224, batch: int = 4,
               seed: int = 0, providers: Optional[list] = None) -> float:
    """Return the max abs logit difference between PyTorch (CPU, FP32) and ONNX Runtime."""
    import onnxruntime as ort

    x = np.random.default_rng(seed).standard_normal((batch, 3, image_size, image_size)).astype(np.float32)
    session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {"input": x})
    with torch.no_grad():
        torch_out = model.cpu().eval()(torch.from_numpy(x)).numpy()
    return float(np.abs(onnx_out - torch_out).max())
