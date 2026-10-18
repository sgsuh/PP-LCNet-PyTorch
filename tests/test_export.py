import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from pplcnet import PPLCNet  # noqa: E402
from pplcnet.export import check_onnx, export_onnx  # noqa: E402


def _randomize_bn(model):
    # Fresh BN layers are identity at eval time; random stats exercise BN folding.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    return model


@pytest.mark.parametrize("batch", [1, 3])
def test_onnx_matches_torch(tmp_path, batch):
    model = _randomize_bn(PPLCNet(scale=0.25)).eval()
    path = export_onnx(model, str(tmp_path / "m.onnx"))
    with torch.no_grad():
        scale = model(torch.randn(2, 3, 224, 224)).abs().max().item()
    assert scale > 0.1
    assert check_onnx(model, path, batch=batch) < 1e-4 * max(1.0, scale)
