import copy
import io
import pickle

import pytest
import torch
import torch.nn as nn

import pplcnet
from pplcnet import NET_CONFIG, PPLCNet, param_groups
from pplcnet.convert import load_pdparams, paddle_to_torch_state_dict, pretrained_url
from pplcnet.model import DownscaleInInferDropout
from pplcnet.profile import count_flops, count_params

# Paper Table 2: (params in M, FLOPs in M) at 224x224, counted the Paddle way.
EXPECTED = {
    "x0_25": (1.52, 18), "x0_35": (1.65, 29), "x0_5": (1.89, 47), "x0_75": (2.37, 99),
    "x1_0": (2.96, 161), "x1_5": (4.52, 342), "x2_0": (6.54, 590), "x2_5": (9.04, 906),
}


@pytest.mark.parametrize("scale", list(EXPECTED))
def test_params_and_flops_match_paper(scale):
    model = getattr(pplcnet, f"PPLCNet_{scale}")()
    params, flops = EXPECTED[scale]
    assert round(count_params(model) / 1e6, 2) == params
    assert round(count_flops(model) / 1e6) == flops


def test_forward_shapes():
    model = PPLCNet(scale=1.0, class_num=10).eval()
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 10)
    stages = model.forward_stages(x)
    assert [t.shape[1] for t in stages] == model.out_channels == [32, 64, 128, 256, 512]
    assert [t.shape[-1] for t in stages] == [112, 56, 28, 14, 7]


def test_architecture_follows_table1():
    model = PPLCNet()
    blocks = [b for name in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6")
              for b in getattr(model, name)]
    assert len(blocks) == 13
    kernels = [b.dw_conv.conv.kernel_size[0] for b in blocks]
    assert kernels == [3] * 6 + [5] * 7
    assert [b.use_se for b in blocks] == [False] * 11 + [True] * 2
    assert all(isinstance(m, nn.Hardswish) for m in model.modules()
               if isinstance(m, (nn.ReLU, nn.Hardswish)) and not _inside_se(model, m))
    assert model.last_conv.bias is None


def _inside_se(model, module):
    return any(module is m for se in model.modules() if isinstance(se, pplcnet.model.SEModule)
               for m in se.modules())


def test_stride_list_does_not_leak_into_global_config():
    before = copy.deepcopy(NET_CONFIG)
    model = PPLCNet(stride_list=[2, 2, 2, 1, 1])
    assert NET_CONFIG == before
    assert model.forward_stages(torch.randn(1, 3, 64, 64))[-1].shape[-1] == 8
    assert PPLCNet().forward_stages(torch.randn(1, 3, 64, 64))[-1].shape[-1] == 2


def test_ablation_switches():
    model = PPLCNet(scale=0.5, act="relu", use_last_conv=False).eval()
    assert model.last_conv is None
    assert model.fc.in_features == 256
    assert not any(isinstance(m, nn.Hardswish) for m in model.modules())
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 1000)


def test_downscale_in_infer_dropout():
    drop = DownscaleInInferDropout(p=0.2)
    x = torch.ones(10000)
    drop.train()
    y = drop(x)
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    assert y.mean().item() == pytest.approx(0.8, abs=0.03)
    drop.eval()
    assert torch.allclose(drop(x), x * 0.8)


def test_param_groups():
    model = PPLCNet(lr_mult_list=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    groups = param_groups(model, lr=0.8, weight_decay=3e-5)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    lookup = {id(p): g for g in groups for p in g["params"]}
    assert lookup[id(model.conv1.bn.weight)]["weight_decay"] == 0.0
    assert lookup[id(model.conv1.conv.weight)]["lr"] == pytest.approx(0.08)
    assert lookup[id(model.blocks5[0].pw_conv.conv.weight)]["lr"] == pytest.approx(0.4)
    assert lookup[id(model.fc.weight)]["lr"] == pytest.approx(0.8)
    assert lookup[id(model.fc.weight)]["weight_decay"] == 3e-5
    torch.optim.SGD(groups, lr=0.8, momentum=0.9)


def _fake_pdparams(model):
    """Build a Paddle-style checkpoint from a torch model's state dict."""
    state = {}
    for k, v in model.state_dict().items():
        if k.endswith("num_batches_tracked"):
            continue
        k = k.replace("running_mean", "_mean").replace("running_var", "_variance")
        v = v.numpy()
        state[k] = v.T.copy() if k == "fc.weight" else v
    state["StructuredToParameterName@@"] = {"conv1.conv.weight": "conv2d_0.w_0"}
    return state


def test_paddle_checkpoint_roundtrip(tmp_path):
    src = PPLCNet(scale=0.5).eval()
    for m in src.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 2)
    path = tmp_path / "fake.pdparams"
    with open(path, "wb") as f:
        pickle.dump(_fake_pdparams(src), f, protocol=4)

    dst = PPLCNet(scale=0.5).eval()
    dst.load_state_dict(paddle_to_torch_state_dict(load_pdparams(str(path)), dst))
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(src(x), dst(x), atol=1e-5)


def test_protocol2_checkpoint(tmp_path):
    # Protocol 2 stores array bytes via _codecs.encode, as the SSLD checkpoints do.
    model = PPLCNet(scale=0.25)
    path = tmp_path / "proto2.pdparams"
    with open(path, "wb") as f:
        pickle.dump(_fake_pdparams(model), f, protocol=2)
    state = paddle_to_torch_state_dict(load_pdparams(str(path)), model)
    assert torch.equal(state["fc.weight"], model.fc.weight.detach())


def test_unpickler_rejects_arbitrary_globals(tmp_path):
    path = tmp_path / "evil.pdparams"
    with open(path, "wb") as f:
        pickle.dump({"x": io.BytesIO()}, f)
    with pytest.raises(pickle.UnpicklingError):
        load_pdparams(str(path))


def test_pretrained_url():
    assert pretrained_url(1.0).endswith("PPLCNet_x1_0_pretrained.pdparams")
    assert pretrained_url(2.5, use_ssld=True).endswith("PPLCNet_x2_5_ssld_pretrained.pdparams")
    with pytest.raises(ValueError):
        pretrained_url(0.35, use_ssld=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    model = PPLCNet().cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 224, 224, device="cuda"))
    assert out.shape == (4, 1000) and torch.isfinite(out).all()
