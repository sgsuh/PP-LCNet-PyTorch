import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from train import build_model, build_net_config, get_parser, warmup_cosine  # noqa: E402

from pplcnet import NET_CONFIG  # noqa: E402


def test_build_net_config_switches():
    before = [b[:] for blocks in NET_CONFIG.values() for b in blocks]
    config = build_net_config(use_5x5=False, use_se=False)
    blocks = [b for bs in config.values() for b in bs]
    assert all(b[0] == 3 and not b[4] for b in blocks)
    assert [b[:] for bs in NET_CONFIG.values() for b in bs] == before


def test_basenet_model():
    args = get_parser().parse_args(["--root", "x", "--act", "relu", "--no-5x5", "--no-se",
                                     "--no-last-conv", "--dropout", "0"])
    model = build_model(args, num_classes=10)
    assert model.last_conv is None and model.fc.out_features == 10
    assert not any(b.use_se for b in model.blocks6)


def test_warmup_cosine():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5)
    assert f(99) < 0.01
