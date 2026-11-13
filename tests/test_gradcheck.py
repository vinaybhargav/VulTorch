"""gradcheck: analytic gradients must match central finite differences.

These are the tests that make a ported or rewritten kernel acceptable: a kernel
whose backward pass disagrees with numerical differentiation is rejected here
before it ever reaches a model.
"""

import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn
from vultorch.autograd import Node
from vultorch.testing import gradcheck

DEVICES = ["cpu"] + (["vk:0"] if torch.vulkan.device_count() > 0 else [])


def _rand(shape, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_elementwise(dev):
    x = np.array([-1.5, -0.4, 0.7, 1.9], np.float32)
    cases = {
        "add": lambda t: (t + 1.5).sum(),
        "mul_scalar": lambda t: (t * 3.0).sum(),
        "mul": lambda t: (t * t).sum(),
        "pow": lambda t: (t ** 3.0).sum(),
        "neg": lambda t: (-t).sum(),
        "exp": lambda t: t.exp().sum(),
        "tanh": lambda t: t.tanh().sum(),
        "sigmoid": lambda t: t.sigmoid().sum(),
        "gelu": lambda t: t.gelu().sum(),
        "silu": lambda t: t.silu().sum(),
    }
    for name, fn in cases.items():
        assert gradcheck(fn, x, device=dev), name


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_positive_domain(dev):
    x = np.array([0.6, 1.2, 2.4], np.float32)
    for name, fn in {
        "log": lambda t: t.log().sum(),
        "sqrt": lambda t: t.sqrt().sum(),
        "relu": lambda t: t.relu().sum(),
    }.items():
        assert gradcheck(fn, x, device=dev), name


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_reductions(dev):
    x = np.array([[-1.0, 2.0, 0.5], [3.0, -2.0, 1.0]], np.float32)
    for name, fn in {
        "sum_dim": lambda t: (t.sum(dim=1) ** 2).sum(),
        "mean_dim": lambda t: (t.mean(dim=0) ** 2).sum(),
        "sum_all": lambda t: t.sum() * 2.0,
    }.items():
        assert gradcheck(fn, x, device=dev), name


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_broadcast_binary(dev):
    a = np.array([[1.0], [2.0]], np.float32)
    b = np.array([0.5, -1.0, 2.0], np.float32)
    assert gradcheck(lambda x, y: ((x * y) ** 2).sum(), a, b, device=dev)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_matmul(dev):
    a, b = _rand((2, 3), 1), _rand((3, 2), 2)
    assert gradcheck(lambda x, y: ((x @ y) ** 2).sum(), a, b, device=dev)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_shape_ops(dev):
    x = _rand((2, 3), 3)
    for name, fn in {
        "transpose": lambda t: (t.transpose(0, 1) ** 2).sum(),
        "reshape": lambda t: (t.reshape(3, 2) ** 2).sum(),
        "unsqueeze": lambda t: (t.unsqueeze(0) ** 2).sum(),
    }.items():
        assert gradcheck(fn, x, device=dev), name


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_softmax(dev):
    x = _rand((2, 3), 4)
    for name, fn in {
        "softmax": lambda t: (t.softmax(-1) ** 2).sum(),
        "log_softmax": lambda t: (t.log_softmax(-1) ** 2).sum(),
        "softmax_dim0": lambda t: (t.softmax(0) ** 2).sum(),
    }.items():
        assert gradcheck(fn, x, device=dev), name


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_indexing_and_max(dev):
    x = _rand((2, 3), 5)
    for name, fn in {
        "getitem": lambda t: (t[0:1] ** 2).sum(),
        "max_all": lambda t: (t.max() ** 2),
        "max_dim": lambda t: (t.max(dim=1) ** 2).sum(),
    }.items():
        assert gradcheck(fn, x, device=dev), name


def test_gradcheck_layernorm():
    ln = nn.LayerNorm(3, device="cpu")
    ln.weight.copy_(np.array([1.0, 0.5, 2.0], np.float32))
    ln.bias.copy_(np.array([0.0, 0.1, -0.2], np.float32))
    x = _rand((2, 3), 6)
    assert gradcheck(lambda t: (ln(t) ** 2).sum(), x, device="cpu")


def test_gradcheck_detects_a_wrong_gradient():
    """The harness must fail on an intentionally incorrect gradient."""

    def wrong(t):
        out = t * 2.0
        # Correct gradient is 2.0; pretend it is 4.0.
        out.grad_fn = Node("Wrong", lambda g: g * 4.0, [t])
        return out.sum()

    with pytest.raises(AssertionError):
        gradcheck(wrong, np.array([1.0, 2.0], np.float32), device="cpu")
