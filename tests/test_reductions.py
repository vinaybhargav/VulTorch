"""Reductions along every axis, on every device, against NumPy."""

import numpy as np
import pytest

import vultorch as torch
from vultorch.testing import gradcheck

GPU_DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())]
DEVICES = ["cpu"] + GPU_DEVICES


def _rand(shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_sum_along_dim_matches_numpy(dev, dim, keepdim):
    x_np = _rand((4, 5), 1)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(
        x.sum(dim=dim, keepdim=keepdim).numpy(),
        x_np.sum(axis=dim, keepdims=keepdim),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_sum_3d_along_dim_matches_numpy(dev, dim):
    x_np = _rand((2, 3, 4), 2)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(
        x.sum(dim=dim).numpy(), x_np.sum(axis=dim), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_mean_along_dim_matches_numpy(dev, dim):
    x_np = _rand((3, 7), 3)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(
        x.mean(dim=dim).numpy(), x_np.mean(axis=dim), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_max_along_dim_matches_numpy(dev, dim):
    x_np = _rand((5, 6), 4)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(
        x.max(dim=dim).numpy(), x_np.max(axis=dim), rtol=1e-6
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_total_sum_and_max(dev):
    x_np = _rand((3, 4), 5)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(float(x.sum().item()), float(x_np.sum()), rtol=1e-5)
    np.testing.assert_allclose(float(x.max().item()), float(x_np.max()), rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_reduction_of_non_contiguous_input_is_still_correct(dev):
    """A transposed tensor is materialised first, so results must still match."""
    x_np = _rand((4, 5), 6)
    x = torch.tensor(x_np, device=dev)
    xt = x.transpose(0, 1)
    np.testing.assert_allclose(xt.sum(dim=0).numpy(), x_np.T.sum(axis=0), rtol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1])
def test_gradcheck_reductions(dev, dim):
    x = _rand((3, 4), 7)
    assert gradcheck(lambda t: (t.sum(dim=dim) ** 2).sum(), x, device=dev)
    assert gradcheck(lambda t: (t.mean(dim=dim) ** 2).sum(), x, device=dev)


@pytest.mark.parametrize("dev", DEVICES)
def test_bias_gradient_shape(dev):
    """sum(dim=0) over a 2-D tensor is the classic bias-gradient case."""
    import vultorch.nn as nn

    layer = nn.Linear(8, 4, device=dev)
    x = torch.tensor(_rand((16, 8), 8), device=dev)
    layer(x).sum().backward()

    assert layer.bias.grad is not None
    np.testing.assert_allclose(
        layer.bias.grad.numpy(), np.full(4, 16.0, np.float32), rtol=1e-5
    )
