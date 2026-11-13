"""Shape operations must be layout-correct and, on GPU, zero-copy where possible."""

import numpy as np
import pytest

import vultorch as torch
from vultorch.testing import gradcheck

GPU_DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())]
DEVICES = ["cpu"] + GPU_DEVICES


def _rand(shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


@pytest.mark.parametrize("dev", DEVICES)
def test_transpose_matches_numpy(dev):
    x_np = _rand((3, 5), 1)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(x.transpose(0, 1).numpy(), x_np.T, rtol=1e-6)
    np.testing.assert_allclose(x.t().numpy(), x_np.T, rtol=1e-6)
    np.testing.assert_allclose(x.transpose(-1, -2).numpy(), x_np.T, rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_transpose_3d_non_adjacent(dev):
    x_np = _rand((2, 3, 4), 2)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(x.transpose(0, 2).numpy(), np.swapaxes(x_np, 0, 2), rtol=1e-6)
    np.testing.assert_allclose(x.transpose(1, 2).numpy(), np.swapaxes(x_np, 1, 2), rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_transpose_is_not_a_no_op(dev):
    """A row-major -> column-major transpose must move data, not just shapes."""
    x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_array_equal(x.transpose(0, 1).numpy(), x_np.T)


@pytest.mark.parametrize("dev", DEVICES)
def test_reshape_matches_numpy(dev):
    x_np = _rand((2, 6), 3)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(x.reshape(3, 4).numpy(), x_np.reshape(3, 4), rtol=1e-6)
    np.testing.assert_allclose(x.reshape(-1).numpy(), x_np.reshape(-1), rtol=1e-6)
    np.testing.assert_allclose(x.flatten().numpy(), x_np.flatten(), rtol=1e-6)


@pytest.mark.skipif(not GPU_DEVICES, reason="requires a Vulkan device")
def test_reshape_shares_storage_on_gpu():
    """reshape is a view: writing through the base must be visible in the view."""
    x = torch.tensor(np.zeros(6, np.float32), device="vk:0")
    y = x.reshape(2, 3)
    x.copy_(torch.tensor(np.full(6, 7.0, np.float32), device="vk:0"))
    np.testing.assert_allclose(y.numpy(), np.full((2, 3), 7.0, np.float32))


@pytest.mark.skipif(not GPU_DEVICES, reason="requires a Vulkan device")
def test_detach_shares_storage_on_gpu():
    x = torch.tensor(np.zeros(4, np.float32), device="vk:0", requires_grad=True)
    d = x.detach()
    assert d.requires_grad is False
    x.copy_(torch.tensor(np.full(4, 3.0, np.float32), device="vk:0"))
    np.testing.assert_allclose(d.numpy(), np.full(4, 3.0, np.float32))


@pytest.mark.parametrize("dev", DEVICES)
def test_unsqueeze_squeeze_roundtrip(dev):
    x_np = _rand((2, 3), 4)
    x = torch.tensor(x_np, device=dev)
    assert x.unsqueeze(0).shape == (1, 2, 3)
    assert x.unsqueeze(-1).shape == (2, 3, 1)
    np.testing.assert_allclose(x.unsqueeze(1).squeeze(1).numpy(), x_np, rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_expand_matches_numpy(dev):
    x_np = _rand((1, 4), 5)
    x = torch.tensor(x_np, device=dev)
    np.testing.assert_allclose(x.expand(3, 4).numpy(), np.broadcast_to(x_np, (3, 4)), rtol=1e-6)

    v_np = _rand((3,), 6)
    v = torch.tensor(v_np, device=dev)
    np.testing.assert_allclose(v.expand(2, 3).numpy(), np.broadcast_to(v_np, (2, 3)), rtol=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_expand_invalid_raises(dev):
    x = torch.tensor(_rand((2, 3), 7), device=dev)
    with pytest.raises(ValueError):
        _ = x.expand(4, 3)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_shape_ops(dev):
    x = _rand((2, 3), 8)
    assert gradcheck(lambda t: (t.transpose(0, 1) ** 2).sum(), x, device=dev)
    assert gradcheck(lambda t: (t.reshape(6) ** 2).sum(), x, device=dev)
    assert gradcheck(lambda t: (t.unsqueeze(0) ** 2).sum(), x, device=dev)

    v = _rand((1, 3), 9)
    assert gradcheck(lambda t: (t.expand(4, 3) ** 2).sum(), v, device=dev)
