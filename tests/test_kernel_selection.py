"""Device capabilities and GEMM variant selection.

The selection mechanism is only trustworthy if (a) the reported capabilities are
real and (b) every variant the framework might pick is numerically correct.
"""

import numpy as np
import pytest

import vultorch as torch

GPU_DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())]

pytestmark = pytest.mark.skipif(not GPU_DEVICES, reason="requires a Vulkan device")


def _rand(shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


def test_capabilities_are_populated():
    for i in range(torch.vulkan.device_count()):
        info = torch.vulkan.get_device_properties(i)
        caps = info["capabilities"]

        # A subgroup is at least one invocation; 0 means the properties query
        # silently failed (chaining it onto the features query does exactly that).
        assert caps["subgroup_size"] in (4, 8, 16, 32, 64, 128), caps
        assert caps["subgroup_size"] > 0
        assert caps["max_shared_memory_bytes"] >= 16384
        assert caps["max_workgroup_invocations"] >= 128
        assert isinstance(caps["float16_arithmetic"], bool)
        assert isinstance(caps["storage_16bit"], bool)


def test_capabilities_are_consistent_with_device_count():
    assert len(torch.vulkan.get_all_devices()) == torch.vulkan.device_count()


@pytest.mark.parametrize("shader", ["matmul", "matmul_tiled64", "matmul_tiled"])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),        # exactly one 64 tile
        (129, 200, 97),      # ragged in every dimension
        (256, 512, 384),     # several tiles, K not a multiple of 16
    ],
)
def test_gemm_variant_matches_numpy(shader, shape, monkeypatch):
    monkeypatch.setenv("VULTORCH_MATMUL_KERNEL", shader)
    rows, inner, cols = shape

    a_np, b_np = _rand((rows, inner), 1), _rand((inner, cols), 2)
    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")

    np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, rtol=1e-3, atol=1e-3)


def test_tiny_matmul_still_matches_numpy(monkeypatch):
    """Below the autotune threshold the baseline kernel must still be correct."""
    monkeypatch.delenv("VULTORCH_MATMUL_KERNEL", raising=False)
    a_np, b_np = _rand((4, 5), 3), _rand((5, 6), 4)
    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")
    np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, rtol=1e-4, atol=1e-4)


def test_repeated_shapes_are_stable(monkeypatch):
    """The cached choice must produce identical results on every call."""
    monkeypatch.delenv("VULTORCH_MATMUL_KERNEL", raising=False)
    a = torch.tensor(_rand((512, 512), 5), device="vk:0")
    b = torch.tensor(_rand((512, 512), 6), device="vk:0")

    reference = (a @ b).numpy()
    for _ in range(3):
        np.testing.assert_allclose((a @ b).numpy(), reference, rtol=1e-6, atol=1e-6)


def test_autograd_through_tuned_matmul(monkeypatch):
    monkeypatch.delenv("VULTORCH_MATMUL_KERNEL", raising=False)
    import vultorch.nn as nn

    layer = nn.Linear(256, 512).to("vk:0")
    x = torch.tensor(_rand((128, 256), 7), device="vk:0", requires_grad=True)
    layer(x).sum().backward()

    assert x.grad is not None
    assert np.isfinite(x.grad.numpy()).all()
    assert layer.weight.grad is not None
