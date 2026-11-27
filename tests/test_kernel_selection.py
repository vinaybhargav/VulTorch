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


GEMM_VARIANTS = [
    "matmul",
    "matmul_tiled64",
    "matmul_tiled",
    # Slang generic SGEMM (shaders/gemm.slang)
    "gemm_t32_k16_r2",
    "gemm_t64_k16_r4",
    "gemm_t64_k8_r4",
    "gemm_t64_k16_r8",
    "gemm_t128_k8_r8",
    "gemm_t128_k16_r8",
]


@pytest.mark.parametrize("device", GPU_DEVICES)
@pytest.mark.parametrize("shader", GEMM_VARIANTS)
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 64),  # exactly one 64 tile
        (129, 200, 97),  # ragged in every dimension
        (256, 512, 384),  # several tiles, K not a multiple of 16
        (300, 7, 131),  # K smaller than one K step
    ],
)
def test_gemm_variant_matches_numpy(device, shader, shape, monkeypatch):
    monkeypatch.setenv("VULTORCH_MATMUL_KERNEL", shader)
    rows, inner, cols = shape

    a_np, b_np = _rand((rows, inner), 1), _rand((inner, cols), 2)
    a = torch.tensor(a_np, device=device)
    b = torch.tensor(b_np, device=device)

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


def _has_fp16_coopmat() -> bool:
    if torch.vulkan.device_count() == 0:
        return False
    return bool(torch.vulkan.get_device_properties(0)["capabilities"]["fp16_coopmat_16x16"])


@pytest.mark.skipif(not _has_fp16_coopmat(), reason="requires 16x16x16 f16 cooperative matrix")
@pytest.mark.parametrize("shape", [(64, 64, 64), (129, 200, 97), (256, 512, 384)])
def test_coopmat_variant_matches_reference(shape, monkeypatch):
    """The cooperative-matrix kernel is opt-in, but must stay correct.

    Operands are narrowed to f16 with f32 accumulation, so the expected error
    grows like sqrt(K) times the f16 epsilon. The tolerance below is derived
    from that, not tuned until the test passed; a structurally wrong kernel
    produces errors orders of magnitude larger.
    """
    monkeypatch.setenv("VULTORCH_MATMUL_KERNEL", "matmul_coopmat")
    rows, inner, cols = shape

    a_np, b_np = _rand((rows, inner), 11), _rand((inner, cols), 12)
    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")

    atol = 5e-3 * np.sqrt(inner)
    np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, rtol=1e-2, atol=atol)


@pytest.mark.skipif(not _has_fp16_coopmat(), reason="requires 16x16x16 f16 cooperative matrix")
def test_coopmat_is_not_selected_automatically(monkeypatch):
    """A variant that loses the measurement must not be picked on its own."""
    monkeypatch.delenv("VULTORCH_MATMUL_KERNEL", raising=False)
    a = torch.tensor(_rand((128, 128), 13), device="vk:0")
    b = torch.tensor(_rand((128, 128), 14), device="vk:0")
    # If the coopmat kernel were auto-selected the result would carry f16 error;
    # the automatic path must stay on the fp32 SIMT kernels.
    np.testing.assert_allclose((a @ b).numpy(), a.numpy() @ b.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", GPU_DEVICES)
@pytest.mark.parametrize("shader", [v for v in GEMM_VARIANTS if v.startswith("gemm_")])
def test_batched_gemm_variant_matches_numpy(device, shader, monkeypatch):
    monkeypatch.setenv("VULTORCH_MATMUL_KERNEL", shader)
    a_np, b_np = _rand((3, 70, 45), 3), _rand((3, 45, 150), 4)
    out = torch.tensor(a_np, device=device) @ torch.tensor(b_np, device=device)
    np.testing.assert_allclose(out.numpy(), a_np @ b_np, rtol=1e-3, atol=1e-3)
