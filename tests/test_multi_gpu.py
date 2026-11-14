import pytest
import numpy as np
import vultorch as torch
import vultorch.nn as nn

@pytest.mark.skipif(torch.vulkan.device_count() < 2, reason="Multi-GPU test requires at least 2 Vulkan devices")
def test_cross_gpu_transfer():
    x_amd = torch.randn(64, 64, device="vk:0")
    # Transfer from AMD 7900 XTX to NVIDIA RTX 3080
    x_nvidia = x_amd.to("vk:1")
    assert x_nvidia.device == torch.device("vk:1")
    np.testing.assert_allclose(x_amd.numpy(), x_nvidia.numpy(), rtol=1e-5, atol=1e-5)

    # Transfer back
    x_back = x_nvidia.to("vk:0")
    assert x_back.device == torch.device("vk:0")
    np.testing.assert_allclose(x_amd.numpy(), x_back.numpy(), rtol=1e-5, atol=1e-5)

@pytest.mark.skipif(torch.vulkan.device_count() < 2, reason="Multi-GPU test requires at least 2 Vulkan devices")
def test_concurrent_multi_gpu_compute():
    # Compute on AMD GPU
    a0 = torch.randn(128, 128, device="vk:0")
    b0 = torch.randn(128, 128, device="vk:0")
    c0 = a0 @ b0

    # Compute on NVIDIA GPU
    a1 = torch.randn(128, 128, device="vk:1")
    b1 = torch.randn(128, 128, device="vk:1")
    c1 = a1 @ b1

    np.testing.assert_allclose(c0.numpy(), a0.numpy() @ b0.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c1.numpy(), a1.numpy() @ b1.numpy(), rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(torch.vulkan.device_count() < 2, reason="Multi-GPU test requires at least 2 Vulkan devices")
def test_data_parallel_heterogeneous():
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    ).to("vk:0")

    dp_model = nn.DataParallel(model, device_ids=["vk:0", "vk:1"])
    x = torch.randn(16, 32, device="vk:0")
    out = dp_model(x)

    assert out.shape == (16, 4)
    assert out.device == torch.device("vk:0")
    assert sum(dp_model.last_split_sizes) == 16
    assert all(s >= 1 for s in dp_model.last_split_sizes)

    # Verify DataParallel output matches single-device reference
    ref = model(x)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(torch.vulkan.device_count() < 2, reason="Multi-GPU test requires at least 2 Vulkan devices")
def test_tflops_load_balancer_weights():
    t0 = torch.vulkan.get_device_tflops(0)
    t1 = torch.vulkan.get_device_tflops(1)
    assert t0 > 0.0 and t1 > 0.0

    splits = torch.vulkan.compute_split_sizes(100, ["vk:0", "vk:1"])
    assert sum(splits) == 100
    # More capable GPU receives a strictly larger share of the batch
    if t1 > t0:
        assert splits[1] > splits[0]
    elif t0 > t1:
        assert splits[0] > splits[1]
