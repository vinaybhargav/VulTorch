import pytest
import vultorch as torch
import time
import os


def test_pipeline_cache_persistence():
    cache_dir = os.path.expanduser("~/.cache/vultorch")
    assert os.path.exists(cache_dir)
    # Exercise a dispatch on vk:0 and vk:1
    for dev_id in range(torch.vulkan.device_count()):
        dev = f"vk:{dev_id}"
        x = torch.randn(64, 64, device=dev)
        y = torch.sin(x) + torch.cos(x)
        torch.vulkan.synchronize(dev)
        bin_path = os.path.join(cache_dir, f"pipeline_cache_dev{dev_id}.bin")
        assert os.path.exists(bin_path)
        assert os.path.getsize(bin_path) > 16


def test_subgroup_wave_reduction_accuracy():
    for dev_id in range(torch.vulkan.device_count()):
        dev = f"vk:{dev_id}"
        x = torch.randn(128, 512, device=dev)
        s = x.sum(dim=1)
        m = x.max(dim=1)
        torch.vulkan.synchronize(dev)

        # Verify correctness against CPU
        x_cpu = x.cpu()
        assert torch.allclose(s.cpu(), x_cpu.sum(dim=1), rtol=1e-4, atol=1e-4)
        assert torch.allclose(m.cpu(), x_cpu.max(dim=1), rtol=1e-4, atol=1e-4)


def test_subgroup_softmax_accuracy():
    for dev_id in range(torch.vulkan.device_count()):
        dev = f"vk:{dev_id}"
        x = torch.randn(64, 256, device=dev)
        sm = torch.softmax(x, dim=1)
        log_sm = torch.log_softmax(x, dim=1)
        torch.vulkan.synchronize(dev)

        x_cpu = x.cpu()
        assert torch.allclose(sm.cpu(), torch.softmax(x_cpu, dim=1), rtol=1e-4, atol=1e-4)
        assert torch.allclose(log_sm.cpu(), torch.log_softmax(x_cpu, dim=1), rtol=1e-4, atol=1e-4)


def test_device_capabilities_reported():
    for dev_id in range(torch.vulkan.device_count()):
        props = torch.vulkan.get_device_properties(dev_id)
        caps = props["capabilities"]
        assert "subgroup_size" in caps
        assert "subgroup_basic" in caps
        assert "subgroup_arithmetic" in caps
        assert "shader_atomic_float" in caps
        assert caps["subgroup_size"] in [16, 32, 64]
