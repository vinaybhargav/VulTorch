import pytest
import vultorch as torch


def test_vulkan_device_discovery():
    assert torch.vulkan.is_available()
    count = torch.vulkan.device_count()
    assert count >= 1

    for i in range(count):
        props = torch.vulkan.get_device_properties(i)
        assert "name" in props
        assert "vendor" in props
        assert "total_memory" in props
        assert props["total_memory"] > 0
        print(
            f"Discovered GPU {i}: {props['name']} ({props['vendor']}) - Memory: {props['total_memory'] / (1024**3):.2f} GB"
        )


def test_device_object():
    d_cpu = torch.device("cpu")
    assert d_cpu.type == "cpu"
    assert d_cpu.device_id == -1

    d_vk0 = torch.device("vk:0")
    assert d_vk0.type == "vk"
    assert d_vk0.device_id == 0

    if torch.vulkan.device_count() > 1:
        d_vk1 = torch.device("vk:1")
        assert d_vk1.type == "vk"
        assert d_vk1.device_id == 1
