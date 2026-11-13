"""Execution-engine tests: batching, memory pooling, and thread safety.

These target the failure modes introduced by deferring submission:
stale reads, descriptor-set lifetime, buffer reuse, and concurrent access.
"""

import threading

import numpy as np
import pytest

import vultorch as torch

GPU_DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())]
HAS_GPU = bool(GPU_DEVICES)


def _rand(shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_few_ops_batched_together_are_correct():
    x_np = _rand((32, 32), 1)
    x = torch.tensor(x_np, device="vk:0")
    y = x
    for _ in range(50):
        y = y + 0.25
    np.testing.assert_allclose(y.numpy(), x_np + 50 * 0.25, rtol=1e-5, atol=1e-4)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_many_ops_cross_the_batch_boundary():
    """More dispatches than kMaxOpsPerBatch must still be ordered correctly."""
    x_np = _rand((64,), 2)
    x = torch.tensor(x_np, device="vk:0")
    y = x
    for _ in range(600):
        y = y + 1.0
    np.testing.assert_allclose(y.numpy(), x_np + 600.0, rtol=1e-5, atol=1e-2)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_descriptor_pool_pressure():
    """Exceed the pending-descriptor threshold and stay correct."""
    x = torch.tensor(np.zeros(64, np.float32), device="vk:0")
    y = x
    for _ in range(1500):
        y = y + 1.0
    np.testing.assert_allclose(y.numpy(), np.full(64, 1500.0, np.float32), rtol=1e-5)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_host_read_forces_completion():
    """Reading a tensor must observe every op recorded before it."""
    x_np = _rand((16,), 3)
    x = torch.tensor(x_np, device="vk:0")

    y = x * 2.0
    mid = y.numpy()  # must flush the open batch
    np.testing.assert_allclose(mid, x_np * 2.0, rtol=1e-6)

    z = y * 3.0
    np.testing.assert_allclose(z.numpy(), x_np * 6.0, rtol=1e-6)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_recycled_buffers_do_not_leak_stale_values():
    """A pooled buffer handed to a new tensor must not carry old contents."""
    for _ in range(50):
        filled = torch.ones(1024, device="vk:0")
        assert float(filled.numpy().sum()) == 1024.0
        del filled

        fresh = torch.zeros(1024, device="vk:0")
        assert float(fresh.numpy().sum()) == 0.0
        del fresh


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_allocation_churn_stays_correct():
    for i in range(100):
        t = torch.randn(256, device="vk:0") * 2.0
        assert np.isfinite(t.numpy()).all(), f"corrupt tensor at iteration {i}"


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_concurrent_threads_share_one_device():
    """Several threads dispatching to one device must not interleave state.

    Each thread records many dispatches, so descriptor allocation, the pool and
    the command-buffer ring are all exercised under contention.
    """
    results = {}
    errors = []

    def work(k):
        try:
            x = torch.tensor(np.full(256, float(k), np.float32), device="vk:0")
            y = x
            for _ in range(100):
                y = y + 1.0
            results[k] = y.numpy()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == 6
    for k, value in results.items():
        np.testing.assert_allclose(value, np.full(256, k + 100.0, np.float32), rtol=1e-5)


@pytest.mark.skipif(len(GPU_DEVICES) < 2, reason="requires two Vulkan devices")
def test_concurrent_threads_on_separate_devices():
    results = {}
    errors = []

    def work(device):
        try:
            x = torch.tensor(np.full(256, 2.0, np.float32), device=device)
            results[device] = (x * 3.0).numpy()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(dev,)) for dev in GPU_DEVICES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for value in results.values():
        np.testing.assert_allclose(value, np.full(256, 6.0, np.float32))


@pytest.mark.skipif(len(GPU_DEVICES) < 2, reason="requires two Vulkan devices")
def test_cross_device_transfer_is_ordered():
    x_np = _rand((64, 64), 4)
    x = torch.tensor(x_np, device="vk:0")
    # Queue up work, then transfer without an explicit synchronise.
    y = x * 2.0
    moved = y.to("vk:1")
    np.testing.assert_allclose(moved.numpy(), x_np * 2.0, rtol=1e-6)


@pytest.mark.skipif(not HAS_GPU, reason="requires a Vulkan device")
def test_train_step_with_batched_backward():
    """End-to-end: many dispatches with a host read only at the end."""
    import vultorch.nn as nn
    import vultorch.optim as optim

    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).to("vk:0")
    opt = optim.SGD(model.parameters(), lr=0.1)
    x = torch.tensor(_rand((8, 16), 5), device="vk:0")
    y = torch.tensor(np.zeros(8, np.float32), device="vk:0")
    crit = nn.CrossEntropyLoss()

    first = None
    for _ in range(10):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        value = loss.item()
        if first is None:
            first = value
    assert value < first, "batched training should reduce loss"
