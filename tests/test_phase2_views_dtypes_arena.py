"""
Phase 2 Acceptance Tests:
- Zero-Copy Strided Views (< 5 us, 0 GPU dispatches, shared storage_id)
- Chained Non-Contiguous Forward + Backward + Gradcheck
- Multi-Dtype System (float32, float16, bfloat16, int32, int64, bool, uint8)
- Native FP16 GEMM (50% VRAM)
- 2-Level Arena Sub-Allocator & Persistent Mapped Staging Ring
"""

import time
import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


DEVICES = ["cpu"] + [f"vk:{i}" for i in range(torch.vulkan.device_count())]
GPU_DEVICES = [d for d in DEVICES if d.startswith("vk:")]


def _dev_id(dev: str) -> int:
    return -1 if dev == "cpu" else int(dev.split(":")[1])


def _rand_np(shape, seed=0, low=-1.5, high=1.5, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(dtype)


# ---------------------------------------------------------------------------
# 1. Zero-Copy Strided Views
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_zero_copy_views_latency_dispatches_and_storage(dev):
    torch.vulkan.synchronize_all()
    did = _dev_id(dev)

    x_np = _rand_np((64, 32, 16), seed=11)
    x = torch.tensor(x_np, device=dev)
    if did >= 0:
        torch.vulkan.synchronize(did)

    # Warm up view wrappers
    _ = x.transpose(0, 1)
    _ = x.permute(2, 0, 1)
    _ = x[:, 4:12].narrow(2, 2, 8)
    _ = x.reshape(64, 512)
    _ = torch.tensor(np.ones((64, 1), dtype=np.float32), device=dev).expand(64, 64)

    y = torch.tensor(np.arange(64, dtype=np.float32).reshape(64, 1), device=dev)

    if did >= 0:
        d_before = torch.vulkan.recorded_dispatches(did)
    else:
        d_before = 0

    views = {}
    t0 = time.perf_counter()
    iters = 200
    for _ in range(iters):
        v1 = x.transpose(1, 2)
        v2 = x.permute(2, 0, 1)
        v3 = y.expand(64, 64)
        v4 = x.narrow(1, 4, 16)
        v5 = x[8:24]
        v6 = x.as_strided((32, 16), (x.strides[1], x.strides[2]), storage_offset=x.strides[0] * 3)
        v7 = x.reshape(64, 512)
    elapsed_us_per_view = ((time.perf_counter() - t0) * 1e6) / (iters * 7)

    views = {"transpose": v1, "permute": v2, "expand": v3, "narrow": v4, "slice": v5, "as_strided": v6, "reshape": v7}

    if did >= 0:
        d_after = torch.vulkan.recorded_dispatches(did)
        assert d_after == d_before, f"View ops triggered {d_after - d_before} unexpected GPU dispatches"

    assert elapsed_us_per_view < 10.0, f"View creation too slow: {elapsed_us_per_view:.2f} us/view"

    # Median single-op latency check < 5 us on steady-state view ops
    for name, fn in [
        ("transpose", lambda: x.transpose(1, 2)),
        ("permute", lambda: x.permute(2, 0, 1)),
        ("expand", lambda: y.expand(64, 64)),
        ("narrow", lambda: x.narrow(1, 4, 16)),
        ("slice", lambda: x[8:24]),
        ("as_strided", lambda: x.as_strided((32, 16), (x.strides[1], x.strides[2]), 16)),
        ("reshape", lambda: x.reshape(64, 512)),
    ]:
        # Warm function & branch predictor
        for _ in range(5):
            _ = fn()
        samples = []
        for _ in range(100):
            ts = time.perf_counter()
            _ = fn()
            samples.append((time.perf_counter() - ts) * 1e6)
        med = float(np.median(samples))
        assert med < 5.0, f"{name} median latency {med:.2f} us >= 5 us"

    base_sid = x.storage_id()
    for name, v in views.items():
        if name != "expand":
            assert v.storage_id() == base_sid, f"{name} did not share storage_id"
    assert v3.storage_id() == y.storage_id()

    # Exact values through to_numpy()
    np.testing.assert_allclose(v1.numpy(), x_np.transpose(0, 2, 1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v2.numpy(), x_np.transpose(2, 0, 1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v3.numpy(), np.broadcast_to(y.numpy(), (64, 64)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v4.numpy(), x_np[:, 4:20, :], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v5.numpy(), x_np[8:24], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v6.numpy(), x_np[3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v7.numpy(), x_np.reshape(64, 512), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_chained_non_contiguous_forward_backward_and_gradcheck(dev):
    x_np = _rand_np((64, 64), seed=101)
    y_np = _rand_np((64, 1), seed=102)
    z_np = _rand_np((64, 64), seed=103)

    x = torch.tensor(x_np, device=dev, requires_grad=True)
    y = torch.tensor(y_np, device=dev, requires_grad=True)
    z = torch.tensor(z_np, device=dev, requires_grad=True)

    # Zero-copy chained expression directly into elementwise + GEMM kernels
    out = (x.t() + y.expand(64, 64)) @ z.t()
    loss = (out * out).sum()
    loss.backward()

    # NumPy exact reference
    expected_s = x_np.T + np.broadcast_to(y_np, (64, 64))
    expected_out = expected_s @ z_np.T
    expected_loss = float(np.sum(expected_out ** 2))
    grad_out = 2.0 * expected_out
    expected_ds = grad_out @ z_np
    expected_dz = (expected_s.T @ grad_out).T
    expected_dx = expected_ds.T
    expected_dy = expected_ds.sum(axis=1, keepdims=True)

    assert abs(loss.item() - expected_loss) / max(abs(expected_loss), 1.0) < 1e-4
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x.grad.numpy(), expected_dx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y.grad.numpy(), expected_dy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(z.grad.numpy(), expected_dz, rtol=1e-4, atol=1e-4)

    # Finite-difference gradcheck on smaller non-contiguous graph
    sx = torch.tensor(_rand_np((5, 4), seed=21), device=dev, requires_grad=True)
    sy = torch.tensor(_rand_np((4, 1), seed=22), device=dev, requires_grad=True)
    sz = torch.tensor(_rand_np((6, 5), seed=23), device=dev, requires_grad=True)
    assert torch.gradcheck(
        lambda a, b, c: (((a.t() + b.expand(4, 5)) @ c.t())[:, 1:5:2]).sum(),
        sx, sy, sz,
        eps=2e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_strided_offset_slicing_and_narrow_backward(dev):
    x_np = _rand_np((6, 8, 4), seed=55)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    # Double slice + narrow + permute (creates non-zero storage_offset + non-contiguous strides)
    sub = x[1:5, 2:6:2, :].narrow(2, 1, 2).permute(2, 0, 1)
    assert sub.storage_id() == x.storage_id()
    np.testing.assert_allclose(sub.numpy(), x_np[1:5, 2:6:2, 1:3].transpose(2, 0, 1), rtol=1e-5, atol=1e-5)

    w = torch.tensor(_rand_np((2, 4, 2), seed=56), device=dev)
    loss = (sub * w).sum()
    loss.backward()

    expected_grad = np.zeros_like(x_np)
    expected_grad[1:5, 2:6:2, 1:3] = w.numpy().transpose(1, 2, 0)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Multi-Dtype System & Native FP16 GEMM
# ---------------------------------------------------------------------------

ALL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.uint8,
]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dt", ALL_DTYPES, ids=lambda d: str(d))
def test_multi_dtype_creation_and_numpy_roundtrip(dev, dt):
    if dt in (torch.float32, torch.float16, torch.bfloat16):
        arr = np.array([[1.25, -2.5], [0.5, 3.75]], dtype=np.float32)
        t = torch.tensor(arr, device=dev, dtype=dt)
        assert t.dtype == dt
        expected_itemsize = 4 if dt == torch.float32 else 2
        assert t.itemsize == expected_itemsize
        assert t.nbytes == t.numel * expected_itemsize
        tol = 1e-5 if dt == torch.float32 else 2e-2
        np.testing.assert_allclose(t.numpy(), arr, rtol=tol, atol=tol)
    elif dt in (torch.int32, torch.int64, torch.uint8):
        arr = np.array([[1, 7], [13, 42]], dtype=dt.numpy_dtype)
        t = torch.tensor(arr, device=dev, dtype=dt)
        assert t.dtype == dt
        np.testing.assert_array_equal(t.numpy(), arr)
        # Auto-inferred from ndarray without explicit dtype
        t_auto = torch.from_numpy(arr, device=dev)
        assert t_auto.dtype == dt
        np.testing.assert_array_equal(t_auto.numpy(), arr)
    else:  # bool
        arr = np.array([[True, False], [False, True]], dtype=np.bool_)
        t = torch.tensor(arr, device=dev, dtype=torch.bool)
        assert t.dtype == torch.bool
        assert t.itemsize == 1
        np.testing.assert_array_equal(t.numpy(), arr)


@pytest.mark.parametrize("dev", DEVICES)
def test_multi_dtype_casts_arithmetic_reductions_and_indices(dev):
    base = torch.tensor([[-3.0, 2.5], [0.0, 4.0]], device=dev)
    assert base.half().dtype == torch.float16
    assert base.bfloat16().dtype == torch.bfloat16
    assert base.int().dtype == torch.int32
    assert base.long().dtype == torch.int64
    assert base.bool().dtype == torch.bool
    assert base.byte().dtype == torch.uint8
    assert base.to(dtype=torch.float16).float().dtype == torch.float32

    np.testing.assert_array_equal(base.int().numpy(), np.array([[-3, 2], [0, 4]], dtype=np.int32))
    np.testing.assert_array_equal(base.long().numpy(), np.array([[-3, 2], [0, 4]], dtype=np.int64))
    np.testing.assert_array_equal(base.bool().numpy(), np.array([[True, True], [False, True]], dtype=np.bool_))
    np.testing.assert_array_equal(torch.tensor([0, 5, 255], device=dev, dtype=torch.uint8).numpy(),
                                  np.array([0, 5, 255], dtype=np.uint8))

    # Comparisons return vultorch.bool
    cmp_gt = base > 1.0
    cmp_eq = base == 0.0
    cmp_ne = base != 0.0
    assert cmp_gt.dtype == torch.bool
    assert cmp_eq.dtype == torch.bool
    assert cmp_ne.dtype == torch.bool
    np.testing.assert_array_equal(cmp_gt.numpy(), np.array([[False, True], [False, True]]))
    np.testing.assert_array_equal(cmp_eq.numpy(), np.array([[False, False], [True, False]]))
    np.testing.assert_array_equal(cmp_ne.numpy(), np.array([[True, True], [False, True]]))

    # Arithmetic + reductions across int64 & float16
    i1 = torch.tensor([[2, 4], [6, 8]], device=dev, dtype=torch.int64)
    i2 = torch.tensor([[1, 2], [3, 4]], device=dev, dtype=torch.int64)
    i_sum = i1 + i2
    assert i_sum.dtype == torch.int64
    np.testing.assert_array_equal(i_sum.numpy(), np.array([[3, 6], [9, 12]], dtype=np.int64))
    assert i_sum.sum().item() == 30
    assert i_sum.max().item() == 12

    # Embedding & CrossEntropyLoss directly accept int64 / int32 indices
    emb = nn.Embedding(8, 4).to(dev)
    idx_i64 = torch.tensor([[0, 3, 7], [2, 1, 5]], device=dev, dtype=torch.int64)
    emb_out = emb(idx_i64)
    assert emb_out.shape == (2, 3, 4)
    np.testing.assert_allclose(emb_out.numpy(), emb.weight.numpy()[idx_i64.numpy()], rtol=1e-5, atol=1e-5)

    logits = torch.tensor(_rand_np((4, 5), seed=99), device=dev, requires_grad=True)
    targets_i64 = torch.tensor(np.array([0, 2, 4, 1], dtype=np.int64), device=dev)
    ce = nn.CrossEntropyLoss()(logits, targets_i64)
    ce.backward()
    assert logits.grad is not None and np.isfinite(logits.grad.numpy()).all()

    # Cast backward for floating point types
    h = torch.tensor(_rand_np((4, 4), seed=71), device=dev, requires_grad=True)
    loss_h = (h.half() * 2.0).sum().float()
    loss_h.backward()
    np.testing.assert_allclose(h.grad.numpy(), np.full((4, 4), 2.0, dtype=np.float32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dev", DEVICES)
def test_native_fp16_gemm_and_memory_savings(dev):
    a_np = _rand_np((64, 48), seed=31, low=-1.0, high=1.0)
    b_np = _rand_np((48, 32), seed=32, low=-1.0, high=1.0)

    a_f32 = torch.tensor(a_np, device=dev)
    b_f32 = torch.tensor(b_np, device=dev)
    a_f16 = a_f32.half()
    b_f16 = b_f32.half()

    assert a_f16.nbytes * 2 == a_f32.nbytes
    assert b_f16.nbytes * 2 == b_f32.nbytes

    c_f16 = a_f16 @ b_f16
    assert c_f16.dtype == torch.float16
    assert c_f16.nbytes == 64 * 32 * 2

    ref = a_np.astype(np.float16).astype(np.float32) @ b_np.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(c_f16.numpy().astype(np.float32), ref, rtol=1e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# 3. Arena Sub-Allocator & Persistent Mapped Staging Ring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dev", GPU_DEVICES)
def test_arena_suballocator_low_vk_allocs_and_zero_leak_training(dev):
    did = _dev_id(dev)
    torch.vulkan.synchronize(did)

    # 10,000 tensor allocations and frees of varying shapes should trigger <= 8 new VkDeviceMemory allocations
    rng = np.random.default_rng(7)
    base_allocs = torch.vulkan.driver_allocation_count(did)
    # Force one small allocation to ensure initial slab exists before counting incremental slabs
    _warm = torch.empty((16, 16), device=dev)
    torch.vulkan.synchronize(did)
    slab_baseline = torch.vulkan.driver_allocation_count(did)
    active_baseline = torch.vulkan.active_arena_bytes(did)

    shapes = [(32, 32), (128, 64), (256, 256), (17, 513), (64, 128), (1024, 32)]
    for i in range(10000):
        shp = shapes[i % len(shapes)]
        t = torch.empty(shp, device=dev)
        # Drop immediately so arena recycles slices in O(log N)
        del t
    torch.vulkan.synchronize(did)
    after_allocs = torch.vulkan.driver_allocation_count(did)
    delta_allocs = after_allocs - slab_baseline
    assert delta_allocs <= 8, f"Arena triggered {delta_allocs} new VkDeviceMemory blocks (expected <= 8)"
    assert torch.vulkan.active_arena_bytes(did) <= active_baseline + 4096
    del _warm

    # 1,000-step training loop shows zero monotonic active arena growth
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    ).to(dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    x_fix = torch.tensor(rng.standard_normal((16, 32)).astype(np.float32), device=dev)
    y_fix = torch.tensor(rng.standard_normal((16, 16)).astype(np.float32), device=dev)

    active_after_warmup = None
    for step in range(1000):
        opt.zero_grad()
        pred = model(x_fix)
        diff = pred - y_fix
        loss = (diff * diff).mean()
        loss.backward()
        opt.step()
        if step == 20:
            torch.vulkan.synchronize(did)
            active_after_warmup = torch.vulkan.active_arena_bytes(did)

    torch.vulkan.synchronize(did)
    active_after_1000 = torch.vulkan.active_arena_bytes(did)
    assert active_after_1000 <= active_after_warmup + 4096, (
        f"Monotonic arena leak detected: step 20 active={active_after_warmup}, "
        f"step 1000 active={active_after_1000}"
    )


@pytest.mark.parametrize("dev", GPU_DEVICES)
def test_staging_ring_4mib_h2d_upload(dev):
    did = _dev_id(dev)
    payload = np.ones((1024, 1024), dtype=np.float32) * 3.25  # exact 4 MiB

    # Warm staging ring
    t_warm = torch.from_numpy(payload, device=dev)
    torch.vulkan.synchronize(did)
    del t_warm

    # Repeated 4 MiB uploads reuse the persistent mapped staging ring with zero new slab allocs
    allocs_before = torch.vulkan.driver_allocation_count(did)
    t0 = time.perf_counter()
    for _ in range(10):
        gpu_t = torch.from_numpy(payload, device=dev)
    torch.vulkan.synchronize(did)
    elapsed_ms = ((time.perf_counter() - t0) * 1e3) / 10.0
    allocs_after = torch.vulkan.driver_allocation_count(did)

    assert allocs_after - allocs_before <= 1
    assert elapsed_ms < 25.0, f"4 MiB H2D upload too slow: {elapsed_ms:.2f} ms"
    np.testing.assert_allclose(gpu_t.numpy()[:2, :4], payload[:2, :4], rtol=1e-6)
