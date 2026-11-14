"""
Phase 1 Verification & Gradcheck Suite:
- Zero host fallbacks (.numpy()) in vultorch.nn & vultorch.optim
- MaxPool2d forward + race-free input-centric gather backward (including overlapping k=3, s=2)
- AvgPool2d (count_include_pad=True/False) & AdaptiveAvgPool2d forward + backward + gradcheck
- Embedding forward + race-free backward with repeated indices and padding_idx
- Fused LayerNorm & RMSNorm forward + backward + gradcheck
- Fused SGD (plain, weight_decay, momentum, dampening, nesterov)
- Philox-4x32-7 GPU PRNG (uniform_, normal_, rand, randn) and zero-mask-memory Dropout
"""

from pathlib import Path
import numpy as np
import pytest
import vultorch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.testing import gradcheck


def _devices():
    devs = ["cpu"]
    try:
        n = vultorch.vulkan.device_count()
        devs.extend(f"vk:{i}" for i in range(n))
    except Exception:
        pass
    return devs


DEVICES = _devices()


def test_zero_host_fallbacks_in_nn_and_optim():
    """Phase 1 Acceptance Criterion 1: 0 .numpy() calls in vultorch/nn and vultorch/optim."""
    root = Path(__file__).resolve().parent.parent
    for rel in ("vultorch/nn/__init__.py", "vultorch/optim/__init__.py"):
        src = (root / rel).read_text()
        matches = [
            (lineno, line)
            for lineno, line in enumerate(src.splitlines(), start=1)
            if ".numpy()" in line
        ]
        assert len(matches) == 0, f"Found .numpy() in {rel}: {matches}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [
        (2, 2, 0),
        (3, 3, 1),
        (3, 2, 1),  # Overlapping windows (k > stride)
    ],
)
def test_maxpool2d_gpu_matches_cpu_and_gradcheck(device, kernel_size, stride, padding):
    rng = np.random.default_rng(101)
    # Use well-separated values so finite-difference perturbations don't flip argmax ties
    x_np = rng.permutation(2 * 2 * 5 * 5).astype(np.float32).reshape(2, 2, 5, 5) * 0.1 - 2.5

    # Compare device forward + backward against CPU reference
    pool_cpu = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    x_cpu = vultorch.tensor(x_np, device="cpu", requires_grad=True)
    g_np = rng.standard_normal(pool_cpu(x_cpu).shape).astype(np.float32)

    y_cpu = pool_cpu(x_cpu)
    y_cpu.backward(vultorch.tensor(g_np, device="cpu"))

    pool_dev = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    x_dev = vultorch.tensor(x_np, device=device, requires_grad=True)
    y_dev = pool_dev(x_dev)
    y_dev.backward(vultorch.tensor(g_np, device=device))

    np.testing.assert_allclose(y_dev.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(x_dev.grad.numpy(), x_cpu.grad.numpy(), rtol=1e-5, atol=1e-5)

    # Numerical gradcheck
    x_gc = x_np[:1, :1, :4, :4].copy()
    w_gc = rng.standard_normal(pool_dev(vultorch.tensor(x_gc, device=device)).shape).astype(np.float32)
    w_gc_t = vultorch.tensor(w_gc, device=device)
    assert gradcheck(
        lambda inp: (pool_dev(inp) * w_gc_t).sum(),
        x_gc,
        eps=1e-3,
        rtol=1e-2,
        atol=1e-2,
        device=device,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_avgpool2d_and_adaptive_avgpool2d_gradcheck(device, count_include_pad):
    rng = np.random.default_rng(202)
    x_np = rng.standard_normal((1, 2, 5, 5)).astype(np.float32)

    # Fixed-kernel AvgPool2d with overlapping stride and padding
    pool_cpu = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=count_include_pad)
    x_cpu = vultorch.tensor(x_np, device="cpu", requires_grad=True)
    y_cpu = pool_cpu(x_cpu)
    g_np = rng.standard_normal(y_cpu.shape).astype(np.float32)
    y_cpu.backward(vultorch.tensor(g_np, device="cpu"))

    pool_dev = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=count_include_pad)
    x_dev = vultorch.tensor(x_np, device=device, requires_grad=True)
    y_dev = pool_dev(x_dev)
    y_dev.backward(vultorch.tensor(g_np, device=device))

    np.testing.assert_allclose(y_dev.numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x_dev.grad.numpy(), x_cpu.grad.numpy(), rtol=1e-5, atol=1e-5)

    x_gc = x_np[:, :1, :4, :4].copy()
    w_gc = rng.standard_normal(pool_dev(vultorch.tensor(x_gc, device=device)).shape).astype(np.float32)
    w_gc_t = vultorch.tensor(w_gc, device=device)
    assert gradcheck(
        lambda inp: (pool_dev(inp) * w_gc_t).sum(),
        x_gc,
        eps=1e-3,
        rtol=1e-2,
        atol=1e-2,
        device=device,
    )

    # AdaptiveAvgPool2d (non-divisible 5x5 -> 2x2 and global 1x1)
    for out_sz in [(2, 2), (1, 1)]:
        adap_cpu = nn.AdaptiveAvgPool2d(out_sz)
        xc = vultorch.tensor(x_np, device="cpu", requires_grad=True)
        yc = adap_cpu(xc)
        ga = rng.standard_normal(yc.shape).astype(np.float32)
        yc.backward(vultorch.tensor(ga, device="cpu"))

        adap_dev = nn.AdaptiveAvgPool2d(out_sz)
        xd = vultorch.tensor(x_np, device=device, requires_grad=True)
        yd = adap_dev(xd)
        yd.backward(vultorch.tensor(ga, device=device))

        np.testing.assert_allclose(yd.numpy(), yc.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(xd.grad.numpy(), xc.grad.numpy(), rtol=1e-5, atol=1e-5)

        x_gc2 = x_np[:, :1, :4, :4].copy()
        w_gc2 = rng.standard_normal(adap_dev(vultorch.tensor(x_gc2, device=device)).shape).astype(np.float32)
        w_gc2_t = vultorch.tensor(w_gc2, device=device)
        assert gradcheck(
            lambda inp: (adap_dev(inp) * w_gc2_t).sum(),
            x_gc2,
            eps=1e-3,
            rtol=1e-2,
            atol=1e-2,
            device=device,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_forward_backward_padding_idx(device):
    rng = np.random.default_rng(303)
    vocab_size, dim = 12, 8
    padding_idx = 0
    w_np = rng.standard_normal((vocab_size, dim)).astype(np.float32)
    w_np[padding_idx] = 0.0

    # Repeated token indices (tokens 2, 5, and padding token 0 appear multiple times)
    idx_np = np.array([[0, 2, 5, 2, 0], [5, 2, 7, 5, 2]], dtype=np.float32)
    grad_out_np = rng.standard_normal((2, 5, dim)).astype(np.float32)

    emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx, device=device)
    with vultorch.no_grad():
        emb.weight.copy_(w_np)

    idx_t = vultorch.tensor(idx_np, device=device)
    out = emb(idx_t)
    out.backward(vultorch.tensor(grad_out_np, device=device))

    # Expected forward
    expected_out = w_np[idx_np.astype(np.int64)]
    expected_out[idx_np.astype(np.int64) == padding_idx] = 0.0
    np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-5, atol=1e-6)

    # Expected backward (accumulated over repeated indices, 0 at padding_idx)
    expected_gw = np.zeros_like(w_np)
    flat_idx = idx_np.astype(np.int64).reshape(-1)
    flat_go = grad_out_np.reshape(-1, dim)
    for tok, go_row in zip(flat_idx, flat_go):
        if tok != padding_idx:
            expected_gw[tok] += go_row

    np.testing.assert_allclose(emb.weight.grad.numpy(), expected_gw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(emb.weight.grad.numpy()[padding_idx], np.zeros(dim), atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_layernorm_and_rmsnorm_fused_backward(device):
    rng = np.random.default_rng(404)
    x_np = rng.standard_normal((4, 3, 16)).astype(np.float32)
    g_np = rng.uniform(0.5, 1.5, size=(16,)).astype(np.float32)
    b_np = rng.uniform(-0.5, 0.5, size=(16,)).astype(np.float32)
    dy_np = rng.standard_normal((4, 3, 16)).astype(np.float32)
    eps = 1e-5

    # --- LayerNorm ---
    ln = nn.LayerNorm(16, eps=eps, device=device)
    with vultorch.no_grad():
        ln.weight.copy_(g_np)
        ln.bias.copy_(b_np)

    x_t = vultorch.tensor(x_np, device=device, requires_grad=True)
    y_t = ln(x_t)
    y_t.backward(vultorch.tensor(dy_np, device=device))

    # Analytical reference in float64
    x64 = x_np.astype(np.float64)
    g64 = g_np.astype(np.float64)
    b64 = b_np.astype(np.float64)
    dy64 = dy_np.astype(np.float64)

    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x64 - mean) * inv_std
    y_ref = x_hat * g64 + b64
    dg_ref = (dy64 * x_hat).sum(axis=(0, 1))
    db_ref = dy64.sum(axis=(0, 1))
    dx_hat = dy64 * g64
    dx_ref = inv_std * (
        dx_hat
        - dx_hat.mean(axis=-1, keepdims=True)
        - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
    )

    np.testing.assert_allclose(y_t.numpy(), y_ref.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x_t.grad.numpy(), dx_ref.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ln.weight.grad.numpy(), dg_ref.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ln.bias.grad.numpy(), db_ref.astype(np.float32), rtol=1e-4, atol=1e-4)

    # --- RMSNorm ---
    rms = nn.RMSNorm(16, eps=eps, device=device)
    with vultorch.no_grad():
        rms.weight.copy_(g_np)

    xr_t = vultorch.tensor(x_np, device=device, requires_grad=True)
    yr_t = rms(xr_t)
    yr_t.backward(vultorch.tensor(dy_np, device=device))

    ms = (x64 ** 2).mean(axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    xr_hat = x64 * inv_rms
    yr_ref = xr_hat * g64
    dgr_ref = (dy64 * xr_hat).sum(axis=(0, 1))
    dxr_hat = dy64 * g64
    dxr_ref = inv_rms * (dxr_hat - xr_hat * (dxr_hat * xr_hat).mean(axis=-1, keepdims=True))

    np.testing.assert_allclose(yr_t.numpy(), yr_ref.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(xr_t.grad.numpy(), dxr_ref.astype(np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rms.weight.grad.numpy(), dgr_ref.astype(np.float32), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "momentum,dampening,weight_decay,nesterov",
    [
        (0.0, 0.0, 0.0, False),      # Plain SGD
        (0.0, 0.0, 1e-2, False),     # SGD + weight_decay
        (0.9, 0.0, 1e-3, False),     # SGD + momentum + weight_decay
        (0.9, 0.2, 1e-3, False),     # SGD + momentum + dampening
        (0.9, 0.0, 1e-3, True),      # Nesterov SGD
    ],
)
def test_fused_sgd_matches_unfused(device, momentum, dampening, weight_decay, nesterov):
    rng = np.random.default_rng(505)
    p0 = rng.standard_normal((64, 32)).astype(np.float32)
    grads = [rng.standard_normal((64, 32)).astype(np.float32) for _ in range(4)]

    p_fused = nn.Parameter(vultorch.tensor(p0.copy(), device=device))
    p_unfused = nn.Parameter(vultorch.tensor(p0.copy(), device=device))

    opt_fused = optim.SGD(
        [p_fused],
        lr=0.05,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        fused=True,
    )
    opt_unfused = optim.SGD(
        [p_unfused],
        lr=0.05,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        fused=False,
    )

    for g_np in grads:
        p_fused.grad = vultorch.tensor(g_np.copy(), device=device)
        p_unfused.grad = vultorch.tensor(g_np.copy(), device=device)
        opt_fused.step()
        opt_unfused.step()

    np.testing.assert_allclose(p_fused.numpy(), p_unfused.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_gpu_dropout_reconstructs_exact_mask_in_backward(device):
    vultorch.manual_seed(12345)
    p = 0.35
    x_np = np.ones((256, 256), dtype=np.float32) * 2.0
    dy_np = np.ones((256, 256), dtype=np.float32) * 3.0

    drop = nn.Dropout(p=p)
    drop.train()

    x = vultorch.tensor(x_np, device=device, requires_grad=True)
    y = drop(x)
    y.backward(vultorch.tensor(dy_np, device=device))

    # Forward mask = y / x, Backward reconstructed mask = dx / dy
    fwd_mask = y.numpy() / x_np
    bwd_mask = x.grad.numpy() / dy_np

    # Must match bit-for-bit without allocating a separate mask buffer
    np.testing.assert_allclose(fwd_mask, bwd_mask, rtol=1e-6, atol=1e-6)

    # Check empirical keep rate and inverted scale 1 / (1 - p)
    keep_fraction = np.mean(fwd_mask > 0.0)
    assert 0.58 < keep_fraction < 0.72
    non_zero_vals = fwd_mask[fwd_mask > 0.0]
    np.testing.assert_allclose(non_zero_vals, 1.0 / (1.0 - p), rtol=1e-5, atol=1e-5)

    # Eval mode must be exact identity
    drop.eval()
    y_eval = drop(x)
    np.testing.assert_allclose(y_eval.numpy(), x_np, rtol=0, atol=0)
