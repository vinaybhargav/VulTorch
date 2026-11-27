"""Regression tests for the Phase 1 correctness fixes.

Every test here corresponds to a bug that previously produced a wrong result
silently (or crashed) instead of raising, so each one is a guard against a
regression rather than a feature test.
"""

import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn

DEVICES = ["cpu"] + (["vk:0"] if torch.vulkan.device_count() > 0 else [])


def _rand(shape, seed=0):
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


# ---------------------------------------------------------------- broadcasting


def test_cpu_broadcast_matches_numpy():
    cases = [
        ((2, 3), (3,)),
        ((4, 1, 8), (1, 6, 8)),
        ((1,), (2, 2)),
        ((5, 1, 1), (1, 4, 1)),
        ((3,), (3,)),
    ]
    for i, (sa, sb) in enumerate(cases):
        a, b = _rand(sa, i), _rand(sb, i + 1)
        for dev in DEVICES:
            out = (torch.tensor(a, device=dev) + torch.tensor(b, device=dev)).numpy()
            assert out.shape == np.broadcast_shapes(sa, sb), (sa, sb, dev)
            np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-6)


def test_incompatible_broadcast_raises():
    a = torch.tensor(np.zeros((2, 3), np.float32), device="cpu")
    b = torch.tensor(np.zeros((4,), np.float32), device="cpu")
    with pytest.raises(RuntimeError):
        _ = a + b


@pytest.mark.parametrize("name", ["exp", "log", "sqrt", "tanh", "sigmoid", "relu", "gelu", "silu"])
def test_unary_ops_cpu_matches_gpu(name):
    """The CPU fallback must implement the same op table as the shader.

    `gelu` and `silu` used to fall through to the identity function on CPU.
    """
    x_np = np.array([0.4, 0.9, 1.7, 2.5], np.float32)
    cpu = getattr(torch.tensor(x_np, device="cpu"), name)().numpy()

    if "vk:0" in DEVICES:
        gpu = getattr(torch.tensor(x_np, device="vk:0"), name)().numpy()
        np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-5)

    # Independent references for the two ops that used to be no-ops.
    if name == "gelu":
        ref = 0.5 * x_np * (1.0 + np.tanh(0.7978845608 * (x_np + 0.044715 * x_np**3)))
        np.testing.assert_allclose(cpu, ref, rtol=1e-4, atol=1e-5)
    elif name == "silu":
        ref = x_np / (1.0 + np.exp(-x_np))
        np.testing.assert_allclose(cpu, ref, rtol=1e-4, atol=1e-5)
    elif name == "relu":
        np.testing.assert_allclose(cpu, np.maximum(x_np, 0.0), rtol=1e-6)


# --------------------------------------------------------------------- softmax


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_axis(dim):
    x_np = _rand((4, 3))
    shifted = x_np - x_np.max(axis=dim, keepdims=True)
    ref = np.exp(shifted)
    ref /= ref.sum(axis=dim, keepdims=True)

    for dev in DEVICES:
        np.testing.assert_allclose(
            torch.tensor(x_np, device=dev).softmax(dim=dim).numpy(), ref, rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("dim", [0, -1])
def test_log_softmax_axis(dim):
    x_np = _rand((3, 5), seed=3)
    shifted = x_np - x_np.max(axis=dim, keepdims=True)
    ref = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))

    for dev in DEVICES:
        np.testing.assert_allclose(
            torch.tensor(x_np, device=dev).log_softmax(dim=dim).numpy(), ref, rtol=1e-4, atol=1e-5
        )


def test_softmax_invalid_dim_raises():
    x = torch.tensor(_rand((2, 3)), device="cpu")
    with pytest.raises(RuntimeError):
        _ = x._native.softmax(5, False)


# -------------------------------------------------------------------- indexing


def test_getitem_propagates_gradient():
    x = torch.tensor(np.ones((2, 3), np.float32), device="cpu", requires_grad=True)
    x[0:1].sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[1, 1, 1], [0, 0, 0]])


def test_advanced_index_accumulates_duplicates():
    x = torch.tensor(np.zeros(3, np.float32), device="cpu", requires_grad=True)
    x[[0, 0, 2]].sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 0.0, 1.0])


def test_getitem_scalar_index_gradient():
    x = torch.tensor(np.ones((2, 2), np.float32), device="cpu", requires_grad=True)
    x[1].sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[0, 0], [1, 1]])


# ------------------------------------------------------------------------ max


def test_max_routes_gradient_to_first_argmax():
    x = torch.tensor(np.array([1.0, 5.0, 5.0], np.float32), device="cpu", requires_grad=True)
    x.max().backward()
    np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0, 0.0])


def test_max_over_dim_gradient():
    x_np = np.array([[1.0, 3.0], [4.0, 2.0]], np.float32)
    x = torch.tensor(x_np, device="cpu", requires_grad=True)
    x.max(dim=0).sum().backward()
    # max of column 0 is x[1, 0]; max of column 1 is x[0, 1].
    np.testing.assert_allclose(x.grad.numpy(), [[0.0, 1.0], [1.0, 0.0]])


# --------------------------------------------------------------------- matmul


def test_matmul_broadcasts_unbatched_weight():
    a, b = _rand((2, 3, 4), 7), _rand((4, 5), 8)
    for dev in DEVICES:
        out = (torch.tensor(a, device=dev) @ torch.tensor(b, device=dev)).numpy()
        assert out.shape == (2, 3, 5)
        np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)


def test_linear_accepts_3d_input():
    layer = nn.Linear(4, 5).to("cpu")
    x = torch.tensor(_rand((2, 3, 4)), device="cpu")
    assert layer(x).shape == (2, 3, 5)


def test_linear_fused_and_transposed_matmul_match_numpy():
    x_np = _rand((33, 67), 91)
    w_np = _rand((45, 67), 92)
    b_np = _rand((45,), 93)
    g_np = _rand((33, 45), 94)

    expected_y = x_np @ w_np.T + b_np
    expected_gx = g_np @ w_np
    expected_gw = g_np.T @ x_np
    expected_gb = g_np.sum(axis=0)

    for dev in DEVICES:
        layer = nn.Linear(67, 45, device=dev)
        layer.weight.copy_(w_np)
        layer.bias.copy_(b_np)
        x = torch.tensor(x_np, device=dev, requires_grad=True)
        g = torch.tensor(g_np, device=dev)

        y = layer(x)
        np.testing.assert_allclose(y.numpy(), expected_y, rtol=1e-4, atol=1e-4)

        y.backward(g)
        np.testing.assert_allclose(x.grad.numpy(), expected_gx, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(layer.weight.grad.numpy(), expected_gw, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(layer.bias.grad.numpy(), expected_gb, rtol=1e-4, atol=1e-4)


def test_strided_outer_reduction_matches_numpy():
    x_np = _rand((65, 130), 95)
    for dev in DEVICES:
        x = torch.tensor(x_np, device=dev)
        np.testing.assert_allclose(x.sum(dim=0).numpy(), x_np.sum(axis=0), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(x.max(dim=0).numpy(), x_np.max(axis=0), rtol=1e-4, atol=1e-4)


def test_matmul_batch_mismatch_raises():
    a = torch.tensor(np.zeros((2, 3, 4), np.float32), device="cpu")
    b = torch.tensor(np.zeros((3, 4, 5), np.float32), device="cpu")
    with pytest.raises(RuntimeError):
        _ = a @ b


# ------------------------------------------------------------------- conv2d


def test_conv2d_cpu_gradients_match_gpu():
    x_np, w_np = _rand((2, 2, 5, 5), 11), _rand((3, 2, 3, 3), 12)

    grads = {}
    for dev in DEVICES:
        conv = nn.Conv2d(2, 3, 3, padding=1, bias=False, device=dev)
        conv.weight.copy_(w_np)
        x = torch.tensor(x_np, device=dev, requires_grad=True)
        conv(x).sum().backward()
        grads[dev] = (x.grad.numpy().copy(), conv.weight.grad.numpy().copy())

    gx, gw = grads["cpu"]
    assert np.abs(gx).sum() > 0.0, "CPU conv input gradient must not be all zeros"
    assert np.abs(gw).sum() > 0.0, "CPU conv weight gradient must not be all zeros"

    if "vk:0" in grads:
        np.testing.assert_allclose(gx, grads["vk:0"][0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gw, grads["vk:0"][1], rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------- layernorm


def _layernorm_reference(x_np, gamma, beta, eps=1e-5):
    mu = x_np.mean(-1, keepdims=True)
    var = ((x_np - mu) ** 2).mean(-1, keepdims=True)
    return (x_np - mu) / np.sqrt(var + eps) * gamma + beta


def test_layernorm_matches_reference():
    x_np = _rand((3, 4), 21)
    gamma = np.array([1.5, 0.5, 2.0, -1.0], np.float32)
    beta = np.array([0.1, -0.2, 0.3, 0.0], np.float32)

    ln = nn.LayerNorm(4, device="cpu")
    ln.weight.copy_(gamma)
    ln.bias.copy_(beta)

    ref = _layernorm_reference(x_np, gamma, beta)
    np.testing.assert_allclose(
        ln(torch.tensor(x_np, device="cpu")).numpy(), ref, rtol=1e-4, atol=1e-5
    )


def test_layernorm_weight_grad_matches_reference():
    x_np = _rand((3, 4), 22)
    ln = nn.LayerNorm(4, device="cpu")
    x = torch.tensor(x_np, device="cpu", requires_grad=True)
    ln(x).sum().backward()

    mu = x_np.mean(-1, keepdims=True)
    var = ((x_np - mu) ** 2).mean(-1, keepdims=True)
    xhat = (x_np - mu) / np.sqrt(var + 1e-5)

    np.testing.assert_allclose(ln.weight.grad.numpy(), xhat.sum(0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ln.bias.grad.numpy(), np.full(4, 3.0, np.float32), rtol=1e-5)


# -------------------------------------------------------------------- device


def test_device_is_hashable():
    d = torch.device("vk:0")
    assert {d: "gpu"}[torch.device("vk:0")] == "gpu"
    assert torch.device("cpu") != "vk:0"
    assert torch.device("cpu") == "cpu"
