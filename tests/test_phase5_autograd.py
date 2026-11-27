import pytest
import numpy as np
import vultorch
import vultorch.nn as nn
from vultorch.autograd import Function, grad
from vultorch.utils.checkpoint import checkpoint


def _devices():
    devs = ["cpu"]
    if vultorch.vulkan.is_available():
        for i in range(vultorch.vulkan.device_count()):
            devs.append(f"vk:{i}")
    return devs


class _MulAddFn(Function):
    @staticmethod
    def forward(ctx, x, y, scale=2.0):
        ctx.save_for_backward(x, y)
        ctx.scale = scale
        return x * y * scale

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        s = ctx.scale
        return grad_out * y * s, grad_out * x * s, None


class _SplitSquareFn(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * x + y, x - y * y

    @staticmethod
    def backward(ctx, ga, gb):
        x, y = ctx.saved_tensors
        gx = ga * (2.0 * x) + gb
        gy = ga - gb * (2.0 * y)
        return gx, gy


@pytest.mark.parametrize("device", _devices())
def test_custom_autograd_function_single_and_multi_output(device):
    x = vultorch.tensor([[1.2, -0.7], [0.5, 1.5]], device=device, requires_grad=True)
    y = vultorch.tensor([[0.8, 1.3], [-1.1, 0.4]], device=device, requires_grad=True)

    assert vultorch.gradcheck(lambda a, b: _MulAddFn.apply(a, b, 1.5).sum(), x, y, device=device)
    assert vultorch.gradcheck(
        lambda a, b: (_SplitSquareFn.apply(a, b)[0] + _SplitSquareFn.apply(a, b)[1]).sum(),
        x,
        y,
        device=device,
    )

    # Multi-output backward via sum
    a, b = _SplitSquareFn.apply(x, y)
    loss = a.sum() + b.sum()
    loss.backward()
    expected_gx = 2.0 * x.numpy() + 1.0
    expected_gy = 1.0 - 2.0 * y.numpy()
    np.testing.assert_allclose(x.grad.numpy(), expected_gx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y.grad.numpy(), expected_gy, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _devices())
def test_autograd_grad_and_higher_order(device):
    x = vultorch.tensor([1.5, -2.0, 0.5], device=device, requires_grad=True)
    y = (x * x * x).sum()  # x^3 -> dy/dx = 3x^2 -> d2y/dx2 = 6x

    (g1,) = grad(y, x, create_graph=True)
    np.testing.assert_allclose(g1.numpy(), 3.0 * (x.numpy() ** 2), rtol=1e-4, atol=1e-4)

    (g2,) = grad(g1.sum(), x)
    np.testing.assert_allclose(g2.numpy(), 6.0 * x.numpy(), rtol=1e-4, atol=1e-4)

    # allow_unused check
    unused = vultorch.tensor([1.0], device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="not have been used"):
        grad(y, (x, unused), allow_unused=False)
    gx, gu = grad(y, (x, unused), allow_unused=True)
    assert gx is not None and gu is None


@pytest.mark.parametrize("device", _devices())
def test_inplace_version_counter_raises(device):
    x = vultorch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    y = vultorch.tensor([[2.0, 1.0], [0.5, 1.5]], device=device, requires_grad=True)
    z = _MulAddFn.apply(x * 1.0, y, 2.0)

    # Mutate the tensor saved for backward (x * 1.0 is saved, or mutate y directly)
    y.zero_()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        z.sum().backward()


@pytest.mark.parametrize("device", _devices())
def test_activation_checkpointing_matches_uncheckpointed(device):
    vultorch.manual_seed(123)
    layers = [nn.Linear(32, 32, device=device) for _ in range(6)]
    x = vultorch.randn(8, 32, device=device, requires_grad=True)

    def run_blocks(inp):
        h = inp
        for layer in layers:
            h = layer(h).relu()
        return h

    # Uncheckpointed reference
    out_ref = run_blocks(x)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    ref_x_grad = x.grad.numpy().copy()
    ref_w_grads = [l.weight.grad.numpy().copy() for l in layers]

    x.zero_grad()
    for l in layers:
        l.zero_grad()

    # Checkpointed
    def seg1(inp):
        h = inp
        for layer in layers[:3]:
            h = layer(h).relu()
        return h

    def seg2(inp):
        h = inp
        for layer in layers[3:]:
            h = layer(h).relu()
        return h

    out_ckpt = checkpoint(seg2, checkpoint(seg1, x))
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()

    np.testing.assert_allclose(x.grad.numpy(), ref_x_grad, rtol=1e-4, atol=1e-4)
    for l, rg in zip(layers, ref_w_grads):
        np.testing.assert_allclose(l.weight.grad.numpy(), rg, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", _devices())
def test_slang_differentiable_ops_gradcheck(device):
    import vultorch.nn.functional as F

    x = vultorch.tensor(
        [[-1.5, -0.2, 0.4, 1.8], [0.1, -2.5, 2.2, -0.7]], device=device, requires_grad=True
    )
    y = vultorch.tensor(
        [[0.5, -0.8, 1.1, 0.9], [-0.3, -1.2, 1.0, 0.2]], device=device, requires_grad=True
    )

    assert vultorch.gradcheck(lambda t: F.mish(t).sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(
        lambda t: F.hardswish(t).sum(), x, device=device, atol=1e-3, rtol=1e-3
    )
    assert vultorch.gradcheck(
        lambda t: F.leaky_relu(t, 0.1).sum(), x, device=device, atol=1e-3, rtol=1e-3
    )
    assert vultorch.gradcheck(lambda t: F.elu(t, 1.0).sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(
        lambda a, b: F.smooth_l1_loss(a, b, reduction="sum", beta=1.0),
        x,
        y,
        device=device,
        atol=1e-3,
        rtol=1e-3,
    )
    assert vultorch.gradcheck(
        lambda a, b: F.cosine_similarity(a, b, dim=-1).sum(),
        x,
        y,
        device=device,
        atol=1e-3,
        rtol=1e-3,
    )

    # Module parity
    assert nn.Mish()(x).shape == x.shape
    assert nn.Hardswish()(x).shape == x.shape
    assert nn.LeakyReLU(0.2)(x).shape == x.shape
    assert nn.ELU(1.0)(x).shape == x.shape
    assert nn.SmoothL1Loss()(x, y).numel == 1
    assert nn.CosineSimilarity(dim=-1)(x, y).shape == (2,)
