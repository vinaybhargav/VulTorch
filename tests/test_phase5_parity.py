import pytest
import numpy as np
import vultorch
import vultorch.nn as nn
import vultorch.nn.functional as F
from vultorch.utils.data import TensorDataset, DataLoader


def _devices():
    devs = ["cpu"]
    if vultorch.vulkan.is_available():
        for i in range(vultorch.vulkan.device_count()):
            devs.append(f"vk:{i}")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_math_and_rounding_helpers_parity(device):
    arr = np.array([-2.7, -1.3, -0.4, 0.4, 1.3, 2.7], dtype=np.float32)
    t = vultorch.tensor(arr, device=device)

    np.testing.assert_allclose(vultorch.round(t).numpy(), np.round(arr), atol=1e-5)
    np.testing.assert_allclose(vultorch.ceil(t).numpy(), np.ceil(arr), atol=1e-5)
    np.testing.assert_allclose(vultorch.floor(t).numpy(), np.floor(arr), atol=1e-5)
    np.testing.assert_allclose(vultorch.trunc(t).numpy(), np.trunc(arr), atol=1e-5)
    np.testing.assert_allclose(vultorch.frac(t).numpy(), arr - np.trunc(arr), atol=1e-5)
    np.testing.assert_allclose(vultorch.square(t).numpy(), arr * arr, atol=1e-5)
    np.testing.assert_allclose(vultorch.reciprocal(t).numpy(), 1.0 / arr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.neg(t).numpy(), -arr, atol=1e-5)
    np.testing.assert_allclose(vultorch.pow(t, 2).numpy(), arr**2, rtol=1e-5, atol=1e-5)

    pos = np.array([0.2, 0.5, 0.8, 1.5, 2.5], dtype=np.float32)
    tp = vultorch.tensor(pos, device=device)
    np.testing.assert_allclose(vultorch.exp2(tp).numpy(), np.exp2(pos), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.expm1(tp).numpy(), np.expm1(pos), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.log2(tp).numpy(), np.log2(pos), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.log10(tp).numpy(), np.log10(pos), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.log1p(tp).numpy(), np.log1p(pos), rtol=1e-5, atol=1e-5)

    unit = np.array([-0.6, -0.2, 0.0, 0.3, 0.7], dtype=np.float32)
    tu = vultorch.tensor(unit, device=device)
    np.testing.assert_allclose(vultorch.tan(tu).numpy(), np.tan(unit), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vultorch.asin(tu).numpy(), np.arcsin(unit), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vultorch.acos(tu).numpy(), np.arccos(unit), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vultorch.atan(tu).numpy(), np.arctan(unit), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vultorch.sinh(tu).numpy(), np.sinh(unit), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.cosh(tu).numpy(), np.cosh(unit), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.asinh(tu).numpy(), np.arcsinh(unit), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.atanh(tu).numpy(), np.arctanh(unit), rtol=1e-5, atol=1e-5)

    # erf / erfc against known values
    import math

    expected_erf = np.array([math.erf(float(v)) for v in unit], dtype=np.float32)
    np.testing.assert_allclose(vultorch.erf(tu).numpy(), expected_erf, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vultorch.erfc(tu).numpy(), 1.0 - expected_erf, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_nan_inf_comparison_and_binary_helpers(device):
    special = np.array([1.0, np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    ts = vultorch.tensor(special, device=device)
    np.testing.assert_array_equal(vultorch.isnan(ts).numpy(), np.isnan(special))
    np.testing.assert_array_equal(vultorch.isinf(ts).numpy(), np.isinf(special))
    np.testing.assert_array_equal(vultorch.isfinite(ts).numpy(), np.isfinite(special))

    cleaned = vultorch.nan_to_num(ts, nan=0.0, posinf=100.0, neginf=-100.0).numpy()
    np.testing.assert_allclose(cleaned, [1.0, 0.0, 100.0, -100.0, 0.0], atol=1e-5)

    a = vultorch.tensor([1.0, 4.0, 2.0, 5.0], device=device)
    b = vultorch.tensor([2.0, 3.0, 2.0, 1.0], device=device)
    np.testing.assert_allclose(vultorch.maximum(a, b).numpy(), [2.0, 4.0, 2.0, 5.0])
    np.testing.assert_allclose(vultorch.minimum(a, b).numpy(), [1.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(vultorch.lerp(a, b, 0.25).numpy(), [1.25, 3.75, 2.0, 4.0])
    assert vultorch.allclose(a, a.clone())
    assert vultorch.equal(a, a.clone())
    assert vultorch.min(a).item() == pytest.approx(1.0)
    assert vultorch.amax(a).item() == pytest.approx(5.0)
    assert vultorch.argsort(a).tolist() == [0, 2, 1, 3]


@pytest.mark.parametrize("device", _devices())
def test_differentiable_math_and_losses_gradcheck(device):
    x = vultorch.tensor([-0.4, 0.2, 0.5, -0.1], device=device, requires_grad=True)
    y = vultorch.tensor([0.3, -0.5, 0.1, 0.6], device=device, requires_grad=True)
    pos = vultorch.tensor([0.6, 1.2, 1.8, 0.9], device=device, requires_grad=True)

    assert vultorch.gradcheck(lambda t: t.square().sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(
        lambda t: t.reciprocal().sum(), pos, device=device, atol=1e-3, rtol=1e-3
    )
    assert vultorch.gradcheck(lambda t: t.erf().sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(lambda t: t.tan().sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(lambda t: t.asin().sum(), x, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(lambda t: t.log1p().sum(), pos, device=device, atol=1e-3, rtol=1e-3)
    assert vultorch.gradcheck(
        lambda a, b: a.lerp(b, 0.3).sum(), x, y, device=device, atol=1e-3, rtol=1e-3
    )
    assert vultorch.gradcheck(
        lambda a, b: F.l1_loss(a, b, reduction="sum"), x, y, device=device, atol=1e-3, rtol=1e-3
    )

    targets = vultorch.tensor([0.0, 1.0, 1.0, 0.0], device=device)
    assert vultorch.gradcheck(
        lambda a: F.binary_cross_entropy_with_logits(a, targets, reduction="sum"),
        x,
        device=device,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("device", _devices())
def test_constructors_tensor_api_and_functional_parity(device):
    vultorch.manual_seed(42)
    r_int = vultorch.randint(0, 10, (64,), device=device)
    assert r_int.shape == (64,)
    assert r_int.dtype == vultorch.int64
    arr_int = r_int.numpy()
    assert arr_int.min() >= 0 and arr_int.max() < 10

    perm = vultorch.randperm(16, device=device)
    assert sorted(perm.tolist()) == list(range(16))

    base = vultorch.randn(4, 8, device=device)
    assert base.size() == (4, 8)
    assert base.size(1) == 8
    assert base.dim() == 2
    assert base.is_floating_point()
    assert vultorch.full_like(base, 3.0).shape == (4, 8)
    assert vultorch.rand_like(base).shape == (4, 8)
    assert vultorch.randn_like(base).shape == (4, 8)

    # DataLoader on-GPU shuffle
    ds = TensorDataset(
        vultorch.randn(20, 4, device=device), vultorch.randint(0, 3, (20,), device=device)
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batches = list(loader)
    assert sum(bx.shape[0] for bx, _ in batches) == 20

    # F.* & nn.* wrappers
    img = vultorch.randn(2, 3, 8, 8, device=device, requires_grad=True)
    w = vultorch.randn(4, 3, 3, 3, device=device, requires_grad=True)
    conv_out = F.conv2d(img, w, padding=1)
    pool_out = F.max_pool2d(conv_out, 2)
    up_out = nn.Upsample(scale_factor=2, mode="bilinear")(pool_out)
    assert up_out.shape == (2, 4, 8, 8)
    up_out.sum().backward()
    assert img.grad is not None and w.grad is not None

    md = nn.ModuleDict({"id": nn.Identity(), "bce": nn.BCEWithLogitsLoss()})
    assert "id" in md and len(md) == 2


def test_phase5_multigpu_all():
    if not vultorch.vulkan.is_available() or vultorch.vulkan.device_count() < 2:
        pytest.skip("Requires at least 2 Vulkan GPUs for multi_gpu(['all']) test")

    from vultorch.autograd import Function
    from vultorch.utils.checkpoint import checkpoint

    class _ScaleShiftFn(Function):
        @staticmethod
        def forward(ctx, x, scale=0.5):
            ctx.save_for_backward(x)
            ctx.scale = scale
            return x * scale + 0.1

        @staticmethod
        def backward(ctx, g):
            _ = ctx.saved_tensors
            return g * ctx.scale, None

    class _Phase5Block(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.mish = nn.Mish()
            self.fc2 = nn.Linear(dim, dim)
            self.hswish = nn.Hardswish()

        def forward(self, x):
            h = self.mish(self.fc1(x))
            h = self.hswish(self.fc2(h))
            return _ScaleShiftFn.apply(h, 0.5)

    class _Phase5Model(nn.Module):
        def __init__(self, dim=32):
            super().__init__()
            self.b1 = _Phase5Block(dim)
            self.b2 = _Phase5Block(dim)
            self.head = nn.Linear(dim, 8)

        def forward(self, x):
            h = checkpoint(self.b1, x)
            h = self.b2(h)
            h = h.lerp(h.erf(), 0.25)
            return self.head(h)

    vultorch.manual_seed(42)
    base = _Phase5Model(32).to("vk:0")
    parallel = base.multi_gpu(["all"])
    opt = vultorch.optim.AdamW(base.parameters(), lr=1e-2)
    crit = nn.SmoothL1Loss()

    x = vultorch.randn(64, 32, device="vk:0")
    y = vultorch.randn(64, 8, device="vk:0")
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    losses = []
    for _ in range(3):
        epoch_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            out = parallel(bx)
            loss = crit(out, by)
            loss.backward()
            for p in base.parameters():
                assert p.grad is not None
                assert not vultorch.isnan(p.grad).any().item()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    assert np.isfinite(losses[-1])
    assert losses[-1] <= losses[0] + 0.1
