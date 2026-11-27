"""
Phase 3.3 Tests:
- argmax, argmin, prod, all, any, var, std, norm
- cumsum, cumprod, sort, topk
- Analytical & finite-difference gradcheck across cpu, vk:0, vk:1
"""

import numpy as np
import pytest

import vultorch as torch


DEVICES = ["cpu"] + [f"vk:{i}" for i in range(torch.vulkan.device_count())]


def _rand_np(shape, seed=0, low=-1.5, high=1.5):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(np.float32)


@pytest.mark.parametrize("dev", DEVICES)
def test_argmax_argmin_prod_all_any(dev):
    x_np = np.array([[1.0, 5.0, 2.0, 5.0], [-3.0, 0.0, -3.0, 2.0]], dtype=np.float32)
    x = torch.tensor(x_np, device=dev)

    assert int(x.argmax().item()) == int(np.argmax(x_np))
    assert int(x.argmin().item()) == int(np.argmin(x_np))
    np.testing.assert_array_equal(x.argmax(dim=1).numpy(), np.argmax(x_np, axis=1))
    np.testing.assert_array_equal(x.argmin(dim=0).numpy(), np.argmin(x_np, axis=0))
    assert x.argmax(dim=1).dtype == torch.int64

    p_np = np.array([[1.5, 2.0, 3.0], [0.5, 4.0, 2.0]], dtype=np.float32)
    p = torch.tensor(p_np, device=dev)
    np.testing.assert_allclose(p.prod(dim=1).numpy(), np.prod(p_np, axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(p.prod().item()), float(np.prod(p_np)), rtol=1e-5, atol=1e-5)

    b_np = np.array([[True, True, False], [True, True, True]], dtype=np.bool_)
    b = torch.tensor(b_np, device=dev)
    np.testing.assert_array_equal(b.all(dim=1).numpy(), np.all(b_np, axis=1))
    np.testing.assert_array_equal(b.any(dim=0).numpy(), np.any(b_np, axis=0))
    assert bool(b.all().item()) is False
    assert bool(b.any().item()) is True


@pytest.mark.parametrize("dev", DEVICES)
def test_var_std_norm_and_gradcheck(dev):
    x_np = _rand_np((3, 5), seed=11)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    np.testing.assert_allclose(
        x.var(dim=1, unbiased=True).numpy(), np.var(x_np, axis=1, ddof=1), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        x.std(dim=1, unbiased=False).numpy(), np.std(x_np, axis=1, ddof=0), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        x.norm(p=2.0, dim=1).numpy(), np.linalg.norm(x_np, ord=2, axis=1), rtol=1e-4, atol=1e-4
    )

    assert torch.gradcheck(
        lambda t: (t.var(dim=1, unbiased=True) + t.std(dim=1, unbiased=False)).sum(),
        x,
        eps=2e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_cumsum_cumprod_and_gradcheck(dev):
    x_np = _rand_np((3, 6), seed=22, low=0.4, high=1.5)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    np.testing.assert_allclose(
        x.cumsum(dim=1).numpy(), np.cumsum(x_np, axis=1), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        x.cumprod(dim=1).numpy(), np.cumprod(x_np, axis=1), rtol=1e-4, atol=1e-4
    )

    assert torch.gradcheck(
        lambda t: (t.cumsum(dim=1) ** 2).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )
    assert torch.gradcheck(
        lambda t: t.cumprod(dim=1).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_sort_and_topk_and_gradcheck(dev):
    x_np = np.array([[3.1, -1.2, 5.4, 0.2, 2.7], [4.0, 1.5, -2.3, 8.8, 0.0]], dtype=np.float32)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    vals_asc, idx_asc = x.sort(dim=1, descending=False)
    np.testing.assert_allclose(vals_asc.numpy(), np.sort(x_np, axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(idx_asc.numpy(), np.argsort(x_np, axis=1, kind="stable"))

    top_vals, top_idx = x.topk(3, dim=1, largest=True)
    ref_order = np.argsort(-x_np, axis=1, kind="stable")[:, :3]
    ref_vals = np.take_along_axis(x_np, ref_order, axis=1)
    np.testing.assert_allclose(top_vals.numpy(), ref_vals, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(top_idx.numpy(), ref_order)

    assert torch.gradcheck(
        lambda t: (t.topk(3, dim=1, largest=True).values ** 2).sum(),
        x,
        eps=2e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )
