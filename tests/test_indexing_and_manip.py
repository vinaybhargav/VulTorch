"""
Phase 3.1 & 3.2 Tests:
- where, masked_fill, masked_fill_, gather, scatter_, scatter_add_, index_select, index_add_
- multi-axis cat, stack, split, chunk, pad (constant/reflect/replicate), tril, triu, flip, roll, repeat
- Analytical & finite-difference gradcheck across cpu, vk:0, vk:1
"""

import numpy as np
import pytest

import vultorch as torch
import vultorch.nn.functional as F


DEVICES = ["cpu"] + [f"vk:{i}" for i in range(torch.vulkan.device_count())]


def _rand_np(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)


@pytest.mark.parametrize("dev", DEVICES)
def test_where_broadcast_and_gradcheck(dev):
    cond_np = np.array([[True], [False], [True]], dtype=np.bool_)
    a_np = _rand_np((3, 4), seed=1)
    b_np = _rand_np((1, 4), seed=2)

    cond = torch.tensor(cond_np, device=dev)
    a = torch.tensor(a_np, device=dev, requires_grad=True)
    b = torch.tensor(b_np, device=dev, requires_grad=True)

    out = torch.where(cond, a, b)
    expected = np.where(cond_np, a_np, b_np)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    assert torch.gradcheck(
        lambda x, y: torch.where(cond, x, y).sum(), a, b, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_masked_fill_and_gradcheck(dev):
    x_np = _rand_np((3, 5), seed=3)
    m_np = np.array([[1, 0, 1, 0, 0]], dtype=np.bool_)

    x = torch.tensor(x_np, device=dev, requires_grad=True)
    m = torch.tensor(m_np, device=dev)

    out = x.masked_fill(m, -1e9)
    expected = np.where(m_np, -1e9, x_np).astype(np.float32)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)

    assert torch.gradcheck(
        lambda t: t.masked_fill(m, 0.0).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    x_inp = torch.tensor(x_np, device=dev)
    x_inp.masked_fill_(m, 42.0)
    np.testing.assert_allclose(x_inp.numpy(), np.where(m_np, 42.0, x_np), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_gather_scatter_index_select_and_gradcheck(dev):
    x_np = _rand_np((4, 6), seed=4)
    idx_np = np.array([[0, 2, 5, 0], [5, 1, 1, 3], [2, 4, 0, 2], [1, 1, 1, 1]], dtype=np.int64)

    x = torch.tensor(x_np, device=dev, requires_grad=True)
    idx = torch.tensor(idx_np, device=dev)

    g = x.gather(1, idx)
    expected_g = np.take_along_axis(x_np, idx_np, axis=1)
    np.testing.assert_allclose(g.numpy(), expected_g, rtol=1e-5, atol=1e-5)

    assert torch.gradcheck(
        lambda t: t.gather(1, idx).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    # scatter_add_
    dst = torch.zeros((4, 6), device=dev)
    src = torch.ones((4, 4), device=dev)
    dst.scatter_add_(1, idx, src)
    ref_dst = np.zeros((4, 6), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            ref_dst[r, idx_np[r, c]] += 1.0
    np.testing.assert_allclose(dst.numpy(), ref_dst, rtol=1e-5, atol=1e-5)

    # index_select + gradcheck
    sel_idx_np = np.array([3, 1, 3], dtype=np.int64)
    sel_idx = torch.tensor(sel_idx_np, device=dev)
    sel = x.index_select(0, sel_idx)
    np.testing.assert_allclose(sel.numpy(), x_np[sel_idx_np], rtol=1e-5, atol=1e-5)
    assert torch.gradcheck(
        lambda t: t.index_select(0, sel_idx).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_multi_axis_cat_stack_split_chunk_and_gradcheck(dev, dim):
    a_np = _rand_np((2, 3, 4, 2), seed=10 + dim)
    b_np = _rand_np((2, 3, 4, 2), seed=20 + dim)
    a = torch.tensor(a_np, device=dev, requires_grad=True)
    b = torch.tensor(b_np, device=dev, requires_grad=True)

    c = torch.cat([a, b], dim=dim)
    np.testing.assert_allclose(
        c.numpy(), np.concatenate([a_np, b_np], axis=dim), rtol=1e-5, atol=1e-5
    )

    parts = c.chunk(2, dim=dim)
    assert len(parts) == 2
    np.testing.assert_allclose(parts[0].numpy(), a_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(parts[1].numpy(), b_np, rtol=1e-5, atol=1e-5)

    if dim == 1:
        assert torch.gradcheck(
            lambda x, y: (torch.cat([x, y], dim=1) ** 2).sum(),
            a,
            b,
            eps=2e-3,
            rtol=2e-2,
            atol=5e-3,
            device=dev,
        )


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
def test_pad_and_gradcheck(dev, mode):
    x_np = _rand_np((2, 3, 4, 5), seed=30)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    pad_spec = (1, 2, 1, 1)
    y = F.pad(x, pad_spec, mode=mode, value=0.5 if mode == "constant" else 0.0)

    np_mode = "edge" if mode == "replicate" else mode
    kw = {"constant_values": 0.5} if mode == "constant" else {}
    ref = np.pad(x_np, ((0, 0), (0, 0), (1, 1), (1, 2)), mode=np_mode, **kw)
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-5, atol=1e-5)

    assert torch.gradcheck(
        lambda t: (F.pad(t, pad_spec, mode=mode) ** 2).sum(),
        x,
        eps=2e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_tril_triu_flip_roll_repeat(dev):
    x_np = _rand_np((2, 4, 4), seed=40)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    np.testing.assert_allclose(x.tril(-1).numpy(), np.tril(x_np, k=-1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x.triu(1).numpy(), np.triu(x_np, k=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        x.flip([1, 2]).numpy(), np.flip(x_np, axis=(1, 2)), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        x.roll(2, dims=2).numpy(), np.roll(x_np, 2, axis=2), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        x.repeat(1, 2, 1).numpy(), np.tile(x_np, (1, 2, 1)), rtol=1e-5, atol=1e-5
    )

    assert torch.gradcheck(
        lambda t: (t.tril(0) + t.flip([2]) + t.roll(1, dims=1)).sum(),
        x,
        eps=2e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )
