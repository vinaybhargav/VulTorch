"""The fused Adam/AdamW kernel must match the composed-op reference exactly enough."""

import numpy as np
import pytest

import vultorch as torch
import vultorch.optim as optim

DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())] + ["cpu"]


def _run(opt_cls, device, fused, shape, steps=5, **kwargs):
    rng = np.random.RandomState(0)
    p = torch.tensor(rng.randn(*shape).astype(np.float32), device=device, requires_grad=True)
    opt = opt_cls([p], fused=fused, **kwargs)
    for step in range(steps):
        g = np.random.RandomState(100 + step).randn(*shape).astype(np.float32)
        p.grad = torch.tensor(g, device=device)
        opt.step()
    return p.numpy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opt_cls", [optim.Adam, optim.AdamW])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("shape", [(1,), (257,), (33, 65), (1024, 130)])
def test_fused_matches_reference(device, opt_cls, weight_decay, shape):
    kwargs = dict(lr=1e-2, weight_decay=weight_decay)
    fused = _run(opt_cls, device, True, shape, **kwargs)
    ref = _run(opt_cls, device, False, shape, **kwargs)
    np.testing.assert_allclose(fused, ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_updates_moments_in_place(device):
    p = torch.tensor(np.ones(8, np.float32), device=device, requires_grad=True)
    opt = optim.Adam([p], lr=0.1)
    p.grad = torch.tensor(np.full(8, 2.0, np.float32), device=device)
    opt.step()
    state = opt.state[p]
    assert state["step"] == 1
    np.testing.assert_allclose(state["exp_avg"].numpy(), 0.2, rtol=1e-5)
    np.testing.assert_allclose(state["exp_avg_sq"].numpy(), 0.004, rtol=1e-4)
    # First Adam step moves each weight by ~lr regardless of gradient scale.
    np.testing.assert_allclose(p.numpy(), 0.9, rtol=1e-4)
