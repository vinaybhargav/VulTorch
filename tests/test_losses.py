"""Fused softmax cross-entropy: loss values and gradients against references."""

import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn
from vultorch.testing import gradcheck

GPU_DEVICES = [f"vk:{i}" for i in range(torch.vulkan.device_count())]
DEVICES = ["cpu"] + GPU_DEVICES


def _logits(rows=4, cols=5, seed=0):
    return np.random.RandomState(seed).randn(rows, cols).astype(np.float32) * 2.0


def _reference(logits, targets, reduction="mean"):
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True)) + logits.max(
        axis=1, keepdims=True
    )
    nll = logsumexp[:, 0] - logits[np.arange(len(targets)), targets]
    return nll.mean() if reduction == "mean" else nll.sum()


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_matches_reference(dev, reduction):
    logits_np = _logits()
    targets_np = np.array([0, 2, 1, 4], np.float32)

    logits = torch.tensor(logits_np, device=dev)
    targets = torch.tensor(targets_np, device=dev)

    loss = nn.CrossEntropyLoss(reduction=reduction)(logits, targets)
    assert loss.numel == 1
    np.testing.assert_allclose(
        float(loss.numpy().reshape(-1)[0]),
        float(_reference(logits_np, targets_np.astype(int), reduction)),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_gradient_matches_softmax_minus_onehot(dev):
    logits_np = _logits(seed=3)
    targets_np = np.array([0, 2, 1, 4], np.float32)

    logits = torch.tensor(logits_np, device=dev, requires_grad=True)
    targets = torch.tensor(targets_np, device=dev)

    nn.CrossEntropyLoss()(logits, targets).backward()

    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    expected = probs.copy()
    expected[np.arange(len(targets_np)), targets_np.astype(int)] -= 1.0
    expected /= len(targets_np)

    np.testing.assert_allclose(logits.grad.numpy(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_gradcheck(dev, reduction):
    logits_np = _logits(rows=3, cols=4, seed=4)
    targets = torch.tensor(np.array([0, 3, 1], np.float32), device=dev)
    criterion = nn.CrossEntropyLoss(reduction=reduction)

    assert gradcheck(lambda t: criterion(t, targets), logits_np, device=dev)


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_two_dimensional_target(dev):
    """A (N, 1) index tensor is accepted and matches the (N,) form."""
    logits_np = _logits(seed=5)
    logits = torch.tensor(logits_np, device=dev)

    flat = torch.tensor(np.array([0, 2, 1, 4], np.float32), device=dev)
    column = torch.tensor(np.array([[0], [2], [1], [4]], np.float32), device=dev)

    crit = nn.CrossEntropyLoss()
    assert abs(float(crit(logits, flat).item()) - float(crit(logits, column).item())) < 1e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_rejects_bad_target_shape(dev):
    logits = torch.tensor(_logits(), device=dev)
    bad = torch.tensor(np.array([0, 1], np.float32), device=dev)
    with pytest.raises(RuntimeError):
        _ = nn.CrossEntropyLoss()(logits, bad)
