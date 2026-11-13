import pytest
import numpy as np
import vultorch as torch

def test_autograd_basic():
    x = torch.tensor([2.0, 3.0], device="vk:0", requires_grad=True)
    y = x * 3.0
    z = y.sum()
    z.backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), [3.0, 3.0])

def test_autograd_chain_rule():
    # z = (x + y) * (x - y) = x^2 - y^2
    # dz/dx = 2x, dz/dy = -2y
    x_val = np.array([3.0, 4.0], dtype=np.float32)
    y_val = np.array([1.0, 2.0], dtype=np.float32)

    x = torch.tensor(x_val, device="vk:0", requires_grad=True)
    y = torch.tensor(y_val, device="vk:0", requires_grad=True)

    z = ((x + y) * (x - y)).sum()
    z.backward()

    np.testing.assert_allclose(x.grad.numpy(), 2.0 * x_val, rtol=1e-4)
    np.testing.assert_allclose(y.grad.numpy(), -2.0 * y_val, rtol=1e-4)

def test_autograd_matmul():
    a = torch.randn(4, 5, device="vk:0", requires_grad=True)
    b = torch.randn(5, 6, device="vk:0", requires_grad=True)
    c = a @ b
    loss = c.sum()
    loss.backward()

    # dloss/da = ones(4,6) @ b^T
    expected_da = np.ones((4, 6), dtype=np.float32) @ b.numpy().T
    expected_db = a.numpy().T @ np.ones((4, 6), dtype=np.float32)

    np.testing.assert_allclose(a.grad.numpy(), expected_da, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(b.grad.numpy(), expected_db, rtol=1e-4, atol=1e-4)

def test_no_grad_context():
    x = torch.randn(5, device="vk:0", requires_grad=True)
    with torch.no_grad():
        y = x * 2.0
    assert y.requires_grad is False
    assert y.grad_fn is None
