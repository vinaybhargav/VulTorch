import pytest
import numpy as np
import vultorch as torch

def test_basic_arithmetic():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")

    np.testing.assert_allclose((a + b).numpy(), a_np + b_np)
    np.testing.assert_allclose((a - b).numpy(), a_np - b_np)
    np.testing.assert_allclose((a * b).numpy(), a_np * b_np)
    np.testing.assert_allclose((a / b).numpy(), a_np / b_np)
    np.testing.assert_allclose((-a).numpy(), -a_np)

def test_broadcasting():
    a_np = np.random.randn(4, 1, 8).astype(np.float32)
    b_np = np.random.randn(1, 6, 8).astype(np.float32)

    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")

    c = a + b
    assert c.shape == (4, 6, 8)
    np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5, atol=1e-5)

def test_matmul():
    a_np = np.random.randn(64, 48).astype(np.float32)
    b_np = np.random.randn(48, 32).astype(np.float32)

    a = torch.tensor(a_np, device="vk:0")
    b = torch.tensor(b_np, device="vk:0")

    c = a @ b
    assert c.shape == (64, 32)
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-4, atol=1e-4)

def test_activations():
    x_np = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
    x = torch.tensor(x_np, device="vk:0")

    # ReLU
    np.testing.assert_allclose(x.relu().numpy(), np.maximum(x_np, 0.0))

    # Sigmoid
    np.testing.assert_allclose(x.sigmoid().numpy(), 1.0 / (1.0 + np.exp(-x_np)), rtol=1e-4, atol=1e-4)

    # Tanh
    np.testing.assert_allclose(x.tanh().numpy(), np.tanh(x_np), rtol=1e-4, atol=1e-4)

    # Exp & Log
    pos_np = np.linspace(0.1, 5.0, 50, dtype=np.float32)
    pos = torch.tensor(pos_np, device="vk:0")
    np.testing.assert_allclose(pos.exp().numpy(), np.exp(pos_np), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pos.log().numpy(), np.log(pos_np), rtol=1e-4, atol=1e-4)

def test_reductions():
    x_np = np.random.randn(10, 20).astype(np.float32)
    x = torch.tensor(x_np, device="vk:0")

    # Total sum
    np.testing.assert_allclose(x.sum().item(), np.sum(x_np), rtol=1e-4, atol=1e-4)

    # Sum along last dim
    np.testing.assert_allclose(x.sum(dim=-1).numpy(), np.sum(x_np, axis=-1), rtol=1e-4, atol=1e-4)

    # Mean along last dim
    np.testing.assert_allclose(x.mean(dim=-1).numpy(), np.mean(x_np, axis=-1), rtol=1e-4, atol=1e-4)

def test_softmax():
    x_np = np.random.randn(8, 16).astype(np.float32)
    x = torch.tensor(x_np, device="vk:0")

    sm = x.softmax(dim=-1).numpy()
    np.testing.assert_allclose(np.sum(sm, axis=-1), np.ones(8), rtol=1e-4, atol=1e-4)
    assert np.all(sm >= 0.0)
