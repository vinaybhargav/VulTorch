import pytest
import numpy as np
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


def test_linear_layer():
    layer = nn.Linear(10, 5).to("vk:0")
    x = torch.randn(4, 10, device="vk:0")
    y = layer(x)
    assert y.shape == (4, 5)
    assert y.device == torch.device("vk:0")


def test_conv2d_layer():
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1).to("vk:0")
    x = torch.randn(2, 3, 32, 32, device="vk:0", requires_grad=True)
    out = conv(x)
    assert out.shape == (2, 16, 32, 32)
    loss = out.sum()
    loss.backward()
    assert conv.weight.grad is not None
    assert x.grad is not None


def test_cross_entropy_and_adam():
    model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 3)).to("vk:0")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 20, device="vk:0")
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device="vk:0")

    initial_loss = criterion(model(x), y).item()

    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), y).item()
    assert final_loss < initial_loss
