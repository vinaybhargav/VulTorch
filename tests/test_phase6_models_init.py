import pytest
import vultorch as torch
import vultorch.nn as nn


def test_nn_init_methods():
    w = torch.empty(64, 32, device="vk:0")
    nn.init.xavier_uniform_(w)
    assert not torch.allclose(w, torch.zeros_like(w))

    nn.init.kaiming_normal_(w)
    assert not torch.allclose(w, torch.zeros_like(w))

    nn.init.constant_(w, 3.14)
    assert torch.allclose(w, torch.full_like(w, 3.14))


def test_lstm_forward_backward():
    lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
    lstm.to("vk:0")
    x = torch.randn(4, 8, 16, device="vk:0", requires_grad=True)
    out, (hn, cn) = lstm(x)

    assert out.shape == (4, 8, 32)
    assert hn.shape == (2, 4, 32)
    assert cn.shape == (2, 4, 32)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gru_forward_backward():
    gru = nn.GRU(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
    gru.to("vk:0")
    x = torch.randn(4, 8, 16, device="vk:0", requires_grad=True)
    out, hn = gru(x)

    assert out.shape == (4, 8, 32)
    assert hn.shape == (2, 4, 32)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_transformer_encoder():
    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
    encoder = nn.TransformerEncoder(layer, num_layers=2)
    encoder.to("vk:0")
    x = torch.randn(2, 10, 32, device="vk:0", requires_grad=True)
    out = encoder(x)
    assert out.shape == (2, 10, 32)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
