import pytest
import vultorch as torch


def test_contractions():
    # dot
    a = torch.tensor([1.0, 2.0, 3.0], device="vk:0")
    b = torch.tensor([4.0, 5.0, 6.0], device="vk:0")
    d = torch.dot(a, b)
    assert d.item() == pytest.approx(32.0)

    # mv
    M = torch.randn(4, 3, device="vk:0")
    v = torch.randn(3, device="vk:0")
    mv_out = torch.mv(M, v)
    assert mv_out.shape == (4,)
    assert torch.allclose(mv_out.cpu(), (M.cpu() @ v.cpu().unsqueeze(1)).squeeze(1))

    # outer
    o = torch.outer(a, b)
    assert o.shape == (3, 3)
    assert torch.allclose(o.cpu(), a.cpu().unsqueeze(1) @ b.cpu().unsqueeze(0))

    # addmm
    C = torch.zeros(4, 5, device="vk:0")
    A = torch.randn(4, 3, device="vk:0")
    B = torch.randn(3, 5, device="vk:0")
    out = torch.addmm(C, A, B, alpha=2.0, beta=0.0)
    assert torch.allclose(out.cpu(), (A.cpu() @ B.cpu()) * 2.0, rtol=1e-4, atol=1e-4)


def test_einsum():
    A = torch.randn(3, 4, 5, device="vk:0")
    B = torch.randn(4, 6, device="vk:0")
    out = torch.einsum("ijk,jl->ikl", A, B)
    assert out.shape == (3, 5, 6)


def test_linalg_and_fft():
    # norm
    x = torch.tensor([[3.0, 4.0]], device="vk:0")
    n = torch.linalg.norm(x)
    assert n.item() == pytest.approx(5.0)

    # qr
    A = torch.randn(4, 4, device="vk:0")
    Q, R = torch.linalg.qr(A)
    assert Q.shape == (4, 4)
    assert R.shape == (4, 4)

    # fft
    sig = torch.randn(64, device="vk:0")
    f = torch.fft.fft(sig)
    assert f.shape == (64,)
