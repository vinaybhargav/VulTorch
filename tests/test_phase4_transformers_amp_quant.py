"""
Phase 4 Verification Suite:
- 4.1 Fused Slang FlashAttention-2 (`F.scaled_dot_product_attention`, `nn.MultiheadAttention`)
      including causal & non-causal masks, GQA (`H_q=8, H_kv=2`), memory & speedup gate at S=2048.
- 4.2 Fused `RoPE` (`nn.RotaryEmbedding`, `F.apply_rotary_pos_emb`), `SwiGLU`/`GeGLU`, and
      Slang GEMM `IEpilogue` specializations (`ReLUEpilogue`, `GELUEpilogue`, `SiLUEpilogue`).
- 4.3 Automatic Mixed Precision (`vultorch.amp.autocast`, `GradScaler`) and Weight-Only
      `int4`/`int8` `nn.QuantizedLinear` (including `Linear(4096, 4096)` on 2 GB MX450).
- 4.4 Zero-Boilerplate Ergonomics (`nn.MultiGPU(model, device_ids=['all'])`, `model.multi_gpu(['all'])`
      with automatic backward gradient reduction, `vultorch.quantize(model, bits=4)`,
      `vultorch.set_default_device`, `with vultorch.device(...)`, `DataLoader`, and actionable
      device mismatch error messages).
"""

import math
import time
import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn
import vultorch.nn.functional as F
import vultorch.optim as optim
from vultorch.utils.data import TensorDataset, DataLoader


DEVICES = ["cpu"] + [f"vk:{i}" for i in range(torch.vulkan.device_count())]
GPU_DEVICES = [d for d in DEVICES if d.startswith("vk:")]


def _rand_np(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.7, 0.7, size=shape).astype(np.float32)


def _naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
    h_q = q.shape[1]
    h_kv = k.shape[1]
    if h_q != h_kv:
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
    if is_causal:
        s_q, s_k = q.shape[2], k.shape[2]
        mask = torch.ones((s_q, s_k), device=q.device, dtype=torch.bool).triu(
            diagonal=1 + (s_k - s_q)
        )
        scores = scores.masked_fill(mask, -1e9)
    probs = scores.softmax(dim=-1)
    return probs @ v


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("heads", [(4, 4), (8, 2)])  # MHA and GQA (H_q=8, H_kv=2)
def test_flash_attention_matches_naive_and_gradcheck(dev, is_causal, heads):
    h_q, h_kv = heads
    B, S, D = 2, 12, 16
    q_np = _rand_np((B, h_q, S, D), seed=10)
    k_np = _rand_np((B, h_kv, S, D), seed=11)
    v_np = _rand_np((B, h_kv, S, D), seed=12)
    w_np = _rand_np((B, h_q, S, D), seed=13)

    # 1. Forward + Backward parity in FP32
    q_fa = torch.tensor(q_np, device=dev, requires_grad=True)
    k_fa = torch.tensor(k_np, device=dev, requires_grad=True)
    v_fa = torch.tensor(v_np, device=dev, requires_grad=True)
    w = torch.tensor(w_np, device=dev)

    out_fa = F.scaled_dot_product_attention(q_fa, k_fa, v_fa, is_causal=is_causal)
    (out_fa * w).sum().backward()

    q_nv = torch.tensor(q_np, device=dev, requires_grad=True)
    k_nv = torch.tensor(k_np, device=dev, requires_grad=True)
    v_nv = torch.tensor(v_np, device=dev, requires_grad=True)
    out_nv = _naive_attention(q_nv, k_nv, v_nv, is_causal=is_causal)
    (out_nv * w).sum().backward()

    np.testing.assert_allclose(out_fa.numpy(), out_nv.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(q_fa.grad.numpy(), q_nv.grad.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(k_fa.grad.numpy(), k_nv.grad.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(v_fa.grad.numpy(), v_nv.grad.numpy(), rtol=1e-3, atol=1e-3)

    # 2. Forward parity in FP16
    q_16 = torch.tensor(q_np, device=dev, dtype=torch.float16)
    k_16 = torch.tensor(k_np, device=dev, dtype=torch.float16)
    v_16 = torch.tensor(v_np, device=dev, dtype=torch.float16)
    out_16 = F.scaled_dot_product_attention(q_16, k_16, v_16, is_causal=is_causal)
    assert out_16.dtype == torch.float16
    np.testing.assert_allclose(out_16.float().numpy(), out_nv.numpy(), rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("dev", GPU_DEVICES)
def test_flash_attention_s2048_memory_and_speed_gate(dev):
    did = int(dev.split(":")[1])
    B, H, S, D = 2, 8, 2048, 64
    q = torch.randn(B, H, S, D, device=dev)
    k = torch.randn(B, H, S, D, device=dev)
    v = torch.randn(B, H, S, D, device=dev)

    # Warmup
    _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.vulkan.synchronize(did)

    mem_before = torch.vulkan.active_arena_bytes(did)
    out_fa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.vulkan.synchronize(did)
    mem_after = torch.vulkan.active_arena_bytes(did)

    # Activation memory delta: O is [2, 8, 2048, 64] = 8 MiB, L is [2, 8, 2048] = 0.125 MiB (< 35 MiB gate)
    # whereas naive attention materializes [2, 8, 2048, 2048] scores + mask + probs = 768 MiB!
    activation_mib = (mem_after - mem_before) / (1024.0 * 1024.0)
    assert out_fa.shape == (B, H, S, D)
    assert activation_mib < 35.0, (
        f"Expected < 35 MiB activation memory, got {activation_mib:.2f} MiB"
    )

    # Verify causal early-termination speedup: causal FlashAttention skips the upper-triangular tiles (~2x faster than non-causal)
    q_b = q[:, :4, :1024, :]
    k_b = k[:, :4, :1024, :]
    v_b = v[:, :4, :1024, :]
    _ = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
    _ = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False)
    torch.vulkan.synchronize(did)

    t0 = time.perf_counter()
    _ = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
    torch.vulkan.synchronize(did)
    dt_causal = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False)
    torch.vulkan.synchronize(did)
    dt_full = time.perf_counter() - t0

    assert dt_causal < dt_full


@pytest.mark.parametrize("dev", DEVICES)
def test_rope_swiglu_geglu_and_gemm_epilogues(dev):
    # 1. RoPE forward + gradcheck
    B, H, S, D = 2, 2, 6, 8
    x_np = _rand_np((B, H, S, D), seed=21)
    w_np = _rand_np((B, H, S, D), seed=22)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    w = torch.tensor(w_np, device=dev)
    rope = nn.RotaryEmbedding(D, max_position_embeddings=32, seq_dim=2, device=dev)

    y_rope = rope(x)
    assert y_rope.shape == (B, H, S, D)
    # RoPE is an orthogonal rotation: norm must be preserved!
    np.testing.assert_allclose(
        (y_rope**2).sum(dim=-1).numpy(),
        (x**2).sum(dim=-1).numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    assert torch.gradcheck(
        lambda t: (rope(t) * w).sum(), x, eps=1e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    # 2. SwiGLU & GeGLU forward + gradcheck
    a_np = _rand_np((3, 8), seed=23)
    b_np = _rand_np((3, 8), seed=24)
    gw_np = _rand_np((3, 8), seed=25)
    a = torch.tensor(a_np, device=dev, requires_grad=True)
    b = torch.tensor(b_np, device=dev, requires_grad=True)
    gw = torch.tensor(gw_np, device=dev)

    y_swi = F.swiglu(a, b)
    ref_swi = a.silu() * b
    np.testing.assert_allclose(y_swi.numpy(), ref_swi.numpy(), rtol=1e-4, atol=1e-4)
    assert torch.gradcheck(
        lambda t: (F.swiglu(t, b) * gw).sum(), a, eps=1e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    y_ge = F.geglu(a, b)
    ref_ge = a.gelu() * b
    np.testing.assert_allclose(y_ge.numpy(), ref_ge.numpy(), rtol=1e-4, atol=1e-4)
    assert torch.gradcheck(
        lambda t: (F.geglu(t, b) * gw).sum(), a, eps=1e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    # 3. GEMM Epilogues (ReLU, GELU, SiLU) in single dispatch
    lx = torch.tensor(_rand_np((16, 32), seed=26), device=dev)
    lin = nn.Linear(32, 24, device=dev)
    with torch.no_grad():
        base_out = lin(lx)
        for ep, ref_out in [
            ("relu", base_out.relu()),
            ("gelu", base_out.gelu()),
            ("silu", base_out.silu()),
        ]:
            fused_out = F.linear(lx, lin.weight, lin.bias, epilogue=ep)
            np.testing.assert_allclose(fused_out.numpy(), ref_out.numpy(), rtol=1e-4, atol=1e-4)


class MicroTransformerBlock(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_kv_heads=2, device="vk"):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, device=device)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, num_kv_heads=n_kv_heads, use_rope=True, device=device
        )
        self.norm2 = nn.RMSNorm(d_model, device=device)
        self.gate_up = nn.Linear(d_model, d_model * 4, bias=False, device=device)
        self.swiglu = nn.SwiGLU()
        self.down = nn.Linear(d_model * 2, d_model, bias=False, device=device)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), is_causal=True)
        x = x + self.down(self.swiglu(self.gate_up(self.norm2(x))))
        return x


class MicroLLaMA(nn.Module):
    def __init__(self, vocab_size=64, d_model=32, n_layers=4, device="vk"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [
                MicroTransformerBlock(d_model=d_model, n_heads=4, n_kv_heads=2, device=device)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model, device=device)
        self.head = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(self, tokens):
        h = self.emb(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)


@pytest.mark.parametrize("dev", DEVICES)
def test_amp_autocast_gradscaler_and_4layer_transformer(dev):
    model = MicroLLaMA(vocab_size=32, d_model=32, n_layers=4, device=dev)
    opt = optim.AdamW(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(init_scale=128.0)
    crit = nn.CrossEntropyLoss()

    tokens = torch.randint(0, 32, (4, 16), device=dev)
    targets = torch.randint(0, 32, (4 * 16,), device=dev)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        with torch.amp.autocast(dtype=torch.float16):
            logits = model(tokens)
            # Inside autocast, Linear outputs float16
            assert logits.dtype == torch.float16
            loss = crit(logits.float().reshape(64, 32), targets)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())

    assert all(np.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]

    # Verify GradScaler skips step and backs off scale when Inf gradient is injected
    opt.zero_grad()
    with torch.amp.autocast(dtype=torch.float16):
        loss = crit(model(tokens).float().reshape(64, 32), targets)
    scaler.scale(loss).backward()
    first_param = next(model.parameters())
    w_before = first_param.numpy().copy()
    first_param.grad.fill_(float("inf"))
    old_scale = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    assert scaler.get_scale() == old_scale * 0.5
    np.testing.assert_allclose(first_param.numpy(), w_before)


@pytest.mark.parametrize("dev", DEVICES)
def test_quantized_linear_int4_int8_and_4096_gate(dev):
    torch.manual_seed(42)
    lin = nn.Linear(128, 64, bias=True, device=dev)
    x = torch.randn(8, 128, device=dev)
    ref = lin(x).numpy()

    for bits, tol in [(8, 2e-2), (4, 1.5e-1)]:
        qlin = nn.QuantizedLinear.from_float(lin, bits=bits, group_size=32)
        out_q = qlin(x).numpy()
        np.testing.assert_allclose(out_q, ref, rtol=tol, atol=tol)

    # Check int4 memory reduction ratio on Linear(4096, 4096) and run inference on 2 GB MX450
    lin_4k = nn.Linear(4096, 4096, bias=False, device="cpu")
    fp32_bytes = lin_4k.weight.nbytes
    fp16_bytes = fp32_bytes // 2

    q4_4k = nn.QuantizedLinear.from_float(lin_4k, bits=4, group_size=64).to(dev)
    q4_bytes = q4_4k.weight_nbytes

    assert fp16_bytes / q4_bytes >= 3.5, (
        f"Expected >= 3.5x reduction over fp16, got {fp16_bytes / q4_bytes:.2f}x"
    )
    assert fp32_bytes / q4_bytes >= 7.0

    x_4k = torch.randn(2, 4096, device=dev)
    y_4k = q4_4k(x_4k)
    assert y_4k.shape == (2, 4096)
    assert np.isfinite(y_4k.numpy()).all()


def test_phase4_ease_of_use_ergonomics():
    # 1. Default device & `with torch.device(...)` context manager
    orig_dev = str(torch.get_default_device())
    try:
        with torch.device("cpu"):
            t_cpu = torch.randn(4, 8)
            m_cpu = nn.Linear(8, 4)
            assert str(t_cpu.device) == "cpu"
            assert str(m_cpu.weight.device) == "cpu"

        # 2. Actionable error message on device mismatch
        if torch.vulkan.device_count() > 0:
            t_gpu = torch.randn(4, 8, device="vk:0")
            with pytest.raises(RuntimeError, match="Call tensor.to"):
                _ = m_cpu(t_gpu)

        # 3. One-line whole-model quantization via `torch.quantize(model, bits=4)`
        mlp = nn.Sequential(
            nn.Linear(32, 64, device="cpu"),
            nn.SiLU(),
            nn.Linear(64, 16, device="cpu"),
        )
        q_mlp = torch.quantize(mlp, bits=4, group_size=32)
        assert isinstance(q_mlp[0], nn.QuantizedLinear)
        assert isinstance(q_mlp[2], nn.QuantizedLinear)
        x_test = torch.randn(4, 32, device="cpu")
        np.testing.assert_allclose(q_mlp(x_test).numpy(), mlp(x_test).numpy(), rtol=0.2, atol=0.2)

        # 4. DataLoader + MultiGPU(['all']) zero-boilerplate training loop
        dev_target = "vk:0" if torch.vulkan.device_count() > 0 else "cpu"
        X_data = torch.randn(32, 16, device=dev_target)
        Y_data = torch.randint(0, 4, (32,), device=dev_target)
        dataset = TensorDataset(X_data, Y_data)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        base_net = nn.Sequential(
            nn.Linear(16, 32, device=dev_target),
            nn.ReLU(),
            nn.Linear(32, 4, device=dev_target),
        )
        # Wrap using `.multi_gpu(['all'])` (or `nn.MultiGPU(base_net, device_ids=['all'])`)
        model = base_net.multi_gpu(["all"])
        opt = optim.AdamW(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()

        epoch_losses = []
        for _ in range(3):
            total_loss = 0.0
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(model(bx), by)
                # Notice: NO manual model.reduce_gradients() or model.zero_grad() needed!
                loss.backward()
                opt.step()
                total_loss += loss.item()
            epoch_losses.append(total_loss)

        assert epoch_losses[-1] < epoch_losses[0]
    finally:
        torch.set_default_device(orig_dev)
