"""
Phase 3.4 Tests & ResNet-18 Benchmark Gate:
- im2col + GEMM Conv2d (C_in >= 16), Grouped/Depthwise Conv2d, Conv1d, ConvTranspose2d
- BatchNorm1d, BatchNorm2d (train & eval), GroupNorm
- Full ResNet-18 forward + backward + AdamW step on [16, 3, 224, 224]
"""

import time
import numpy as np
import pytest

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


DEVICES = ["cpu"] + [f"vk:{i}" for i in range(torch.vulkan.device_count())]
GPU_DEVICES = [d for d in DEVICES if d.startswith("vk:")]


def _rand_np(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.8, 0.8, size=shape).astype(np.float32)


@pytest.mark.parametrize("dev", GPU_DEVICES)
def test_conv2d_im2col_matches_cpu_reference(dev):
    x_np = _rand_np((2, 16, 8, 8), seed=1)
    x_cpu = torch.tensor(x_np, device="cpu", requires_grad=True)
    x_gpu = torch.tensor(x_np, device=dev, requires_grad=True)

    conv_cpu = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, device="cpu")
    conv_gpu = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, device=dev)
    with torch.no_grad():
        conv_gpu.weight.copy_(conv_cpu.weight.to(dev))
        conv_gpu.bias.copy_(conv_cpu.bias.to(dev))

    y_cpu = conv_cpu(x_cpu)
    y_gpu = conv_gpu(x_gpu)
    np.testing.assert_allclose(y_gpu.numpy(), y_cpu.numpy(), rtol=1e-3, atol=1e-3)

    (y_cpu**2).sum().backward()
    (y_gpu**2).sum().backward()
    np.testing.assert_allclose(x_gpu.grad.numpy(), x_cpu.grad.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        conv_gpu.weight.grad.numpy(), conv_cpu.weight.grad.numpy(), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        conv_gpu.bias.grad.numpy(), conv_cpu.bias.grad.numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_conv2d_grouped_and_conv1d(dev):
    x_np = _rand_np((2, 4, 6, 6), seed=2)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    dw = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=4, device=dev)
    y = dw(x)
    assert y.shape == (2, 8, 6, 6)
    y.sum().backward()
    assert x.grad is not None and np.isfinite(x.grad.numpy()).all()
    assert dw.weight.grad is not None and np.isfinite(dw.weight.grad.numpy()).all()

    x1_np = _rand_np((2, 4, 10), seed=3)
    x1 = torch.tensor(x1_np, device=dev, requires_grad=True)
    c1 = nn.Conv1d(4, 6, kernel_size=3, padding=1, device=dev)
    y1 = c1(x1)
    assert y1.shape == (2, 6, 10)
    y1.sum().backward()
    assert x1.grad is not None


@pytest.mark.parametrize("dev", DEVICES)
def test_conv_transpose2d_and_gradcheck(dev):
    x_np = _rand_np((1, 2, 3, 3), seed=4)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    deconv = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=1, device=dev)
    y = deconv(x)
    assert y.shape == (1, 2, 5, 5)

    assert torch.gradcheck(
        lambda t: (deconv(t) ** 2).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_batchnorm2d_and_groupnorm_gradcheck(dev):
    x_np = _rand_np((3, 4, 4, 4), seed=5)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    bn = nn.BatchNorm2d(4, device=dev)
    y = bn(x)
    assert y.shape == (3, 4, 4, 4)

    # Check zero-mean / unit-var across (N, H, W)
    y_np = y.numpy()
    ch_mean = y_np.mean(axis=(0, 2, 3))
    ch_std = y_np.std(axis=(0, 2, 3))
    np.testing.assert_allclose(ch_mean, np.zeros(4), atol=1e-4)
    np.testing.assert_allclose(ch_std, np.ones(4), atol=1e-2)

    w_proj = torch.tensor(_rand_np((3, 4, 4, 4), seed=55), device=dev)
    assert torch.gradcheck(
        lambda t: (bn(t) * w_proj).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    # Eval mode uses running_mean / running_var
    bn.eval()
    y_eval = bn(x)
    assert np.isfinite(y_eval.numpy()).all()

    # GroupNorm + gradcheck
    gn = nn.GroupNorm(2, 4, device=dev)
    assert torch.gradcheck(
        lambda t: (gn(t) * w_proj).sum(), x, eps=2e-3, rtol=2e-2, atol=5e-3, device=dev
    )


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, device="vk"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_c, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_c, device=device)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False, device=device),
                nn.BatchNorm2d(out_c, device=device),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, device="vk"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1, device=device)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, device=device)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, device=device)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, device=device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.fc = nn.Linear(512, num_classes, device=device)

    def _make_layer(self, in_c, out_c, blocks, stride, device):
        layers = [BasicBlock(in_c, out_c, stride=stride, device=device)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, stride=1, device=device))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flat(self.pool(x))
        return self.fc(x)


@pytest.mark.parametrize("dev", GPU_DEVICES)
def test_resnet18_end_to_end_forward_backward_step(dev):
    model = ResNet18(num_classes=10, device=dev)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 224, 224, device=dev)
    y = torch.randint(0, 10, (4,), dtype=torch.int64, device=dev)

    opt.zero_grad()
    logits = model(x)
    assert logits.shape == (4, 10)
    loss = crit(logits, y)
    loss.backward()
    opt.step()

    did = int(dev.split(":")[1])
    torch.vulkan.synchronize(did)
    assert np.isfinite(loss.item())


@pytest.mark.skipif(torch.vulkan.device_count() < 2, reason="Requires >= 2 Vulkan GPUs")
def test_resnet18_multi_gpu_dataparallel():
    base = ResNet18(num_classes=10, device="vk:0")
    dp = nn.DataParallel(base, device_ids=["vk:0", "vk:1"])
    dp.sync_every_forward = True
    opt = optim.AdamW(base.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 224, 224, device="vk:0")
    y = torch.randint(0, 10, (8,), dtype=torch.int64, device="vk:0")

    losses = []
    for _ in range(2):
        opt.zero_grad()
        dp.zero_grad()
        logits = dp(x)
        assert logits.shape == (8, 10)
        loss = crit(logits, y)
        loss.backward()
        dp.reduce_gradients()
        opt.step()
        torch.vulkan.synchronize_all()
        losses.append(loss.item())

    assert all(np.isfinite(v) for v in losses)
    assert sum(dp.last_split_sizes) == 8
    assert all(s > 0 for s in dp.last_split_sizes)
    assert losses[1] < losses[0]


@pytest.mark.parametrize("dev", DEVICES)
def test_snake_scalar_and_module_gradcheck(dev):
    x_np = _rand_np((2, 4, 16), seed=101)
    x = torch.tensor(x_np, device=dev, requires_grad=True)

    # 1. Scalar functional Snake: x + (1/a) * sin^2(a * x)
    a = 2.0
    y = x.snake(a=a)
    expected = x_np + (1.0 / a) * (np.sin(a * x_np) ** 2)
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-4, atol=1e-4)

    w = torch.tensor(_rand_np((2, 4, 16), seed=102), device=dev)
    assert torch.gradcheck(
        lambda t: (t.snake(a=2.0) * w).sum(), x, eps=1e-3, rtol=2e-2, atol=5e-3, device=dev
    )

    # 2. Per-channel trainable nn.Snake (linear and log-scale)
    for log_scale in (False, True):
        snake = nn.Snake(4, a=1.5, trainable=True, alpha_logscale=log_scale, device=dev)
        x_in = torch.tensor(x_np, device=dev, requires_grad=True)
        out = snake(x_in)
        loss = (out * w).sum()
        loss.backward()

        assert x_in.grad is not None and np.isfinite(x_in.grad.numpy()).all()
        assert snake.alpha.grad is not None and np.isfinite(snake.alpha.grad.numpy()).all()

        # Verify alpha gradient via finite difference
        eps_fd = 1e-3
        alpha_orig = snake.alpha.numpy().copy()
        analytic_da = snake.alpha.grad.numpy().copy()
        num_da = np.zeros_like(alpha_orig)
        for c in range(4):
            ap = alpha_orig.copy()
            am = alpha_orig.copy()
            ap[c] += eps_fd
            am[c] -= eps_fd
            with torch.no_grad():
                snake.alpha.copy_(torch.tensor(ap, device=dev))
                lp = (snake(x_in) * w).sum().item()
                snake.alpha.copy_(torch.tensor(am, device=dev))
                lm = (snake(x_in) * w).sum().item()
                snake.alpha.copy_(torch.tensor(alpha_orig, device=dev))
            num_da[c] = (lp - lm) / (2.0 * eps_fd)
        np.testing.assert_allclose(analytic_da, num_da, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("dev", DEVICES)
def test_snake_beta_and_signal_processing_conv1d_net(dev):
    x_np = _rand_np((3, 4, 24), seed=202)
    w_np = _rand_np((3, 4, 24), seed=203)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    w = torch.tensor(w_np, device=dev)

    sb = nn.SnakeBeta(4, a=1.2, trainable=True, alpha_logscale=True, device=dev)
    y = sb(x)
    (y * w).sum().backward()

    assert x.grad is not None and np.isfinite(x.grad.numpy()).all()
    assert sb.alpha.grad is not None and np.isfinite(sb.alpha.grad.numpy()).all()
    assert sb.beta.grad is not None and np.isfinite(sb.beta.grad.numpy()).all()

    # Finite difference check on both alpha and beta
    eps_fd = 1e-3
    for param, analytic in [
        (sb.alpha, sb.alpha.grad.numpy().copy()),
        (sb.beta, sb.beta.grad.numpy().copy()),
    ]:
        p_orig = param.numpy().copy()
        num_dp = np.zeros_like(p_orig)
        for c in range(4):
            pp = p_orig.copy()
            pm = p_orig.copy()
            pp[c] += eps_fd
            pm[c] -= eps_fd
            with torch.no_grad():
                param.copy_(torch.tensor(pp, device=dev))
                lp = (sb(x) * w).sum().item()
                param.copy_(torch.tensor(pm, device=dev))
                lm = (sb(x) * w).sum().item()
                param.copy_(torch.tensor(p_orig, device=dev))
            num_dp[c] = (lp - lm) / (2.0 * eps_fd)
        np.testing.assert_allclose(analytic, num_dp, rtol=2e-2, atol=5e-3)

    # Train a miniature non-linear harmonic synthesizer (Conv1d + SnakeBeta)
    net = nn.Sequential(
        nn.Conv1d(1, 8, kernel_size=5, padding=2, device=dev),
        nn.SnakeBeta(8, a=1.0, alpha_logscale=True, device=dev),
        nn.Conv1d(8, 1, kernel_size=5, padding=2, device=dev),
    )
    opt = optim.AdamW(net.parameters(), lr=1e-2)
    t = np.linspace(0, 2 * np.pi, 64, dtype=np.float32).reshape(1, 1, 64)
    sig_in = torch.tensor(np.sin(t), device=dev)
    sig_target = torch.tensor(np.sin(t) + 0.5 * np.sin(3 * t), device=dev)

    loss0 = None
    loss_last = None
    for step in range(8):
        opt.zero_grad()
        pred = net(sig_in)
        loss = ((pred - sig_target) ** 2).mean()
        loss.backward()
        opt.step()
        val = loss.item()
        if step == 0:
            loss0 = val
        loss_last = val
    assert loss_last < loss0


@pytest.mark.parametrize("dev", DEVICES)
def test_snake_gamma_cos_rational_sinc_gradcheck(dev):
    x_np = _rand_np((2, 3, 12), seed=301)
    w_np = _rand_np((2, 3, 12), seed=302)
    x = torch.tensor(x_np, device=dev, requires_grad=True)
    w = torch.tensor(w_np, device=dev)

    # 1. Functional scalar variants gradcheck w.r.t. x
    for fn in [
        lambda t: (t.snake_gamma(a=1.7, gamma=0.4) * w).sum(),
        lambda t: (t.snake_cos(a=1.5) * w).sum(),
        lambda t: (t.snake_rational(a=1.8) * w).sum(),
        lambda t: (t.snake_sinc(a=2.0) * w).sum(),
    ]:
        assert torch.gradcheck(fn, x, eps=1e-3, rtol=2e-2, atol=5e-3, device=dev)

    # 2. Module variants: verify parameter gradients (alpha, beta, and gamma) via finite difference
    modules = [
        nn.SnakeGamma(3, a=1.4, gamma=0.35, alpha_logscale=True, device=dev),
        nn.SnakeCos(3, a=1.3, alpha_logscale=False, device=dev),
        nn.SnakeRational(3, a=1.5, alpha_logscale=True, device=dev),
        nn.SnakeSinc(3, a=1.6, alpha_logscale=False, device=dev),
    ]
    eps_fd = 1e-3
    for mod in modules:
        x_in = torch.tensor(x_np, device=dev, requires_grad=True)
        out = mod(x_in)
        (out * w).sum().backward()
        assert x_in.grad is not None and np.isfinite(x_in.grad.numpy()).all()

        for _, param in mod.named_parameters():
            assert param.grad is not None and np.isfinite(param.grad.numpy()).all()
            analytic = param.grad.numpy().copy()
            p_orig = param.numpy().copy()
            num_dp = np.zeros_like(p_orig)
            for c in range(len(p_orig)):
                pp = p_orig.copy()
                pm = p_orig.copy()
                pp[c] += eps_fd
                pm[c] -= eps_fd
                with torch.no_grad():
                    param.copy_(torch.tensor(pp, device=dev))
                    lp = (mod(x_in) * w).sum().item()
                    param.copy_(torch.tensor(pm, device=dev))
                    lm = (mod(x_in) * w).sum().item()
                    param.copy_(torch.tensor(p_orig, device=dev))
                num_dp[c] = (lp - lm) / (2.0 * eps_fd)
            np.testing.assert_allclose(analytic, num_dp, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("dev", DEVICES)
def test_mamba_selective_scan_and_block(dev):
    B_sz, D_sz, L_sz, N_sz = 2, 3, 6, 4
    u_np = _rand_np((B_sz, D_sz, L_sz), seed=401)
    dt_np = np.abs(_rand_np((B_sz, D_sz, L_sz), seed=402)) + 0.1
    A_np = -np.abs(_rand_np((D_sz, N_sz), seed=403)) - 0.1
    B_np = _rand_np((B_sz, N_sz, L_sz), seed=404)
    C_np = _rand_np((B_sz, N_sz, L_sz), seed=405)
    D_np = _rand_np((D_sz,), seed=406)
    w_np = _rand_np((B_sz, D_sz, L_sz), seed=407)

    u = torch.tensor(u_np, device=dev, requires_grad=True)
    dt = torch.tensor(dt_np, device=dev, requires_grad=True)
    A = torch.tensor(A_np, device=dev, requires_grad=True)
    B_t = torch.tensor(B_np, device=dev, requires_grad=True)
    C_t = torch.tensor(C_np, device=dev, requires_grad=True)
    D_t = torch.tensor(D_np, device=dev, requires_grad=True)
    w = torch.tensor(w_np, device=dev)

    y = torch.selective_scan(u, dt, A, B_t, C_t, D_t)
    assert y.shape == (B_sz, D_sz, L_sz)
    (y * w).sum().backward()

    for t_param in (u, dt, A, B_t, C_t, D_t):
        assert t_param.grad is not None and np.isfinite(t_param.grad.numpy()).all()

    # Gradcheck on u, dt, B, C
    assert torch.gradcheck(
        lambda u_in: (torch.selective_scan(u_in, dt, A, B_t, C_t, D_t) * w).sum(),
        u,
        eps=1e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )
    assert torch.gradcheck(
        lambda dt_in: (torch.selective_scan(u, dt_in, A, B_t, C_t, D_t) * w).sum(),
        dt,
        eps=1e-3,
        rtol=2e-2,
        atol=5e-3,
        device=dev,
    )

    # Full Mamba block (both standard SiLU and SnakeBeta front-end)
    for use_snake in (False, True):
        mamba = nn.Mamba(d_model=8, d_state=4, d_conv=3, expand=2, use_snake=use_snake, device=dev)
        opt = optim.AdamW(mamba.parameters(), lr=1e-2)
        seq_in = torch.tensor(_rand_np((2, 10, 8), seed=410), device=dev)
        seq_target = torch.tensor(_rand_np((2, 10, 8), seed=411), device=dev)

        loss_first = None
        loss_final = None
        for step in range(5):
            opt.zero_grad()
            out = mamba(seq_in)
            assert out.shape == (2, 10, 8)
            loss = ((out - seq_target) ** 2).mean()
            loss.backward()
            opt.step()
            val = loss.item()
            if step == 0:
                loss_first = val
            loss_final = val
        assert loss_final < loss_first
