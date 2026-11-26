import time
import pytest
import numpy as np
import vultorch
import vultorch.nn as nn
import vultorch.optim as optim


def _vk_devices():
    if not vultorch.vulkan.is_available():
        return []
    return [f"vk:{i}" for i in range(vultorch.vulkan.device_count())]


@pytest.mark.parametrize("device", _vk_devices())
def test_vulkan_graph_capture_replay_parity_and_speedup(device):
    def make_mlp():
        vultorch.manual_seed(7)
        return nn.Sequential(
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 10, device=device),
        )

    vultorch.manual_seed(99)
    x_eager = vultorch.randn(32, 256, device=device)
    y_eager = vultorch.randint(0, 10, (32,), device=device, dtype=vultorch.int64)
    x_graph = x_eager.clone()
    y_graph = y_eager.clone()

    m_eager = make_mlp()
    m_graph = make_mlp()
    opt_eager = optim.SGD(m_eager.parameters(), lr=0.01)
    opt_graph = optim.SGD(m_graph.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    def step_eager():
        opt_eager.zero_grad()
        loss = crit(m_eager(x_eager), y_eager)
        loss.backward()
        opt_eager.step()
        return loss

    # Warmup eager
    for _ in range(3):
        step_eager()
    vultorch.vulkan.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(15):
        step_eager()
    vultorch.vulkan.synchronize(device)
    eager_dt = time.perf_counter() - t0

    # Match 3 warmup steps on graph model before capture
    def step_graph():
        opt_graph.zero_grad()
        loss = crit(m_graph(x_graph), y_graph)
        loss.backward()
        opt_graph.step()
        return loss

    for _ in range(3):
        step_graph()
    vultorch.vulkan.synchronize(device)

    g = vultorch.VulkanGraph(device)
    with g:
        loss_g = step_graph()
    assert g.is_captured
    assert g.num_dispatches > 10

    # 15 replays to match 15 eager steps after warmup
    for _ in range(15):
        g.replay()
    vultorch.vulkan.synchronize(device)

    for pe, pg in zip(m_eager.parameters(), m_graph.parameters()):
        np.testing.assert_allclose(pg.numpy(), pe.numpy(), rtol=1e-5, atol=1e-5)

    # Measure replay throughput on small-batch 6-layer MLP (overhead-bound)
    t1 = time.perf_counter()
    for _ in range(15):
        g.replay()
    vultorch.vulkan.synchronize(device)
    replay_dt = time.perf_counter() - t1

    assert replay_dt < eager_dt, f"Expected replay ({replay_dt:.4f}s) < eager ({eager_dt:.4f}s)"
    g.release()


@pytest.mark.parametrize("device", _vk_devices())
def test_vultorch_compile_module_and_dynamic_shapes(device):
    vultorch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64, device=device),
        nn.GELU(),
        nn.Linear(64, 16, device=device),
    )

    x1 = vultorch.randn(8, 32, device=device)
    with vultorch.no_grad():
        ref1 = model(x1).numpy()

    compiled = vultorch.compile(model)
    with vultorch.no_grad():
        out1_a = compiled(x1).numpy()
        out1_b = compiled(vultorch.randn(8, 32, device=device)).numpy()
        assert out1_a.shape == (8, 16)
        assert out1_b.shape == (8, 16)
        np.testing.assert_allclose(out1_a, ref1, rtol=1e-4, atol=1e-4)

        # Dynamic shape change triggers transparent re-capture
        x2 = vultorch.randn(16, 32, device=device)
        out2 = compiled(x2).numpy()
        assert out2.shape == (16, 16)
