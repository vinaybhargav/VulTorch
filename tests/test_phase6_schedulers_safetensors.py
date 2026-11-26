import pytest
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.optim import lr_scheduler
import tempfile
import os

def test_lr_schedulers():
    model = nn.Linear(10, 10)
    opt = optim.SGD(model.parameters(), lr=0.1)

    # StepLR
    sched = lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    # CosineAnnealingLR
    opt2 = optim.Adam(model.parameters(), lr=0.1)
    cos_sched = lr_scheduler.CosineAnnealingLR(opt2, T_max=10, eta_min=0.01)
    assert opt2.param_groups[0]["lr"] == pytest.approx(0.1)
    cos_sched.step(5) # Halfway
    assert opt2.param_groups[0]["lr"] == pytest.approx(0.055, abs=1e-3)
    cos_sched.step(10) # Min
    assert opt2.param_groups[0]["lr"] == pytest.approx(0.01, abs=1e-3)

def test_clip_grad_norm():
    p1 = torch.tensor([1.0, 2.0], requires_grad=True, device='vk:0')
    p2 = torch.tensor([3.0, 4.0], requires_grad=True, device='vk:0')
    loss = (p1 * p1 + p2 * p2).sum()
    loss.backward()

    # Original norm = sqrt(4 + 16 + 36 + 64) = sqrt(120) ~= 10.954
    total_norm = nn.utils.clip_grad_norm_([p1, p2], max_norm=2.0)
    assert total_norm.item() == pytest.approx(10.95445, rel=1e-3)

    # Post-clipping norm should equal 2.0
    new_norm = (p1.grad * p1.grad).sum() + (p2.grad * p2.grad).sum()
    assert (new_norm.sqrt()).item() == pytest.approx(2.0, rel=1e-3)

def test_safetensors_roundtrip():
    tensors = {
        "weight": torch.randn(8, 16, device='vk:0'),
        "bias": torch.randn(8, device='vk:0'),
        "count": torch.tensor([1, 2, 3, 4], dtype=torch.int64, device='vk:0'),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.safetensors")
        torch.save_safetensors(tensors, path, metadata={"format": "pt"})
        assert os.path.exists(path)

        loaded = torch.load_safetensors(path, device='vk:1')
        assert set(loaded.keys()) == set(tensors.keys())
        assert torch.allclose(loaded["weight"].cpu(), tensors["weight"].cpu())
        assert torch.allclose(loaded["bias"].cpu(), tensors["bias"].cpu())
        assert (loaded["count"].cpu().numpy() == tensors["count"].cpu().numpy()).all()
