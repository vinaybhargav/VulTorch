# VulTorch 🚀

> **Open-Source, Vendor-Agnostic Deep Learning Powered by Pure Vulkan.**
> No CUDA. No ROCm. True multi-vendor GPU acceleration.

VulTorch is a high-performance deep learning framework designed to bring full hardware acceleration to GPUs across vendors (AMD, NVIDIA, Intel, Apple/MoltenVK) using the open **Vulkan Compute & SPIR-V** standard.

---

## ⚡ Features
- **Zero Proprietary Driver Dependencies**: No CUDA toolkit or ROCm runtimes required. Runs directly on standard Vulkan 1.3/1.4 drivers.
- **Heterogeneous Multi-GPU**: Execute computations simultaneously across AMD (e.g. Radeon RX 7900 XTX) and NVIDIA (e.g. RTX 3080) GPUs in the same Python process.
- **`vultorch.nn.DataParallel`**: Distribute mini-batch workloads concurrently across mixed-vendor GPUs.
- **Tiled & Vectorized SPIR-V Shaders**:
  - 128-bit memory coalesced operations (`vec4`).
  - Shared-memory tiled GEMM with register blocking for matrix multiplication.
  - Subgroup tree reductions.
- **PyTorch-Compatible API**:
  - `vultorch.Tensor`
  - `vultorch.autograd` (Reverse-mode dynamic autodiff DAG)
  - `vultorch.nn` (`Linear`, `Conv2d`, `MaxPool2d`, `LayerNorm`, `CrossEntropyLoss`, etc.)
  - `vultorch.optim` (`SGD`, `Adam`, `AdamW`)
  - `vultorch.utils.data` (`Dataset`, `DataLoader`)

---

## 💻 Quickstart

```python
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim

# Check detected Vulkan GPUs
print("Vulkan Devices:", torch.vulkan.device_count())
for i in range(torch.vulkan.device_count()):
    print(f"[{i}] {torch.vulkan.get_device_name(i)}")

# Target AMD 7900 XTX ('vk:0') or NVIDIA 3080 ('vk:1')
device = torch.device('vk:0')

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

x = torch.randn(32, 128, device=device)
y = torch.tensor([1] * 32, device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Forward + Backward + Step
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

### Heterogeneous Multi-GPU Training
```python
# Train across both AMD and NVIDIA GPUs concurrently!
parallel_model = nn.DataParallel(model, device_ids=['vk:0', 'vk:1'])
output = parallel_model(large_batch_x)
```
