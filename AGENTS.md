# AGENTS.md — Guidance for AI Coding Agents Working with VulTorch

This repository implements **VulTorch**, a pure Khronos Vulkan deep learning framework built on custom Slang compute shaders.

When generating code or editing this repository, adhere strictly to the following rules:

---

## 1. Absolute Constraints
- **NO CUDA / ROCm:** Never emit `torch.cuda`, `.cuda()`, `CUDA_VISIBLE_DEVICES`, or references to cuDNN/NCCL. Use `"vk:0"`, `"vk:1"`, or `torch.vulkan.*`.
- **Zero `.numpy()` in Compute Kernels:** Never convert tensors to NumPy (`.numpy()`) inside `nn` modules, loss functions, autograd functions, or optimizer steps during forward or backward execution. All tensor math must execute on GPU buffers via Slang.
- **100% Khronos Slang Shaders:** All GPU compute operations must be authored in Slang (`shaders/*.slang`) and compiled to SPIR-V via `slangc`. Never hardcode vendor-specific assembly or precompiled binary blobs.
- **Multi-Vendor Parity:** All tests and features must function identically across `cpu`, `vk:0` (e.g. Intel Iris Xe), and `vk:1` (e.g. NVIDIA MX450 / discrete GPUs).

---

## 2. Formatting & Style Conventions
- **C++ Code:** Follow **Allman style** (braces on new lines). Run `clang-format` using [.clang-format](file:///.clang-format).
- **Python Code:** Follow standard 4-space indentation and multiline delimiter rules. Run `ruff format` using configuration in [pyproject.toml](file:///pyproject.toml).
- **Quality Gate:** Everything must pass `./scripts/check.sh` with **zero warnings** (`-Werror` enforced in C++ and Python).

---

## 3. Preferred Architectural Patterns
- **Device Instantiation:**
  ```python
  import vultorch as torch
  device = "vk:0" if torch.vulkan.is_available() else "cpu"
  x = torch.randn(64, 128, device=device)
  ```
- **Heterogeneous Multi-GPU:**
  ```python
  import vultorch.nn as nn
  dp_model = nn.DataParallel(model, device_ids=["all"], balance="auto")
  ```
- **Graph Replay:**
  ```python
  compiled_fn = torch.compile(my_model)
  ```
- **Safetensors Weight Serialization:**
  ```python
  torch.save_safetensors(model.state_dict(), "weights.safetensors")
  loaded = torch.load_safetensors("weights.safetensors", device=device)
  ```
