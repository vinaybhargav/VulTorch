#!/usr/bin/env python3
"""
VulTorch Benchmark: GPU vs CPU Matrix Multiplication and Neural Network Inference
"""

import time
import numpy as np
import vultorch as torch
import vultorch.nn as nn

def benchmark_matmul(size=2048, iterations=10):
    print("=" * 70)
    print(f"📊 Matrix Multiplication Benchmark (Size: {size}x{size}, Float32)")
    print("=" * 70)

    # 1. CPU (NumPy)
    a_cpu = np.random.randn(size, size).astype(np.float32)
    b_cpu = np.random.randn(size, size).astype(np.float32)

    t0 = time.time()
    for _ in range(iterations):
        _ = a_cpu @ b_cpu
    cpu_time = (time.time() - t0) / iterations
    cpu_gflops = (2 * size**3 * 1e-9) / cpu_time
    print(f"🖥️  CPU (Host OpenBLAS):        {cpu_time*1000:7.2f} ms | {cpu_gflops:7.1f} GFLOPS")

    # 2. Vulkan GPU 0 (AMD Radeon RX 7900 XTX)
    gpu0_name = torch.vulkan.get_device_name(0)
    a_gpu0 = torch.randn(size, size, device="vk:0")
    b_gpu0 = torch.randn(size, size, device="vk:0")

    # Warmup
    _ = a_gpu0 @ b_gpu0
    torch.vulkan.synchronize(0)

    t0 = time.time()
    for _ in range(iterations):
        _ = a_gpu0 @ b_gpu0
    torch.vulkan.synchronize(0)
    gpu0_time = (time.time() - t0) / iterations
    gpu0_gflops = (2 * size**3 * 1e-9) / gpu0_time
    speedup0 = cpu_time / gpu0_time
    print(f"⚡ GPU 0 ({gpu0_name}): {gpu0_time*1000:7.2f} ms | {gpu0_gflops:7.1f} GFLOPS (Speedup: {speedup0:.1f}x)")

    # 3. Vulkan GPU 1 (NVIDIA RTX 3080 if available)
    if torch.vulkan.device_count() > 1:
        gpu1_name = torch.vulkan.get_device_name(1)
        a_gpu1 = torch.randn(size, size, device="vk:1")
        b_gpu1 = torch.randn(size, size, device="vk:1")

        # Warmup
        _ = a_gpu1 @ b_gpu1
        torch.vulkan.synchronize(1)

        t0 = time.time()
        for _ in range(iterations):
            _ = a_gpu1 @ b_gpu1
        torch.vulkan.synchronize(1)
        gpu1_time = (time.time() - t0) / iterations
        gpu1_gflops = (2 * size**3 * 1e-9) / gpu1_time
        speedup1 = cpu_time / gpu1_time
        print(f"⚡ GPU 1 ({gpu1_name}): {gpu1_time*1000:7.2f} ms | {gpu1_gflops:7.1f} GFLOPS (Speedup: {speedup1:.1f}x)")

    print("=" * 70)

if __name__ == "__main__":
    benchmark_matmul(size=2048, iterations=10)
