#!/usr/bin/env python3
"""
VulTorch Example: Heterogeneous Multi-GPU DataParallel Training
Concurrently trains a neural network across both AMD Radeon RX 7900 XTX (vk:0)
and NVIDIA GeForce RTX 3080 (vk:1) in the exact same process!
"""

import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


def main():
    dev_count = torch.vulkan.device_count()
    print("=" * 70)
    print("🚀 VulTorch Heterogeneous Multi-GPU DataParallel Demo")
    print(f"📦 Total Vulkan Devices Available: {dev_count}")
    for i in range(dev_count):
        print(f"   GPU {i}: {torch.vulkan.get_device_name(i)} [vk:{i}]")
    print("🛡️  Proprietary Stack: None! Zero CUDA, Zero ROCm.")
    print("=" * 70)

    if dev_count < 2:
        print("Warning: Only 1 Vulkan device found. Need 2 devices for multi-GPU demo.")
        devices = ["vk:0"]
    else:
        devices = ["vk:0", "vk:1"]
        print(f"🎯 Distributing mini-batches across: {devices}")

    # Base model on master GPU (vk:0)
    in_features = 256
    hidden_dim = 128
    out_features = 10

    base_model = nn.Sequential(
        nn.Linear(in_features, hidden_dim, device="vk:0"),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, device="vk:0"),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_features, device="vk:0"),
    ).to("vk:0")

    parallel_model = nn.DataParallel(base_model, device_ids=devices)
    optimizer = optim.Adam(base_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    num_batches = 20

    print(f"\n⚡ Starting Multi-GPU Training ({num_batches} steps, batch size {batch_size})...")
    start_time = time.time()

    for step in range(1, num_batches + 1):
        # Synthetic data batch
        x = torch.randn(batch_size, in_features, device="vk:0")
        y = torch.tensor(np.random.randint(0, out_features, size=batch_size), device="vk:0")

        optimizer.zero_grad()

        # Concurrent forward pass across AMD and NVIDIA GPUs!
        preds = parallel_model(x)

        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"  Step {step:2d}/{num_batches} | Loss: {loss.item():.4f}")

    torch.vulkan.synchronize_all()
    elapsed = time.time() - start_time
    samples_per_sec = (batch_size * num_batches) / elapsed

    print("-" * 70)
    print(f"✅ Heterogeneous Multi-GPU training completed in {elapsed:.2f}s")
    print(f"🚀 Throughput: {samples_per_sec:.1f} samples/sec across AMD 7900 XTX + NVIDIA 3080!")
    print("=" * 70)


if __name__ == "__main__":
    main()
