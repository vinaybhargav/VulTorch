#!/usr/bin/env python3
"""
Pre-train Universal 10M Parameter Neural Text Codec on English Wikipedia (enwik8)
using Heterogeneous Multi-GPU DataParallel (AMD Radeon RX 7900 XTX + NVIDIA RTX 3080)
via VulTorch Pure Vulkan Compute.
"""

import os
import sys
import time
import hashlib
import numpy as np
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.models.compressor_10m import NeuralCompressor10M

def pretrain_universal_codec(corpus_path: str = "benchmarks/data/enwik8",
                             output_weights: str = "models/universal_10m_q8.bin",
                             num_steps: int = 150,
                             batch_size: int = 256,
                             learning_rate: float = 0.003):
    print("=" * 80)
    print("🌍 VulTorch Universal Neural Codec: Multi-GPU Foundation Pre-Training")
    print("=" * 80)

    # 1. Discover available GPUs
    dev_count = torch.vulkan.device_count()
    print(f"📦 Discovered {dev_count} Vulkan Hardware Devices:")
    devices = []
    for i in range(dev_count):
        name = torch.vulkan.get_device_name(i)
        print(f"   GPU {i}: {name} [vk:{i}]")
        devices.append(f"vk:{i}")

    if len(devices) >= 2:
        use_devices = ["vk:0", "vk:1"]
        print(f"\n🚀 Enabling Heterogeneous Multi-GPU Acceleration:")
        print(f"   Master GPU: {torch.vulkan.get_device_name(0)} (vk:0)")
        print(f"   Worker GPU: {torch.vulkan.get_device_name(1)} (vk:1)")
    else:
        use_devices = ["vk:0"]
        print(f"\n🚀 Single GPU Mode: {devices[0]}")

    # 2. Load Wikipedia Corpus (enwik8)
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}. Run examples/download_benchmark_data.py first.")

    file_size = os.path.getsize(corpus_path)
    print(f"\n📚 Training Corpus: {corpus_path} ({file_size:,} bytes, {file_size/(1024*1024):.1f} MB)")

    # Read training stream (stream slices for active memory efficiency)
    with open(corpus_path, "rb") as f:
        # Load first 1,000,000 bytes (1 MB slice) of diverse Wikipedia articles for fast high-density pre-training
        TRAIN_BYTES = min(file_size, 1_000_000)
        train_data = f.read(TRAIN_BYTES)

    print(f"📄 Active Training Window: {len(train_data):,} bytes")

    # 3. Model Setup
    K = 32 # 32-byte receptive field
    padded = bytes([0] * K) + train_data
    N = len(train_data)

    print(f"\n🔧 Initializing 10M Parameter Neural Compressor...")
    base_model = NeuralCompressor10M(context_len=K, embed_dim=512, hidden_dim=1024, device="vk:0")
    total_params = sum(p.numpy().size for p in base_model.parameters())
    print(f"   Total Model Parameters: {total_params:,}")

    # Wrap in Multi-GPU DataParallel
    if len(use_devices) > 1:
        parallel_model = nn.DataParallel(base_model, device_ids=use_devices)
    else:
        parallel_model = base_model

    optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Pre-generate sliding window sample index pool
    print(f"⚡ Launching Multi-GPU Training ({num_steps} iterations, batch size {batch_size})...")
    start_time = time.time()

    for step in range(1, num_steps + 1):
        # Random sample batch from the 1MB Wikipedia stream
        start_indices = np.random.randint(0, N, size=batch_size)
        batch_x = []
        batch_y = []
        for idx in start_indices:
            batch_x.append(list(padded[idx:idx + K]))
            batch_y.append(padded[idx + K])

        bx = torch.tensor(np.array(batch_x, dtype=np.int64), device="vk:0")
        by = torch.tensor(np.array(batch_y, dtype=np.int64), device="vk:0")

        parallel_model.zero_grad()

        # Concurrent forward pass across AMD and NVIDIA GPUs!
        preds = parallel_model(bx)

        loss = criterion(preds, by)
        loss.backward()
        if hasattr(parallel_model, "reduce_gradients"):
            parallel_model.reduce_gradients()
        optimizer.step()

        if step % 25 == 0 or step == 1 or step == num_steps:
            loss_val = float(loss.numpy()[0])
            bpc = loss_val / np.log(2)
            elapsed = time.time() - start_time
            throughput = (step * batch_size) / elapsed
            print(f"   Step {step:3d}/{num_steps:3d} | Loss: {loss_val:.4f} nats | Cross-Entropy: {bpc:.2f} bits/byte | {throughput:.1f} samples/sec", flush=True)

    torch.vulkan.synchronize_all()
    total_time = time.time() - start_time
    print(f"\n✅ Foundation Pre-Training Completed in {total_time:.2f}s ({num_steps * batch_size / total_time:.1f} tokens/sec)!")

    # 4. Quantize and Export to Standalone Binary
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    print(f"\n💎 Quantizing and Exporting Foundation Weights to {output_weights}...")
    base_model.export_q8_binary(output_weights)
    print("🎉 Universal Foundation Codec Model is ready for standalone C++ deployment!")

if __name__ == "__main__":
    pretrain_universal_codec()
