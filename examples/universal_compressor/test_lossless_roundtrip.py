#!/usr/bin/env python3
"""
Byte-by-Byte Lossless Roundtrip Verification for VulTorch Neural Compressor
"""

import time
import hashlib
import struct
import numpy as np
import vultorch as torch
import vultorch.nn as nn
from examples.neural_text_compressor import (
    NeuralCompressor1M,
    train_codec_model,
    compress_file,
    decompress_file,
)


def verify_lossless_roundtrip(data: bytes, device: str = "vk:0", devices=None):
    print("=" * 70)
    print("🔍 VulTorch Neural Compressor: Byte-for-Byte Lossless Verification")
    print("=" * 70)
    orig_len = len(data)
    orig_sha256 = hashlib.sha256(data).hexdigest()
    print(f"📄 Original Data Size:   {orig_len:,} bytes")
    print(f"🔑 Original SHA-256:     {orig_sha256}")

    if devices is None:
        count = torch.vulkan.device_count()
        devices = [f"vk:{i}" for i in range(count)] if count > 1 else [device]

    # 1. Train model on GPU(s) (Heterogeneous Multi-GPU DataParallel)
    dev_str = f"Multi-GPU ({', '.join(devices)})" if len(devices) > 1 else f"Single GPU ({device})"
    print(f"\n1️⃣  Training 1M Parameter Model on {dev_str}...")
    model = train_codec_model(
        data, epochs=10, batch_size=256, lr=0.005, device=device, devices=devices
    )

    # 2. Compress
    print(f"\n2️⃣  Compressing on GPU ({device})...")
    t0 = time.time()
    compressed = compress_file(model, data, device=device)
    comp_time = time.time() - t0

    comp_len = len(compressed)
    ratio = (comp_len / orig_len) * 100
    bpc = (comp_len * 8) / orig_len
    print(f"   Compressed Size:      {comp_len:,} bytes ({ratio:.2f}%, {bpc:.2f} bits/byte)")
    print(f"   Compression Time:     {comp_time * 1000:.1f} ms")

    # 3. Decompress (using exact same device for bit-exact entropy probability matching)
    print(f"\n3️⃣  Decompressing back to original bytes on GPU ({device})...")
    t0 = time.time()
    restored = decompress_file(model, compressed, device=device)
    decomp_time = time.time() - t0

    restored_len = len(restored)
    restored_sha256 = hashlib.sha256(restored).hexdigest()

    print(f"   Decompressed Size:    {restored_len:,} bytes")
    print(f"   Decompressed SHA-256: {restored_sha256}")
    print(f"   Decompression Time:   {decomp_time:.2f}s")

    # 4. Rigorous Byte-by-Byte Verification
    print("\n" + "=" * 70)
    print("🔬 VERIFICATION CHECKS:")
    print("=" * 70)

    # Check 1: Length
    assert orig_len == restored_len, f"❌ Length mismatch! {orig_len} != {restored_len}"
    print(f"  [PASS] Byte Length Exact Match: {orig_len:,} == {restored_len:,}")

    # Check 2: Cryptographic Hash
    assert orig_sha256 == restored_sha256, (
        f"❌ Hash mismatch!\nOrig: {orig_sha256}\nRest: {restored_sha256}"
    )
    print(f"  [PASS] SHA-256 Exact Match:    {orig_sha256}")

    # Check 3: Every individual byte index
    mismatches = []
    for idx in range(orig_len):
        if data[idx] != restored[idx]:
            mismatches.append((idx, data[idx], restored[idx]))
            if len(mismatches) > 5:
                break

    assert len(mismatches) == 0, f"❌ Byte differences found at indices: {mismatches}"
    print(f"  [PASS] Byte-for-Byte Check:     All {orig_len:,} bytes are 100% IDENTICAL!")

    # Check 4: Sample text print
    print("-" * 70)
    print("📝 Decoded Text Sample (First 200 characters):")
    print(restored[:200].decode("utf-8", errors="replace"))
    print("=" * 70)
    print("🎉 100% BIT-EXACT LOSSLESS VERIFICATION CONFIRMED!")
    print("=" * 70)


if __name__ == "__main__":
    import os
    import sys

    # Check if a custom file was passed or default to sample_book.txt in the same directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sample_book_path = os.path.join(cur_dir, "sample_book.txt")

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], "rb") as f:
            data = f.read()
    elif os.path.isfile(sample_book_path):
        with open(sample_book_path, "rb") as f:
            # Test on first 1024 bytes of sample_book for fast verification
            data = f.read()[:1024]
    else:
        sample_text = (
            """VulTorch: A High-Performance Pure Vulkan Deep Learning Library.
Eliminating NVIDIA CUDA lock-in by running full tensor compute, reverse-mode autodiff,
and deep neural networks across AMD Radeon, Intel, and NVIDIA GPUs using standard Vulkan.
Compressing and decompressing text losslessly with deep neural language models!
The quick brown fox jumps over the lazy dog. 0123456789!
"""
            * 3
        )
        data = sample_text.encode("utf-8")

    verify_lossless_roundtrip(data)
