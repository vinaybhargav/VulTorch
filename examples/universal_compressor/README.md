# Universal Neural Text Compressor Example

This example project showcases **VulTorch**'s capabilities for deep autoregressive probability modeling, multi-GPU training, and lossless range coding.

## Overview
1. **Architecture:** A deep 1M-parameter embedding + dense autoregressive MLP predicting byte probability distributions $P(b_t \mid b_{<t})$.
2. **Multi-GPU Training:** Trains on heterogeneous Vulkan GPUs (e.g. Intel Iris Xe + NVIDIA RTX) concurrently via `vultorch.nn.DataParallel`.
3. **Lossless Range Coding:** Combines GPU neural likelihood estimation with 32-bit integer arithmetic range coding for 100% bit-exact lossless roundtrips.

## Files
- `sample_book.txt`: Standard text corpus for training and compression testing.
- `test_lossless_roundtrip.py`: Verification script confirming byte-for-byte exact matches and SHA-256 parity between original and decompressed outputs.

## Running Verification

```bash
# Verify lossless compression roundtrip on sample_book.txt
python3 -m examples.universal_compressor.test_lossless_roundtrip
```
