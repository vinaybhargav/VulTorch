# Universal Neural Text Compressor Example Project

A complete, self-contained neural text compression project showcasing **VulTorch**'s capabilities for deep autoregressive probability modeling, multi-GPU training, and lossless arithmetic range coding.

## Key Features Showcased
1. **Deep Autoregressive Architecture:**
   - 1M-parameter embedding + dense autoregressive MLP with residual connections predicting byte probability distributions $P(b_t \mid b_{<t})$.
2. **Heterogeneous Multi-GPU Training:**
   - Seamlessly balances training batches across mixed-vendor GPUs (e.g. Intel Iris Xe + NVIDIA RTX) via `vultorch.nn.DataParallel`.
3. **Lossless Arithmetic Range Coding:**
   - Custom 32-bit integer arithmetic range encoder & decoder driven by GPU neural likelihood estimates, delivering 100% bit-exact lossless roundtrips.

## Files
- `neural_text_compressor.py`: Core model architecture, arithmetic range coder, and train/compress/decompress functions.
- `compress_cli.py`: Production CLI tool to compress (`-c`) and decompress (`-d`) arbitrary files.
- `test_lossless_roundtrip.py`: Verification script confirming byte-for-byte exact matches and SHA-256 parity.
- `pretrain_universal_codec.py`: Script to pretrain a universal neural codec across multi-domain text corpora.
- `download_benchmark_data.py`: Downloads standard compression benchmarks (e.g., enwik8 / Calgary corpus).
- `benchmark_vs_cpu.py`: Comprehensive benchmark suite comparing VulTorch Multi-GPU against CPU compressors (LZMA, gzip).
- `sample_book.txt`: Standard text corpus for training and compression testing.

## Quick Start

### 1. Verify Lossless Roundtrip
```bash
python3 -m examples.universal_compressor.test_lossless_roundtrip
```

### 2. Compress & Decompress Files via CLI
```bash
# Compress
python3 -m examples.universal_compressor.compress_cli -c input.txt -o compressed.vulc --device vk:0

# Decompress
python3 -m examples.universal_compressor.compress_cli -d compressed.vulc -o restored.txt --device vk:0
```
