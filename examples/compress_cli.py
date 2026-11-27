#!/usr/bin/env python3
"""
VulTorch Neural Text Compressor CLI
Compress and decompress text files using a ~1M parameter deep neural network on Vulkan GPUs.
"""

import sys
import os
import argparse
import time
import subprocess
from examples.neural_text_compressor import (
    NeuralCompressor1M,
    train_codec_model,
    compress_file,
    decompress_file,
)
import vultorch as torch


def main():
    parser = argparse.ArgumentParser(description="VulTorch ~1M Parameter Neural Text Compressor")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Compress
    p_comp = subparsers.add_parser("compress", help="Compress a text file")
    p_comp.add_argument("input", help="Path to input text file")
    p_comp.add_argument("output", help="Path to output .vt file")
    p_comp.add_argument("--device", default="vk:0", help="Vulkan device (default: vk:0)")
    p_comp.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")

    # Decompress
    p_decomp = subparsers.add_parser("decompress", help="Decompress a .vt file")
    p_decomp.add_argument("input", help="Path to input .vt file")
    p_decomp.add_argument("output", help="Path to restored text file")
    p_decomp.add_argument("--model", default="model.pt", help="Path to model weights")
    p_decomp.add_argument("--device", default="vk:0", help="Vulkan device (default: vk:0)")

    # Benchmark
    p_bench = subparsers.add_parser(
        "benchmark", help="Benchmark VulTorch vs LZMA -9 on a text file"
    )
    p_bench.add_argument("input", help="Path to text file")
    p_bench.add_argument("--device", default="vk:0", help="Vulkan device (default: vk:0)")

    args = parser.parse_args()

    if args.cmd == "compress":
        with open(args.input, "rb") as f:
            data = f.read()
        print(f"Reading {args.input} ({len(data):,} bytes)...")
        print(f"Training 1M parameter model on {torch.vulkan.get_device_name(0)}...")
        model = train_codec_model(data, epochs=args.epochs, device=args.device)

        print(f"Encoding bitstream...")
        t0 = time.time()
        compressed = compress_file(model, data, device=args.device)
        enc_time = time.time() - t0

        with open(args.output, "wb") as f:
            f.write(compressed)

        # Save model weights alongside
        model_path = args.output + ".model.pt"
        torch.save(model.state_dict(), model_path)

        orig_sz = len(data)
        comp_sz = len(compressed)
        ratio = comp_sz / orig_sz * 100
        bpc = (comp_sz * 8) / orig_sz

        print(f"✅ Compressed successfully to {args.output}!")
        print(f"   Original:   {orig_sz:,} bytes")
        print(f"   Compressed: {comp_sz:,} bytes ({ratio:.2f}%, {bpc:.2f} bits/byte)")
        print(f"   Saved shared model weights to {model_path}")

    elif args.cmd == "benchmark":
        with open(args.input, "rb") as f:
            data = f.read()
        orig_sz = len(data)

        # LZMA -9
        t0 = time.time()
        res = subprocess.run(["lzma", "-9", "-c", args.input], stdout=subprocess.PIPE)
        lzma_sz = len(res.stdout)
        lzma_time = time.time() - t0

        # VulTorch
        print(f"🏋️  Training 1M parameter model on {torch.vulkan.get_device_name(0)}...")
        model = train_codec_model(data, epochs=15, device=args.device)
        t0 = time.time()
        vt_compressed = compress_file(model, data, device=args.device)
        vt_time = time.time() - t0
        vt_sz = len(vt_compressed)

        print("=" * 70)
        print(f"📊 COMPRESSION BENCHMARK: {args.input}")
        print("=" * 70)
        print(f"Original Size:        {orig_sz:,} bytes (100.0%)")
        print(
            f"LZMA -9 Size:         {lzma_sz:,} bytes ({lzma_sz / orig_sz * 100:.2f}%, {(lzma_sz * 8) / orig_sz:.2f} bpc)"
        )
        print(
            f"VulTorch 1M Size:     {vt_sz:,} bytes ({vt_sz / orig_sz * 100:.2f}%, {(vt_sz * 8) / orig_sz:.2f} bpc)"
        )
        print("-" * 70)
        diff = lzma_sz - vt_sz
        if diff > 0:
            print(
                f"🎉 VulTorch BEAT LZMA -9 by {(diff / lzma_sz) * 100:.1f}% ({diff:,} fewer bytes)!"
            )
        else:
            print(f"LZMA vs VulTorch: {diff} bytes")
        print("=" * 70)


if __name__ == "__main__":
    main()
