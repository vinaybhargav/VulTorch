#!/usr/bin/env python3
"""Per-variant SGEMM throughput.

    VULTORCH_MATMUL_KERNEL=gemm_t64_k16_r4 python benchmarks/bench_gemm_variants.py --device vk:1

Shapes are N (cube) or MxKxN. The GPU is warmed for 1 s first so power
management, not the kernel, does not decide the result.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vultorch as torch


def parse_shape(text):
    dims = [int(x) for x in text.split("x")]
    return dims * 3 if len(dims) == 1 else dims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="vk:0")
    p.add_argument("--shapes", default="512,1024,2048,512x1024x2048")
    args = p.parse_args()
    dev = int(args.device.split(":")[1])
    kernel = os.environ.get("VULTORCH_MATMUL_KERNEL", "auto")

    warm = torch.randn(1024, 1024, device=args.device)
    start = time.perf_counter()
    while time.perf_counter() - start < 1.0:
        _ = warm @ warm
        torch.vulkan.synchronize(dev)

    for text in args.shapes.split(","):
        M, K, N = parse_shape(text)
        a = torch.randn(M, K, device=args.device)
        b = torch.randn(K, N, device=args.device)
        for _ in range(3):
            _ = a @ b
        torch.vulkan.synchronize(dev)
        flops = 2.0 * M * N * K
        iters = max(5, int(2e10 / flops))
        start = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.vulkan.synchronize(dev)
        ms = (time.perf_counter() - start) / iters * 1e3
        print(f"{kernel:18s} {M}x{K}x{N:<6} {ms:9.3f} ms {flops / ms / 1e9:6.2f} TFLOPS")


if __name__ == "__main__":
    main()
