#!/usr/bin/env python3
"""Dispatch and throughput benchmarks for the Vulkan execution engine.

These numbers are the ones the project tracks over time. Run with:

    .venv/bin/python benchmarks/bench_dispatch.py

Every measurement synchronises before stopping the clock, so a batched
implementation is not accidentally credited for work it has not finished.
"""

import argparse
import os
import sys
import time

import numpy as np

# Allow running this file directly from a source checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


def _sync():
    torch.vulkan.synchronize_all()


def time_per_op(fn, iters: int) -> float:
    """Milliseconds per call, including the final synchronisation."""
    fn()  # warm up pipelines and pools
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) / iters * 1e3


def bench_small_ops(device: str, elements: int = 1024, iters: int = 500) -> float:
    x = torch.randn(elements, device=device)
    return time_per_op(lambda: x + 1.0, iters)


def bench_bandwidth(device: str, megabytes: int = 4, iters: int = 50) -> tuple:
    elements = megabytes * 1024 * 1024 // 4
    x = torch.randn(elements, device=device)
    ms = time_per_op(lambda: x * 1.0, iters)
    # read + write of one tensor per op
    gbs = (2 * elements * 4) / (ms / 1e3) / 1e9
    return ms, gbs


def bench_matmul(device: str, n: int = 1024, iters: int = 10) -> tuple:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    ms = time_per_op(lambda: a @ b, iters)
    tflops = (2 * n**3) / (ms / 1e3) / 1e12
    return ms, tflops


def bench_two_device_concurrency(n: int = 1024, total_ops: int = 22) -> tuple:
    """Same total work, split across devices by TFLOPS, run serialised vs overlapped.

    Serial: drain device 0 before submitting anything to device 1.
    Concurrent: submit to both, then wait once.
    """
    a0 = torch.randn(n, n, device="vk:0")
    b0 = torch.randn(n, n, device="vk:0")
    a1 = torch.randn(n, n, device="vk:1")
    b1 = torch.randn(n, n, device="vk:1")

    ops0, ops1 = torch.vulkan.compute_split_sizes(total_ops, ["vk:0", "vk:1"], min_chunk=2)

    def serial():
        for _ in range(ops0):
            _ = a0 @ b0
        torch.vulkan.synchronize(0)
        for _ in range(ops1):
            _ = a1 @ b1
        torch.vulkan.synchronize(1)

    def concurrent():
        for _ in range(ops0):
            _ = a0 @ b0
        for _ in range(ops1):
            _ = a1 @ b1
        torch.vulkan.synchronize_all()

    serial()
    _sync()
    start = time.perf_counter()
    serial()
    t_serial = time.perf_counter() - start

    concurrent()
    _sync()
    start = time.perf_counter()
    concurrent()
    t_concurrent = time.perf_counter() - start

    return t_serial * 1e3, t_concurrent * 1e3, t_serial / t_concurrent, (ops0, ops1)


def bench_train_step(device: str, iters: int = 10, batch: int = 512) -> float:
    model = nn.Linear(1024, 2048).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 1024, device=device)
    y = torch.tensor(np.random.randint(0, 2048, size=(batch,)).astype(np.float32), device=device)
    crit = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    _sync()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
    args = parser.parse_args()

    print(f"devices: {torch.vulkan.device_count()}")
    for i in range(torch.vulkan.device_count()):
        print(f"  [{i}] {torch.vulkan.get_device_name(i)}")
    print()

    ms = bench_small_ops(args.device)
    print(f"small op (1024 floats, +scalar) : {ms * 1e3:8.2f} us/op")

    ms, gbs = bench_bandwidth(args.device)
    print(f"4 MiB elementwise (read+write)  : {ms:8.3f} ms  -> {gbs:6.1f} GB/s")

    for n in (512, 1024):
        ms, tflops = bench_matmul(args.device, n=n)
        print(f"matmul {n}^3 fp32               : {ms:8.3f} ms  -> {tflops:6.2f} TFLOPS")

    if torch.vulkan.device_count() >= 2:
        single, split, speedup, (ops0, ops1) = bench_two_device_concurrency()
        print(f"2 devices ({ops0}+{ops1} matmuls) serial : {single:8.3f} ms")
        print(f"2 devices, posted concurrently : {split:8.3f} ms  -> {speedup:4.2f}x")

    ms = bench_train_step(args.device)
    print(f"train step (Linear 1024x2048)   : {ms:8.2f} ms")


if __name__ == "__main__":
    main()
