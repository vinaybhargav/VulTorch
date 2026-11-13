# GEMM: from roofline to measured kernel selection

This note records the derivation and the measurements behind the SGEMM
implementation. It follows the project rule: **derive, implement, measure,
publish** — no kernel is adopted on the grounds that another project shipped
something similar.

## 1. The bound

A discrete GPU is either compute-bound or bandwidth-bound. The ratio that
decides which is

$$R = \frac{\text{peak FLOP/s}}{\text{peak bytes/s}}
\qquad \text{(FLOP per byte)}$$

For the RX 7900 XTX: ≈61 TFLOPS fp32 (dual-issue) over ≈960 GB/s HBM →
**R ≈ 64 FLOP/byte**.

A $T_M \times T_N$ output tile stepped with depth $B_K$ reaches

$$\text{AI} = \frac{2\,T_M T_N}{(T_M + T_N)\,s}\ \ \text{FLOP/byte}$$

where $s$ is bytes per element. A kernel is only compute-bound when
$\text{AI} \ge R$.

| $s$ | tile | AI | verdict at $R = 64$ |
|---|---|---|---|
| fp32 (4 B) | 32×32 | 8 | bandwidth-bound (4× short) |
| fp32 (4 B) | 64×64 | 16 | bandwidth-bound (2× short) |
| fp32 (4 B) | 128×128 | 32 | *close*; cache reuse supplies the rest |
| fp16 (2 B) | 128×128 | 64 | compute-bound |

This is why the original 32×32 kernel plateaued at ~6 TFLOPS: it was not
poorly written, it was structurally unable to exceed the bandwidth roof.

## 2. The second constraint: occupancy

Tile size is not free. Each thread owns $T_M/16 \times T_N/16$ accumulators in
registers:

| tile | accumulators/lane | shared (fp32, double-free) | workgroups at 1024³ |
|---|---|---|---|
| 128×128 | 64 | ~16.5 KiB | 64 |
| 64×64 | 16 | ~8.3 KiB | 256 |
| 32×32 | 4 | 0 | 1024 |

Sixty-four accumulator registers per lane halve occupancy, and 64 workgroups on
a 96-CU device cannot fill it. This predicts exactly what we measured: the
128-tile wins on large problems and loses badly on mid-sized ones.

## 3. Measured

RX 7900 XTX, fp32, 1 s warmup so clocks are at steady state (without this the
numbers move by 1.6×, which is power management, not the kernel):

| shape | 32×32 | 64×64 | 128×128 | **auto** |
|---|---|---|---|---|
| 512³ | **4.94** | 3.06 | 1.52 | 4.88 |
| 1024³ | 9.32 | **10.20** | 6.60 | 10.06 |
| 2048³ | 11.10 | 13.84 | **15.34** | 15.35 |

(tabulated in TFLOPS; the winning kernel per row is bold)

Baseline before this work: **5.9 TFLOPS** at 1024³, 6.7 at 2048³. The best
variant is 2.6× the baseline at 2048³, and no single variant is best
everywhere — which is the whole argument against hardcoding one.

## 4. Why selection is measured, not hardcoded

The winner depends on the device as much as the shape:

| device | subgroup | shared memory | consequence |
|---|---|---|---|
| RX 7900 XTX (RADV) | 64 | 64 KiB | 128-tile fits comfortably |
| RTX 3080 | 32 | 48 KiB | 128-tile still fits, different occupancy |

And on shapes the training loop actually reuses: a `Linear(1024, 2048)` at batch
512 gives 4×16 = 64 workgroups with a 128-tile — an occupancy cliff a static
heuristic cannot see.

So `select_matmul_shader` does:

1. **Filter by capability.** Drop any variant whose shared-memory footprint or
   workgroup requirement the device does not report. This is the only place
   device differences are assumed — and they are *queried*, not inferred from a
   vendor ID.
2. **Honour an explicit override.** `VULTORCH_MATMUL_KERNEL=<shader>` pins one
   variant for benchmarking. It is re-read on every call so tests and benchmark
   harnesses can switch mid-process, and it bypasses the cache.
3. **Skip tiny problems.** Under 1 MFLOP the launch cost dominates; tiling
   cannot pay for itself and a sweep would cost more than the multiply.
4. **Measure the survivors once per (device, M, N, K)** and cache the winner.
   Each candidate gets a fixed *time* budget of warmup (20 ms), because the
   first runs of a kernel happen while the GPU ramps clocks and the decision
   would otherwise be made by the power governor.

`VULTORCH_AUTOTUNE=0` disables step 4 and takes the largest viable tile.

The measured-selection result above reproduces exactly this: auto picks the
row-wise winner at all three sizes.

## 5. What this is not

- **Not cooperative matrix.** Both target GPUs expose
  `VK_KHR_cooperative_matrix` with 16×16×16 shapes (verified on the device), but
  exploiting it requires fp16 operands in shared memory and a different
  fragment layout. That is the next step, and it should roughly double the
  compute roof.
- **Not fp16 yet.** `float16_arithmetic` and `storage_16bit` are reported by
  both cards and surfaced through `torch.vulkan.get_device_properties`, but the
  tensor dtype is still fp32-only.
- **Not double-buffered.** The K-loop still serialises global→shared load
  against compute; overlapping them is the next incremental win (~20-30%).
- **Not autotuned over more than tile size.** `B_K`, thread count and
  vector width are fixed per variant.

## 6. Reproducing

```bash
# compare variants on identical inputs
VULTORCH_MATMUL_KERNEL=matmul        .venv/bin/python benchmarks/bench_dispatch.py
VULTORCH_MATMUL_KERNEL=matmul_tiled64 .venv/bin/python benchmarks/bench_dispatch.py
VULTORCH_MATMUL_KERNEL=matmul_tiled   .venv/bin/python benchmarks/bench_dispatch.py
.venv/bin/python benchmarks/bench_dispatch.py          # auto
```

Correctness of every variant, including ragged shapes, is pinned by
`tests/test_kernel_selection.py`; capability reporting is pinned by the same
file.
