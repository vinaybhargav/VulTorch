// Generic register-blocked SGEMM:  C[b] = epilogue(A[b] @ B[b])
//
// One source, many kernels. Every tile configuration below is a separate entry
// point, compiled to its own SPIR-V module (see SLANG_KERNELS in CMakeLists.txt)
// and chosen per (device, shape) by the measured selection in src/tensor.cpp.
// Adding a configuration is one line here plus one line in the candidate table.
//
// Parameters
//   TILE  output tile edge (BM = BN = TILE); the dispatcher assumes square tiles
//   BK    depth of one K step staged through shared memory
//   R     register block edge: each thread owns an R x R patch of the tile
//
// Design choices vs. the GLSL kernels it replaces
//   * Strided ownership: thread (tx, ty) owns rows ty + i*DIM and columns
//     tx + j*DIM rather than an R x R contiguous patch. Neighbouring lanes then
//     read neighbouring shared-memory words (no bank conflicts) and write
//     neighbouring global addresses (coalesced stores).
//   * Register prefetch: the next K tile is fetched from global memory into
//     registers while the current one is consumed from shared memory, hiding
//     DRAM latency. This matters most on small GPUs (e.g. MX450, 14 SMs) where
//     occupancy alone cannot cover it.
//   * Epilogue is a generic interface, so elementwise ops (bias, activation,
//     scaling) can be fused into the store instead of costing a memory pass.

struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint batch_count;
    uint strideA_b;
    uint strideA_m;
    uint strideA_k;
    uint strideB_b;
    uint strideB_k;
    uint strideB_n;
    uint strideC_b;
    uint strideC_m;
    uint strideC_n;
};

[[vk::push_constant]] ConstantBuffer<GemmParams> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] StructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;

interface IEpilogue {
    static float apply(float value, uint m, uint n);
};

struct Identity : IEpilogue {
    static float apply(float value, uint m, uint n) { return value; }
};

void gemm<let TILE : int, let BK : int, let R : int, E : IEpilogue>(uint3 lid, uint3 gid) {
    const int DIM = TILE / R;         // threads per tile edge
    const int T = DIM * DIM;          // threads per workgroup
    const int LOADS = TILE * BK / T;  // elements of A (and of B) each thread stages

    // +1 column on sA: the A staging writes walk k fastest, which would
    // otherwise map a warp onto a single bank.
    static groupshared float sA[BK][TILE + 1];
    static groupshared float sB[BK][TILE];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * DIM + tx;

    const uint bm = gid.y * TILE;
    const uint bn = gid.x * TILE;
    const uint batch = gid.z;

    const uint baseA = batch * pc.strideA_b;
    const uint baseB = batch * pc.strideB_b;
    const uint baseC = batch * pc.strideC_b;

    float acc[R][R];
    [ForceUnroll] for (int i = 0; i < R; ++i)
        [ForceUnroll] for (int j = 0; j < R; ++j)
            acc[i][j] = 0.0;

    float ra[LOADS];
    float rb[LOADS];

    // Fetch K tile starting at kb into registers. Loads walk the contiguous
    // dimension fastest (k for row-major A, n for row-major B) so they coalesce.
    uint kb = 0;
    [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
        const uint idx = tid + l * T;
        const uint m = idx / BK, k = idx % BK;
        const uint gm = bm + m, gk = kb + k;
        ra[l] = (gm < pc.M && gk < pc.K) ? A[baseA + gm * pc.strideA_m + gk * pc.strideA_k] : 0.0;
        const uint k2 = idx / TILE, n = idx % TILE;
        const uint gk2 = kb + k2, gn = bn + n;
        rb[l] = (gk2 < pc.K && gn < pc.N) ? B[baseB + gk2 * pc.strideB_k + gn * pc.strideB_n] : 0.0;
    }

    const uint num_tiles = (pc.K + BK - 1) / BK;
    for (uint t = 0; t < num_tiles; ++t) {
        [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
            const uint idx = tid + l * T;
            sA[idx % BK][idx / BK] = ra[l];
            sB[idx / TILE][idx % TILE] = rb[l];
        }
        GroupMemoryBarrierWithGroupSync();

        // Prefetch the next tile while this one is consumed.
        if (t + 1 < num_tiles) {
            kb = (t + 1) * BK;
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                const uint m = idx / BK, k = idx % BK;
                const uint gm = bm + m, gk = kb + k;
                ra[l] = (gm < pc.M && gk < pc.K) ? A[baseA + gm * pc.strideA_m + gk * pc.strideA_k] : 0.0;
                const uint k2 = idx / TILE, n = idx % TILE;
                const uint gk2 = kb + k2, gn = bn + n;
                rb[l] = (gk2 < pc.K && gn < pc.N) ? B[baseB + gk2 * pc.strideB_k + gn * pc.strideB_n] : 0.0;
            }
        }

        [ForceUnroll] for (int k = 0; k < BK; ++k) {
            float a[R];
            float b[R];
            [ForceUnroll] for (int i = 0; i < R; ++i) a[i] = sA[k][ty + i * DIM];
            [ForceUnroll] for (int j = 0; j < R; ++j) b[j] = sB[k][tx + j * DIM];
            [ForceUnroll] for (int i = 0; i < R; ++i)
                [ForceUnroll] for (int j = 0; j < R; ++j)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll] for (int i = 0; i < R; ++i) {
        const uint gm = bm + ty + i * DIM;
        if (gm >= pc.M) continue;
        [ForceUnroll] for (int j = 0; j < R; ++j) {
            const uint gn = bn + tx + j * DIM;
            if (gn >= pc.N) continue;
            C[baseC + gm * pc.strideC_m + gn * pc.strideC_n] = E::apply(acc[i][j], gm, gn);
        }
    }
}

// ---------------------------------------------------------------------------
// Configurations. numthreads must equal (TILE / R) x (TILE / R).
// Shared memory = 4 * BK * (2 * TILE + 1) bytes.

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_t32_k16_r2(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<32, 16, 2, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_t64_k16_r4(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<64, 16, 4, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_t64_k8_r4(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<64, 8, 4, Identity>(lid, gid); }

[shader("compute")] [numthreads(8, 8, 1)]
void gemm_t64_k16_r8(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<64, 16, 8, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_t128_k8_r8(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<128, 8, 8, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_t128_k16_r8(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<128, 16, 8, Identity>(lid, gid); }

