// Generic register-blocked SGEMM and cooperative-matrix SGEMM in Slang:
//   C[b] = epilogue(op(A[b]) @ op(B[b]) + (has_bias ? bias : 0))
//
// One source file compiles every GEMM entry point used by VulTorch:
//   - gemm_t32_k16_r2   (and legacy alias `matmul`)
//   - gemm_t64_k16_r4   (and legacy alias `matmul_tiled64`)
//   - gemm_t64_k8_r4
//   - gemm_t64_k16_r8
//   - gemm_t128_k8_r8
//   - gemm_t128_k16_r8  (and legacy alias `matmul_tiled`)
//   - matmul_coopmat    (16x16x16 fp16 cooperative matrix with fp32 accumulate)
//
// Coalesced transpose support:
//   * Normal A (strideA_k == 1): threads walk k fastest when loading A.
//   * Transposed A (strideA_m == 1): threads walk m fastest when loading A.
//   * Normal B (strideB_n == 1): threads walk n fastest when loading B.
//   * Transposed B (strideB_k == 1): threads walk k fastest when loading B.
//   Both sA and sB are padded by +1 (`[BK][TILE + 1]`) so both row-wise and
//   column-wise shared-memory writes are bank-conflict-free.

using namespace linalg;

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
    uint has_bias;
    uint offsetA;
    uint offsetB;
    uint offsetC;
};

[[vk::push_constant]] GemmParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> A;
[[vk::binding(1, 0)]] StructuredBuffer<float> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> C;
[[vk::binding(3, 0)]] StructuredBuffer<float> bias_buf;

interface IEpilogue {
    static float apply(float value, uint m, uint n);
};

struct Identity : IEpilogue {
    static float apply(float value, uint m, uint n) { return value; }
};

[ForceInline]
float load_a<let TILE : int, let BK : int>(uint baseA, uint bm, uint kb, uint idx) {
    uint m, k;
    if (pc.strideA_k == 1u) {
        m = idx / BK;
        k = idx % BK;
    } else {
        m = idx % TILE;
        k = idx / TILE;
    }
    const uint gm = bm + m;
    const uint gk = kb + k;
    return (gm < pc.M && gk < pc.K) ? A[baseA + gm * pc.strideA_m + gk * pc.strideA_k] : 0.0;
}

[ForceInline]
float load_b<let TILE : int, let BK : int>(uint baseB, uint bn, uint kb, uint idx) {
    uint k, n;
    if (pc.strideB_n == 1u) {
        k = idx / TILE;
        n = idx % TILE;
    } else {
        k = idx % BK;
        n = idx / BK;
    }
    const uint gk = kb + k;
    const uint gn = bn + n;
    return (gk < pc.K && gn < pc.N) ? B[baseB + gk * pc.strideB_k + gn * pc.strideB_n] : 0.0;
}

void gemm<let TILE : int, let BK : int, let R : int, E : IEpilogue>(uint3 lid, uint3 gid) {
    const int DIM = TILE / R;         // threads per tile edge
    const int T = DIM * DIM;          // threads per workgroup
    const int LOADS = TILE * BK / T;  // elements of A (and of B) each thread stages

    // Both sA and sB have odd row stride (TILE + 1) so strided column writes
    // (when A is normal or B is transposed) are free of shared-memory bank conflicts.
    // Combined into [2][BK][TILE + 1] so Slang's identical-groupshared-type CSE pass
    // does not alias sA and sB onto the same shared-memory buffer.
    static groupshared float smem[2][BK][TILE + 1];

    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint tid = ty * DIM + tx;

    const uint bm = gid.y * TILE;
    const uint bn = gid.x * TILE;
    const uint batch = gid.z;

    const uint baseA = pc.offsetA + batch * pc.strideA_b;
    const uint baseB = pc.offsetB + batch * pc.strideB_b;
    const uint baseC = pc.offsetC + batch * pc.strideC_b;

    float acc[R][R];
    [ForceUnroll] for (int i = 0; i < R; ++i)
        [ForceUnroll] for (int j = 0; j < R; ++j)
            acc[i][j] = 0.0;

    float ra[LOADS];
    float rb[LOADS];

    uint kb = 0;
    [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
        const uint idx = tid + l * T;
        ra[l] = load_a<TILE, BK>(baseA, bm, kb, idx);
        rb[l] = load_b<TILE, BK>(baseB, bn, kb, idx);
    }

    const uint num_tiles = (pc.K + BK - 1) / BK;
    for (uint t = 0; t < num_tiles; ++t) {
        if (pc.strideA_k == 1u) {
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                smem[0][idx % BK][idx / BK] = ra[l];
            }
        } else {
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                smem[0][idx / TILE][idx % TILE] = ra[l];
            }
        }

        if (pc.strideB_n == 1u) {
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                smem[1][idx / TILE][idx % TILE] = rb[l];
            }
        } else {
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                smem[1][idx % BK][idx / BK] = rb[l];
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Prefetch the next tile while this one is consumed.
        if (t + 1 < num_tiles) {
            kb = (t + 1) * BK;
            [ForceUnroll] for (int l = 0; l < LOADS; ++l) {
                const uint idx = tid + l * T;
                ra[l] = load_a<TILE, BK>(baseA, bm, kb, idx);
                rb[l] = load_b<TILE, BK>(baseB, bn, kb, idx);
            }
        }

        [ForceUnroll] for (int k = 0; k < BK; ++k) {
            float a[R];
            float b[R];
            [ForceUnroll] for (int i = 0; i < R; ++i) a[i] = smem[0][k][ty + i * DIM];
            [ForceUnroll] for (int j = 0; j < R; ++j) b[j] = smem[1][k][tx + j * DIM];
            [ForceUnroll] for (int i = 0; i < R; ++i)
                [ForceUnroll] for (int j = 0; j < R; ++j)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float b_reg[R];
    [ForceUnroll] for (int j = 0; j < R; ++j) {
        const uint gn = bn + tx + j * DIM;
        b_reg[j] = (pc.has_bias != 0u && gn < pc.N) ? bias_buf[gn] : 0.0;
    }

    [ForceUnroll] for (int i = 0; i < R; ++i) {
        const uint gm = bm + ty + i * DIM;
        if (gm >= pc.M) continue;
        [ForceUnroll] for (int j = 0; j < R; ++j) {
            const uint gn = bn + tx + j * DIM;
            if (gn >= pc.N) continue;
            C[baseC + gm * pc.strideC_m + gn * pc.strideC_n] = E::apply(acc[i][j] + b_reg[j], gm, gn);
        }
    }
}

// ---------------------------------------------------------------------------
// SIMT configurations. numthreads = (TILE / R) x (TILE / R).
// Shared memory = 4 * BK * 2 * (TILE + 1) bytes.

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

// Legacy entry-point aliases kept for VULTORCH_MATMUL_KERNEL compatibility.
[shader("compute")] [numthreads(16, 16, 1)]
void matmul(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<32, 16, 2, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void matmul_tiled64(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<64, 16, 4, Identity>(lid, gid); }

[shader("compute")] [numthreads(16, 16, 1)]
void matmul_tiled(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) { gemm<128, 16, 8, Identity>(lid, gid); }

// ---------------------------------------------------------------------------
// Cooperative-matrix 16x16x16 fp16 -> fp32 SGEMM (64x64 workgroup tile).

static const uint CM_BM = 64u;
static const uint CM_BN = 64u;
static const uint CM_BK = 16u;
static const uint CM_SIZE = 16u;
static const uint CM_BKP = 17u;
static const uint CM_BNP = 68u;
static const uint CM_SOP = 68u;

static groupshared float16_t cm_sA[CM_BM * CM_BKP];
static groupshared float16_t cm_sB[CM_BK * CM_BNP];
static groupshared float cm_sOut[CM_BM * CM_SOP];

[shader("compute")] [numthreads(64, 1, 1)]
void matmul_coopmat(uint3 lid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid = lid.x;
    const uint wave_size = WaveGetLaneCount();
    const uint num_subgroups = 64u / wave_size;
    const uint subgroup = tid / wave_size;

    const uint block_y = gid.y;
    const uint block_x = gid.x;
    const uint batch = gid.z;

    const uint cols_per_subgroup = CM_BN / num_subgroups;
    const uint col_tiles = cols_per_subgroup / CM_SIZE;
    const uint col_base = subgroup * cols_per_subgroup;

    typealias FragA = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
    typealias FragB = CoopMat<float16_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
    typealias Accum = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

    Accum acc[4][4];
    [ForceUnroll] for (int i = 0; i < 4; ++i)
        [ForceUnroll] for (int j = 0; j < 4; ++j)
            acc[i][j] = Accum(0.0);

    const uint baseA = pc.offsetA + batch * pc.strideA_b;
    const uint baseB = pc.offsetB + batch * pc.strideB_b;
    const uint baseC = pc.offsetC + batch * pc.strideC_b;
    const uint num_k_tiles = (pc.K + CM_BK - 1u) / CM_BK;

    for (uint tile = 0; tile < num_k_tiles; ++tile) {
        const uint k_base = tile * CM_BK;

        for (uint idx = tid; idx < CM_BM * CM_BK; idx += 64u) {
            const uint m = idx / CM_BK;
            const uint k = idx % CM_BK;
            const uint gm = block_y * CM_BM + m;
            const uint gk = k_base + k;
            const float v = (gm < pc.M && gk < pc.K)
                                ? A[baseA + gm * pc.strideA_m + gk * pc.strideA_k]
                                : 0.0;
            cm_sA[m * CM_BKP + k] = float16_t(v);
        }

        for (uint idx = tid; idx < CM_BK * CM_BN; idx += 64u) {
            const uint k = idx / CM_BN;
            const uint n = idx % CM_BN;
            const uint gk = k_base + k;
            const uint gn = block_x * CM_BN + n;
            const float v = (gk < pc.K && gn < pc.N)
                                ? B[baseB + gk * pc.strideB_k + gn * pc.strideB_n]
                                : 0.0;
            cm_sB[k * CM_BNP + n] = float16_t(v);
        }

        GroupMemoryBarrierWithGroupSync();

        [ForceUnroll] for (int rt = 0; rt < 4; ++rt) {
            FragA frag_a = FragA.Load<CoopMatMatrixLayout.RowMajor>(cm_sA, rt * CM_SIZE * CM_BKP, CM_BKP);
            [ForceUnroll] for (int ct = 0; ct < 4; ++ct) {
                if (uint(ct) < col_tiles) {
                    FragB frag_b = FragB.Load<CoopMatMatrixLayout.RowMajor>(cm_sB, col_base + uint(ct) * CM_SIZE, CM_BNP);
                    acc[rt][ct] = coopMatMulAdd<float, false>(frag_a, frag_b, acc[rt][ct]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll] for (int rt = 0; rt < 4; ++rt) {
        [ForceUnroll] for (int ct = 0; ct < 4; ++ct) {
            if (uint(ct) < col_tiles) {
                acc[rt][ct].Store<CoopMatMatrixLayout.RowMajor>(
                    cm_sOut, uint(rt) * CM_SIZE * CM_SOP + col_base + uint(ct) * CM_SIZE, CM_SOP);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    for (uint idx = tid; idx < CM_BM * CM_BN; idx += 64u) {
        const uint m = idx / CM_BN;
        const uint n = idx % CM_BN;
        const uint gm = block_y * CM_BM + m;
        const uint gn = block_x * CM_BN + n;
        if (gm < pc.M && gn < pc.N) {
            const float b_val = (pc.has_bias != 0u) ? bias_buf[gn] : 0.0;
            C[baseC + gm * pc.strideC_m + gn * pc.strideC_n] = cm_sOut[m * CM_SOP + n] + b_val;
        }
    }
}
