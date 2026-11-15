// Native float16 GEMM kernels for VulTorch:
//   - gemm_fp16: 64x64 tiled SIMT GEMM with fp16 storage and fp32 accumulation
//   - matmul_coopmat_fp16: 16x16x16 VK_KHR_cooperative_matrix fp16 GEMM (fp16 storage, fp16/fp32 accum)

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

[[vk::binding(0, 0)]] StructuredBuffer<half> A;
[[vk::binding(1, 0)]] StructuredBuffer<half> B;
[[vk::binding(2, 0)]] RWStructuredBuffer<half> C;
[[vk::binding(3, 0)]] StructuredBuffer<half> Bias;

static const uint TILE_M = 64u;
static const uint TILE_N = 64u;
static const uint TILE_K = 16u;

// Single 2x16x65 shared tile array prevents Slang -O3 from aliasing separate
// identically-shaped groupshared buffers into a single SPIR-V OpVariable.
static groupshared float smem[2][TILE_K][TILE_M + 1u];

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_fp16(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint batch = gid.z;
    if (batch >= pc.batch_count) return;

    const uint baseA = pc.offsetA + batch * pc.strideA_b;
    const uint baseB = pc.offsetB + batch * pc.strideB_b;
    const uint baseC = pc.offsetC + batch * pc.strideC_b;

    const uint wg_m = gid.y * TILE_M;
    const uint wg_n = gid.x * TILE_N;
    const uint ty = ltid.y;
    const uint tx = ltid.x;
    const uint local_idx = ty * 16u + tx;

    float acc[4][4];
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        [unroll]
        for (uint j = 0u; j < 4u; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (uint k_base = 0u; k_base < pc.K; k_base += TILE_K) {
        [unroll]
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = local_idx + i * 256u;
            uint r = idx >> 6u;
            uint c = idx & 63u;

            uint global_m = wg_m + c;
            uint global_k_a = k_base + r;
            float a_val = 0.0f;
            if (global_m < pc.M && global_k_a < pc.K) {
                a_val = float(A[baseA + global_m * pc.strideA_m + global_k_a * pc.strideA_k]);
            }
            smem[0][r][c] = a_val;

            uint global_k_b = k_base + r;
            uint global_n = wg_n + c;
            float b_val = 0.0f;
            if (global_k_b < pc.K && global_n < pc.N) {
                b_val = float(B[baseB + global_k_b * pc.strideB_k + global_n * pc.strideB_n]);
            }
            smem[1][r][c] = b_val;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0u; k < TILE_K; ++k) {
            float regA[4];
            float regB[4];
            [unroll]
            for (uint i = 0u; i < 4u; ++i) {
                regA[i] = smem[0][k][ty * 4u + i];
            }
            [unroll]
            for (uint j = 0u; j < 4u; ++j) {
                regB[j] = smem[1][k][tx * 4u + j];
            }
            [unroll]
            for (uint i = 0u; i < 4u; ++i) {
                [unroll]
                for (uint j = 0u; j < 4u; ++j) {
                    acc[i][j] = fma(regA[i], regB[j], acc[i][j]);
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        uint out_m = wg_m + ty * 4u + i;
        if (out_m >= pc.M) continue;
        [unroll]
        for (uint j = 0u; j < 4u; ++j) {
            uint out_n = wg_n + tx * 4u + j;
            if (out_n >= pc.N) continue;
            float val = acc[i][j];
            if (pc.has_bias != 0u) {
                val += float(Bias[out_n]);
            }
            C[baseC + out_m * pc.strideC_m + out_n * pc.strideC_n] = half(val);
        }
    }
}

static groupshared float coop_out_smem[256];

[shader("compute")] [numthreads(32, 1, 1)]
void matmul_coopmat_fp16(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    const uint batch = gid.z;
    if (batch >= pc.batch_count) return;

    const uint baseA = pc.offsetA + batch * pc.strideA_b;
    const uint baseB = pc.offsetB + batch * pc.strideB_b;
    const uint baseC = pc.offsetC + batch * pc.strideC_b;

    const uint row0 = gid.y * 16u;
    const uint col0 = gid.x * 16u;

    linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator> matC;
    matC.fill(0.0f);

    for (uint k_base = 0u; k_base < pc.K; k_base += 16u) {
        linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA> matA;
        linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB> matB;

        if (pc.strideA_k == 1u && row0 + 16u <= pc.M && k_base + 16u <= pc.K) {
            matA = linalg.coopMatLoad<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA, linalg.CoopMatMatrixLayout.RowMajor>(
                A, baseA + row0 * pc.strideA_m + k_base, pc.strideA_m);
        } else {
            matA.fill(half(0.0));
        }

        if (pc.strideB_n == 1u && k_base + 16u <= pc.K && col0 + 16u <= pc.N) {
            matB = linalg.coopMatLoad<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB, linalg.CoopMatMatrixLayout.RowMajor>(
                B, baseB + k_base * pc.strideB_k + col0, pc.strideB_k);
        } else {
            matB.fill(half(0.0));
        }

        matC = linalg.coopMatMulAdd<float, false>(matA, matB, matC);
    }

    matC.Store<linalg.CoopMatMatrixLayout.RowMajor>(coop_out_smem, 0u, 16u);
    GroupMemoryBarrierWithGroupSync();

    for (uint elem = ltid.x; elem < 256u; elem += 32u) {
        uint lr = elem >> 4u;
        uint lc = elem & 15u;
        uint gm = row0 + lr;
        uint gn = col0 + lc;
        if (gm < pc.M && gn < pc.N) {
            float val = coop_out_smem[elem];
            if (pc.has_bias != 0u) {
                val += float(Bias[gn]);
            }
            C[baseC + gm * pc.strideC_m + gn * pc.strideC_n] = half(val);
        }
    }
}
