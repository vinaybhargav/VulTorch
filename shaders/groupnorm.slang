// Fused GroupNorm Forward & Backward Kernels.
// Operates on [N, C, L] where G groups each have C_per_g = C / G channels.

struct GroupNormParams {
    uint N;
    uint C;
    uint G;
    uint L;
    float eps;
    uint has_affine;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> gnIn0;
[[vk::binding(1, 0)]] StructuredBuffer<float> gnIn1;
[[vk::binding(2, 0)]] StructuredBuffer<float> gnIn2;
[[vk::binding(3, 0)]] StructuredBuffer<float> gnIn3;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> gnOut0;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> gnOut1;

groupshared float gn_s0[256];
groupshared float gn_s1[256];

// Forward: 1 workgroup per (n, g) pair -> N * G workgroups.
// gnIn0 = x, gnIn1 = gamma, gnIn2 = beta, gnOut0 = y, gnOut1 = mean_invstd [2 * N * G]
[shader("compute")] [numthreads(256, 1, 1)]
void groupnorm_forward(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                       uniform GroupNormParams pc) {
    uint ng = gid.x;
    if (ng >= pc.N * pc.G) return;

    uint n = ng / pc.G;
    uint g = ng % pc.G;
    uint C_per_g = pc.C / pc.G;
    uint group_elems = C_per_g * pc.L;
    uint tid = lid.x;

    float sum = 0.0;
    float sum_sq = 0.0;
    uint base_offset = (n * pc.C + g * C_per_g) * pc.L;

    for (uint i = tid; i < group_elems; i += 256u) {
        float v = gnIn0[base_offset + i];
        sum += v;
        sum_sq += v * v;
    }

    gn_s0[tid] = sum;
    gn_s1[tid] = sum_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            gn_s0[tid] += gn_s0[tid + s];
            gn_s1[tid] += gn_s1[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float mean = gn_s0[0] / float(group_elems);
    float var = max(0.0, (gn_s1[0] / float(group_elems)) - mean * mean);
    float invstd = 1.0 / sqrt(var + pc.eps);

    if (tid == 0u) {
        gnOut1[ng] = mean;
        gnOut1[pc.N * pc.G + ng] = invstd;
    }

    for (uint i = tid; i < group_elems; i += 256u) {
        uint c = g * C_per_g + (i / pc.L);
        float xhat = (gnIn0[base_offset + i] - mean) * invstd;
        gnOut0[base_offset + i] = (pc.has_affine != 0u) ? (xhat * gnIn1[c] + gnIn2[c]) : xhat;
    }
}

// Backward dx: 1 workgroup per (n, g) pair.
// gnIn0 = grad_out, gnIn1 = x, gnIn2 = gamma, gnIn3 = mean_invstd [2 * N * G], gnOut0 = dx
[shader("compute")] [numthreads(256, 1, 1)]
void groupnorm_backward_dx(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                           uniform GroupNormParams pc) {
    uint ng = gid.x;
    if (ng >= pc.N * pc.G) return;

    uint n = ng / pc.G;
    uint g = ng % pc.G;
    uint C_per_g = pc.C / pc.G;
    uint group_elems = C_per_g * pc.L;
    uint tid = lid.x;

    float mean = gnIn3[ng];
    float invstd = gnIn3[pc.N * pc.G + ng];
    uint base_offset = (n * pc.C + g * C_per_g) * pc.L;

    float s_dyg = 0.0;
    float s_dyg_xhat = 0.0;
    for (uint i = tid; i < group_elems; i += 256u) {
        uint c = g * C_per_g + (i / pc.L);
        float gamma = (pc.has_affine != 0u) ? gnIn2[c] : 1.0;
        float dyg = gnIn0[base_offset + i] * gamma;
        float xhat = (gnIn1[base_offset + i] - mean) * invstd;
        s_dyg += dyg;
        s_dyg_xhat += dyg * xhat;
    }

    gn_s0[tid] = s_dyg;
    gn_s1[tid] = s_dyg_xhat;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            gn_s0[tid] += gn_s0[tid + s];
            gn_s1[tid] += gn_s1[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float total_dyg = gn_s0[0];
    float total_dyg_xhat = gn_s1[0];
    float M = float(group_elems);

    for (uint i = tid; i < group_elems; i += 256u) {
        uint c = g * C_per_g + (i / pc.L);
        float gamma = (pc.has_affine != 0u) ? gnIn2[c] : 1.0;
        float dyg = gnIn0[base_offset + i] * gamma;
        float xhat = (gnIn1[base_offset + i] - mean) * invstd;
        gnOut0[base_offset + i] = invstd * (dyg - (total_dyg / M) - xhat * (total_dyg_xhat / M));
    }
}

// Backward dgamma & dbeta: 1 workgroup per channel c -> C workgroups.
// gnIn0 = grad_out, gnIn1 = x, gnIn2 = mean_invstd [2 * N * G], gnOut0 = dgamma, gnOut1 = dbeta
[shader("compute")] [numthreads(256, 1, 1)]
void groupnorm_backward_gamma_beta(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                                   uniform GroupNormParams pc) {
    uint c = gid.x;
    if (c >= pc.C) return;

    uint tid = lid.x;
    uint C_per_g = pc.C / pc.G;
    uint g = c / C_per_g;
    uint NL = pc.N * pc.L;

    float dgamma = 0.0;
    float dbeta = 0.0;
    for (uint i = tid; i < NL; i += 256u) {
        uint n = i / pc.L;
        uint l = i % pc.L;
        uint ng = n * pc.G + g;
        float mean = gnIn2[ng];
        float invstd = gnIn2[pc.N * pc.G + ng];
        uint idx = (n * pc.C + c) * pc.L + l;
        float dy = gnIn0[idx];
        float xhat = (gnIn1[idx] - mean) * invstd;
        dgamma += dy * xhat;
        dbeta += dy;
    }

    gn_s0[tid] = dgamma;
    gn_s1[tid] = dbeta;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            gn_s0[tid] += gn_s0[tid + s];
            gn_s1[tid] += gn_s1[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        gnOut0[c] = gn_s0[0];
        gnOut1[c] = gn_s1[0];
    }
}
