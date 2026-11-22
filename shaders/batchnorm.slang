// Fused BatchNorm1d/2d Forward & Backward Kernels.
// Operates on [N, C, L] (with L = H * W for 2D or L = 1 for 1D).

struct BatchNormParams {
    uint N;
    uint C;
    uint L;
    float eps;
    float momentum;
    uint training;
    uint has_affine;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> bnIn0;
[[vk::binding(1, 0)]] StructuredBuffer<float> bnIn1;
[[vk::binding(2, 0)]] StructuredBuffer<float> bnIn2;
[[vk::binding(3, 0)]] StructuredBuffer<float> bnIn3;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> bnOut0;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> bnOut1;

groupshared float bn_s0[256];
groupshared float bn_s1[256];

// Compute batch mean & inv_std per channel (1 workgroup per channel c)
// bnIn0 = x, bnOut0 = save_mean [C], bnOut1 = save_invstd [C]
[shader("compute")] [numthreads(256, 1, 1)]
void batchnorm_stats(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                     uniform BatchNormParams pc) {
    uint c = gid.x;
    if (c >= pc.C) return;

    uint tid = lid.x;
    uint NL = pc.N * pc.L;
    float sum = 0.0;
    float sum_sq = 0.0;

    for (uint i = tid; i < NL; i += 256u) {
        uint n = i / pc.L;
        uint l = i % pc.L;
        float v = bnIn0[(n * pc.C + c) * pc.L + l];
        sum += v;
        sum_sq += v * v;
    }

    bn_s0[tid] = sum;
    bn_s1[tid] = sum_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            bn_s0[tid] += bn_s0[tid + s];
            bn_s1[tid] += bn_s1[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        float mean = bn_s0[0] / float(NL);
        float var = max(0.0, (bn_s1[0] / float(NL)) - mean * mean);
        float invstd = 1.0 / sqrt(var + pc.eps);
        bnOut0[c] = mean;
        bnOut1[c] = invstd;
    }
}

// Forward normalize + affine:
// bnIn0 = x, bnIn1 = mean, bnIn2 = invstd_or_var, bnIn3 = gamma, bnOut0 = beta, bnOut1 = y
[shader("compute")] [numthreads(256, 1, 1)]
void batchnorm_forward(uint3 tid : SV_DispatchThreadID, uniform BatchNormParams pc) {
    uint total = pc.N * pc.C * pc.L;
    uint idx = tid.x;
    if (idx >= total) return;

    uint c = (idx / pc.L) % pc.C;
    float mean = bnIn1[c];
    float invstd = (pc.training != 0u) ? bnIn2[c] : (1.0 / sqrt(max(0.0, bnIn2[c]) + pc.eps));
    float xhat = (bnIn0[idx] - mean) * invstd;
    if (pc.has_affine != 0u) {
        bnOut1[idx] = xhat * bnIn3[c] + bnOut0[c];
    } else {
        bnOut1[idx] = xhat;
    }
}

// Backward channel reductions: computes sum(dy) and sum(dy * xhat) per channel
// bnIn0 = grad_out, bnIn1 = x, bnIn2 = mean, bnIn3 = invstd
// bnOut0 = dbeta_or_sum_dy [C], bnOut1 = dgamma_or_sum_dy_xhat [C]
[shader("compute")] [numthreads(256, 1, 1)]
void batchnorm_backward_reduce(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                               uniform BatchNormParams pc) {
    uint c = gid.x;
    if (c >= pc.C) return;

    uint tid = lid.x;
    uint NL = pc.N * pc.L;
    float mean = bnIn2[c];
    float invstd = bnIn3[c];

    float s_dy = 0.0;
    float s_dy_xhat = 0.0;
    for (uint i = tid; i < NL; i += 256u) {
        uint n = i / pc.L;
        uint l = i % pc.L;
        uint idx = (n * pc.C + c) * pc.L + l;
        float dy = bnIn0[idx];
        float xhat = (bnIn1[idx] - mean) * invstd;
        s_dy += dy;
        s_dy_xhat += dy * xhat;
    }

    bn_s0[tid] = s_dy;
    bn_s1[tid] = s_dy_xhat;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            bn_s0[tid] += bn_s0[tid + s];
            bn_s1[tid] += bn_s1[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        bnOut0[c] = bn_s0[0];
        bnOut1[c] = bn_s1[0];
    }
}

// Backward dx elementwise:
// bnIn0 = grad_out, bnIn1 = x, bnIn2 = mean, bnIn3 = invstd
// bnOut0 = gamma (read), bnOut1 = dx (write)
// Wait: we also need sum_dy and sum_dy_xhat, so pack them via separate kernel or
// store [sum_dy, sum_dy_xhat] interleaved? Even cleaner: compute dx using
// bnIn0=grad_out, bnIn1=x, bnIn2=mean_invstd_packed? No, just 6 bindings!
// We already have bnIn0..3 (4) + bnOut0..1 (2) = 6 bindings! Let's pack sum_dy & sum_dy_xhat
// in bnOut0[0..C-1] and bnOut0[C..2C-1] if needed, or read gamma from bnIn3?
// Let's keep it super clean: dx_kernel takes:
// bnIn0 = grad_out, bnIn1 = x, bnIn2 = stats_packed [4 * C: mean, invstd, sum_dy, sum_dy_xhat],
// bnIn3 = gamma, bnOut0 = dx!
[shader("compute")] [numthreads(256, 1, 1)]
void batchnorm_backward_dx(uint3 tid : SV_DispatchThreadID, uniform BatchNormParams pc) {
    uint total = pc.N * pc.C * pc.L;
    uint idx = tid.x;
    if (idx >= total) return;

    uint c = (idx / pc.L) % pc.C;
    float mean = bnIn2[c];
    float invstd = bnIn2[pc.C + c];
    float sum_dy = bnIn2[2u * pc.C + c];
    float sum_dy_xhat = bnIn2[3u * pc.C + c];

    float gamma = (pc.has_affine != 0u) ? bnIn3[c] : 1.0;
    float NL = float(pc.N * pc.L);
    float xhat = (bnIn1[idx] - mean) * invstd;
    float dy = bnIn0[idx];

    bnOut0[idx] = (gamma * invstd / NL) * (NL * dy - sum_dy - xhat * sum_dy_xhat);
}
