struct LayerNormParams {
    uint num_rows;
    uint cols;
    float eps;
    uint has_affine;
};

[[vk::push_constant]] LayerNormParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> gamma;
[[vk::binding(2, 0)]] StructuredBuffer<float> beta;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dataOut;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> auxOut;

static const uint WG_SIZE = 256u;
// Single [2][WG_SIZE] shared memory array avoids Slang identical-type groupshared aliasing.
static groupshared float s_red[2][WG_SIZE];

[shader("compute")] [numthreads(256, 1, 1)]
void layernorm(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;

    float sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        sum += dataIn[row_offset + i];
    }
    s_red[0][tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[0][tid] += s_red[0][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_mean = s_red[0][0] / float(pc.cols);
    GroupMemoryBarrierWithGroupSync();

    float var_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float diff = dataIn[row_offset + i] - row_mean;
        var_sum += diff * diff;
    }
    s_red[1][tid] = var_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[1][tid] += s_red[1][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_var = s_red[1][0] / float(pc.cols);
    float inv_std = rsqrt(row_var + pc.eps);
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float norm = (dataIn[row_offset + i] - row_mean) * inv_std;
        if (pc.has_affine != 0u) {
            norm = norm * gamma[i] + beta[i];
        }
        dataOut[row_offset + i] = norm;
    }
}

// Fused LayerNorm backward for dx (one workgroup per row) and optional (gradOut * xhat) output.
// Bindings: dataIn = x (0), gamma = gradOut (1), beta = gamma (2), dataOut = dx (3), auxOut = gy_xhat (4).
// pc.has_affine bit 0 (1u): gamma affine present
// pc.has_affine bit 1 (2u): writedx to dataOut
// pc.has_affine bit 2 (4u): write (gradOut * xhat) to auxOut for dgamma reduction
[shader("compute")] [numthreads(256, 1, 1)]
void layernorm_backward_dx(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;
    float inv_cols = 1.0 / float(pc.cols);

    float sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        sum += dataIn[row_offset + i];
    }
    s_red[0][tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[0][tid] += s_red[0][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_mean = s_red[0][0] * inv_cols;
    GroupMemoryBarrierWithGroupSync();

    float var_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float diff = dataIn[row_offset + i] - row_mean;
        var_sum += diff * diff;
    }
    s_red[1][tid] = var_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[1][tid] += s_red[1][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_std = rsqrt(s_red[1][0] * inv_cols + pc.eps);
    GroupMemoryBarrierWithGroupSync();

    bool has_w = (pc.has_affine & 1u) != 0u;
    bool write_dx = (pc.has_affine & 2u) != 0u;
    bool write_dgamma_buf = (pc.has_affine & 4u) != 0u;

    if (!write_dx) {
        if (write_dgamma_buf) {
            for (uint i = tid; i < pc.cols; i += WG_SIZE) {
                float xhat = (dataIn[row_offset + i] - row_mean) * inv_std;
                auxOut[row_offset + i] = gamma[row_offset + i] * xhat;
            }
        }
        return;
    }

    float sum_gy = 0.0;
    float sum_gy_xhat = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float xhat = (dataIn[row_offset + i] - row_mean) * inv_std;
        float g = gamma[row_offset + i];
        if (write_dgamma_buf) {
            auxOut[row_offset + i] = g * xhat;
        }
        float w = has_w ? beta[i] : 1.0;
        float gy = g * w;
        sum_gy += gy;
        sum_gy_xhat += gy * xhat;
    }
    s_red[0][tid] = sum_gy;
    s_red[1][tid] = sum_gy_xhat;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_red[0][tid] += s_red[0][tid + s];
            s_red[1][tid] += s_red[1][tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float mean_gy = s_red[0][0] * inv_cols;
    float mean_gy_xhat = s_red[1][0] * inv_cols;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float xhat = (dataIn[row_offset + i] - row_mean) * inv_std;
        float w = has_w ? beta[i] : 1.0;
        float gy = gamma[row_offset + i] * w;
        dataOut[row_offset + i] = (gy - mean_gy - xhat * mean_gy_xhat) * inv_std;
    }
}

// RMSNorm forward: y = (x * rsqrt(mean(x^2) + eps)) * gamma
[shader("compute")] [numthreads(256, 1, 1)]
void rmsnorm(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;

    float sq_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float v = dataIn[row_offset + i];
        sq_sum += v * v;
    }
    s_red[0][tid] = sq_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[0][tid] += s_red[0][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_rms = rsqrt(s_red[0][0] / float(pc.cols) + pc.eps);
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float norm = dataIn[row_offset + i] * inv_rms;
        if (pc.has_affine != 0u) {
            norm *= gamma[i];
        }
        dataOut[row_offset + i] = norm;
    }
}

// RMSNorm backward:
// Bindings: dataIn = x (0), gamma = gradOut (1), beta = gamma (2), dataOut = dx (3), auxOut = gy_xhat (4).
[shader("compute")] [numthreads(256, 1, 1)]
void rmsnorm_backward_dx(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;
    float inv_cols = 1.0 / float(pc.cols);

    float sq_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float v = dataIn[row_offset + i];
        sq_sum += v * v;
    }
    s_red[0][tid] = sq_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[0][tid] += s_red[0][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float inv_rms = rsqrt(s_red[0][0] * inv_cols + pc.eps);
    GroupMemoryBarrierWithGroupSync();

    bool has_w = (pc.has_affine & 1u) != 0u;
    bool write_dx = (pc.has_affine & 2u) != 0u;
    bool write_dgamma_buf = (pc.has_affine & 4u) != 0u;

    if (!write_dx) {
        if (write_dgamma_buf) {
            for (uint i = tid; i < pc.cols; i += WG_SIZE) {
                float xhat = dataIn[row_offset + i] * inv_rms;
                auxOut[row_offset + i] = gamma[row_offset + i] * xhat;
            }
        }
        return;
    }

    float sum_gy_xhat = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float xhat = dataIn[row_offset + i] * inv_rms;
        float g = gamma[row_offset + i];
        if (write_dgamma_buf) {
            auxOut[row_offset + i] = g * xhat;
        }
        float w = has_w ? beta[i] : 1.0;
        float gy = g * w;
        sum_gy_xhat += gy * xhat;
    }
    s_red[1][tid] = sum_gy_xhat;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_red[1][tid] += s_red[1][tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean_gy_xhat = s_red[1][0] * inv_cols;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float xhat = dataIn[row_offset + i] * inv_rms;
        float w = has_w ? beta[i] : 1.0;
        float gy = gamma[row_offset + i] * w;
        dataOut[row_offset + i] = (gy - xhat * mean_gy_xhat) * inv_rms;
    }
}
