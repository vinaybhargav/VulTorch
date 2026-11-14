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

static const uint WG_SIZE = 256u;
static groupshared float s_mean[WG_SIZE];
static groupshared float s_var[WG_SIZE];

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
    s_mean[tid] = sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_mean = s_mean[0] / float(pc.cols);
    GroupMemoryBarrierWithGroupSync();

    float var_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float diff = dataIn[row_offset + i] - row_mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) s_var[tid] += s_var[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_var = s_var[0] / float(pc.cols);
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
