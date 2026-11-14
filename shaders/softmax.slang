struct SoftmaxParams {
    uint num_rows;
    uint cols;
    uint is_log_softmax; // 0: softmax, 1: log_softmax
};

[[vk::push_constant]] SoftmaxParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

static const uint WG_SIZE = 256u;
static groupshared float s_max[WG_SIZE];
static groupshared float s_sum[WG_SIZE];

[shader("compute")] [numthreads(256, 1, 1)]
void softmax(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;

    float local_max = -3.402823466e+38;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        local_max = max(local_max, dataIn[row_offset + i]);
    }
    s_max[tid] = local_max;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_max[tid] = max(s_max[tid], s_max[tid + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = s_max[0];
    GroupMemoryBarrierWithGroupSync();

    float local_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        local_sum += exp(dataIn[row_offset + i] - row_max);
    }
    s_sum[tid] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_sum = s_sum[0];
    GroupMemoryBarrierWithGroupSync();

    float inv_sum = 1.0 / row_sum;
    float log_sum = log(row_sum);

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float val = dataIn[row_offset + i];
        if (pc.is_log_softmax != 0u) {
            dataOut[row_offset + i] = (val - row_max) - log_sum;
        } else {
            dataOut[row_offset + i] = exp(val - row_max) * inv_sum;
        }
    }
}
