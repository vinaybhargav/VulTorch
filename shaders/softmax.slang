struct SoftmaxParams {
    uint num_rows;
    uint cols;
    uint is_log_softmax; // 0: softmax, 1: log_softmax
};

[[vk::push_constant]] SoftmaxParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

static const uint WG_SIZE = 256u;
static groupshared float s_wave_max[32];
static groupshared float s_wave_sum[32];
static groupshared float s_row_max;
static groupshared float s_row_sum;

[shader("compute")] [numthreads(256, 1, 1)]
void softmax(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;
    uint wave_size = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = tid / wave_size;
    uint num_subgroups = (WG_SIZE + wave_size - 1u) / wave_size;

    // 1. Max reduction across columns using subgroup wave intrinsics
    float local_max = -3.402823466e+38;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        local_max = max(local_max, dataIn[row_offset + i]);
    }
    float wave_max = WaveActiveMax(local_max);
    if (lane_id == 0u) {
        s_wave_max[subgroup_id] = wave_max;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < wave_size) {
        float sub_m = (tid < num_subgroups) ? s_wave_max[tid] : -3.402823466e+38;
        float final_m = WaveActiveMax(sub_m);
        if (tid == 0u) {
            s_row_max = final_m;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float row_max = s_row_max;

    // 2. Sum reduction across exponentials using subgroup wave intrinsics
    float local_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        local_sum += exp(dataIn[row_offset + i] - row_max);
    }
    float wave_sum = WaveActiveSum(local_sum);
    if (lane_id == 0u) {
        s_wave_sum[subgroup_id] = wave_sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < wave_size) {
        float sub_s = (tid < num_subgroups) ? s_wave_sum[tid] : 0.0;
        float final_s = WaveActiveSum(sub_s);
        if (tid == 0u) {
            s_row_sum = final_s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float row_sum = s_row_sum;

    float inv_sum = 1.0 / row_sum;
    float log_sum = log(row_sum);

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float val = dataIn[row_offset + i];
        if (pc.is_log_softmax == 1u) {
            dataOut[row_offset + i] = (val - row_max) - log_sum;
        } else {
            dataOut[row_offset + i] = exp(val - row_max) * inv_sum;
        }
    }
}

// Backward pass for softmax and log_softmax
// grad_out: incoming gradient (binding 0)
// output: forward output tensor (softmax or log_softmax) (binding 1)
// grad_in: outgoing gradient (binding 2)
[shader("compute")] [numthreads(256, 1, 1)]
void softmax_backward(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;
    uint wave_size = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = tid / wave_size;
    uint num_subgroups = (WG_SIZE + wave_size - 1u) / wave_size;

    // For softmax: grad_in = y * (grad_out - sum(grad_out * y))
    // For log_softmax: grad_in = grad_out - exp(y) * sum(grad_out)
    float dot_or_sum = 0.0;
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float gy = dataIn[row_offset + i];
        if (pc.is_log_softmax == 1u) {
            dot_or_sum += gy;
        } else {
            float y = dataOut[row_offset + i];
            dot_or_sum += gy * y;
        }
    }
    float wave_dot = WaveActiveSum(dot_or_sum);
    if (lane_id == 0u) {
        s_wave_sum[subgroup_id] = wave_dot;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < wave_size) {
        float sub_d = (tid < num_subgroups) ? s_wave_sum[tid] : 0.0;
        float final_d = WaveActiveSum(sub_d);
        if (tid == 0u) {
            s_row_sum = final_d;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float total_dot = s_row_sum;

    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        float gy = dataIn[row_offset + i];
        float y = dataOut[row_offset + i];
        if (pc.is_log_softmax == 1u) {
            dataOut[row_offset + i] = gy - exp(y) * total_dot;
        } else {
            dataOut[row_offset + i] = y * (gy - total_dot);
        }
    }
}
