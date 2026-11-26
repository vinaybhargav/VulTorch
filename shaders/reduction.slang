struct ReductionParams {
    uint num_rows;
    uint cols;
    uint op_type; // 0: Sum, 1: Max, 2: Min
};

[[vk::push_constant]] ReductionParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

static const uint WG_SIZE = 256u;
static groupshared float s_wave_acc[32]; // Up to 32 subgroups (min subgroup size 8)

float reduce_op(float a, float b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return max(a, b);
    if (op == 2u) return min(a, b);
    return a + b;
}

float init_val(uint op) {
    if (op == 0u) return 0.0;
    if (op == 1u) return -3.402823466e+38;
    if (op == 2u) return 3.402823466e+38;
    return 0.0;
}

float wave_reduce(float val, uint op) {
    if (op == 0u) return WaveActiveSum(val);
    if (op == 1u) return WaveActiveMax(val);
    if (op == 2u) return WaveActiveMin(val);
    return WaveActiveSum(val);
}

[shader("compute")] [numthreads(256, 1, 1)]
void reduction(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.y;
    if (row >= pc.num_rows) return;

    uint tid = lid.x;
    uint row_offset = row * pc.cols;

    float acc = init_val(pc.op_type);
    for (uint i = tid; i < pc.cols; i += WG_SIZE) {
        acc = reduce_op(acc, dataIn[row_offset + i], pc.op_type);
    }

    // Step 1: Intra-subgroup reduction using single-cycle register operations
    float wave_val = wave_reduce(acc, pc.op_type);

    uint wave_size = WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();
    uint subgroup_id = tid / wave_size;
    uint num_subgroups = (WG_SIZE + wave_size - 1u) / wave_size;

    if (lane_id == 0u) {
        s_wave_acc[subgroup_id] = wave_val;
    }
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Leader subgroup reduces results from all subgroups
    if (tid < wave_size) {
        float sub_val = (tid < num_subgroups) ? s_wave_acc[tid] : init_val(pc.op_type);
        float final_val = wave_reduce(sub_val, pc.op_type);
        if (tid == 0u) {
            dataOut[row] = final_val;
        }
    }
}
