struct ReductionParams {
    uint num_rows;
    uint cols;
    uint op_type; // 0: Sum, 1: Max, 2: Min
};

[[vk::push_constant]] ReductionParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

static const uint WG_SIZE = 256u;
static groupshared float s_data[WG_SIZE];

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

    s_data[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_data[tid] = reduce_op(s_data[tid], s_data[tid + s], pc.op_type);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        dataOut[row] = s_data[0];
    }
}
