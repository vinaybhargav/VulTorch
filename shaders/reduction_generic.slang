// Single-axis reductions over contiguous [outer, dim_size, inner] tensors:
//   out[o, i] = reduce_d( in[o, d, i] )
//
// Two entry points share the same push constants and bindings:
//   * reduction_generic: 1 workgroup of 128 threads per (o, i), reducing along d.
//     Best when `inner` is small and `outer * inner` is modest.
//   * reduction_strided: 32 x 8 tiled workgroup where `tx` (0..31) maps to
//     contiguous columns `i` (100% coalesced warp loads and stores) and `ty`
//     (0..7) steps along `d` and reduces in shared memory with a single barrier.
//     Used when `inner >= 16` (e.g. bias gradients `grad_out.sum(dim=0)`).

struct GenericReductionParams {
    uint outer;
    uint dim_size;
    uint inner;
    uint op_type; // 0: Sum, 1: Max, 2: Min
};

[[vk::push_constant]] GenericReductionParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

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

static groupshared float s_data[128];

[shader("compute")] [numthreads(128, 1, 1)]
void reduction_generic(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint out_idx = gid.x;
    uint total_out = pc.outer * pc.inner;
    if (out_idx >= total_out) return;

    uint o = out_idx / pc.inner;
    uint i = out_idx % pc.inner;
    uint base = o * pc.dim_size * pc.inner + i;

    uint tid = lid.x;

    float acc = init_val(pc.op_type);
    for (uint d = tid; d < pc.dim_size; d += 128u) {
        acc = reduce_op(acc, dataIn[base + d * pc.inner], pc.op_type);
    }

    s_data[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_data[tid] = reduce_op(s_data[tid], s_data[tid + stride], pc.op_type);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        dataOut[out_idx] = s_data[0];
    }
}

static groupshared float s_tile[8][33];

[shader("compute")] [numthreads(32, 8, 1)]
void reduction_strided(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint tx = lid.x; // 0..31 along inner
    const uint ty = lid.y; // 0..7 along dim_size
    const uint i = gid.x * 32u + tx;
    const uint o = gid.y;

    float acc = init_val(pc.op_type);
    if (o < pc.outer && i < pc.inner) {
        const uint base = o * pc.dim_size * pc.inner + i;
        for (uint d = ty; d < pc.dim_size; d += 8u) {
            acc = reduce_op(acc, dataIn[base + d * pc.inner], pc.op_type);
        }
    }

    s_tile[ty][tx] = acc;
    GroupMemoryBarrierWithGroupSync();

    if (ty == 0u && o < pc.outer && i < pc.inner) {
        float res = s_tile[0][tx];
        [ForceUnroll] for (uint k = 1u; k < 8u; ++k) {
            res = reduce_op(res, s_tile[k][tx], pc.op_type);
        }
        dataOut[o * pc.inner + i] = res;
    }
}
