// Fused softmax cross-entropy for a 2-D logits tensor [rows, cols] with one
// integer class index per row (stored as float).
//
// mode 0: per-row loss   out[row]      = logsumexp(logits[row]) - logits[row][target]
// mode 1: gradient       out[row][col] = (softmax(logits[row])[col] - onehot) * row_scale
//
// In mode 1, `scale` (e.g. 1/N for mean reduction) and optional upstream
// `gradOut` (either scalar [1] or per-row [rows]) are fused directly into the
// write so no separate broadcast multiply or expand kernel is needed.

struct CrossEntropyParams {
    uint rows;
    uint cols;
    uint mode;
    uint has_grad_buf;
    uint g_is_scalar;
    float scale;
};

[[vk::push_constant]] CrossEntropyParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> logits;
[[vk::binding(1, 0)]] StructuredBuffer<float> targets;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dataOut;
[[vk::binding(3, 0)]] StructuredBuffer<float> gradOut;

static const uint WG_SIZE = 256u;
static groupshared float s_reduce[WG_SIZE];

float block_max(float value, uint tid) {
    s_reduce[tid] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_reduce[tid] = max(s_reduce[tid], s_reduce[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float result = s_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

float block_sum(float value, uint tid) {
    s_reduce[tid] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float result = s_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

[shader("compute")] [numthreads(256, 1, 1)]
void softmax_cross_entropy(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint row = gid.x;
    if (row >= pc.rows) return;

    uint tid = lid.x;
    uint base = row * pc.cols;
    uint target = uint(targets[row]);

    float local_max = -3.402823466e+38;
    for (uint j = tid; j < pc.cols; j += WG_SIZE) {
        local_max = max(local_max, logits[base + j]);
    }
    float row_max = block_max(local_max, tid);

    float local_sum = 0.0;
    for (uint j = tid; j < pc.cols; j += WG_SIZE) {
        local_sum += exp(logits[base + j] - row_max);
    }
    float row_sum = block_sum(local_sum, tid);

    if (pc.mode == 0u) {
        if (tid == 0u) {
            float target_logit = logits[base + target];
            dataOut[row] = (row_max + log(row_sum)) - target_logit;
        }
        return;
    }

    float row_scale = pc.scale;
    if (pc.has_grad_buf != 0u) {
        row_scale *= gradOut[(pc.g_is_scalar != 0u) ? 0u : row];
    }

    float inv_sum = 1.0 / row_sum;
    for (uint j = tid; j < pc.cols; j += WG_SIZE) {
        float probability = exp(logits[base + j] - row_max) * inv_sum;
        float onehot = (j == target) ? 1.0 : 0.0;
        dataOut[base + j] = (probability - onehot) * row_scale;
    }
}
