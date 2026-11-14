struct BinaryBroadcastParams {
    uint total_elements;
    uint op_type;    // 0: Add, 1: Sub, 2: Mul, 3: Div, 4: Pow, 5: Max, 6: Min, 7: GT, 8: LT, 9: EQ
    uint ndim;
    uint is_b_scalar;
    float scalar_b;
    uint shape[6];
    uint strideA[6];
    uint strideB[6];
};

[[vk::push_constant]] BinaryBroadcastParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<float> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> dataC;

float safe_pow_scalar(float a, float b) {
    float p = pow(abs(a), b);
    if (fmod(floor(abs(b)), 2.0) == 1.0) {
        return p * float(sign(a));
    }
    return p;
}

float apply_op_scalar(float a, float b, uint op) {
    switch (op) {
        case 0u: return a + b;
        case 1u: return a - b;
        case 2u: return a * b;
        case 3u: return a / b;
        case 4u: return safe_pow_scalar(a, b);
        case 5u: return max(a, b);
        case 6u: return min(a, b);
        case 7u: return (a > b) ? 1.0 : 0.0;
        case 8u: return (a < b) ? 1.0 : 0.0;
        case 9u: return (a == b) ? 1.0 : 0.0;
        default: return a + b;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void elementwise_binary_broadcast(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    if (pc.is_b_scalar != 0u) {
        dataC[idx] = apply_op_scalar(dataA[idx], pc.scalar_b, pc.op_type);
        return;
    }

    if (pc.ndim == 0u) {
        dataC[idx] = apply_op_scalar(dataA[idx], dataB[idx], pc.op_type);
        return;
    }

    uint rem = idx;
    uint idxA = 0u;
    uint idxB = 0u;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;

        idxA += coord * pc.strideA[d];
        idxB += coord * pc.strideB[d];
    }

    dataC[idx] = apply_op_scalar(dataA[idxA], dataB[idxB], pc.op_type);
}
