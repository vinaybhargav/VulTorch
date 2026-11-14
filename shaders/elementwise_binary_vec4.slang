struct BinaryVec4Params {
    uint num_vec4;
    uint op_type;    // 0: Add, 1: Sub, 2: Mul, 3: Div, 4: Pow, 5: Max, 6: Min, 7: GT, 8: LT, 9: EQ
    float scalar_b;
    uint is_b_scalar;
};

[[vk::push_constant]] BinaryVec4Params pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> dataA;
[[vk::binding(1, 0)]] StructuredBuffer<float4> dataB;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> dataC;

float safe_pow_elem(float a, float b) {
    float p = pow(abs(a), b);
    float rem = fmod(floor(abs(b)), 2.0);
    return (rem == 1.0) ? p * float(sign(a)) : p;
}

float4 safe_pow(float4 a, float4 b) {
    return float4(
        safe_pow_elem(a.x, b.x),
        safe_pow_elem(a.y, b.y),
        safe_pow_elem(a.z, b.z),
        safe_pow_elem(a.w, b.w)
    );
}

float4 apply_op(float4 a, float4 b, uint op) {
    switch (op) {
        case 0u: return a + b;
        case 1u: return a - b;
        case 2u: return a * b;
        case 3u: return a / b;
        case 4u: return safe_pow(a, b);
        case 5u: return max(a, b);
        case 6u: return min(a, b);
        case 7u: return select(a > b, float4(1.0), float4(0.0));
        case 8u: return select(a < b, float4(1.0), float4(0.0));
        case 9u: return select(a == b, float4(1.0), float4(0.0));
        default: return a + b;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void elementwise_binary_vec4(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.num_vec4) return;

    float4 a = dataA[idx];
    float4 b = (pc.is_b_scalar != 0u) ? float4(pc.scalar_b) : dataB[idx];
    dataC[idx] = apply_op(a, b, pc.op_type);
}
