struct UnaryParams {
    uint num_vec4;
    uint op_type;
    float param1;
    float param2;
};

[[vk::push_constant]] UnaryParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> dataOut;

static const float SQRT_2_OVER_PI = 0.7978845608;
static const float GELU_COEFF = 0.044715;

float4 fast_gelu(float4 x) {
    float4 inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    return 0.5 * x * (float4(1.0) + tanh(inner));
}

float4 apply_op_vec4(float4 x, uint op) {
    switch (op) {
        case 0u: return -x;
        case 1u: return exp(x);
        case 2u: return log(x);
        case 3u: return sqrt(x);
        case 4u: return rsqrt(x);
        case 5u: return abs(x);
        case 6u: return sin(x);
        case 7u: return cos(x);
        case 8u: return tanh(x);
        case 9u: return float4(1.0) / (float4(1.0) + exp(-x));
        case 10u: return max(x, float4(0.0));
        case 11u: return fast_gelu(x);
        case 12u: return x / (float4(1.0) + exp(-x));
        case 13u: return clamp(x, float4(pc.param1), float4(pc.param2));
        case 14u: return float4(sign(x));
        case 15u: return x * pc.param1 + pc.param2;
        default: return x;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void elementwise_unary_vec4(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.num_vec4) return;
    dataOut[idx] = apply_op_vec4(dataIn[idx], pc.op_type);
}
