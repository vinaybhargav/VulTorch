struct UnaryScalarParams {
    uint total_elements;
    uint op_type;
    float param1;
    float param2;
};

[[vk::push_constant]] UnaryScalarParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

static const float SQRT_2_OVER_PI = 0.7978845608;
static const float GELU_COEFF = 0.044715;

float fast_gelu_scalar(float x) {
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

float apply_op_scalar(float x, uint op) {
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
        case 9u: return 1.0 / (1.0 + exp(-x));
        case 10u: return max(x, 0.0);
        case 11u: return fast_gelu_scalar(x);
        case 12u: return x / (1.0 + exp(-x));
        case 13u: return clamp(x, pc.param1, pc.param2);
        case 14u: return float(sign(x));
        case 15u: return x * pc.param1 + pc.param2;
        default: return x;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void elementwise_unary_scalar(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    dataOut[idx] = apply_op_scalar(dataIn[idx], pc.op_type);
}
