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

float erf_s(float x) {
    float s = (x < 0.0) ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-ax * ax));
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
        case 16u: return round(x);
        case 17u: return ceil(x);
        case 18u: return floor(x);
        case 19u: return trunc(x);
        case 20u: return x - trunc(x);
        case 21u: return x * x;
        case 22u: return 1.0 / x;
        case 23u: return exp2(x);
        case 24u: return exp(x) - 1.0;
        case 25u: return log2(x);
        case 26u: return log(x) * 0.4342944819032518;
        case 27u: return log(1.0 + x);
        case 28u: return tan(x);
        case 29u: return asin(x);
        case 30u: return acos(x);
        case 31u: return atan(x);
        case 32u: return sinh(x);
        case 33u: return cosh(x);
        case 34u: return log(x + sqrt(x * x + 1.0));
        case 35u: return log(x + sqrt(x * x - 1.0));
        case 36u: return 0.5 * log((1.0 + x) / (1.0 - x));
        case 37u: return erf_s(x);
        case 38u: return 1.0 - erf_s(x);
        case 39u: return isnan(x) ? 1.0 : 0.0;
        case 40u: return isinf(x) ? 1.0 : 0.0;
        case 41u: return isfinite(x) ? 1.0 : 0.0;
        default: return x;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void elementwise_unary_scalar(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    dataOut[idx] = apply_op_scalar(dataIn[idx], pc.op_type);
}
