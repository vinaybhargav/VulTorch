// Slang Compile-Time Automatic Differentiation ([Differentiable]) Kernels
// Uses Slang's native `bwd_diff(...)` to derive exact backward SPIR-V kernels
// from single `[Differentiable]` forward functions.

struct AutodiffParams {
    uint total_elements;
    uint op_type; // Unary: 0=Mish, 1=HardSwish, 2=LeakyReLU, 3=ELU | Binary: 0=SmoothL1Loss
    float param;  // negative_slope / alpha / beta
    uint _pad;
};

[[vk::push_constant]] AutodiffParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> buf_a;
[[vk::binding(1, 0)]] StructuredBuffer<float> buf_b;
[[vk::binding(2, 0)]] StructuredBuffer<float> buf_grad_out;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> buf_out0;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> buf_out1;

[Differentiable]
float mish_fn(float x) {
    float sp = (x > 20.0) ? x : log(1.0 + exp(x));
    return x * tanh(sp);
}

[Differentiable]
float hardswish_fn(float x) {
    float c = clamp(x + 3.0, 0.0, 6.0);
    return x * c * (1.0 / 6.0);
}

[Differentiable]
float leaky_relu_fn(float x, no_diff float slope) {
    return (x >= 0.0) ? x : (slope * x);
}

[Differentiable]
float elu_fn(float x, no_diff float alpha) {
    return (x >= 0.0) ? x : (alpha * (exp(x) - 1.0));
}

[Differentiable]
float smooth_l1_fn(float pred, float target, no_diff float beta) {
    float diff = pred - target;
    float ad = abs(diff);
    if (ad < beta) {
        return 0.5 * diff * diff / beta;
    }
    return ad - 0.5 * beta;
}

[shader("compute")] [numthreads(256, 1, 1)]
void autodiff_unary_fwd(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    float x = buf_a[idx];
    float y = x;
    switch (pc.op_type) {
        case 0u: y = mish_fn(x); break;
        case 1u: y = hardswish_fn(x); break;
        case 2u: y = leaky_relu_fn(x, pc.param); break;
        case 3u: y = elu_fn(x, pc.param); break;
        default: break;
    }
    buf_out0[idx] = y;
}

[shader("compute")] [numthreads(256, 1, 1)]
void autodiff_unary_bwd(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    float x = buf_a[idx];
    float g = buf_grad_out[idx];
    DifferentialPair<float> dp_x = diffPair(x);
    switch (pc.op_type) {
        case 0u: bwd_diff(mish_fn)(dp_x, g); break;
        case 1u: bwd_diff(hardswish_fn)(dp_x, g); break;
        case 2u: bwd_diff(leaky_relu_fn)(dp_x, pc.param, g); break;
        case 3u: bwd_diff(elu_fn)(dp_x, pc.param, g); break;
        default: dp_x = diffPair(x, g); break;
    }
    buf_out0[idx] = dp_x.d;
}

[shader("compute")] [numthreads(256, 1, 1)]
void autodiff_binary_fwd(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    float a = buf_a[idx];
    float b = buf_b[idx];
    buf_out0[idx] = smooth_l1_fn(a, b, pc.param);
}

[shader("compute")] [numthreads(256, 1, 1)]
void autodiff_binary_bwd(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;
    float a = buf_a[idx];
    float b = buf_b[idx];
    float g = buf_grad_out[idx];
    DifferentialPair<float> dp_a = diffPair(a);
    DifferentialPair<float> dp_b = diffPair(b);
    bwd_diff(smooth_l1_fn)(dp_a, dp_b, pc.param, g);
    buf_out0[idx] = dp_a.d;
    buf_out1[idx] = dp_b.d;
}
