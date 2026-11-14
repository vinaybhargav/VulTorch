// Fused in-place SGD optimizer step (with weight decay, momentum, dampening, Nesterov).

struct SgdParams {
    uint numel;
    uint groups_x;
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    uint nesterov;
    uint first_step;
};

[[vk::push_constant]] SgdParams pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> param;
[[vk::binding(1, 0)]] StructuredBuffer<float> grad;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> momentum_buf;

[shader("compute")] [numthreads(256, 1, 1)]
void sgd_step(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.y * (pc.groups_x * 256u) + tid.x;
    if (i >= pc.numel) return;

    float p = param[i];
    float g = grad[i];

    if (pc.weight_decay != 0.0) {
        g += pc.weight_decay * p;
    }

    float d_p = g;
    if (pc.momentum != 0.0) {
        float buf;
        if (pc.first_step != 0u) {
            buf = g;
        } else {
            buf = pc.momentum * momentum_buf[i] + (1.0 - pc.dampening) * g;
        }
        momentum_buf[i] = buf;

        if (pc.nesterov != 0u) {
            d_p = g + pc.momentum * buf;
        } else {
            d_p = buf;
        }
    }

    param[i] = p - pc.lr * d_p;
}
