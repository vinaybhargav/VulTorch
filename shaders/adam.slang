// Fused Adam / AdamW update, in place.
//
// The composed-op version costs ~12 elementwise dispatches per parameter, each
// a full read/write pass over tensors the size of the weights. On a
// bandwidth-poor GPU that is the whole training step. Here every tensor is read
// once and written once:  traffic = 4 reads + 3 writes per element, one launch.
//
//   decoupled == 0 (Adam):  g += wd * p                 (L2 regularisation)
//   decoupled == 1 (AdamW): p *= 1 - lr * wd            (decoupled decay)
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g*g
//   p -= (lr / bias_c1) * m / (sqrt(v / bias_c2) + eps)

struct AdamParams {
    uint numel;
    uint groups_x;     // workgroups along x, to linearise a 2-D dispatch
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    uint decoupled;
    float bias_c1;     // 1 - beta1^t
    float bias_c2;     // 1 - beta2^t
};

[[vk::push_constant]] ConstantBuffer<AdamParams> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> param;
[[vk::binding(1, 0)]] StructuredBuffer<float> grad;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exp_avg;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> exp_avg_sq;

[shader("compute")] [numthreads(256, 1, 1)]
void adam_step(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint i = (gid.y * pc.groups_x + gid.x) * 256 + lid.x;
    if (i >= pc.numel) return;

    float p = param[i];
    float g = grad[i];

    if (pc.decoupled != 0)
        p *= 1.0 - pc.lr * pc.weight_decay;
    else
        g += pc.weight_decay * p;

    const float m = pc.beta1 * exp_avg[i] + (1.0 - pc.beta1) * g;
    const float v = pc.beta2 * exp_avg_sq[i] + (1.0 - pc.beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;

    const float denom = sqrt(v / pc.bias_c2) + pc.eps;
    param[i] = p - (pc.lr / pc.bias_c1) * m / denom;
}
