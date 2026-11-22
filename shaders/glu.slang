// Fused Gated Linear Units (SwiGLU & GeGLU — Shazeer 2020) Forward & Backward Kernels.
//
// Mode 0 (SwiGLU):
//   gate(a) = SiLU(a) = a * sigmoid(a)
//   y = gate(a) * b
// Mode 1 (GeGLU):
//   gate(a) = GELU(a) = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
//   y = gate(a) * b

struct GLUParams {
    uint total_elements;
    uint mode; // 0 = SwiGLU, 1 = GeGLU
};

[[vk::binding(0, 0)]] StructuredBuffer<float> gluIn0;  // a (fwd) or grad_out (bwd)
[[vk::binding(1, 0)]] StructuredBuffer<float> gluIn1;  // b (fwd) or a (bwd)
[[vk::binding(2, 0)]] StructuredBuffer<float> gluIn2;  // b (bwd)
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gluOut0; // y (fwd) or da (bwd)
[[vk::binding(4, 0)]] RWStructuredBuffer<float> gluOut1; // db (bwd)

void eval_gate(uint mode, float a, out float g, out float dg_da) {
    if (mode == 0u) {
        // SiLU(a) = a * sig(a)
        float sig = 1.0 / (1.0 + exp(-a));
        g = a * sig;
        dg_da = sig * (1.0 + a * (1.0 - sig));
    } else {
        // Tanh-approx GELU(a)
        const float kAlpha = 0.7978845608028654; // sqrt(2 / pi)
        const float kBeta = 0.044715;
        float a2 = a * a;
        float inner = kAlpha * (a + kBeta * a * a2);
        float th = tanh(inner);
        float sech2 = 1.0 - th * th;
        float dinner = kAlpha * (1.0 + 3.0 * kBeta * a2);
        g = 0.5 * a * (1.0 + th);
        dg_da = 0.5 * (1.0 + th) + 0.5 * a * sech2 * dinner;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void glu_forward(uint3 tid : SV_DispatchThreadID, uniform GLUParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    float a = gluIn0[idx];
    float b = gluIn1[idx];
    float g, dg;
    eval_gate(pc.mode, a, g, dg);
    gluOut0[idx] = g * b;
}

[shader("compute")] [numthreads(256, 1, 1)]
void glu_backward(uint3 tid : SV_DispatchThreadID, uniform GLUParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    float dy = gluIn0[idx];
    float a = gluIn1[idx];
    float b = gluIn2[idx];
    float g, dg;
    eval_gate(pc.mode, a, g, dg);
    gluOut0[idx] = dy * b * dg;
    gluOut1[idx] = dy * g;
}
