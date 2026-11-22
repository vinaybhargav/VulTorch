// Fused Rotary Position Embedding (RoPE — Su et al. 2021) Forward & Backward Kernels.
//
// Applies complex 2D rotation to half-dim pairs (x[d], x[d + D/2]) for d in 0 .. D/2 - 1:
//   Forward:
//     out[d]       = x[d] * cos[s, d] - x[d + D/2] * sin[s, d]
//     out[d + D/2] = x[d + D/2] * cos[s, d] + x[d] * sin[s, d]
//   Backward (transpose of orthogonal rotation matrix R(theta)^T = R(-theta)):
//     dx[d]       = dy[d] * cos[s, d] + dy[d + D/2] * sin[s, d]
//     dx[d + D/2] = dy[d + D/2] * cos[s, d] - dy[d] * sin[s, d]
//
// Input shape: [B, H, S, D] (or [B, S, H, D] when seq_dim == 1), cos/sin shape: [S, D/2] or [S, D].

struct RoPEParams {
    uint total_pairs; // B * H * S * (D / 2)
    uint H;
    uint S;
    uint half_D;      // D / 2
    uint cos_stride_s; // half_D if cos is [S, D/2], or D if cos is [S, D]
    uint seq_dim;     // 2 for [B, H, S, D], 1 for [B, S, H, D]
    uint is_backward; // 0 = forward, 1 = backward
};

[[vk::binding(0, 0)]] StructuredBuffer<float> ropeIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> ropeCos;
[[vk::binding(2, 0)]] StructuredBuffer<float> ropeSin;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> ropeOut;

[shader("compute")] [numthreads(256, 1, 1)]
void rope_kernel(uint3 tid : SV_DispatchThreadID, uniform RoPEParams pc) {
    uint pair_idx = tid.x;
    if (pair_idx >= pc.total_pairs) return;

    uint d = pair_idx % pc.half_D;
    uint rem = pair_idx / pc.half_D;
    uint s, h, b;
    if (pc.seq_dim == 2u) {
        // Layout: [B, H, S, D]
        s = rem % pc.S;
        uint bh = rem / pc.S;
        h = bh % pc.H;
        b = bh / pc.H;
    } else {
        // Layout: [B, S, H, D]
        h = rem % pc.H;
        uint bs = rem / pc.H;
        s = bs % pc.S;
        b = bs / pc.S;
    }

    uint D = pc.half_D * 2u;
    uint row_offset = rem * D;
    uint idx0 = row_offset + d;
    uint idx1 = row_offset + d + pc.half_D;

    uint cs_idx = s * pc.cos_stride_s + d;
    float c = ropeCos[cs_idx];
    float sn = ropeSin[cs_idx];

    float x0 = ropeIn[idx0];
    float x1 = ropeIn[idx1];

    if (pc.is_backward == 0u) {
        ropeOut[idx0] = x0 * c - x1 * sn;
        ropeOut[idx1] = x1 * c + x0 * sn;
    } else {
        ropeOut[idx0] = x0 * c + x1 * sn;
        ropeOut[idx1] = x1 * c - x0 * sn;
    }
}
