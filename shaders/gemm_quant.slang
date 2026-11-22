// Weight-Only int8 & int4 Group-Wise Quantized Linear / GEMM Kernel (`nn.QuantizedLinear`)
// and GPU-side AMP inf/nan gradient check & unscale kernel (`vultorch.amp.GradScaler`).
//
// Quantized GEMM computes:
//   Y[m, n] = sum_{k=0..K-1} X[m, k] * dequant(W_q[n, k]) + (has_bias ? Bias[n] : 0)
// where W is stored row-major as [N, K] (matching nn.Linear weight [out_features, in_features]):
//   - bits == 8: 4 signed int8 weights packed per uint32 word (little-endian bytes 0..3)
//                w = float(int8(byte)) * scale[n, k / group_size]
//   - bits == 4: 8 unsigned 4-bit nibbles (0..15, symmetric zero-point 8) packed per uint32 word
//                w = float(int(nibble) - 8) * scale[n, k / group_size]

struct QuantGemmParams {
    uint M;
    uint N;
    uint K;
    uint group_size;
    uint bits;      // 4 or 8
    uint has_bias;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> qX;       // [M, K]
[[vk::binding(1, 0)]] StructuredBuffer<uint> qW;        // packed [N, K / (32 / bits)]
[[vk::binding(2, 0)]] StructuredBuffer<float> qScale;   // [N, K / group_size]
[[vk::binding(3, 0)]] StructuredBuffer<float> qBias;    // [N]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> qY;     // [M, N]
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> ampFlag; // [1] (used by amp_unscale_check)

[shader("compute")] [numthreads(16, 16, 1)]
void gemm_quant_kernel(uint3 tid : SV_DispatchThreadID, uniform QuantGemmParams pc) {
    uint n = tid.x;
    uint m = tid.y;
    if (m >= pc.M || n >= pc.N) return;

    uint K = pc.K;
    uint gs = pc.group_size;
    uint num_groups = (K + gs - 1u) / gs;
    uint x_row = m * K;
    uint scale_row = n * num_groups;

    float acc = (pc.has_bias != 0u) ? qBias[n] : 0.0;

    if (pc.bits == 4u) {
        uint words_per_row = K / 8u;
        uint w_row = n * words_per_row;
        for (uint w_idx = 0u; w_idx < words_per_row; ++w_idx) {
            uint k_base = w_idx * 8u;
            float sc = qScale[scale_row + (k_base / gs)];
            uint packed = qW[w_row + w_idx];
            [ForceUnroll]
            for (uint j = 0u; j < 8u; ++j) {
                int nib = int((packed >> (j * 4u)) & 0xFu) - 8;
                float w_val = float(nib) * sc;
                acc = fma(qX[x_row + k_base + j], w_val, acc);
            }
        }
    } else {
        uint words_per_row = K / 4u;
        uint w_row = n * words_per_row;
        for (uint w_idx = 0u; w_idx < words_per_row; ++w_idx) {
            uint k_base = w_idx * 4u;
            float sc = qScale[scale_row + (k_base / gs)];
            uint packed = qW[w_row + w_idx];
            [ForceUnroll]
            for (uint j = 0u; j < 4u; ++j) {
                uint byte_val = (packed >> (j * 8u)) & 0xFFu;
                int s8 = (byte_val >= 128u) ? (int(byte_val) - 256) : int(byte_val);
                float w_val = float(s8) * sc;
                acc = fma(qX[x_row + k_base + j], w_val, acc);
            }
        }
    }

    qY[m * pc.N + n] = acc;
}

struct QuantizePackParams {
    uint N;
    uint K;
    uint group_size;
    uint bits;
};

// Quantizes FP32 weight matrix qX [N, K] into packed uint32 qW [N, K / (32/bits)]
// and per-group FP32 scales qY [N, K / group_size].
// Dispatched with 1 thread per (n, group_idx) or 1 thread per word (n, w_idx).
[shader("compute")] [numthreads(16, 16, 1)]
void quantize_pack_weights(uint3 tid : SV_DispatchThreadID, uniform QuantizePackParams pc) {
    uint w_idx = tid.x;
    uint n = tid.y;
    uint pack_factor = (pc.bits == 4u) ? 8u : 4u;
    uint words_per_row = pc.K / pack_factor;
    if (n >= pc.N || w_idx >= words_per_row) return;

    uint gs = pc.group_size;
    uint num_groups = (pc.K + gs - 1u) / gs;
    uint k_base = w_idx * pack_factor;
    uint g_idx = k_base / gs;
    uint g_start = g_idx * gs;
    uint g_end = min(g_start + gs, pc.K);

    float max_abs = 0.0;
    uint row_off = n * pc.K;
    for (uint k = g_start; k < g_end; ++k) {
        max_abs = max(max_abs, abs(qX[row_off + k]));
    }

    float max_q = (pc.bits == 4u) ? 7.0 : 127.0;
    float sc = max(max_abs / max_q, 1e-8);
    if ((k_base % gs) == 0u) {
        qY[n * num_groups + g_idx] = sc;
    }

    uint packed = 0u;
    if (pc.bits == 4u) {
        for (uint j = 0u; j < 8u; ++j) {
            float v = qX[row_off + k_base + j] / sc;
            int qv = clamp(int(round(v)), -7, 7) + 8;
            packed |= (uint(qv) & 0xFu) << (j * 4u);
        }
    } else {
        for (uint j = 0u; j < 4u; ++j) {
            float v = qX[row_off + k_base + j] / sc;
            int qv = clamp(int(round(v)), -127, 127);
            uint u8 = uint(qv) & 0xFFu;
            packed |= u8 << (j * 8u);
        }
    }
    ampFlag[n * words_per_row + w_idx] = packed;
}

struct AmpCheckParams {
    uint total_elements;
    float inv_scale;
};

// Unscales gradient buffer qY in-place by `inv_scale` and sets ampFlag[0] = 1 if any element is NaN or Inf.
[shader("compute")] [numthreads(256, 1, 1)]
void amp_unscale_check(uint3 tid : SV_DispatchThreadID, uniform AmpCheckParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    float v = qY[idx] * pc.inv_scale;
    qY[idx] = v;
    uint bits = asuint(v);
    bool is_nan_or_inf = ((bits & 0x7F800000u) == 0x7F800000u);
    if (is_nan_or_inf) {
        uint old_val;
        InterlockedOr(ampFlag[0], 1u, old_val);
    }
}
