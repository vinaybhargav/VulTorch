// Stateless counter-based Philox-4x32-7 PRNG for GPU uniform, normal, and
// zero-mask-memory Dropout forward/backward.

struct RandomParams {
    uint numel;
    uint seed_lo;
    uint seed_hi;
    uint offset;
    float p1;     // low (uniform), mean (normal), p (dropout)
    float p2;     // high (uniform), std (normal), inv_keep = 1/(1-p) (dropout)
    uint mode;    // 0: uniform, 1: normal, 2: dropout
};

[[vk::push_constant]] RandomParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

void mulhilo32(uint a, uint b, out uint hi, out uint lo) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    lo = uint(prod);
    hi = uint(prod >> 32u);
}

uint4 philox4x32_7(uint4 ctr, uint2 key) {
    const uint M0 = 0xD2511F53u;
    const uint M1 = 0xCD9E8D57u;
    const uint W0 = 0x9E3779B9u;
    const uint W1 = 0xBB67AE85u;

    [ForceUnroll] for (int r = 0; r < 7; ++r) {
        uint hi0, lo0, hi1, lo1;
        mulhilo32(M0, ctr.x, hi0, lo0);
        mulhilo32(M1, ctr.z, hi1, lo1);
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += W0;
        key.y += W1;
    }
    return ctr;
}

float uint_to_uniform01(uint u) {
    // Maps uint32 to (0, 1) strictly inside open interval for Box-Muller log() safety.
    return (float(u >> 8u) + 0.5) * (1.0 / 16777216.0);
}

[shader("compute")] [numthreads(256, 1, 1)]
void philox_rng(uint3 tid : SV_DispatchThreadID) {
    uint base_idx = tid.x * 4u;
    if (base_idx >= pc.numel) return;

    uint4 ctr = uint4(tid.x, pc.offset, 0u, 0u);
    uint2 key = uint2(pc.seed_lo, pc.seed_hi);
    uint4 r = philox4x32_7(ctr, key);

    float u0 = uint_to_uniform01(r.x);
    float u1 = uint_to_uniform01(r.y);
    float u2 = uint_to_uniform01(r.z);
    float u3 = uint_to_uniform01(r.w);

    float vals[4];
    if (pc.mode == 0u) {
        float span = pc.p2 - pc.p1;
        vals[0] = pc.p1 + u0 * span;
        vals[1] = pc.p1 + u1 * span;
        vals[2] = pc.p1 + u2 * span;
        vals[3] = pc.p1 + u3 * span;
    } else if (pc.mode == 1u) {
        const float TWO_PI = 6.28318530718;
        float rad0 = sqrt(-2.0 * log(u0));
        float rad1 = sqrt(-2.0 * log(u2));
        vals[0] = pc.p1 + pc.p2 * (rad0 * cos(TWO_PI * u1));
        vals[1] = pc.p1 + pc.p2 * (rad0 * sin(TWO_PI * u1));
        vals[2] = pc.p1 + pc.p2 * (rad1 * cos(TWO_PI * u3));
        vals[3] = pc.p1 + pc.p2 * (rad1 * sin(TWO_PI * u3));
    } else {
        float us[4] = { u0, u1, u2, u3 };
        [ForceUnroll] for (uint k = 0; k < 4u; ++k) {
            uint i = base_idx + k;
            if (i < pc.numel) {
                vals[k] = (us[k] >= pc.p1) ? (dataIn[i] * pc.p2) : 0.0;
            } else {
                vals[k] = 0.0;
            }
        }
    }

    [ForceUnroll] for (uint k = 0; k < 4u; ++k) {
        uint i = base_idx + k;
        if (i < pc.numel) {
            dataOut[i] = vals[k];
        }
    }
}
