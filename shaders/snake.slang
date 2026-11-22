// Fused Snake Periodic Non-Linear Activation Family (Ziyin et al. 2020; BigVGAN / DAC; SnakeGamma / SnakeCos / SnakeSinc).
//
// Mode 0 (Snake / SnakeBeta):
//   y = x + (1 / b) * sin^2(a * x)
// Mode 1 (SnakeGamma — phase-shifted periodic activation):
//   y = x + (1 / b) * sin^2(a * x + gamma)
// Mode 2 (SnakeCos — complimentary cosine-squared periodic activation):
//   y = x + (1 / b) * cos^2(a * x)
// Mode 3 (SnakeRational / damped periodic activation):
//   y = x + (1 / b) * sin^2(a * x) / (1 + (a * x)^2)
// Mode 4 (SnakeSinc — unnormalized sinc^2 wavepacket activation):
//   y = x + (1 / b) * (sin(a * x) / (a * x))^2   (limit a*x -> 0 is x + 1/b)
//
// Supports scalar frequency `a` as well as per-channel trainable `alpha`, `beta`, and `gamma`
// in either linear scale (a = alpha, b = beta) or log-scale (a = exp(alpha), b = exp(beta)).

struct SnakeParams {
    uint total_elements; // outer_size * channels * inner_size
    uint channels;       // C
    uint inner_size;     // L (elements per channel slice; 1 if channel is last dim)
    uint has_beta;       // 0 = b = a, 1 = independent beta
    uint log_scale;      // 0 = linear parameter, 1 = exp(alpha), exp(beta)
    uint is_scalar_a;    // 1 = use scalar_a, 0 = read alphaBuf[c]
    uint mode;           // 0 = Snake/SnakeBeta, 1 = SnakeGamma, 2 = SnakeCos, 3 = SnakeRational, 4 = SnakeSinc
    uint has_gamma;      // 1 = read phase shift gamma from snIn3 (forward) or pack after beta
    float scalar_a;
    float scalar_gamma;
    float eps;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> snIn0;
[[vk::binding(1, 0)]] StructuredBuffer<float> snIn1;
[[vk::binding(2, 0)]] StructuredBuffer<float> snIn2;
[[vk::binding(3, 0)]] StructuredBuffer<float> snIn3;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> snOut0;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> snOut1;

groupshared float sn_s0[256];
groupshared float sn_s1[256];
groupshared float sn_s2[256];

void load_abg(uint c, uniform SnakeParams pc, out float a, out float b, out float g_phase) {
    if (pc.is_scalar_a != 0u) {
        a = pc.scalar_a;
        b = pc.scalar_a;
        g_phase = pc.scalar_gamma;
        return;
    }
    float raw_a = snIn1[c];
    a = (pc.log_scale != 0u) ? exp(raw_a) : raw_a;
    if (pc.has_beta != 0u) {
        float raw_b = snIn2[c];
        b = (pc.log_scale != 0u) ? exp(raw_b) : raw_b;
    } else {
        b = a;
    }
    if (pc.has_gamma != 0u) {
        // When gamma is per-channel, it is stored in snIn2 at offset pc.channels (or snIn2[c] if has_beta==0)
        uint g_idx = (pc.has_beta != 0u) ? (pc.channels + c) : c;
        g_phase = snIn2[g_idx];
    } else {
        g_phase = pc.scalar_gamma;
    }
}

float safe_inv(float b, float eps) {
    float sign_b = (b >= 0.0) ? 1.0 : -1.0;
    float abs_b = abs(b);
    return (abs_b < eps) ? (sign_b / eps) : (1.0 / b);
}

// Evaluates periodic basis P(u) and derivative dP/du where y = x + (1/b) * P(u), u = a * x + g_phase
void eval_periodic_basis(uint mode, float ax, float g_phase, out float P, out float dP_du) {
    if (mode == 0u || mode == 1u) {
        float u = ax + ((mode == 1u) ? g_phase : 0.0);
        float s = sin(u);
        P = s * s;
        dP_du = sin(2.0 * u);
    } else if (mode == 2u) {
        // SnakeCos: cos^2(a*x)
        float c = cos(ax);
        P = c * c;
        dP_du = -sin(2.0 * ax);
    } else if (mode == 3u) {
        // SnakeRational: sin^2(u) / (1 + u^2)
        float u = ax;
        float s = sin(u);
        float s2 = s * s;
        float denom = 1.0 + u * u;
        float inv_d = 1.0 / denom;
        P = s2 * inv_d;
        dP_du = (sin(2.0 * u) * denom - 2.0 * u * s2) * (inv_d * inv_d);
    } else {
        // SnakeSinc: (sin(u) / u)^2 with Taylor expansion near u = 0:
        // sinc(u) ~ 1 - u^2/6 + u^4/120 => sinc^2(u) ~ 1 - u^2/3 + 2*u^4/45
        float u = ax;
        float u2 = u * u;
        if (u2 < 1e-4) {
            P = 1.0 - u2 / 3.0 + (2.0 * u2 * u2) / 45.0;
            dP_du = (-2.0 / 3.0) * u + (8.0 / 45.0) * (u * u2);
        } else {
            float s = sin(u);
            float c = cos(u);
            float inv_u = 1.0 / u;
            float sinc_u = s * inv_u;
            P = sinc_u * sinc_u;
            // d/du (sin(u)/u)^2 = 2 * (sin(u)/u) * (u*cos(u) - sin(u)) / u^2
            dP_du = 2.0 * sinc_u * (u * c - s) * (inv_u * inv_u);
        }
    }
}

// Forward:
// snIn0 = x, snIn1 = alpha, snIn2 = beta (and optional gamma at +C), snOut0 = y
[shader("compute")] [numthreads(256, 1, 1)]
void snake_forward(uint3 tid : SV_DispatchThreadID, uniform SnakeParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint c = (idx / pc.inner_size) % pc.channels;
    float a, b, g_phase;
    load_abg(c, pc, a, b, g_phase);

    float x = snIn0[idx];
    float P, dP_du;
    eval_periodic_basis(pc.mode, a * x, g_phase, P, dP_du);
    float inv_b = safe_inv(b, pc.eps);
    snOut0[idx] = x + P * inv_b;
}

// Backward dx:
// snIn0 = grad_out, snIn1 = alpha, snIn2 = beta (+gamma), snIn3 = x, snOut0 = dx
[shader("compute")] [numthreads(256, 1, 1)]
void snake_backward_dx(uint3 tid : SV_DispatchThreadID, uniform SnakeParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint c = (idx / pc.inner_size) % pc.channels;
    float a, b, g_phase;
    load_abg(c, pc, a, b, g_phase);

    float x = snIn3[idx];
    float dy = snIn0[idx];
    float P, dP_du;
    eval_periodic_basis(pc.mode, a * x, g_phase, P, dP_du);
    float inv_b = safe_inv(b, pc.eps);
    snOut0[idx] = dy * (1.0 + a * inv_b * dP_du);
}

// Backward dalpha, dbeta, dgamma (1 workgroup per channel c -> C workgroups):
// snIn0 = grad_out, snIn1 = alpha, snIn2 = beta (+gamma), snIn3 = x
// snOut0 = dalpha [C], snOut1 = dbeta [C] (and dgamma at snOut1[C + c] when has_gamma != 0)
[shader("compute")] [numthreads(256, 1, 1)]
void snake_backward_params(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                           uniform SnakeParams pc) {
    uint c = gid.x;
    if (c >= pc.channels) return;

    uint tid = lid.x;
    uint outer_size = pc.total_elements / (pc.channels * pc.inner_size);
    uint slice_elems = outer_size * pc.inner_size;

    float a, b, g_phase;
    load_abg(c, pc, a, b, g_phase);
    float inv_b = safe_inv(b, pc.eps);

    float sum_da = 0.0;
    float sum_db = 0.0;
    float sum_dg = 0.0;

    for (uint i = tid; i < slice_elems; i += 256u) {
        uint o = i / pc.inner_size;
        uint l = i % pc.inner_size;
        uint idx = (o * pc.channels + c) * pc.inner_size + l;

        float dy = snIn0[idx];
        float x = snIn3[idx];
        float P, dP_du;
        eval_periodic_basis(pc.mode, a * x, g_phase, P, dP_du);

        if (pc.has_beta != 0u) {
            sum_da += dy * (inv_b * x * dP_du);
            sum_db += dy * (-P * inv_b * inv_b);
        } else {
            sum_da += dy * inv_b * (x * dP_du - P * inv_b);
        }
        if (pc.has_gamma != 0u && pc.mode == 1u) {
            sum_dg += dy * (inv_b * dP_du);
        }
    }

    if (pc.log_scale != 0u) {
        sum_da *= a;
        sum_db *= b;
    }

    sn_s0[tid] = sum_da;
    sn_s1[tid] = sum_db;
    sn_s2[tid] = sum_dg;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sn_s0[tid] += sn_s0[tid + s];
            sn_s1[tid] += sn_s1[tid + s];
            sn_s2[tid] += sn_s2[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        snOut0[c] = sn_s0[0];
        if (pc.has_beta != 0u) {
            snOut1[c] = sn_s1[0];
        }
        if (pc.has_gamma != 0u) {
            uint g_out_idx = (pc.has_beta != 0u) ? (pc.channels + c) : c;
            snOut1[g_out_idx] = sn_s2[0];
        }
    }
}
