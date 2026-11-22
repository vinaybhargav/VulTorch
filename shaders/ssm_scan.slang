// Mamba (Gu & Dao 2023) Selective State Space Model (S6) Parallel Associative Scan Kernels.
//
// Discretized Selective State Space Recurrence (per batch b, channel d, state n, timestep t in 0..L-1):
//   dt = softplus(delta[b, d, t])   (already softplus-activated by caller or raw dt > 0)
//   A_bar = exp(dt * A[d, n])
//   B_bar_u = dt * B[b, n, t] * u[b, d, t]
//   h_t = A_bar * h_{t-1} + B_bar_u      (with h_{-1} = 0)
//   y[b, d, t] = sum_n (C[b, n, t] * h_t[b, d, n]) + D[d] * u[b, d, t]
//
// Backward recurrence (reverse time t = L-1 down to 0):
//   dh_t = dy[b, d, t] * C[b, n, t] + A_bar_{t+1} * dh_{t+1}
//   dC[b, n, t] = sum_d (dy[b, d, t] * h_t[b, d, n])
//   d(B_bar_u)_t = dh_t
//   d(A_bar)_t   = dh_t * h_{t-1}

struct SSMParams {
    uint B;      // Batch size
    uint D;      // Inner model dimension (d_inner)
    uint L;      // Sequence length
    uint N;      // SSM state dimension (d_state, typically <= 32)
    uint has_D;  // 1 if skip vector D[d] is provided
    uint has_z;  // 1 if gated output y * SiLU(z) is fused
};

[[vk::binding(0, 0)]] StructuredBuffer<float> ssmIn0; // u [B, D, L]
[[vk::binding(1, 0)]] StructuredBuffer<float> ssmIn1; // delta [B, D, L] (post-softplus > 0)
[[vk::binding(2, 0)]] StructuredBuffer<float> ssmIn2; // A [D, N]
[[vk::binding(3, 0)]] StructuredBuffer<float> ssmIn3; // B_mat [B, N, L]
[[vk::binding(4, 0)]] StructuredBuffer<float> ssmIn4; // C_mat [B, N, L]
[[vk::binding(5, 0)]] StructuredBuffer<float> ssmIn5; // D_vec [D] (forward) or grad_out [B, D, L] (backward)
[[vk::binding(6, 0)]] RWStructuredBuffer<float> ssmOut0; // y [B, D, L] (forward) or du [B, D, L] (backward)
[[vk::binding(7, 0)]] RWStructuredBuffer<float> ssmOut1; // packed [dDelta (B*D*L), dA (B*D*N), dB (B*D*N*L), dC (B*D*N*L), dD (B*D)]

// Forward kernel: 1 thread per (b, d) series -> B * D threads total.
// Loops over t = 0..L-1 maintaining state vector h[0..N-1] in registers (up to N <= 32).
[shader("compute")] [numthreads(64, 1, 1)]
void ssm_scan_forward(uint3 tid : SV_DispatchThreadID, uniform SSMParams pc) {
    uint bd = tid.x;
    if (bd >= pc.B * pc.D) return;

    uint b = bd / pc.D;
    uint d = bd % pc.D;
    uint L = pc.L;
    uint N = pc.N;

    float h[32];
    for (uint n = 0u; n < N; ++n) {
        h[n] = 0.0;
    }

    float A_loc[32];
    for (uint n = 0u; n < N; ++n) {
        A_loc[n] = ssmIn2[d * N + n];
    }

    float D_val = (pc.has_D != 0u) ? ssmIn5[d] : 0.0;
    uint u_base = bd * L;
    uint bc_base = b * N * L;

    for (uint t = 0u; t < L; ++t) {
        float u_t = ssmIn0[u_base + t];
        float dt_t = ssmIn1[u_base + t];
        float y_t = D_val * u_t;

        for (uint n = 0u; n < N; ++n) {
            float b_nt = ssmIn3[bc_base + n * L + t];
            float c_nt = ssmIn4[bc_base + n * L + t];
            float a_bar = exp(dt_t * A_loc[n]);
            h[n] = a_bar * h[n] + (dt_t * b_nt * u_t);
            y_t += c_nt * h[n];
        }
        ssmOut0[u_base + t] = y_t;
    }
}

// Backward kernel: 1 thread per (b, d) series -> B * D threads total.
// Recomputes forward trajectory per chunk or stores/recomputes h_{t-1} and runs reverse scan.
// Writes per-(b, d) contributions into ssmOut0 (du [B, D, L]) and ssmOut1 (packed workspace):
//   offset 0:                  dDelta [B, D, L]
//   offset B*D*L:              dA_bd  [B, D, N]
//   offset B*D*L + B*D*N:      dB_bd  [B, D, N, L]
//   offset B*D*L + B*D*N + B*D*N*L: dC_bd [B, D, N, L]
//   offset B*D*L + B*D*N + 2*B*D*N*L: dD_bd [B, D]
[shader("compute")] [numthreads(64, 1, 1)]
void ssm_scan_backward(uint3 tid : SV_DispatchThreadID, uniform SSMParams pc) {
    uint bd = tid.x;
    uint BD = pc.B * pc.D;
    if (bd >= BD) return;

    uint b = bd / pc.D;
    uint d = bd % pc.D;
    uint L = pc.L;
    uint N = pc.N;

    float A_loc[32];
    for (uint n = 0u; n < N; ++n) {
        A_loc[n] = ssmIn2[d * N + n];
    }

    uint u_base = bd * L;
    uint bc_base = b * N * L;

    uint off_ddelta = 0u;
    uint off_dA     = BD * L;
    uint off_dB     = off_dA + BD * N;
    uint off_dC     = off_dB + BD * N * L;
    uint off_dD     = off_dC + BD * N * L;

    // 1. Forward pass to record h_t in dC_bd workspace temporarily:
    //    slot: off_dC + (bd * N + n) * L + t holds h_t[b, d, n]
    float h[32];
    for (uint n = 0u; n < N; ++n) {
        h[n] = 0.0;
    }

    for (uint t = 0u; t < L; ++t) {
        float u_t = ssmIn0[u_base + t];
        float dt_t = ssmIn1[u_base + t];
        for (uint n = 0u; n < N; ++n) {
            float b_nt = ssmIn3[bc_base + n * L + t];
            float a_bar = exp(dt_t * A_loc[n]);
            h[n] = a_bar * h[n] + (dt_t * b_nt * u_t);
            ssmOut1[off_dC + (bd * N + n) * L + t] = h[n];
        }
    }

    // 2. Reverse scan t = L-1 down to 0
    float dh[32];
    float dA_acc[32];
    for (uint n = 0u; n < N; ++n) {
        dh[n] = 0.0;
        dA_acc[n] = 0.0;
    }
    float dD_acc = 0.0;

    for (int ti = int(L) - 1; ti >= 0; --ti) {
        uint t = uint(ti);
        float u_t = ssmIn0[u_base + t];
        float dt_t = ssmIn1[u_base + t];
        float dy_t = ssmIn5[u_base + t];

        dD_acc += dy_t * u_t;
        float du_t = 0.0;
        float ddt_t = 0.0;

        for (uint n = 0u; n < N; ++n) {
            uint h_slot = off_dC + (bd * N + n) * L + t;
            float h_t = ssmOut1[h_slot];
            float h_prev = (t > 0u) ? ssmOut1[h_slot - 1u] : 0.0;

            float b_nt = ssmIn3[bc_base + n * L + t];
            float c_nt = ssmIn4[bc_base + n * L + t];
            float a_val = A_loc[n];
            float a_bar = exp(dt_t * a_val);

            // Overwrite h_slot with actual dC_bd = dy_t * h_t
            ssmOut1[h_slot] = dy_t * h_t;

            float dh_cur = dh[n] + dy_t * c_nt;
            float da_bar = dh_cur * h_prev;

            du_t += dh_cur * dt_t * b_nt;
            ddt_t += dh_cur * b_nt * u_t + da_bar * a_bar * a_val;
            dA_acc[n] += da_bar * a_bar * dt_t;

            ssmOut1[off_dB + (bd * N + n) * L + t] = dh_cur * dt_t * u_t;

            // Propagate to t - 1
            dh[n] = dh_cur * a_bar;
        }

        ssmOut0[u_base + t] = du_t;
        ssmOut1[off_ddelta + u_base + t] = ddt_t;
    }

    for (uint n = 0u; n < N; ++n) {
        ssmOut1[off_dA + bd * N + n] = dA_acc[n];
    }
    ssmOut1[off_dD + bd] = dD_acc;
}
