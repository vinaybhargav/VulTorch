// Fused FlashAttention-2 (Dao et al. 2023) Forward and Backward Compute Shaders in Slang.
//
// Avoids materializing the O(B * H * S_q * S_k) attention matrix in global memory by
// streaming tiles of K and V through workgroup shared memory and maintaining running
// online softmax statistics (row max m_i, exp sum l_i, and output accumulator O_i).
// Saves only O [B, H_q, S_q, D] and logsumexp L = m + log(l) [B, H_q, S_q] for backward.
// Supports arbitrary head dimension D <= 128, causal masking, and Grouped-Query Attention (GQA).

struct FlashAttnParams {
    uint B;          // Batch size
    uint H_q;        // Number of query heads
    uint H_kv;       // Number of key/value heads (H_q % H_kv == 0)
    uint S_q;        // Query sequence length
    uint S_k;        // Key/Value sequence length
    uint D;          // Head dimension (<= 128)
    uint is_causal;  // 1 = causal mask (col > row + (S_k - S_q) masked to -inf)
    float scale;     // 1.0 / sqrt(D)
};

[[vk::binding(0, 0)]] StructuredBuffer<float> faIn0; // Q [B, H_q, S_q, D]
[[vk::binding(1, 0)]] StructuredBuffer<float> faIn1; // K [B, H_kv, S_k, D]
[[vk::binding(2, 0)]] StructuredBuffer<float> faIn2; // V [B, H_kv, S_k, D]
[[vk::binding(3, 0)]] StructuredBuffer<float> faIn3; // L [B, H_q, S_q] (bwd)
[[vk::binding(4, 0)]] StructuredBuffer<float> faIn4; // dO [B, H_q, S_q, D] (bwd)
[[vk::binding(5, 0)]] StructuredBuffer<float> faIn5; // delta [B, H_q, S_q] = rowsum(dO * O) (bwd)
[[vk::binding(6, 0)]] RWStructuredBuffer<float> faOut0; // O (fwd) or dQ (bwd) or delta (pre)
[[vk::binding(7, 0)]] RWStructuredBuffer<float> faOut1; // L (fwd) or dK_hq (bwd)
[[vk::binding(8, 0)]] RWStructuredBuffer<float> faOut2; // dV_hq (bwd)

static const uint FA_BC = 32u;
static const uint FA_MAX_D = 128u;

groupshared float fa_sK[FA_BC][FA_MAX_D + 1u];
groupshared float fa_sV[FA_BC][FA_MAX_D + 1u];

// Forward kernel:
// Workgroup size = 64 threads (each thread owns 1 query row i = gid.x * 64 + lid.x).
// gid.y = b * H_q + h_q.
// With causal masking, tiles where k_start > max_q_in_block + causal_shift are skipped completely!
[shader("compute")] [numthreads(64, 1, 1)]
void flash_attention_forward(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                             uniform FlashAttnParams pc) {
    uint tid = lid.x;
    uint bh_q = gid.y;
    if (bh_q >= pc.B * pc.H_q) return;

    uint b = bh_q / pc.H_q;
    uint h_q = bh_q % pc.H_q;
    uint group_ratio = pc.H_q / pc.H_kv;
    uint h_kv = h_q / group_ratio;
    uint bh_kv = b * pc.H_kv + h_kv;

    uint q_row = gid.x * 64u + tid;
    bool valid_q = (q_row < pc.S_q);
    uint D = pc.D;

    uint q_base = (bh_q * pc.S_q + q_row) * D;
    uint kv_head_base = bh_kv * pc.S_k * D;

    float q_reg[FA_MAX_D];
    float o_acc[FA_MAX_D];
    for (uint d = 0u; d < D; ++d) {
        q_reg[d] = valid_q ? faIn0[q_base + d] : 0.0;
        o_acc[d] = 0.0;
    }

    float m_i = -1e30;
    float l_i = 0.0;
    int causal_shift = int(pc.S_k) - int(pc.S_q);

    // Causal early tile termination: highest query row in this workgroup is gid.x * 64 + 63
    uint max_k_limit = pc.S_k;
    if (pc.is_causal != 0u) {
        int max_k_for_wg = int(min(gid.x * 64u + 63u, pc.S_q - 1u)) + causal_shift + 1;
        max_k_limit = uint(clamp(max_k_for_wg, 0, int(pc.S_k)));
    }

    uint num_k_tiles = (max_k_limit + FA_BC - 1u) / FA_BC;
    for (uint kt = 0u; kt < num_k_tiles; ++kt) {
        uint k_start = kt * FA_BC;

        // Collaboratively load FA_BC x D of K and V into shared memory
        uint tile_elems = FA_BC * D;
        for (uint idx = tid; idx < tile_elems; idx += 64u) {
            uint kr = idx / D;
            uint kd = idx % D;
            uint g_kr = k_start + kr;
            if (g_kr < pc.S_k) {
                uint kv_off = kv_head_base + g_kr * D + kd;
                fa_sK[kr][kd] = faIn1[kv_off];
                fa_sV[kr][kd] = faIn2[kv_off];
            } else {
                fa_sK[kr][kd] = 0.0;
                fa_sV[kr][kd] = 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid_q) {
            uint k_end = min(FA_BC, pc.S_k - k_start);
            if (pc.is_causal != 0u) {
                int row_max_k = int(q_row) + causal_shift + 1 - int(k_start);
                k_end = uint(clamp(row_max_k, 0, int(k_end)));
            }
            for (uint j = 0u; j < k_end; ++j) {
                float dot_qk = 0.0;
                for (uint d = 0u; d < D; ++d) {
                    dot_qk = fma(q_reg[d], fa_sK[j][d], dot_qk);
                }
                float s_ij = dot_qk * pc.scale;
                float m_new = max(m_i, s_ij);
                float alpha = exp(m_i - m_new);
                float p_ij = exp(s_ij - m_new);
                l_i = l_i * alpha + p_ij;
                for (uint d = 0u; d < D; ++d) {
                    o_acc[d] = fma(p_ij, fa_sV[j][d], o_acc[d] * alpha);
                }
                m_i = m_new;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid_q) {
        float inv_l = (l_i > 0.0) ? (1.0 / l_i) : 0.0;
        for (uint d = 0u; d < D; ++d) {
            faOut0[q_base + d] = o_acc[d] * inv_l;
        }
        faOut1[bh_q * pc.S_q + q_row] = (l_i > 0.0) ? (m_i + log(l_i)) : -1e30;
    }
}

// Backward preprocess: computes delta[b, h_q, i] = sum_d (dO[b, h_q, i, d] * O[b, h_q, i, d])
// faIn0 = O [B, H_q, S_q, D], faIn4 = dO [B, H_q, S_q, D], faOut0 = delta [B, H_q, S_q]
[shader("compute")] [numthreads(256, 1, 1)]
void flash_attention_backward_prep(uint3 tid : SV_DispatchThreadID, uniform FlashAttnParams pc) {
    uint row_idx = tid.x;
    uint total_rows = pc.B * pc.H_q * pc.S_q;
    if (row_idx >= total_rows) return;

    uint D = pc.D;
    uint base = row_idx * D;
    float sum_do_o = 0.0;
    for (uint d = 0u; d < D; ++d) {
        sum_do_o = fma(faIn4[base + d], faIn0[base + d], sum_do_o);
    }
    faOut0[row_idx] = sum_do_o;
}

// Backward dQ kernel: 1 thread per query row i (64 threads per workgroup), gid.y = bh_q.
[shader("compute")] [numthreads(64, 1, 1)]
void flash_attention_backward_dq(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                                 uniform FlashAttnParams pc) {
    uint tid = lid.x;
    uint bh_q = gid.y;
    if (bh_q >= pc.B * pc.H_q) return;

    uint b = bh_q / pc.H_q;
    uint h_q = bh_q % pc.H_q;
    uint group_ratio = pc.H_q / pc.H_kv;
    uint h_kv = h_q / group_ratio;
    uint bh_kv = b * pc.H_kv + h_kv;

    uint q_row = gid.x * 64u + tid;
    bool valid_q = (q_row < pc.S_q);
    uint D = pc.D;

    uint q_base = (bh_q * pc.S_q + q_row) * D;
    uint kv_head_base = bh_kv * pc.S_k * D;

    float q_reg[FA_MAX_D];
    float do_reg[FA_MAX_D];
    float dq_acc[FA_MAX_D];
    for (uint d = 0u; d < D; ++d) {
        q_reg[d] = valid_q ? faIn0[q_base + d] : 0.0;
        do_reg[d] = valid_q ? faIn4[q_base + d] : 0.0;
        dq_acc[d] = 0.0;
    }

    float L_i = valid_q ? faIn3[bh_q * pc.S_q + q_row] : 0.0;
    float delta_i = valid_q ? faIn5[bh_q * pc.S_q + q_row] : 0.0;
    int causal_shift = int(pc.S_k) - int(pc.S_q);

    uint max_k_limit = pc.S_k;
    if (pc.is_causal != 0u) {
        int max_k_for_wg = int(min(gid.x * 64u + 63u, pc.S_q - 1u)) + causal_shift + 1;
        max_k_limit = uint(clamp(max_k_for_wg, 0, int(pc.S_k)));
    }

    uint num_k_tiles = (max_k_limit + FA_BC - 1u) / FA_BC;
    for (uint kt = 0u; kt < num_k_tiles; ++kt) {
        uint k_start = kt * FA_BC;
        uint tile_elems = FA_BC * D;
        for (uint idx = tid; idx < tile_elems; idx += 64u) {
            uint kr = idx / D;
            uint kd = idx % D;
            uint g_kr = k_start + kr;
            if (g_kr < pc.S_k) {
                uint kv_off = kv_head_base + g_kr * D + kd;
                fa_sK[kr][kd] = faIn1[kv_off];
                fa_sV[kr][kd] = faIn2[kv_off];
            } else {
                fa_sK[kr][kd] = 0.0;
                fa_sV[kr][kd] = 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid_q) {
            uint k_end = min(FA_BC, pc.S_k - k_start);
            if (pc.is_causal != 0u) {
                int row_max_k = int(q_row) + causal_shift + 1 - int(k_start);
                k_end = uint(clamp(row_max_k, 0, int(k_end)));
            }
            for (uint j = 0u; j < k_end; ++j) {
                float dot_qk = 0.0;
                float dp_ij = 0.0;
                for (uint d = 0u; d < D; ++d) {
                    dot_qk = fma(q_reg[d], fa_sK[j][d], dot_qk);
                    dp_ij = fma(do_reg[d], fa_sV[j][d], dp_ij);
                }
                float s_ij = dot_qk * pc.scale;
                float p_ij = exp(s_ij - L_i);
                float ds_ij = p_ij * (dp_ij - delta_i) * pc.scale;
                for (uint d = 0u; d < D; ++d) {
                    dq_acc[d] = fma(ds_ij, fa_sK[j][d], dq_acc[d]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid_q) {
        for (uint d = 0u; d < D; ++d) {
            faOut0[q_base + d] = dq_acc[d];
        }
    }
}

// Backward dK & dV kernel: 1 thread per key/value row j (64 threads per workgroup), gid.y = bh_q.
[shader("compute")] [numthreads(64, 1, 1)]
void flash_attention_backward_dkv(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                                  uniform FlashAttnParams pc) {
    uint tid = lid.x;
    uint bh_q = gid.y;
    if (bh_q >= pc.B * pc.H_q) return;

    uint b = bh_q / pc.H_q;
    uint h_q = bh_q % pc.H_q;
    uint group_ratio = pc.H_q / pc.H_kv;
    uint h_kv = h_q / group_ratio;
    uint bh_kv = b * pc.H_kv + h_kv;

    uint k_row = gid.x * 64u + tid;
    bool valid_k = (k_row < pc.S_k);
    uint D = pc.D;

    uint kv_in_base = (bh_kv * pc.S_k + k_row) * D;
    uint kv_out_base = (bh_q * pc.S_k + k_row) * D;
    uint q_head_base = bh_q * pc.S_q * D;

    float k_reg[FA_MAX_D];
    float v_reg[FA_MAX_D];
    float dk_acc[FA_MAX_D];
    float dv_acc[FA_MAX_D];
    for (uint d = 0u; d < D; ++d) {
        k_reg[d] = valid_k ? faIn1[kv_in_base + d] : 0.0;
        v_reg[d] = valid_k ? faIn2[kv_in_base + d] : 0.0;
        dk_acc[d] = 0.0;
        dv_acc[d] = 0.0;
    }

    int causal_shift = int(pc.S_k) - int(pc.S_q);
    uint num_q_tiles = (pc.S_q + FA_BC - 1u) / FA_BC;

    for (uint qt = 0u; qt < num_q_tiles; ++qt) {
        uint q_start = qt * FA_BC;
        uint tile_elems = FA_BC * D;
        for (uint idx = tid; idx < tile_elems; idx += 64u) {
            uint qr = idx / D;
            uint qd = idx % D;
            uint g_qr = q_start + qr;
            if (g_qr < pc.S_q) {
                uint q_off = q_head_base + g_qr * D + qd;
                fa_sK[qr][qd] = faIn0[q_off];
                fa_sV[qr][qd] = faIn4[q_off];
            } else {
                fa_sK[qr][qd] = 0.0;
                fa_sV[qr][qd] = 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (valid_k) {
            uint q_end = min(FA_BC, pc.S_q - q_start);
            for (uint i = 0u; i < q_end; ++i) {
                uint g_qi = q_start + i;
                if (pc.is_causal != 0u && int(k_row) > int(g_qi) + causal_shift) {
                    continue;
                }
                float L_i = faIn3[bh_q * pc.S_q + g_qi];
                float delta_i = faIn5[bh_q * pc.S_q + g_qi];

                float dot_qk = 0.0;
                float dp_ij = 0.0;
                for (uint d = 0u; d < D; ++d) {
                    dot_qk = fma(fa_sK[i][d], k_reg[d], dot_qk);
                    dp_ij = fma(fa_sV[i][d], v_reg[d], dp_ij);
                }
                float s_ij = dot_qk * pc.scale;
                float p_ij = exp(s_ij - L_i);
                float ds_ij = p_ij * (dp_ij - delta_i) * pc.scale;
                for (uint d = 0u; d < D; ++d) {
                    dv_acc[d] = fma(p_ij, fa_sV[i][d], dv_acc[d]);
                    dk_acc[d] = fma(ds_ij, fa_sK[i][d], dk_acc[d]);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (valid_k) {
        for (uint d = 0u; d < D; ++d) {
            faOut1[kv_out_base + d] = dk_acc[d];
            faOut2[kv_out_base + d] = dv_acc[d];
        }
    }
}
