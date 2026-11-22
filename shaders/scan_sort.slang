// Extended Reductions, Prefix Scans (Blelloch), and Bitonic Sort / Top-K Kernels.
// All kernels operate over flattened [outer_size, dim_size, inner_size] views.

struct ScanSortParams {
    uint total_slices;   // outer_size * inner_size
    uint dim_size;
    uint inner_size;
    uint mode;           // kernel-specific mode
    uint out_k;          // for topk (k), or dim_size for sort
    uint src_offset;
    uint pad_pow2;       // next power of 2 >= dim_size (for bitonic sort)
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> ssIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> ssOutVal;
[[vk::binding(2, 0)]] RWStructuredBuffer<int> ssOutIdx;

groupshared float smem_f[1024];
groupshared int smem_i[1024];

uint slice_elem_idx(uint slice_id, uint j, uint dim_size, uint inner_size, uint src_offset) {
    uint outer = slice_id / inner_size;
    uint inner = slice_id % inner_size;
    return src_offset + (outer * dim_size + j) * inner_size + inner;
}

uint out_elem_idx(uint slice_id, uint j, uint out_dim_size, uint inner_size) {
    uint outer = slice_id / inner_size;
    uint inner = slice_id % inner_size;
    return (outer * out_dim_size + j) * inner_size + inner;
}

// ----------------------------------------------------------------------------
// 1. Argmax (mode=0) & Argmin (mode=1)
// ----------------------------------------------------------------------------
[shader("compute")] [numthreads(256, 1, 1)]
void argmax_argmin_kernel(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                          uniform ScanSortParams pc) {
    uint slice_id = gid.x;
    if (slice_id >= pc.total_slices) return;

    uint tid = lid.x;
    bool is_min = (pc.mode == 1u);

    float best_v = is_min ? 3.402823466e+38 : -3.402823466e+38;
    int best_i = 0;
    bool has_val = false;

    for (uint j = tid; j < pc.dim_size; j += 256u) {
        float v = ssIn[slice_elem_idx(slice_id, j, pc.dim_size, pc.inner_size, pc.src_offset)];
        bool better = !has_val || (is_min ? (v < best_v) : (v > best_v));
        if (better) {
            best_v = v;
            best_i = int(j);
            has_val = true;
        }
    }

    smem_f[tid] = best_v;
    smem_i[tid] = has_val ? best_i : 0x7FFFFFFF;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            int idx_a = smem_i[tid];
            int idx_b = smem_i[tid + s];
            if (idx_b != 0x7FFFFFFF) {
                if (idx_a == 0x7FFFFFFF) {
                    smem_f[tid] = smem_f[tid + s];
                    smem_i[tid] = idx_b;
                } else {
                    float va = smem_f[tid];
                    float vb = smem_f[tid + s];
                    bool take_b = is_min ? (vb < va || (vb == va && idx_b < idx_a))
                                         : (vb > va || (vb == va && idx_b < idx_a));
                    if (take_b) {
                        smem_f[tid] = vb;
                        smem_i[tid] = idx_b;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        ssOutIdx[slice_id] = smem_i[0];
    }
}

// ----------------------------------------------------------------------------
// 2. Prod (mode=0), All (mode=1), Any (mode=2)
// ----------------------------------------------------------------------------
[shader("compute")] [numthreads(256, 1, 1)]
void prod_all_any_kernel(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                         uniform ScanSortParams pc) {
    uint slice_id = gid.x;
    if (slice_id >= pc.total_slices) return;

    uint tid = lid.x;
    float acc = (pc.mode == 2u) ? 0.0 : 1.0;

    for (uint j = tid; j < pc.dim_size; j += 256u) {
        float v = ssIn[slice_elem_idx(slice_id, j, pc.dim_size, pc.inner_size, pc.src_offset)];
        if (pc.mode == 0u) {
            acc *= v;
        } else if (pc.mode == 1u) {
            acc = (acc != 0.0 && v != 0.0) ? 1.0 : 0.0;
        } else {
            acc = (acc != 0.0 || v != 0.0) ? 1.0 : 0.0;
        }
    }

    smem_f[tid] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            float a = smem_f[tid];
            float b = smem_f[tid + s];
            if (pc.mode == 0u) {
                smem_f[tid] = a * b;
            } else if (pc.mode == 1u) {
                smem_f[tid] = (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
            } else {
                smem_f[tid] = (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        ssOutVal[slice_id] = smem_f[0];
    }
}

// ----------------------------------------------------------------------------
// 3. Inclusive Prefix Scan: Cumsum (mode=0) & Cumprod (mode=1)
// ----------------------------------------------------------------------------
[shader("compute")] [numthreads(256, 1, 1)]
void scan_kernel(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                 uniform ScanSortParams pc) {
    uint slice_id = gid.x;
    if (slice_id >= pc.total_slices) return;

    uint tid = lid.x;
    bool is_prod = (pc.mode == 1u);
    float carry = is_prod ? 1.0 : 0.0;

    for (uint base_j = 0u; base_j < pc.dim_size; base_j += 256u) {
        uint j = base_j + tid;
        float val = is_prod ? 1.0 : 0.0;
        if (j < pc.dim_size) {
            val = ssIn[slice_elem_idx(slice_id, j, pc.dim_size, pc.inner_size, pc.src_offset)];
        }
        smem_f[tid] = val;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele inclusive scan across 256 threads
        for (uint offset = 1u; offset < 256u; offset <<= 1u) {
            float prev = (tid >= offset) ? smem_f[tid - offset] : (is_prod ? 1.0 : 0.0);
            GroupMemoryBarrierWithGroupSync();
            if (tid >= offset) {
                smem_f[tid] = is_prod ? (smem_f[tid] * prev) : (smem_f[tid] + prev);
            }
            GroupMemoryBarrierWithGroupSync();
        }

        float block_val = is_prod ? (carry * smem_f[tid]) : (carry + smem_f[tid]);
        if (j < pc.dim_size) {
            uint oidx = out_elem_idx(slice_id, j, pc.dim_size, pc.inner_size);
            ssOutVal[oidx] = block_val;
        }

        float block_total = smem_f[255];
        carry = is_prod ? (carry * block_total) : (carry + block_total);
        GroupMemoryBarrierWithGroupSync();
    }
}

// ----------------------------------------------------------------------------
// 4. Workgroup Bitonic Sort & Top-K (dim_size <= 1024 in shared memory)
//    mode: 0 = ascending, 1 = descending
// ----------------------------------------------------------------------------
[shader("compute")] [numthreads(256, 1, 1)]
void sort_topk_kernel(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID,
                      uniform ScanSortParams pc) {
    uint slice_id = gid.x;
    if (slice_id >= pc.total_slices) return;

    uint tid = lid.x;
    bool desc = (pc.mode == 1u);
    float pad_val = desc ? -3.402823466e+38 : 3.402823466e+38;
    uint P = pc.pad_pow2;

    for (uint i = tid; i < P; i += 256u) {
        if (i < pc.dim_size) {
            smem_f[i] = ssIn[slice_elem_idx(slice_id, i, pc.dim_size, pc.inner_size, pc.src_offset)];
            smem_i[i] = int(i);
        } else {
            smem_f[i] = pad_val;
            smem_i[i] = int(i);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort over P elements
    for (uint k = 2u; k <= P; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = tid; i < P; i += 256u) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool dir_asc = ((i & k) == 0u);
                    bool want_asc = desc ? !dir_asc : dir_asc;
                    float vi = smem_f[i];
                    float vx = smem_f[ixj];
                    int ii = smem_i[i];
                    int ix = smem_i[ixj];
                    bool swap_cond = want_asc
                        ? (vi > vx || (vi == vx && ii > ix))
                        : (vi < vx || (vi == vx && ii > ix));
                    if (swap_cond) {
                        smem_f[i] = vx;
                        smem_f[ixj] = vi;
                        smem_i[i] = ix;
                        smem_i[ixj] = ii;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    for (uint i = tid; i < pc.out_k; i += 256u) {
        uint oidx = out_elem_idx(slice_id, i, pc.out_k, pc.inner_size);
        ssOutVal[oidx] = smem_f[i];
        ssOutIdx[oidx] = smem_i[i];
    }
}
