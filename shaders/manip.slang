// Tensor Manipulation Kernels: Multi-axis cat (copy_into_slice),
// N-D padding (constant/reflect/replicate + backward), tril/triu, flip, roll.

struct SliceCopyParams {
    uint total_elements;
    uint ndim;
    uint elem_bytes;
    uint src_offset;
    uint dst_offset;
    uint src_shape[6];
    uint src_stride[6];
    uint dst_stride[6];
};

struct PadParams {
    uint total_elements;
    uint ndim;
    uint mode;       // 0 = constant, 1 = reflect, 2 = replicate
    float pad_value;
    uint src_offset;
    uint in_shape[6];
    uint out_shape[6];
    uint in_stride[6];
    uint pad_before[6];
};

struct TriFlipRollParams {
    uint total_elements;
    uint ndim;
    uint mode;       // 0 = tril, 1 = triu, 2 = flip, 3 = roll
    int param;       // diagonal for tril/triu, flip_mask for flip
    uint src_offset;
    uint shape[6];
    uint src_stride[6];
    int shifts[6];
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> mSrc;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> mDst;

void write_subword_cas(uint elem_idx, uint elem_bytes, uint val) {
    uint shift = (elem_bytes == 2u) ? ((elem_idx & 1u) * 16u) : ((elem_idx & 3u) * 8u);
    uint mask = (elem_bytes == 2u) ? (0xFFFFu << shift) : (0xFFu << shift);
    uint word_idx = (elem_bytes == 2u) ? (elem_idx >> 1u) : (elem_idx >> 2u);
    uint ins = (val << shift) & mask;
    uint expected = mDst[word_idx];
    while (true) {
        uint desired = (expected & ~mask) | ins;
        uint prev;
        InterlockedCompareExchange(mDst[word_idx], expected, desired, prev);
        if (prev == expected) break;
        expected = prev;
    }
}

void pad_atomic_add_f32(uint idx, float val) {
    uint expected = mDst[idx];
    while (true) {
        float current = asfloat(expected);
        uint desired = asuint(current + val);
        uint prev;
        InterlockedCompareExchange(mDst[idx], expected, desired, prev);
        if (prev == expected) break;
        expected = prev;
    }
}

int reflect_coord(int c, int len) {
    if (len <= 1) return 0;
    while (c < 0 || c >= len) {
        if (c < 0) c = -c;
        if (c >= len) c = 2 * (len - 1) - c;
    }
    return c;
}

[shader("compute")] [numthreads(256, 1, 1)]
void copy_into_slice(uint3 tid : SV_DispatchThreadID, uniform SliceCopyParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint s_pos = pc.src_offset;
    uint d_pos = pc.dst_offset;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.src_shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        s_pos += coord * pc.src_stride[d];
        d_pos += coord * pc.dst_stride[d];
    }

    if (pc.elem_bytes == 4u) {
        mDst[d_pos] = mSrc[s_pos];
    } else if (pc.elem_bytes == 8u) {
        mDst[d_pos * 2u] = mSrc[s_pos * 2u];
        mDst[d_pos * 2u + 1u] = mSrc[s_pos * 2u + 1u];
    } else if (pc.elem_bytes == 2u) {
        uint v = (mSrc[s_pos >> 1u] >> ((s_pos & 1u) * 16u)) & 0xFFFFu;
        write_subword_cas(d_pos, 2u, v);
    } else {
        uint v = (mSrc[s_pos >> 2u] >> ((s_pos & 3u) * 8u)) & 0xFFu;
        write_subword_cas(d_pos, 1u, v);
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void pad_nd_kernel(uint3 tid : SV_DispatchThreadID, uniform PadParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint in_pos = pc.src_offset;
    bool in_bounds = true;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint out_sz = pc.out_shape[d];
        uint coord = rem % out_sz;
        rem /= out_sz;

        int c = int(coord) - int(pc.pad_before[d]);
        int in_sz = int(pc.in_shape[d]);
        if (c < 0 || c >= in_sz) {
            if (pc.mode == 0u) {
                in_bounds = false;
            } else if (pc.mode == 1u) {
                c = reflect_coord(c, in_sz);
                in_pos += uint(c) * pc.in_stride[d];
            } else {
                c = clamp(c, 0, in_sz - 1);
                in_pos += uint(c) * pc.in_stride[d];
            }
        } else {
            in_pos += uint(c) * pc.in_stride[d];
        }
    }

    mDst[idx] = in_bounds ? mSrc[in_pos] : asuint(pc.pad_value);
}

[shader("compute")] [numthreads(256, 1, 1)]
void pad_nd_backward_kernel(uint3 tid : SV_DispatchThreadID, uniform PadParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint in_pos = 0u;
    bool in_bounds = true;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint out_sz = pc.out_shape[d];
        uint coord = rem % out_sz;
        rem /= out_sz;

        int c = int(coord) - int(pc.pad_before[d]);
        int in_sz = int(pc.in_shape[d]);
        if (c < 0 || c >= in_sz) {
            if (pc.mode == 0u) {
                in_bounds = false;
            } else if (pc.mode == 1u) {
                c = reflect_coord(c, in_sz);
                in_pos += uint(c) * pc.in_stride[d];
            } else {
                c = clamp(c, 0, in_sz - 1);
                in_pos += uint(c) * pc.in_stride[d];
            }
        } else {
            in_pos += uint(c) * pc.in_stride[d];
        }
    }

    if (in_bounds) {
        pad_atomic_add_f32(in_pos, asfloat(mSrc[pc.src_offset + idx]));
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void tril_triu_kernel(uint3 tid : SV_DispatchThreadID, uniform TriFlipRollParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.src_offset;
    int row = 0;
    int col = 0;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_sz = pc.shape[d];
        uint coord = rem % dim_sz;
        rem /= dim_sz;
        src_pos += coord * pc.src_stride[d];
        if (d == int(pc.ndim) - 1) col = int(coord);
        if (d == int(pc.ndim) - 2) row = int(coord);
    }

    int diff = col - row;
    bool keep = (pc.mode == 0u) ? (diff <= pc.param) : (diff >= pc.param);
    mDst[idx] = keep ? mSrc[src_pos] : 0u;
}

[shader("compute")] [numthreads(256, 1, 1)]
void flip_roll_kernel(uint3 tid : SV_DispatchThreadID, uniform TriFlipRollParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.src_offset;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        int dim_sz = int(pc.shape[d]);
        int coord = int(rem % uint(dim_sz));
        rem /= uint(dim_sz);

        int s_coord = coord;
        if (pc.mode == 2u) {
            if ((uint(pc.param) & (1u << uint(d))) != 0u) {
                s_coord = dim_sz - 1 - coord;
            }
        } else {
            int sh = pc.shifts[d] % dim_sz;
            s_coord = (coord - sh + dim_sz) % dim_sz;
        }
        src_pos += uint(s_coord) * pc.src_stride[d];
    }

    mDst[idx] = mSrc[src_pos];
}
