// Conditional, Gather, Scatter, and Index-Select/Add Kernels (up to 6D).

struct WhereParams {
    uint total_elements;
    uint ndim;
    uint offsetC;
    uint offsetA;
    uint offsetB;
    uint shape[6];
    uint strideC[6];
    uint strideA[6];
    uint strideB[6];
};

struct MaskedFillParams {
    uint total_elements;
    uint ndim;
    float fill_value;
    uint offsetX;
    uint offsetM;
    uint shape[6];
    uint strideX[6];
    uint strideM[6];
};

struct GatherScatterParams {
    uint total_elements;
    uint ndim;
    uint dim;
    uint dim_size_target;
    uint offsetA;
    uint offsetIdx;
    uint offsetB;
    float scalar_val;
    uint shape[6];
    uint strideA[6];
    uint strideIdx[6];
    uint strideB[6];
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> buf0;
[[vk::binding(1, 0)]] StructuredBuffer<uint> buf1;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> buf2;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> buf3;

void atomic_add_f32_buf2(uint idx, float val) {
    uint expected = buf2[idx];
    while (true) {
        float current = asfloat(expected);
        uint desired = asuint(current + val);
        uint prev;
        InterlockedCompareExchange(buf2[idx], expected, desired, prev);
        if (prev == expected) break;
        expected = prev;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void where_kernel(uint3 tid : SV_DispatchThreadID, uniform WhereParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    if (pc.ndim == 0u) {
        buf3[idx] = (asfloat(buf0[pc.offsetC]) != 0.0) ? buf1[pc.offsetA] : buf2[pc.offsetB];
        return;
    }

    uint rem = idx;
    uint idxC = pc.offsetC;
    uint idxA = pc.offsetA;
    uint idxB = pc.offsetB;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        idxC += coord * pc.strideC[d];
        idxA += coord * pc.strideA[d];
        idxB += coord * pc.strideB[d];
    }
    buf3[idx] = (asfloat(buf0[idxC]) != 0.0) ? buf1[idxA] : buf2[idxB];
}

[shader("compute")] [numthreads(256, 1, 1)]
void masked_fill_kernel(uint3 tid : SV_DispatchThreadID, uniform MaskedFillParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    if (pc.ndim == 0u) {
        buf2[idx] = (asfloat(buf1[pc.offsetM]) != 0.0) ? asuint(pc.fill_value) : buf0[pc.offsetX];
        return;
    }

    uint rem = idx;
    uint idxX = pc.offsetX;
    uint idxM = pc.offsetM;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        idxX += coord * pc.strideX[d];
        idxM += coord * pc.strideM[d];
    }
    buf2[idx] = (asfloat(buf1[idxM]) != 0.0) ? asuint(pc.fill_value) : buf0[idxX];
}

[shader("compute")] [numthreads(256, 1, 1)]
void gather_kernel(uint3 tid : SV_DispatchThreadID, uniform GatherScatterParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.offsetA;
    uint idx_pos = pc.offsetIdx;
    uint out_pos = pc.offsetB;
    uint dim_stride_src = pc.strideA[pc.dim];

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        idx_pos += coord * pc.strideIdx[d];
        out_pos += coord * pc.strideB[d];
        if (uint(d) != pc.dim) {
            src_pos += coord * pc.strideA[d];
        }
    }

    int k = asint(buf1[idx_pos]);
    if (k < 0) k += int(pc.dim_size_target);
    if (k >= 0 && uint(k) < pc.dim_size_target) {
        src_pos += uint(k) * dim_stride_src;
        buf2[out_pos] = buf0[src_pos];
    } else {
        buf2[out_pos] = 0u;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void scatter_kernel(uint3 tid : SV_DispatchThreadID, uniform GatherScatterParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.offsetA;
    uint idx_pos = pc.offsetIdx;
    uint dst_pos = pc.offsetB;
    uint dim_stride_dst = pc.strideB[pc.dim];

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        src_pos += coord * pc.strideA[d];
        idx_pos += coord * pc.strideIdx[d];
        if (uint(d) != pc.dim) {
            dst_pos += coord * pc.strideB[d];
        }
    }

    int k = asint(buf1[idx_pos]);
    if (k < 0) k += int(pc.dim_size_target);
    if (k >= 0 && uint(k) < pc.dim_size_target) {
        dst_pos += uint(k) * dim_stride_dst;
        buf2[dst_pos] = (pc.offsetA == 0xFFFFFFFFu) ? asuint(pc.scalar_val) : buf0[src_pos];
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void scatter_add_kernel(uint3 tid : SV_DispatchThreadID, uniform GatherScatterParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.offsetA;
    uint idx_pos = pc.offsetIdx;
    uint dst_pos = pc.offsetB;
    uint dim_stride_dst = pc.strideB[pc.dim];

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        src_pos += coord * pc.strideA[d];
        idx_pos += coord * pc.strideIdx[d];
        if (uint(d) != pc.dim) {
            dst_pos += coord * pc.strideB[d];
        }
    }

    int k = asint(buf1[idx_pos]);
    if (k < 0) k += int(pc.dim_size_target);
    if (k >= 0 && uint(k) < pc.dim_size_target) {
        dst_pos += uint(k) * dim_stride_dst;
        atomic_add_f32_buf2(dst_pos, asfloat(buf0[src_pos]));
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void index_select_kernel(uint3 tid : SV_DispatchThreadID, uniform GatherScatterParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.offsetA;
    uint out_pos = pc.offsetB;
    uint dim_coord = 0u;
    uint dim_stride_src = pc.strideA[pc.dim];

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        out_pos += coord * pc.strideB[d];
        if (uint(d) == pc.dim) {
            dim_coord = coord;
        } else {
            src_pos += coord * pc.strideA[d];
        }
    }

    int k = asint(buf1[pc.offsetIdx + dim_coord * pc.strideIdx[0]]);
    if (k < 0) k += int(pc.dim_size_target);
    if (k >= 0 && uint(k) < pc.dim_size_target) {
        src_pos += uint(k) * dim_stride_src;
        buf2[out_pos] = buf0[src_pos];
    } else {
        buf2[out_pos] = 0u;
    }
}

[shader("compute")] [numthreads(256, 1, 1)]
void index_add_kernel(uint3 tid : SV_DispatchThreadID, uniform GatherScatterParams pc) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint rem = idx;
    uint src_pos = pc.offsetA;
    uint dst_pos = pc.offsetB;
    uint dim_coord = 0u;
    uint dim_stride_dst = pc.strideB[pc.dim];

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        src_pos += coord * pc.strideA[d];
        if (uint(d) == pc.dim) {
            dim_coord = coord;
        } else {
            dst_pos += coord * pc.strideB[d];
        }
    }

    int k = asint(buf1[pc.offsetIdx + dim_coord * pc.strideIdx[0]]);
    if (k < 0) k += int(pc.dim_size_target);
    if (k >= 0 && uint(k) < pc.dim_size_target) {
        dst_pos += uint(k) * dim_stride_dst;
        atomic_add_f32_buf2(dst_pos, asfloat(buf0[src_pos]) * pc.scalar_val);
    }
}
