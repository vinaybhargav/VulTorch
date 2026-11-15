// Arbitrary strided copy across up to 6 dimensions for 1, 2, 4, and 8-byte dtypes.
// Uses 32-bit word stores on dataOut so 1-byte and 2-byte outputs are completely race-free
// and require no 8-bit/16-bit storage extensions.

struct CopyStridedParams {
    uint total_elements;
    uint ndim;
    uint src_offset;
    uint elem_bytes;
    uint shape[6];
    uint src_stride[6];
};

[[vk::push_constant]] CopyStridedParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dataOut;

uint compute_src_idx(uint dst_idx) {
    uint rem = dst_idx;
    uint src_idx = pc.src_offset;
    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim_size = pc.shape[d];
        uint coord = rem % dim_size;
        rem /= dim_size;
        src_idx += coord * pc.src_stride[d];
    }
    return src_idx;
}

[shader("compute")] [numthreads(256, 1, 1)]
void copy_strided(uint3 tid : SV_DispatchThreadID) {
    uint word_idx = tid.x;

    if (pc.elem_bytes == 4u) {
        if (word_idx >= pc.total_elements) return;
        uint s = compute_src_idx(word_idx);
        dataOut[word_idx] = dataIn[s];
        return;
    }

    if (pc.elem_bytes == 8u) {
        uint elem_idx = word_idx >> 1u;
        if (elem_idx >= pc.total_elements) return;
        uint sub = word_idx & 1u;
        uint s = compute_src_idx(elem_idx);
        dataOut[word_idx] = dataIn[s * 2u + sub];
        return;
    }

    if (pc.elem_bytes == 2u) {
        uint idx0 = word_idx * 2u;
        if (idx0 >= pc.total_elements) return;
        uint s0 = compute_src_idx(idx0);
        uint low = (dataIn[s0 >> 1u] >> ((s0 & 1u) * 16u)) & 0xFFFFu;

        uint high = 0u;
        uint idx1 = idx0 + 1u;
        if (idx1 < pc.total_elements) {
            uint s1 = compute_src_idx(idx1);
            high = (dataIn[s1 >> 1u] >> ((s1 & 1u) * 16u)) & 0xFFFFu;
        }
        dataOut[word_idx] = low | (high << 16u);
        return;
    }

    // pc.elem_bytes == 1u
    uint base_idx = word_idx * 4u;
    if (base_idx >= pc.total_elements) return;
    uint packed = 0u;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        uint elem = base_idx + k;
        if (elem < pc.total_elements) {
            uint s = compute_src_idx(elem);
            uint b = (dataIn[s >> 2u] >> ((s & 3u) * 8u)) & 0xFFu;
            packed |= (b << (k * 8u));
        }
    }
    dataOut[word_idx] = packed;
}
