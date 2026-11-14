// Materialises an arbitrarily strided view into a contiguous output buffer.
// Covers transpose/permute, expand (stride 0 on broadcast axes), and any other
// layout change that cannot be expressed by reshaping.

struct CopyStridedParams {
    uint total_elements;
    uint ndim;
    uint shape[6];
    uint src_stride[6];
};

[[vk::push_constant]] CopyStridedParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

[shader("compute")] [numthreads(256, 1, 1)]
void copy_strided(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_elements) return;

    uint remaining = idx;
    uint src_idx = 0u;

    for (int d = int(pc.ndim) - 1; d >= 0; --d) {
        uint dim = pc.shape[d];
        uint coord = remaining % dim;
        remaining /= dim;
        src_idx += coord * pc.src_stride[d];
    }

    dataOut[idx] = dataIn[src_idx];
}
