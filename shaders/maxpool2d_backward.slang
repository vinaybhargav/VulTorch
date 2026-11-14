struct MaxPool2dBackwardParams {
    uint total_out_elements;
};

[[vk::push_constant]] MaxPool2dBackwardParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> gradOut;
[[vk::binding(1, 0)]] StructuredBuffer<uint> maxIndices;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gradIn;

[shader("compute")] [numthreads(256, 1, 1)]
void maxpool2d_backward(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= pc.total_out_elements) return;

    uint in_idx = maxIndices[idx];
    gradIn[in_idx] += gradOut[idx];
}
