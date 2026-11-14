struct MaxPool2dBackwardParams {
    uint N;
    uint C;
    uint H_in;
    uint W_in;
    uint H_out;
    uint W_out;
    uint kH;
    uint kW;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
};

[[vk::push_constant]] MaxPool2dBackwardParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> gradOut;
[[vk::binding(1, 0)]] StructuredBuffer<float> maxIndices;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gradIn;

[shader("compute")] [numthreads(16, 16, 1)]
void maxpool2d_backward(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    uint iw = tid.x;
    uint ih = tid.y;
    uint nc = gid.z;

    if (iw >= pc.W_in || ih >= pc.H_in || nc >= (pc.N * pc.C)) return;

    uint in_idx = nc * (pc.H_in * pc.W_in) + ih * pc.W_in + iw;
    uint out_nc_offset = nc * (pc.H_out * pc.W_out);

    // Fast path for non-overlapping pooling windows (e.g. 2x2 stride 2):
    // at most 1 output window can cover (ih, iw).
    if (pc.pad_h == 0u && pc.pad_w == 0u && pc.kH == pc.stride_h && pc.kW == pc.stride_w) {
        uint oh = ih / pc.stride_h;
        uint ow = iw / pc.stride_w;
        float val = 0.0;
        if (oh < pc.H_out && ow < pc.W_out) {
            uint out_idx = out_nc_offset + oh * pc.W_out + ow;
            if (asuint(maxIndices[out_idx]) == in_idx) {
                val = gradOut[out_idx];
            }
        }
        gradIn[in_idx] = val;
        return;
    }

    int h_padded = int(ih + pc.pad_h);
    int w_padded = int(iw + pc.pad_w);

    uint oh_start = (h_padded < int(pc.kH)) ? 0u : uint((h_padded - int(pc.kH)) / int(pc.stride_h) + 1);
    uint oh_end = min(uint(h_padded / int(pc.stride_h)) + 1u, pc.H_out);

    uint ow_start = (w_padded < int(pc.kW)) ? 0u : uint((w_padded - int(pc.kW)) / int(pc.stride_w) + 1);
    uint ow_end = min(uint(w_padded / int(pc.stride_w)) + 1u, pc.W_out);

    float sum = 0.0;
    for (uint oh = oh_start; oh < oh_end; ++oh) {
        for (uint ow = ow_start; ow < ow_end; ++ow) {
            uint out_idx = out_nc_offset + oh * pc.W_out + ow;
            if (asuint(maxIndices[out_idx]) == in_idx) {
                sum += gradOut[out_idx];
            }
        }
    }

    gradIn[in_idx] = sum;
}
