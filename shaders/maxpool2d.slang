struct MaxPool2dParams {
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

[[vk::push_constant]] MaxPool2dParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> maxIndices;

[shader("compute")] [numthreads(16, 16, 1)]
void maxpool2d(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    uint ow = tid.x;
    uint oh = tid.y;
    uint nc = gid.z;

    if (ow >= pc.W_out || oh >= pc.H_out || nc >= (pc.N * pc.C)) return;

    int in_h_base = int(oh * pc.stride_h) - int(pc.pad_h);
    int in_w_base = int(ow * pc.stride_w) - int(pc.pad_w);

    uint in_nc_offset = nc * pc.H_in * pc.W_in;

    float max_val = -3.402823466e+38;
    uint max_idx = 0u;

    for (uint kh = 0; kh < pc.kH; ++kh) {
        int ih = in_h_base + int(kh);
        if (ih >= 0 && ih < int(pc.H_in)) {
            for (uint kw = 0; kw < pc.kW; ++kw) {
                int iw = in_w_base + int(kw);
                if (iw >= 0 && iw < int(pc.W_in)) {
                    uint cur_idx = in_nc_offset + uint(ih) * pc.W_in + uint(iw);
                    float val = dataIn[cur_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = cur_idx;
                    }
                }
            }
        }
    }

    uint out_idx = nc * (pc.H_out * pc.W_out) + oh * pc.W_out + ow;
    dataOut[out_idx] = max_val;
    maxIndices[out_idx] = asfloat(max_idx);
}
