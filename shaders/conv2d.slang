struct Conv2dParams {
    uint N;
    uint C_in;
    uint H_in;
    uint W_in;
    uint C_out;
    uint H_out;
    uint W_out;
    uint kH;
    uint kW;
    uint stride_h;
    uint stride_w;
    uint pad_h;
    uint pad_w;
    uint has_bias;
};

[[vk::push_constant]] Conv2dParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] StructuredBuffer<float> dataWeight;
[[vk::binding(2, 0)]] StructuredBuffer<float> dataBias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> dataOut;

[shader("compute")] [numthreads(8, 8, 4)]
void conv2d(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint ow = tid.x;
    uint oh = tid.y;
    uint c_groups = (pc.C_out + 3u) / 4u;
    uint n = gid.z / c_groups;
    uint real_oc = (gid.z % c_groups) * 4u + lid.z;

    if (ow >= pc.W_out || oh >= pc.H_out || real_oc >= pc.C_out || n >= pc.N) return;

    float acc = (pc.has_bias != 0u) ? dataBias[real_oc] : 0.0;

    int in_h_base = int(oh * pc.stride_h) - int(pc.pad_h);
    int in_w_base = int(ow * pc.stride_w) - int(pc.pad_w);

    uint in_batch_offset = n * pc.C_in * pc.H_in * pc.W_in;
    uint w_out_offset = real_oc * pc.C_in * pc.kH * pc.kW;

    for (uint ic = 0; ic < pc.C_in; ++ic) {
        uint in_ch_offset = in_batch_offset + ic * pc.H_in * pc.W_in;
        uint w_ch_offset = w_out_offset + ic * pc.kH * pc.kW;

        for (uint kh = 0; kh < pc.kH; ++kh) {
            int in_h = in_h_base + int(kh);
            if (in_h >= 0 && in_h < int(pc.H_in)) {
                uint in_row_offset = in_ch_offset + uint(in_h) * pc.W_in;
                uint w_row_offset = w_ch_offset + kh * pc.kW;

                for (uint kw = 0; kw < pc.kW; ++kw) {
                    int in_w = in_w_base + int(kw);
                    if (in_w >= 0 && in_w < int(pc.W_in)) {
                        acc += dataIn[in_row_offset + uint(in_w)] * dataWeight[w_row_offset + kw];
                    }
                }
            }
        }
    }

    uint out_idx = n * (pc.C_out * pc.H_out * pc.W_out) +
                   real_oc * (pc.H_out * pc.W_out) +
                   oh * pc.W_out + ow;
    dataOut[out_idx] = acc;
}
