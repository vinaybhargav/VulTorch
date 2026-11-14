struct Conv2dBackwardParams {
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
};

[[vk::push_constant]] Conv2dBackwardParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> buf0; // gradOut
[[vk::binding(1, 0)]] StructuredBuffer<float> buf1; // weight (input bwd) or dataIn (weight bwd)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> buf2; // gradIn or gradWeight

[shader("compute")] [numthreads(8, 8, 4)]
void conv2d_backward_input(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint iw = tid.x;
    uint ih = tid.y;
    uint ic_groups = (pc.C_in + 3u) / 4u;
    uint ic = (gid.z % ic_groups) * 4u + lid.z;
    uint n = gid.z / ic_groups;

    if (iw >= pc.W_in || ih >= pc.H_in || ic >= pc.C_in || n >= pc.N) return;

    float acc = 0.0;

    for (uint oc = 0; oc < pc.C_out; ++oc) {
        uint grad_oc_offset = n * (pc.C_out * pc.H_out * pc.W_out) + oc * (pc.H_out * pc.W_out);
        uint w_oc_offset = oc * (pc.C_in * pc.kH * pc.kW) + ic * (pc.kH * pc.kW);

        for (uint kh = 0; kh < pc.kH; ++kh) {
            int h_padded = int(ih) + int(pc.pad_h) - int(kh);
            if (h_padded >= 0 && (h_padded % int(pc.stride_h)) == 0) {
                int oh = h_padded / int(pc.stride_h);
                if (oh >= 0 && oh < int(pc.H_out)) {
                    uint grad_row_offset = grad_oc_offset + uint(oh) * pc.W_out;
                    uint w_row_offset = w_oc_offset + kh * pc.kW;

                    for (uint kw = 0; kw < pc.kW; ++kw) {
                        int w_padded = int(iw) + int(pc.pad_w) - int(kw);
                        if (w_padded >= 0 && (w_padded % int(pc.stride_w)) == 0) {
                            int ow = w_padded / int(pc.stride_w);
                            if (ow >= 0 && ow < int(pc.W_out)) {
                                acc += buf0[grad_row_offset + uint(ow)] * buf1[w_row_offset + kw];
                            }
                        }
                    }
                }
            }
        }
    }

    uint in_idx = n * (pc.C_in * pc.H_in * pc.W_in) + ic * (pc.H_in * pc.W_in) + ih * pc.W_in + iw;
    buf2[in_idx] = acc;
}

[shader("compute")] [numthreads(16, 16, 1)]
void conv2d_backward_weight(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint kw = lid.x;
    uint kh = lid.y;

    uint pair_idx = gid.x;
    uint total_pairs = pc.C_out * pc.C_in;
    if (pair_idx >= total_pairs || kh >= pc.kH || kw >= pc.kW) return;

    uint oc = pair_idx / pc.C_in;
    uint ic = pair_idx % pc.C_in;

    float acc = 0.0;

    for (uint n = 0; n < pc.N; ++n) {
        uint grad_base = n * (pc.C_out * pc.H_out * pc.W_out) + oc * (pc.H_out * pc.W_out);
        uint in_base = n * (pc.C_in * pc.H_in * pc.W_in) + ic * (pc.H_in * pc.W_in);

        for (uint oh = 0; oh < pc.H_out; ++oh) {
            int in_h = int(oh * pc.stride_h) - int(pc.pad_h) + int(kh);
            if (in_h >= 0 && in_h < int(pc.H_in)) {
                uint grad_row = grad_base + oh * pc.W_out;
                uint in_row = in_base + uint(in_h) * pc.W_in;

                for (uint ow = 0; ow < pc.W_out; ++ow) {
                    int in_w = int(ow * pc.stride_w) - int(pc.pad_w) + int(kw);
                    if (in_w >= 0 && in_w < int(pc.W_in)) {
                        acc += buf0[grad_row + ow] * buf1[in_row + uint(in_w)];
                    }
                }
            }
        }
    }

    uint w_idx = oc * (pc.C_in * pc.kH * pc.kW) + ic * (pc.kH * pc.kW) + kh * pc.kW + kw;
    buf2[w_idx] = acc;
}
