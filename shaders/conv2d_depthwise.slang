// Grouped / Depthwise Conv2d Forward & Backward Kernels (groups > 1).

struct GroupedConvParams {
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
    uint groups;
    uint has_bias;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> gcA;
[[vk::binding(1, 0)]] StructuredBuffer<float> gcB;
[[vk::binding(2, 0)]] StructuredBuffer<float> gcBias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gcOut;

[shader("compute")] [numthreads(8, 8, 4)]
void conv2d_grouped_forward(uint3 tid : SV_DispatchThreadID, uniform GroupedConvParams pc) {
    uint ow = tid.x;
    uint oh = tid.y;
    uint noc = tid.z;
    if (ow >= pc.W_out || oh >= pc.H_out || noc >= (pc.N * pc.C_out)) return;

    uint n = noc / pc.C_out;
    uint oc = noc % pc.C_out;
    uint c_out_per_g = pc.C_out / pc.groups;
    uint c_in_per_g = pc.C_in / pc.groups;
    uint g = oc / c_out_per_g;
    uint ic_start = g * c_in_per_g;

    float acc = (pc.has_bias != 0u) ? gcBias[oc] : 0.0;
    int ih_base = int(oh * pc.stride_h) - int(pc.pad_h);
    int iw_base = int(ow * pc.stride_w) - int(pc.pad_w);

    for (uint ic_rel = 0u; ic_rel < c_in_per_g; ++ic_rel) {
        uint ic = ic_start + ic_rel;
        for (uint kh = 0u; kh < pc.kH; ++kh) {
            int ih = ih_base + int(kh);
            if (ih >= 0 && ih < int(pc.H_in)) {
                for (uint kw = 0u; kw < pc.kW; ++kw) {
                    int iw = iw_base + int(kw);
                    if (iw >= 0 && iw < int(pc.W_in)) {
                        uint in_idx = ((n * pc.C_in + ic) * pc.H_in + uint(ih)) * pc.W_in + uint(iw);
                        uint w_idx = ((oc * c_in_per_g + ic_rel) * pc.kH + kh) * pc.kW + kw;
                        acc += gcA[in_idx] * gcB[w_idx];
                    }
                }
            }
        }
    }
    uint out_idx = ((n * pc.C_out + oc) * pc.H_out + oh) * pc.W_out + ow;
    gcOut[out_idx] = acc;
}

[shader("compute")] [numthreads(8, 8, 4)]
void conv2d_grouped_backward_input(uint3 tid : SV_DispatchThreadID, uniform GroupedConvParams pc) {
    uint iw = tid.x;
    uint ih = tid.y;
    uint nic = tid.z;
    if (iw >= pc.W_in || ih >= pc.H_in || nic >= (pc.N * pc.C_in)) return;

    uint n = nic / pc.C_in;
    uint ic = nic % pc.C_in;
    uint c_out_per_g = pc.C_out / pc.groups;
    uint c_in_per_g = pc.C_in / pc.groups;
    uint g = ic / c_in_per_g;
    uint ic_rel = ic % c_in_per_g;
    uint oc_start = g * c_out_per_g;

    float acc = 0.0;
    for (uint oc_rel = 0u; oc_rel < c_out_per_g; ++oc_rel) {
        uint oc = oc_start + oc_rel;
        for (uint kh = 0u; kh < pc.kH; ++kh) {
            int oh_num = int(ih + pc.pad_h) - int(kh);
            if (oh_num >= 0 && (oh_num % int(pc.stride_h)) == 0) {
                uint oh = uint(oh_num) / pc.stride_h;
                if (oh < pc.H_out) {
                    for (uint kw = 0u; kw < pc.kW; ++kw) {
                        int ow_num = int(iw + pc.pad_w) - int(kw);
                        if (ow_num >= 0 && (ow_num % int(pc.stride_w)) == 0) {
                            uint ow = uint(ow_num) / pc.stride_w;
                            if (ow < pc.W_out) {
                                uint gout_idx = ((n * pc.C_out + oc) * pc.H_out + oh) * pc.W_out + ow;
                                uint w_idx = ((oc * c_in_per_g + ic_rel) * pc.kH + kh) * pc.kW + kw;
                                acc += gcA[gout_idx] * gcB[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    uint gin_idx = ((n * pc.C_in + ic) * pc.H_in + ih) * pc.W_in + iw;
    gcOut[gin_idx] = acc;
}

[shader("compute")] [numthreads(256, 1, 1)]
void conv2d_grouped_backward_weight(uint3 tid : SV_DispatchThreadID, uniform GroupedConvParams pc) {
    uint c_in_per_g = pc.C_in / pc.groups;
    uint c_out_per_g = pc.C_out / pc.groups;
    uint total_w = pc.C_out * c_in_per_g * pc.kH * pc.kW;
    uint w_idx = tid.x;
    if (w_idx >= total_w) return;

    uint kw = w_idx % pc.kW;
    uint rem = w_idx / pc.kW;
    uint kh = rem % pc.kH;
    rem /= pc.kH;
    uint ic_rel = rem % c_in_per_g;
    uint oc = rem / c_in_per_g;
    uint g = oc / c_out_per_g;
    uint ic = g * c_in_per_g + ic_rel;

    float acc = 0.0;
    for (uint n = 0u; n < pc.N; ++n) {
        for (uint oh = 0u; oh < pc.H_out; ++oh) {
            int ih = int(oh * pc.stride_h + kh) - int(pc.pad_h);
            if (ih >= 0 && ih < int(pc.H_in)) {
                for (uint ow = 0u; ow < pc.W_out; ++ow) {
                    int iw = int(ow * pc.stride_w + kw) - int(pc.pad_w);
                    if (iw >= 0 && iw < int(pc.W_in)) {
                        uint in_idx = ((n * pc.C_in + ic) * pc.H_in + uint(ih)) * pc.W_in + uint(iw);
                        uint gout_idx = ((n * pc.C_out + oc) * pc.H_out + oh) * pc.W_out + ow;
                        acc += gcB[in_idx] * gcA[gout_idx];
                    }
                }
            }
        }
    }
    gcOut[w_idx] = acc;
}
