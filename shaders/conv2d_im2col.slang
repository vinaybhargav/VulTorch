// High-throughput im2col & zero-atomic gather col2im kernels for Implicit-GEMM Conv2d.
// Output col layout: [N, C_in * kH * kW, H_out * W_out] so that
// weight [1, C_out, K] @ col [N, K, HW_out] directly produces [N, C_out, H_out, W_out]
// with zero intermediate transposes.

struct Im2ColParams {
    uint N;
    uint C_in;
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

[[vk::binding(0, 0)]] StructuredBuffer<float> icIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> icOut;

[shader("compute")] [numthreads(256, 1, 1)]
void im2col_kernel(uint3 tid : SV_DispatchThreadID, uniform Im2ColParams pc) {
    uint K = pc.C_in * pc.kH * pc.kW;
    uint HW_out = pc.H_out * pc.W_out;
    uint total = pc.N * K * HW_out;
    uint idx = tid.x;
    if (idx >= total) return;

    uint hw = idx % HW_out;
    uint rem = idx / HW_out;
    uint k = rem % K;
    uint n = rem / K;

    uint oh = hw / pc.W_out;
    uint ow = hw % pc.W_out;

    uint kw = k % pc.kW;
    uint k_rem = k / pc.kW;
    uint kh = k_rem % pc.kH;
    uint c = k_rem / pc.kH;

    int ih = int(oh * pc.stride_h + kh) - int(pc.pad_h);
    int iw = int(ow * pc.stride_w + kw) - int(pc.pad_w);

    float val = 0.0;
    if (ih >= 0 && ih < int(pc.H_in) && iw >= 0 && iw < int(pc.W_in)) {
        uint in_idx = ((n * pc.C_in + c) * pc.H_in + uint(ih)) * pc.W_in + uint(iw);
        val = icIn[in_idx];
    }
    icOut[idx] = val;
}

// Zero-atomic gather col2im: 1 thread per input element (n, c, ih, iw).
[shader("compute")] [numthreads(256, 1, 1)]
void col2im_kernel(uint3 tid : SV_DispatchThreadID, uniform Im2ColParams pc) {
    uint total = pc.N * pc.C_in * pc.H_in * pc.W_in;
    uint idx = tid.x;
    if (idx >= total) return;

    uint iw = idx % pc.W_in;
    uint rem = idx / pc.W_in;
    uint ih = rem % pc.H_in;
    rem /= pc.H_in;
    uint c = rem % pc.C_in;
    uint n = rem / pc.C_in;

    uint K = pc.C_in * pc.kH * pc.kW;
    uint HW_out = pc.H_out * pc.W_out;
    uint col_base = (n * K + c * pc.kH * pc.kW) * HW_out;

    float sum = 0.0;
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
                            uint k_local = kh * pc.kW + kw;
                            sum += icIn[col_base + k_local * HW_out + (oh * pc.W_out + ow)];
                        }
                    }
                }
            }
        }
    }
    icOut[idx] = sum;
}
