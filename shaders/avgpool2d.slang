struct AvgPool2dParams {
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
    uint count_include_pad;
    uint is_adaptive;
};

[[vk::push_constant]] AvgPool2dParams pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> dataIn;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dataOut;

[shader("compute")] [numthreads(16, 16, 1)]
void avgpool2d(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    uint ow = tid.x;
    uint oh = tid.y;
    uint nc = gid.z;

    if (ow >= pc.W_out || oh >= pc.H_out || nc >= (pc.N * pc.C)) return;

    uint in_nc_offset = nc * (pc.H_in * pc.W_in);

    int ih_start, ih_end, iw_start, iw_end;
    if (pc.is_adaptive != 0u) {
        ih_start = int((oh * pc.H_in) / pc.H_out);
        ih_end = int(((oh + 1u) * pc.H_in + pc.H_out - 1u) / pc.H_out);
        iw_start = int((ow * pc.W_in) / pc.W_out);
        iw_end = int(((ow + 1u) * pc.W_in + pc.W_out - 1u) / pc.W_out);
    } else {
        ih_start = int(oh * pc.stride_h) - int(pc.pad_h);
        iw_start = int(ow * pc.stride_w) - int(pc.pad_w);
        ih_end = ih_start + int(pc.kH);
        iw_end = iw_start + int(pc.kW);
    }

    int ih_clamped_start = max(ih_start, 0);
    int iw_clamped_start = max(iw_start, 0);
    int ih_clamped_end = min(ih_end, int(pc.H_in));
    int iw_clamped_end = min(iw_end, int(pc.W_in));

    float sum = 0.0;
    for (int ih = ih_clamped_start; ih < ih_clamped_end; ++ih) {
        for (int iw = iw_clamped_start; iw < iw_clamped_end; ++iw) {
            sum += dataIn[in_nc_offset + uint(ih) * pc.W_in + uint(iw)];
        }
    }

    int denom;
    if (pc.is_adaptive != 0u || pc.count_include_pad == 0u) {
        denom = max(1, (ih_clamped_end - ih_clamped_start) * (iw_clamped_end - iw_clamped_start));
    } else {
        denom = int(pc.kH * pc.kW);
    }

    uint out_idx = nc * (pc.H_out * pc.W_out) + oh * pc.W_out + ow;
    dataOut[out_idx] = sum / float(denom);
}

[shader("compute")] [numthreads(16, 16, 1)]
void avgpool2d_backward(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID) {
    uint iw = tid.x;
    uint ih = tid.y;
    uint nc = gid.z;

    if (iw >= pc.W_in || ih >= pc.H_in || nc >= (pc.N * pc.C)) return;

    uint in_idx = nc * (pc.H_in * pc.W_in) + ih * pc.W_in + iw;
    uint out_nc_offset = nc * (pc.H_out * pc.W_out);

    float sum = 0.0;

    if (pc.is_adaptive != 0u) {
        for (uint oh = 0; oh < pc.H_out; ++oh) {
            uint ih_s = (oh * pc.H_in) / pc.H_out;
            uint ih_e = ((oh + 1u) * pc.H_in + pc.H_out - 1u) / pc.H_out;
            if (ih < ih_s || ih >= ih_e) continue;
            for (uint ow = 0; ow < pc.W_out; ++ow) {
                uint iw_s = (ow * pc.W_in) / pc.W_out;
                uint iw_e = ((ow + 1u) * pc.W_in + pc.W_out - 1u) / pc.W_out;
                if (iw < iw_s || iw >= iw_e) continue;
                uint area = max(1u, (ih_e - ih_s) * (iw_e - iw_s));
                sum += gradOut_or_in(out_nc_offset + oh * pc.W_out + ow) / float(area);
            }
        }
    } else {
        int h_padded = int(ih + pc.pad_h);
        int w_padded = int(iw + pc.pad_w);

        uint oh_start = (h_padded < int(pc.kH)) ? 0u : uint((h_padded - int(pc.kH)) / int(pc.stride_h) + 1);
        uint oh_end = min(uint(h_padded / int(pc.stride_h)) + 1u, pc.H_out);

        uint ow_start = (w_padded < int(pc.kW)) ? 0u : uint((w_padded - int(pc.kW)) / int(pc.stride_w) + 1);
        uint ow_end = min(uint(w_padded / int(pc.stride_w)) + 1u, pc.W_out);

        for (uint oh = oh_start; oh < oh_end; ++oh) {
            int ih_s = max(int(oh * pc.stride_h) - int(pc.pad_h), 0);
            int ih_e = min(int(oh * pc.stride_h) - int(pc.pad_h) + int(pc.kH), int(pc.H_in));
            for (uint ow = ow_start; ow < ow_end; ++ow) {
                int iw_s = max(int(ow * pc.stride_w) - int(pc.pad_w), 0);
                int iw_e = min(int(ow * pc.stride_w) - int(pc.pad_w) + int(pc.kW), int(pc.W_in));
                int denom = (pc.count_include_pad != 0u)
                                ? int(pc.kH * pc.kW)
                                : max(1, (ih_e - ih_s) * (iw_e - iw_s));
                sum += dataIn[out_nc_offset + oh * pc.W_out + ow] / float(denom);
            }
        }
    }

    dataOut[in_idx] = sum;
}

float gradOut_or_in(uint idx) {
    return dataIn[idx];
}
