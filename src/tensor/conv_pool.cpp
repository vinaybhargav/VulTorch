#include "tensor/tensor_internal.hpp"

namespace vultorch
{

std::shared_ptr<NativeTensor> NativeTensor::conv2d(std::shared_ptr<NativeTensor> weight,
                                                   std::shared_ptr<NativeTensor> bias,
                                                   uint32_t stride_h, uint32_t stride_w,
                                                   uint32_t pad_h, uint32_t pad_w, uint32_t groups,
                                                   bool use_im2col)
{
    if (shape.size() != 4 || weight->shape.size() != 4)
    {
        throw std::runtime_error("conv2d requires 4D tensors [N, C, H, W]");
    }
    if (dtype != ScalarType::Float32 || weight->dtype != ScalarType::Float32 ||
        (bias && bias->dtype != ScalarType::Float32))
    {
        if (!is_floating_point(dtype))
        {
            throw std::runtime_error("conv2d requires floating-point tensors");
        }
        auto in_f32 =
            (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
        auto w_f32 =
            (weight->dtype == ScalarType::Float32) ? weight : weight->cast(ScalarType::Float32);
        auto b_f32 =
            (bias && bias->dtype != ScalarType::Float32) ? bias->cast(ScalarType::Float32) : bias;
        return in_f32->conv2d(w_f32, b_f32, stride_h, stride_w, pad_h, pad_w, groups, use_im2col)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->conv2d(weight, bias, stride_h, stride_w, pad_h, pad_w, groups,
                                          use_im2col);
    }
    auto w = weight->force_contiguous();
    auto bias_t = bias ? bias->force_contiguous() : nullptr;

    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t C_in = static_cast<uint32_t>(shape[1]);
    uint32_t H_in = static_cast<uint32_t>(shape[2]);
    uint32_t W_in = static_cast<uint32_t>(shape[3]);

    uint32_t C_out = static_cast<uint32_t>(w->shape[0]);
    uint32_t kH = static_cast<uint32_t>(w->shape[2]);
    uint32_t kW = static_cast<uint32_t>(w->shape[3]);

    uint32_t H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    uint32_t W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    std::vector<int64_t> out_shape = {static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                                      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};

    if (device_id >= 0 && groups > 1u)
    {
        auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
        struct
        {
            uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
        } pc{N,        C_in,  H_in,  W_in,   C_out,
             H_out,    W_out, kH,    kW,     stride_h,
             stride_w, pad_h, pad_w, groups, (bias_t != nullptr) ? 1u : 0u};
        uint32_t gx = (W_out + 7) / 8;
        uint32_t gy = (H_out + 7) / 8;
        uint32_t gz = (N * C_out + 3) / 4;
        std::shared_ptr<VulkanBuffer> b_bias = bias_t ? bias_t->gpu_buf() : out->gpu_buf();
        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_grouped_forward", {gpu_buf(), w->gpu_buf(), b_bias, out->gpu_buf()},
            &pc, sizeof(pc), gx, gy, gz);
        return out;
    }

    if (device_id >= 0 && groups == 1u && use_im2col && (C_in >= 16u || N * H_out * W_out > 4096u))
    {
        uint32_t K = C_in * kH * kW;
        uint32_t HW_out = H_out * W_out;
        const size_t per_sample =
            std::max<size_t>(static_cast<size_t>(K) * HW_out, static_cast<size_t>(C_out) * HW_out);
        const size_t max_ws_elems = static_cast<size_t>(
            VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() /
            sizeof(float));
        const uint32_t chunk_n = static_cast<uint32_t>(
            std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
        if (N > chunk_n)
        {
            std::vector<std::shared_ptr<NativeTensor>> parts;
            for (uint32_t n0 = 0; n0 < N; n0 += chunk_n)
            {
                uint32_t n1 = std::min(N, n0 + chunk_n);
                auto in_c = slice_dim0(n0, n1);
                parts.push_back(
                    in_c->conv2d(w, bias_t, stride_h, stride_w, pad_h, pad_w, groups, true));
            }
            return cat_dim0(parts, device_id);
        }
        auto col = std::make_shared<NativeTensor>(
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(K),
                                 static_cast<int64_t>(HW_out)},
            device_id, ScalarType::Float32);
        struct
        {
            uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w;
        } pc{N, C_in, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w};
        uint32_t gx = (static_cast<uint32_t>(col->numel) + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(
            device_id, "im2col_kernel", {gpu_buf(), col->gpu_buf()}, &pc, sizeof(pc), gx);

        auto w_3d = w->reshape_view({1, static_cast<int64_t>(C_out), static_cast<int64_t>(K)})
                        ->expand_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                                       static_cast<int64_t>(K)});
        auto out_3d = w_3d->matmul(col);
        auto out = out_3d->reshape_view(out_shape);
        if (bias_t)
        {
            out = out->binary_op(0, bias_t->reshape_view({1, static_cast<int64_t>(C_out), 1, 1}));
        }
        return out;
    }

    auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N;
            uint32_t C_in;
            uint32_t H_in;
            uint32_t W_in;
            uint32_t C_out;
            uint32_t H_out;
            uint32_t W_out;
            uint32_t kH;
            uint32_t kW;
            uint32_t stride_h;
            uint32_t stride_w;
            uint32_t pad_h;
            uint32_t pad_w;
            uint32_t has_bias;
        } pc{};

        pc.N = N;
        pc.C_in = C_in;
        pc.H_in = H_in;
        pc.W_in = W_in;
        pc.C_out = C_out;
        pc.H_out = H_out;
        pc.W_out = W_out;
        pc.kH = kH;
        pc.kW = kW;
        pc.stride_h = stride_h;
        pc.stride_w = stride_w;
        pc.pad_h = pad_h;
        pc.pad_w = pad_w;
        pc.has_bias = (bias_t != nullptr) ? 1u : 0u;

        uint32_t gx = (W_out + 7) / 8;
        uint32_t gy = (H_out + 7) / 8;
        uint32_t gz = N * ((C_out + 3) / 4);

        std::shared_ptr<VulkanBuffer> b_bias = bias_t ? bias_t->gpu_buf() : out->gpu_buf();

        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d", {gpu_buf(), w->gpu_buf(), b_bias, out->gpu_buf()}, &pc, sizeof(pc),
            gx, gy, gz);
    }
    else
    {
        const float *in_ptr = cpu_f32();
        const float *w_ptr = w->cpu_f32();
        const float *b_ptr = bias_t ? bias_t->cpu_f32() : nullptr;
        float *out_ptr = out->cpu_f32();
        const uint32_t c_out_per_g = C_out / groups;
        const uint32_t c_in_per_g = C_in / groups;
        for (uint32_t n = 0; n < N; ++n)
        {
            for (uint32_t oc = 0; oc < C_out; ++oc)
            {
                const uint32_t g = oc / c_out_per_g;
                const uint32_t ic_start = g * c_in_per_g;
                float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
                for (uint32_t oh = 0; oh < H_out; ++oh)
                {
                    for (uint32_t ow = 0; ow < W_out; ++ow)
                    {
                        float acc = bias_val;
                        int in_h_base = int(oh * stride_h) - int(pad_h);
                        int in_w_base = int(ow * stride_w) - int(pad_w);

                        for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel)
                        {
                            uint32_t ic = ic_start + ic_rel;
                            for (uint32_t kh = 0; kh < kH; ++kh)
                            {
                                int ih = in_h_base + int(kh);
                                if (ih >= 0 && ih < int(H_in))
                                {
                                    for (uint32_t kw = 0; kw < kW; ++kw)
                                    {
                                        int iw = in_w_base + int(kw);
                                        if (iw >= 0 && iw < int(W_in))
                                        {
                                            acc += in_ptr[n * (C_in * H_in * W_in) +
                                                          ic * (H_in * W_in) + ih * W_in + iw] *
                                                   w_ptr[oc * (c_in_per_g * kH * kW) +
                                                         ic_rel * (kH * kW) + kh * kW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        out_ptr[n * (C_out * H_out * W_out) + oc * (H_out * W_out) + oh * W_out +
                                ow] = acc;
                    }
                }
            }
        }
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::conv2d_backward_input(
    std::shared_ptr<NativeTensor> weight, const std::vector<int64_t> &in_shape, uint32_t stride_h,
    uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t groups, bool use_im2col)
{
    if (dtype != ScalarType::Float32 || weight->dtype != ScalarType::Float32)
    {
        auto g_f32 =
            (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
        auto w_f32 =
            (weight->dtype == ScalarType::Float32) ? weight : weight->cast(ScalarType::Float32);
        return g_f32
            ->conv2d_backward_input(w_f32, in_shape, stride_h, stride_w, pad_h, pad_w, groups,
                                    use_im2col)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->conv2d_backward_input(weight, in_shape, stride_h, stride_w,
                                                         pad_h, pad_w, groups, use_im2col);
    }
    auto w_t = weight->force_contiguous();

    auto grad_in = std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);
    grad_in->zero_();

    uint32_t N = static_cast<uint32_t>(in_shape[0]);
    uint32_t C_in = static_cast<uint32_t>(in_shape[1]);
    uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
    uint32_t W_in = static_cast<uint32_t>(in_shape[3]);

    uint32_t C_out = static_cast<uint32_t>(w_t->shape[0]);
    uint32_t kH = static_cast<uint32_t>(w_t->shape[2]);
    uint32_t kW = static_cast<uint32_t>(w_t->shape[3]);

    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

    if (device_id >= 0 && groups > 1u)
    {
        struct
        {
            uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
        } pc{N,  C_in,     H_in,     W_in,  C_out, H_out,  W_out, kH,
             kW, stride_h, stride_w, pad_h, pad_w, groups, 0u};
        uint32_t gx = (W_in + 7) / 8;
        uint32_t gy = (H_in + 7) / 8;
        uint32_t gz = (N * C_in + 3) / 4;
        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_grouped_backward_input",
            {gpu_buf(), w_t->gpu_buf(), grad_in->gpu_buf(), grad_in->gpu_buf()}, &pc, sizeof(pc),
            gx, gy, gz);
        return grad_in;
    }

    if (device_id >= 0 && groups == 1u && use_im2col && (C_in >= 16u || N * H_out * W_out > 4096u))
    {
        uint32_t K = C_in * kH * kW;
        uint32_t HW_out = H_out * W_out;
        const size_t per_sample =
            std::max<size_t>(static_cast<size_t>(K) * HW_out, static_cast<size_t>(C_out) * HW_out);
        const size_t max_ws_elems = static_cast<size_t>(
            VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() /
            sizeof(float));
        const uint32_t chunk_n = static_cast<uint32_t>(
            std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
        if (N > chunk_n)
        {
            std::vector<std::shared_ptr<NativeTensor>> parts;
            for (uint32_t n0 = 0; n0 < N; n0 += chunk_n)
            {
                uint32_t n1 = std::min(N, n0 + chunk_n);
                std::vector<int64_t> chunk_in = {
                    static_cast<int64_t>(n1 - n0), static_cast<int64_t>(C_in),
                    static_cast<int64_t>(H_in), static_cast<int64_t>(W_in)};
                auto g_c = slice_dim0(n0, n1);
                parts.push_back(g_c->conv2d_backward_input(w_t, chunk_in, stride_h, stride_w, pad_h,
                                                           pad_w, groups, true));
            }
            return cat_dim0(parts, device_id);
        }
        auto g_3d = reshape_view(
            {static_cast<int64_t>(N), static_cast<int64_t>(C_out), static_cast<int64_t>(HW_out)});
        auto w_3d = w_t->reshape_view({1, static_cast<int64_t>(C_out), static_cast<int64_t>(K)})
                        ->expand_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                                       static_cast<int64_t>(K)});
        auto col_grad = w_3d->matmul(g_3d, true, false);
        struct
        {
            uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w;
        } pc{N, C_in, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w};
        uint32_t gx = (static_cast<uint32_t>(grad_in->numel) + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "col2im_kernel",
                                                   {col_grad->gpu_buf(), grad_in->gpu_buf()}, &pc,
                                                   sizeof(pc), gx);
        return grad_in;
    }

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N;
            uint32_t C_in;
            uint32_t H_in;
            uint32_t W_in;
            uint32_t C_out;
            uint32_t H_out;
            uint32_t W_out;
            uint32_t kH;
            uint32_t kW;
            uint32_t stride_h;
            uint32_t stride_w;
            uint32_t pad_h;
            uint32_t pad_w;
        } pc{};

        pc.N = N;
        pc.C_in = C_in;
        pc.H_in = H_in;
        pc.W_in = W_in;
        pc.C_out = C_out;
        pc.H_out = H_out;
        pc.W_out = W_out;
        pc.kH = kH;
        pc.kW = kW;
        pc.stride_h = stride_h;
        pc.stride_w = stride_w;
        pc.pad_h = pad_h;
        pc.pad_w = pad_w;

        uint32_t gx = (W_in + 7) / 8;
        uint32_t gy = (H_in + 7) / 8;
        uint32_t gz = N * ((C_in + 3) / 4);

        VulkanPipelineManager::instance().dispatch(device_id, "conv2d_backward_input",
                                                   {gpu_buf(), w_t->gpu_buf(), grad_in->gpu_buf()},
                                                   &pc, sizeof(pc), gx, gy, gz);
    }
    else
    {
        const float *gout_ptr = cpu_f32();
        const float *w_ptr = w_t->cpu_f32();
        float *gin_ptr = grad_in->cpu_f32();
        const uint32_t c_out_per_g = C_out / groups;
        const uint32_t c_in_per_g = C_in / groups;
        for (uint32_t n = 0; n < N; ++n)
        {
            for (uint32_t oc = 0; oc < C_out; ++oc)
            {
                const uint32_t g_idx = oc / c_out_per_g;
                const uint32_t ic_start = g_idx * c_in_per_g;
                for (uint32_t oh = 0; oh < H_out; ++oh)
                {
                    for (uint32_t ow = 0; ow < W_out; ++ow)
                    {
                        const float g = gout_ptr[((n * C_out + oc) * H_out + oh) * W_out + ow];
                        const int in_h_base =
                            static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        const int in_w_base =
                            static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

                        for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel)
                        {
                            const uint32_t ic = ic_start + ic_rel;
                            for (uint32_t kh = 0; kh < kH; ++kh)
                            {
                                const int ih = in_h_base + static_cast<int>(kh);
                                if (ih < 0 || ih >= static_cast<int>(H_in))
                                    continue;
                                for (uint32_t kw = 0; kw < kW; ++kw)
                                {
                                    const int iw = in_w_base + static_cast<int>(kw);
                                    if (iw < 0 || iw >= static_cast<int>(W_in))
                                        continue;
                                    const float wv =
                                        w_ptr[((oc * c_in_per_g + ic_rel) * kH + kh) * kW + kw];
                                    gin_ptr[((n * C_in + ic) * H_in + ih) * W_in + iw] += g * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_in;
}

std::shared_ptr<NativeTensor>
NativeTensor::conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                     const std::vector<int64_t> &weight_shape, uint32_t stride_h,
                                     uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                                     uint32_t groups, bool use_im2col)
{
    if (dtype != ScalarType::Float32 || input->dtype != ScalarType::Float32)
    {
        auto g_f32 =
            (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
        auto in_f32 =
            (input->dtype == ScalarType::Float32) ? input : input->cast(ScalarType::Float32);
        return g_f32
            ->conv2d_backward_weight(in_f32, weight_shape, stride_h, stride_w, pad_h, pad_w, groups,
                                     use_im2col)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->conv2d_backward_weight(input, weight_shape, stride_h, stride_w,
                                                          pad_h, pad_w, groups, use_im2col);
    }
    auto in_t = input->force_contiguous();

    auto grad_weight = std::make_shared<NativeTensor>(weight_shape, device_id, ScalarType::Float32);
    grad_weight->zero_();

    uint32_t N = static_cast<uint32_t>(in_t->shape[0]);
    uint32_t C_in = static_cast<uint32_t>(in_t->shape[1]);
    uint32_t H_in = static_cast<uint32_t>(in_t->shape[2]);
    uint32_t W_in = static_cast<uint32_t>(in_t->shape[3]);

    uint32_t C_out = static_cast<uint32_t>(weight_shape[0]);
    uint32_t kH = static_cast<uint32_t>(weight_shape[2]);
    uint32_t kW = static_cast<uint32_t>(weight_shape[3]);

    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

    if (device_id >= 0 && groups > 1u)
    {
        struct
        {
            uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
        } pc{N,  C_in,     H_in,     W_in,  C_out, H_out,  W_out, kH,
             kW, stride_h, stride_w, pad_h, pad_w, groups, 0u};
        uint32_t gx = (static_cast<uint32_t>(grad_weight->numel) + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_grouped_backward_weight",
            {gpu_buf(), in_t->gpu_buf(), grad_weight->gpu_buf(), grad_weight->gpu_buf()}, &pc,
            sizeof(pc), gx);
        return grad_weight;
    }

    if (device_id >= 0 && groups == 1u && use_im2col && (C_in >= 16u || N * H_out * W_out > 4096u))
    {
        uint32_t K = C_in * kH * kW;
        uint32_t HW_out = H_out * W_out;
        const size_t per_sample =
            std::max<size_t>(static_cast<size_t>(K) * HW_out, static_cast<size_t>(C_out) * K);
        const size_t max_ws_elems = static_cast<size_t>(
            VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() /
            sizeof(float));
        const uint32_t chunk_n = static_cast<uint32_t>(
            std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
        if (N > chunk_n)
        {
            std::shared_ptr<NativeTensor> dw_acc = nullptr;
            for (uint32_t n0 = 0; n0 < N; n0 += chunk_n)
            {
                uint32_t n1 = std::min(N, n0 + chunk_n);
                auto in_c = in_t->slice_dim0(n0, n1);
                auto g_c = slice_dim0(n0, n1);
                auto dw_c = g_c->conv2d_backward_weight(in_c, weight_shape, stride_h, stride_w,
                                                        pad_h, pad_w, groups, true);
                dw_acc = dw_acc ? dw_acc->binary_op(0, dw_c) : dw_c;
            }
            return dw_acc;
        }
        auto col = std::make_shared<NativeTensor>(
            std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(K),
                                 static_cast<int64_t>(HW_out)},
            device_id, ScalarType::Float32);
        struct
        {
            uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
            uint32_t stride_h, stride_w, pad_h, pad_w;
        } pc{N, C_in, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w};
        uint32_t gx = (static_cast<uint32_t>(col->numel) + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(
            device_id, "im2col_kernel", {in_t->gpu_buf(), col->gpu_buf()}, &pc, sizeof(pc), gx);

        auto g_3d = reshape_view(
            {static_cast<int64_t>(N), static_cast<int64_t>(C_out), static_cast<int64_t>(HW_out)});
        auto dw_3d = g_3d->matmul(col, false, true);
        return dw_3d->reduce_sum(0, false)->reshape_view(weight_shape);
    }

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N;
            uint32_t C_in;
            uint32_t H_in;
            uint32_t W_in;
            uint32_t C_out;
            uint32_t H_out;
            uint32_t W_out;
            uint32_t kH;
            uint32_t kW;
            uint32_t stride_h;
            uint32_t stride_w;
            uint32_t pad_h;
            uint32_t pad_w;
        } pc{};

        pc.N = N;
        pc.C_in = C_in;
        pc.H_in = H_in;
        pc.W_in = W_in;
        pc.C_out = C_out;
        pc.H_out = H_out;
        pc.W_out = W_out;
        pc.kH = kH;
        pc.kW = kW;
        pc.stride_h = stride_h;
        pc.stride_w = stride_w;
        pc.pad_h = pad_h;
        pc.pad_w = pad_w;

        uint32_t total_pairs = C_out * C_in;
        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_backward_weight",
            {gpu_buf(), in_t->gpu_buf(), grad_weight->gpu_buf()}, &pc, sizeof(pc), total_pairs, 1,
            1);
    }
    else
    {
        const float *gout_ptr = cpu_f32();
        const float *in_ptr = in_t->cpu_f32();
        float *gw_ptr = grad_weight->cpu_f32();
        const uint32_t c_out_per_g = C_out / groups;
        const uint32_t c_in_per_g = C_in / groups;
        for (uint32_t n = 0; n < N; ++n)
        {
            for (uint32_t oc = 0; oc < C_out; ++oc)
            {
                const uint32_t g_idx = oc / c_out_per_g;
                const uint32_t ic_start = g_idx * c_in_per_g;
                for (uint32_t oh = 0; oh < H_out; ++oh)
                {
                    for (uint32_t ow = 0; ow < W_out; ++ow)
                    {
                        const float g = gout_ptr[((n * C_out + oc) * H_out + oh) * W_out + ow];
                        const int in_h_base =
                            static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        const int in_w_base =
                            static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

                        for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel)
                        {
                            const uint32_t ic = ic_start + ic_rel;
                            for (uint32_t kh = 0; kh < kH; ++kh)
                            {
                                const int ih = in_h_base + static_cast<int>(kh);
                                if (ih < 0 || ih >= static_cast<int>(H_in))
                                    continue;
                                for (uint32_t kw = 0; kw < kW; ++kw)
                                {
                                    const int iw = in_w_base + static_cast<int>(kw);
                                    if (iw < 0 || iw >= static_cast<int>(W_in))
                                        continue;
                                    const float x =
                                        in_ptr[((n * C_in + ic) * H_in + ih) * W_in + iw];
                                    gw_ptr[((oc * c_in_per_g + ic_rel) * kH + kh) * kW + kw] +=
                                        g * x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_weight;
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::maxpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
                        uint32_t pad_h, uint32_t pad_w)
{
    if (shape.size() != 4)
    {
        throw std::runtime_error("maxpool2d requires 4D input [N, C, H, W]");
    }
    if (dtype != ScalarType::Float32)
    {
        if (!is_floating_point(dtype))
        {
            throw std::runtime_error("maxpool2d requires floating-point tensor");
        }
        auto [out_f32, idx_f32] =
            cast(ScalarType::Float32)->maxpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w);
        return {out_f32->cast(dtype), idx_f32};
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->maxpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w);
    }

    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t C = static_cast<uint32_t>(shape[1]);
    uint32_t H_in = static_cast<uint32_t>(shape[2]);
    uint32_t W_in = static_cast<uint32_t>(shape[3]);

    uint32_t H_out = (H_in + 2u * pad_h - kH) / stride_h + 1u;
    uint32_t W_out = (W_in + 2u * pad_w - kW) / stride_w + 1u;

    std::vector<int64_t> out_shape = {static_cast<int64_t>(N), static_cast<int64_t>(C),
                                      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};
    auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
    auto indices = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w;
        } pc{N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w};

        uint32_t gx = (W_out + 15u) / 16u;
        uint32_t gy = (H_out + 15u) / 16u;
        uint32_t gz = N * C;

        VulkanPipelineManager::instance().dispatch(device_id, "maxpool2d",
                                                   {gpu_buf(), out->gpu_buf(), indices->gpu_buf()},
                                                   &pc, sizeof(pc), gx, gy, gz);
    }
    else
    {
        const float *in_ptr = cpu_f32();
        float *out_ptr = out->cpu_f32();
        float *idx_ptr = indices->cpu_f32();
        for (uint32_t nc = 0; nc < N * C; ++nc)
        {
            uint32_t in_nc_off = nc * H_in * W_in;
            uint32_t out_nc_off = nc * H_out * W_out;
            for (uint32_t oh = 0; oh < H_out; ++oh)
            {
                for (uint32_t ow = 0; ow < W_out; ++ow)
                {
                    int ih_base = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                    int iw_base = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                    float max_val = -3.402823466e+38f;
                    uint32_t max_idx = 0u;
                    for (uint32_t kh = 0; kh < kH; ++kh)
                    {
                        int ih = ih_base + static_cast<int>(kh);
                        if (ih >= 0 && ih < static_cast<int>(H_in))
                        {
                            for (uint32_t kw = 0; kw < kW; ++kw)
                            {
                                int iw = iw_base + static_cast<int>(kw);
                                if (iw >= 0 && iw < static_cast<int>(W_in))
                                {
                                    uint32_t cur = in_nc_off + static_cast<uint32_t>(ih) * W_in +
                                                   static_cast<uint32_t>(iw);
                                    float v = in_ptr[cur];
                                    if (v > max_val)
                                    {
                                        max_val = v;
                                        max_idx = cur;
                                    }
                                }
                            }
                        }
                    }
                    uint32_t out_idx = out_nc_off + oh * W_out + ow;
                    out_ptr[out_idx] = max_val;
                    float idx_bits = 0.0f;
                    std::memcpy(&idx_bits, &max_idx, sizeof(float));
                    idx_ptr[out_idx] = idx_bits;
                }
            }
        }
    }
    return {out, indices};
}

std::shared_ptr<NativeTensor> NativeTensor::maxpool2d_backward(
    std::shared_ptr<NativeTensor> indices, const std::vector<int64_t> &in_shape, uint32_t kH,
    uint32_t kW, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    if (dtype != ScalarType::Float32)
    {
        return cast(ScalarType::Float32)
            ->maxpool2d_backward(indices, in_shape, kH, kW, stride_h, stride_w, pad_h, pad_w)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->maxpool2d_backward(indices, in_shape, kH, kW, stride_h, stride_w,
                                                      pad_h, pad_w);
    }
    auto idx_t = indices->force_contiguous();

    uint32_t N = static_cast<uint32_t>(in_shape[0]);
    uint32_t C = static_cast<uint32_t>(in_shape[1]);
    uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
    uint32_t W_in = static_cast<uint32_t>(in_shape[3]);
    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

    auto grad_in = std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w;
        } pc{N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w};

        uint32_t gx = (W_in + 15u) / 16u;
        uint32_t gy = (H_in + 15u) / 16u;
        uint32_t gz = N * C;

        VulkanPipelineManager::instance().dispatch(
            device_id, "maxpool2d_backward", {gpu_buf(), idx_t->gpu_buf(), grad_in->gpu_buf()}, &pc,
            sizeof(pc), gx, gy, gz);
    }
    else
    {
        grad_in->zero_();
        const float *gout_ptr = cpu_f32();
        const float *idx_ptr = idx_t->cpu_f32();
        float *gin_ptr = grad_in->cpu_f32();
        for (size_t i = 0; i < numel; ++i)
        {
            uint32_t max_idx = 0u;
            std::memcpy(&max_idx, &idx_ptr[i], sizeof(uint32_t));
            if (max_idx < grad_in->numel)
            {
                gin_ptr[max_idx] += gout_ptr[i];
            }
        }
    }
    return grad_in;
}

std::shared_ptr<NativeTensor> NativeTensor::avgpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h,
                                                      uint32_t stride_w, uint32_t pad_h,
                                                      uint32_t pad_w, bool count_include_pad,
                                                      bool is_adaptive, uint32_t adaptive_out_h,
                                                      uint32_t adaptive_out_w)
{
    if (shape.size() != 4)
    {
        throw std::runtime_error("avgpool2d requires 4D input [N, C, H, W]");
    }
    if (dtype != ScalarType::Float32)
    {
        if (!is_floating_point(dtype))
        {
            throw std::runtime_error("avgpool2d requires floating-point tensor");
        }
        return cast(ScalarType::Float32)
            ->avgpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w, count_include_pad, is_adaptive,
                        adaptive_out_h, adaptive_out_w)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->avgpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w,
                                             count_include_pad, is_adaptive, adaptive_out_h,
                                             adaptive_out_w);
    }

    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t C = static_cast<uint32_t>(shape[1]);
    uint32_t H_in = static_cast<uint32_t>(shape[2]);
    uint32_t W_in = static_cast<uint32_t>(shape[3]);

    uint32_t H_out = is_adaptive ? adaptive_out_h : ((H_in + 2u * pad_h - kH) / stride_h + 1u);
    uint32_t W_out = is_adaptive ? adaptive_out_w : ((W_in + 2u * pad_w - kW) / stride_w + 1u);

    std::vector<int64_t> out_shape = {static_cast<int64_t>(N), static_cast<int64_t>(C),
                                      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};
    auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w;
            uint32_t count_include_pad, is_adaptive;
        } pc{N,
             C,
             H_in,
             W_in,
             H_out,
             W_out,
             kH,
             kW,
             stride_h,
             stride_w,
             pad_h,
             pad_w,
             count_include_pad ? 1u : 0u,
             is_adaptive ? 1u : 0u};

        uint32_t gx = (W_out + 15u) / 16u;
        uint32_t gy = (H_out + 15u) / 16u;
        uint32_t gz = N * C;

        VulkanPipelineManager::instance().dispatch(
            device_id, "avgpool2d", {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), gx, gy, gz);
    }
    else
    {
        const float *in_ptr = cpu_f32();
        float *out_ptr = out->cpu_f32();
        for (uint32_t nc = 0; nc < N * C; ++nc)
        {
            uint32_t in_nc_off = nc * H_in * W_in;
            uint32_t out_nc_off = nc * H_out * W_out;
            for (uint32_t oh = 0; oh < H_out; ++oh)
            {
                for (uint32_t ow = 0; ow < W_out; ++ow)
                {
                    int ih_s, ih_e, iw_s, iw_e;
                    if (is_adaptive)
                    {
                        ih_s = static_cast<int>((oh * H_in) / H_out);
                        ih_e = static_cast<int>(((oh + 1u) * H_in + H_out - 1u) / H_out);
                        iw_s = static_cast<int>((ow * W_in) / W_out);
                        iw_e = static_cast<int>(((ow + 1u) * W_in + W_out - 1u) / W_out);
                    }
                    else
                    {
                        ih_s = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        iw_s = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                        ih_e = ih_s + static_cast<int>(kH);
                        iw_e = iw_s + static_cast<int>(kW);
                    }
                    int ih_cs = std::max(ih_s, 0);
                    int iw_cs = std::max(iw_s, 0);
                    int ih_ce = std::min(ih_e, static_cast<int>(H_in));
                    int iw_ce = std::min(iw_e, static_cast<int>(W_in));

                    float sum = 0.0f;
                    for (int ih = ih_cs; ih < ih_ce; ++ih)
                    {
                        for (int iw = iw_cs; iw < iw_ce; ++iw)
                        {
                            sum += in_ptr[in_nc_off + static_cast<uint32_t>(ih) * W_in +
                                          static_cast<uint32_t>(iw)];
                        }
                    }
                    int denom = (is_adaptive || !count_include_pad)
                                    ? std::max(1, (ih_ce - ih_cs) * (iw_ce - iw_cs))
                                    : static_cast<int>(kH * kW);
                    out_ptr[out_nc_off + oh * W_out + ow] = sum / static_cast<float>(denom);
                }
            }
        }
    }
    return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::avgpool2d_backward(const std::vector<int64_t> &in_shape, uint32_t kH, uint32_t kW,
                                 uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
                                 uint32_t pad_w, bool count_include_pad, bool is_adaptive)
{
    if (dtype != ScalarType::Float32)
    {
        return cast(ScalarType::Float32)
            ->avgpool2d_backward(in_shape, kH, kW, stride_h, stride_w, pad_h, pad_w,
                                 count_include_pad, is_adaptive)
            ->cast(dtype);
    }
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->avgpool2d_backward(in_shape, kH, kW, stride_h, stride_w, pad_h,
                                                      pad_w, count_include_pad, is_adaptive);
    }

    uint32_t N = static_cast<uint32_t>(in_shape[0]);
    uint32_t C = static_cast<uint32_t>(in_shape[1]);
    uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
    uint32_t W_in = static_cast<uint32_t>(in_shape[3]);
    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

    auto grad_in = std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);

    if (device_id >= 0)
    {
        struct
        {
            uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w, pad_h, pad_w;
            uint32_t count_include_pad, is_adaptive;
        } pc{N,
             C,
             H_in,
             W_in,
             H_out,
             W_out,
             kH,
             kW,
             stride_h,
             stride_w,
             pad_h,
             pad_w,
             count_include_pad ? 1u : 0u,
             is_adaptive ? 1u : 0u};

        uint32_t gx = (W_in + 15u) / 16u;
        uint32_t gy = (H_in + 15u) / 16u;
        uint32_t gz = N * C;

        VulkanPipelineManager::instance().dispatch(device_id, "avgpool2d_backward",
                                                   {gpu_buf(), grad_in->gpu_buf()}, &pc, sizeof(pc),
                                                   gx, gy, gz);
    }
    else
    {
        grad_in->zero_();
        const float *gout_ptr = cpu_f32();
        float *gin_ptr = grad_in->cpu_f32();
        for (uint32_t nc = 0; nc < N * C; ++nc)
        {
            uint32_t in_nc_off = nc * H_in * W_in;
            uint32_t out_nc_off = nc * H_out * W_out;
            for (uint32_t oh = 0; oh < H_out; ++oh)
            {
                for (uint32_t ow = 0; ow < W_out; ++ow)
                {
                    int ih_s, ih_e, iw_s, iw_e;
                    if (is_adaptive)
                    {
                        ih_s = static_cast<int>((oh * H_in) / H_out);
                        ih_e = static_cast<int>(((oh + 1u) * H_in + H_out - 1u) / H_out);
                        iw_s = static_cast<int>((ow * W_in) / W_out);
                        iw_e = static_cast<int>(((ow + 1u) * W_in + W_out - 1u) / W_out);
                    }
                    else
                    {
                        ih_s = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        iw_s = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
                        ih_e = ih_s + static_cast<int>(kH);
                        iw_e = iw_s + static_cast<int>(kW);
                    }
                    int ih_cs = std::max(ih_s, 0);
                    int iw_cs = std::max(iw_s, 0);
                    int ih_ce = std::min(ih_e, static_cast<int>(H_in));
                    int iw_ce = std::min(iw_e, static_cast<int>(W_in));

                    int denom = (is_adaptive || !count_include_pad)
                                    ? std::max(1, (ih_ce - ih_cs) * (iw_ce - iw_cs))
                                    : static_cast<int>(kH * kW);
                    float g = gout_ptr[out_nc_off + oh * W_out + ow] / static_cast<float>(denom);
                    for (int ih = ih_cs; ih < ih_ce; ++ih)
                    {
                        for (int iw = iw_cs; iw < iw_ce; ++iw)
                        {
                            gin_ptr[in_nc_off + static_cast<uint32_t>(ih) * W_in +
                                    static_cast<uint32_t>(iw)] += g;
                        }
                    }
                }
            }
        }
    }
    return grad_in;
}

} // namespace vultorch
