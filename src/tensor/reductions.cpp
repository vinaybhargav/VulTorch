#include "tensor/tensor_internal.hpp"

namespace vultorch
{

std::shared_ptr<NativeTensor> NativeTensor::reduce_sum(int64_t dim, bool keepdim)
{
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->reduce_sum(dim, keepdim);
    }
    if (dtype != ScalarType::Float32)
    {
        return cast(ScalarType::Float32)->reduce_sum(dim, keepdim)->cast(dtype);
    }

    const int64_t TOTAL_DIM = -999999;
    if (dim == TOTAL_DIM)
    {
        std::vector<int64_t> out_shape =
            keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

        if (device_id >= 0)
        {
            struct
            {
                uint32_t num_rows;
                uint32_t cols;
                uint32_t op_type;
            } pc{1, static_cast<uint32_t>(numel), 0};

            VulkanPipelineManager::instance().dispatch(
                device_id, "reduction", {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), 1, 1, 1);
        }
        else
        {
            const float *in_ptr = cpu_f32();
            float sum = 0.0f;
            for (size_t i = 0; i < numel; ++i)
                sum += in_ptr[i];
            out->cpu_f32()[0] = sum;
        }
        return out;
    }

    if (dim < 0)
        dim += static_cast<int64_t>(shape.size());

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (static_cast<int64_t>(i) == dim)
        {
            if (keepdim)
                out_shape.push_back(1);
        }
        else
        {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty())
        out_shape.push_back(1);

    auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

    if (device_id >= 0 && dim == static_cast<int64_t>(shape.size()) - 1)
    {
        uint32_t cols = static_cast<uint32_t>(shape.back());
        uint32_t rows = static_cast<uint32_t>(numel / cols);

        struct
        {
            uint32_t num_rows;
            uint32_t cols;
            uint32_t op_type;
        } pc{rows, cols, 0};

        VulkanPipelineManager::instance().dispatch(
            device_id, "reduction", {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), 1, rows, 1);
        return out;
    }

    if (device_id >= 0)
    {
        uint32_t outer = 1;
        uint32_t inner = 1;
        for (int64_t i = 0; i < dim; ++i)
            outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
        for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i)
        {
            inner *= static_cast<uint32_t>(shape[i]);
        }

        struct
        {
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t op_type;
        } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner, 0u};

        if (inner >= 16u && outer <= 65535u)
        {
            VulkanPipelineManager::instance().dispatch(device_id, "reduction_strided",
                                                       {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
                                                       (inner + 31u) / 32u, outer);
        }
        else
        {
            VulkanPipelineManager::instance().dispatch(device_id, "reduction_generic",
                                                       {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
                                                       outer * inner);
        }
        return out;
    }

    // CPU fallback.
    auto cpu_out = std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Float32);
    cpu_out->zero_();
    const float *in_ptr = cpu_f32();
    float *out_ptr = cpu_out->cpu_f32();

    for (size_t i = 0; i < numel; ++i)
    {
        size_t rem_idx = i;
        size_t target_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            size_t c = rem_idx % static_cast<size_t>(shape[static_cast<size_t>(d)]);
            rem_idx /= static_cast<size_t>(shape[static_cast<size_t>(d)]);
            if (d != dim)
            {
                int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
                target_idx += c * static_cast<size_t>(cpu_out->strides[static_cast<size_t>(out_d)]);
            }
        }
        out_ptr[target_idx] += in_ptr[i];
    }

    return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_mean(int64_t dim, bool keepdim)
{
    const int64_t TOTAL_DIM = -999999;
    if (dtype != ScalarType::Float32)
    {
        auto f32 = cast(ScalarType::Float32)->reduce_mean(dim, keepdim);
        return is_floating_point(dtype) ? f32->cast(dtype) : f32;
    }
    auto sum = reduce_sum(dim, keepdim);
    float divisor = (dim == TOTAL_DIM)
                        ? static_cast<float>(numel)
                        : static_cast<float>(shape[static_cast<size_t>(
                              dim < 0 ? dim + static_cast<int64_t>(shape.size()) : dim)]);
    return sum->binary_op_scalar(2, 1.0f / divisor);
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_max(int64_t dim, bool keepdim)
{
    if (!is_dense_zero_offset())
    {
        return force_contiguous()->reduce_max(dim, keepdim);
    }
    if (dtype != ScalarType::Float32)
    {
        return cast(ScalarType::Float32)->reduce_max(dim, keepdim)->cast(dtype);
    }

    const int64_t TOTAL_DIM = -999999;
    if (dim == TOTAL_DIM)
    {
        std::vector<int64_t> out_shape =
            keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
        if (device_id >= 0)
        {
            struct
            {
                uint32_t num_rows;
                uint32_t cols;
                uint32_t op_type;
            } pc{1, static_cast<uint32_t>(numel), 1};

            VulkanPipelineManager::instance().dispatch(
                device_id, "reduction", {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), 1, 1, 1);
        }
        else
        {
            const float *in_ptr = cpu_f32();
            float max_v = -1e30f;
            for (size_t i = 0; i < numel; ++i)
                max_v = std::max(max_v, in_ptr[i]);
            out->cpu_f32()[0] = max_v;
        }
        return out;
    }

    if (dim < 0)
        dim += static_cast<int64_t>(shape.size());

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (static_cast<int64_t>(i) == dim)
        {
            if (keepdim)
                out_shape.push_back(1);
        }
        else
        {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty())
        out_shape.push_back(1);

    if (device_id >= 0)
    {
        uint32_t outer = 1;
        uint32_t inner = 1;
        for (int64_t i = 0; i < dim; ++i)
            outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
        for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i)
        {
            inner *= static_cast<uint32_t>(shape[i]);
        }

        auto out = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

        struct
        {
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t op_type;
        } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner, 1u};

        if (inner >= 16u && outer <= 65535u)
        {
            VulkanPipelineManager::instance().dispatch(device_id, "reduction_strided",
                                                       {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
                                                       (inner + 31u) / 32u, outer);
        }
        else
        {
            VulkanPipelineManager::instance().dispatch(device_id, "reduction_generic",
                                                       {gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
                                                       outer * inner);
        }
        return out;
    }

    auto cpu_out = std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Float32);
    cpu_out->fill_(-1e30f);
    const float *in_ptr = cpu_f32();
    float *out_ptr = cpu_out->cpu_f32();

    for (size_t i = 0; i < numel; ++i)
    {
        size_t rem_idx = i;
        size_t target_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            size_t c = rem_idx % static_cast<size_t>(shape[static_cast<size_t>(d)]);
            rem_idx /= static_cast<size_t>(shape[static_cast<size_t>(d)]);
            if (d != dim)
            {
                int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
                target_idx += c * static_cast<size_t>(cpu_out->strides[static_cast<size_t>(out_d)]);
            }
        }
        out_ptr[target_idx] = std::max(out_ptr[target_idx], in_ptr[i]);
    }

    return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::argmax_argmin(int64_t dim, bool keepdim,
                                                          bool is_min) const
{
    if (dim == -999999)
    {
        auto flat = reshape_view({static_cast<int64_t>(numel)});
        auto res = flat->argmax_argmin(0, false, is_min);
        if (keepdim)
        {
            std::vector<int64_t> kd(shape.size(), 1);
            return res->reshape_view(kd);
        }
        return res;
    }

    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t nd = normalize_dim(dim, ndim);
    auto cont = is_contiguous() ? shared_from_this() : contiguous();

    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < nd; ++d)
        outer *= static_cast<uint32_t>(cont->shape[d]);
    const uint32_t dim_sz = static_cast<uint32_t>(cont->shape[nd]);
    for (int64_t d = nd + 1; d < ndim; ++d)
        inner *= static_cast<uint32_t>(cont->shape[d]);

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; ++d)
    {
        if (d == nd)
        {
            if (keepdim)
                out_shape.push_back(1);
        }
        else
        {
            out_shape.push_back(cont->shape[d]);
        }
    }

    const uint32_t total_slices = outer * inner;
    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Int64);
        int64_t *o_ptr = reinterpret_cast<int64_t *>(out->cpu_raw_ptr());
        const uint8_t *i_raw = cont->cpu_raw_ptr();
        for (uint32_t s = 0; s < total_slices; ++s)
        {
            const uint32_t o = s / inner;
            const uint32_t in_r = s % inner;
            float best_v = 0.0f;
            int64_t best_i = 0;
            for (uint32_t j = 0; j < dim_sz; ++j)
            {
                const size_t pos = cont->storage_offset + (o * dim_sz + j) * inner + in_r;
                const float v = read_scalar_as_float(i_raw, pos, cont->dtype);
                if (j == 0 || (is_min ? (v < best_v) : (v > best_v)))
                {
                    best_v = v;
                    best_i = static_cast<int64_t>(j);
                }
            }
            o_ptr[s] = best_i;
        }
        return out;
    }

    auto x_f = (cont->dtype == ScalarType::Float32) ? cont : cont->cast(ScalarType::Float32);
    auto idx_i32 = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Int32);
    if (total_slices > 0)
    {
        struct
        {
            uint32_t total_slices;
            uint32_t dim_size;
            uint32_t inner_size;
            uint32_t mode;
            uint32_t out_k;
            uint32_t src_offset;
            uint32_t pad_pow2;
            uint32_t _pad;
        } pc{total_slices, dim_sz,
             inner,        is_min ? 1u : 0u,
             1u,           static_cast<uint32_t>(x_f->storage_offset),
             1u,           0u};
        VulkanPipelineManager::instance().dispatch(
            device_id, "argmax_argmin_kernel",
            {x_f->gpu_buf(), idx_i32->gpu_buf(), idx_i32->gpu_buf()}, &pc, sizeof(pc),
            total_slices);
    }
    return idx_i32->cast(ScalarType::Int64);
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_prod_all_any(int64_t dim, bool keepdim,
                                                                uint32_t mode) const
{
    if (dim == -999999)
    {
        auto flat = reshape_view({static_cast<int64_t>(numel)});
        auto res = flat->reduce_prod_all_any(0, false, mode);
        if (keepdim)
        {
            std::vector<int64_t> kd(shape.size(), 1);
            return res->reshape_view(kd);
        }
        return res;
    }

    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t nd = normalize_dim(dim, ndim);
    auto cont = is_contiguous() ? shared_from_this() : contiguous();

    uint32_t outer = 1, inner = 1;
    for (int64_t d = 0; d < nd; ++d)
        outer *= static_cast<uint32_t>(cont->shape[d]);
    const uint32_t dim_sz = static_cast<uint32_t>(cont->shape[nd]);
    for (int64_t d = nd + 1; d < ndim; ++d)
        inner *= static_cast<uint32_t>(cont->shape[d]);

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; ++d)
    {
        if (d == nd)
        {
            if (keepdim)
                out_shape.push_back(1);
        }
        else
        {
            out_shape.push_back(cont->shape[d]);
        }
    }

    const ScalarType target_dt = (mode == 0u) ? dtype : ScalarType::Bool;
    const uint32_t total_slices = outer * inner;

    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(out_shape, -1, target_dt);
        const uint8_t *i_raw = cont->cpu_raw_ptr();
        uint8_t *o_raw = out->cpu_raw_ptr();
        for (uint32_t s = 0; s < total_slices; ++s)
        {
            const uint32_t o = s / inner;
            const uint32_t in_r = s % inner;
            double acc = (mode == 2u) ? 0.0 : 1.0;
            for (uint32_t j = 0; j < dim_sz; ++j)
            {
                const size_t pos = cont->storage_offset + (o * dim_sz + j) * inner + in_r;
                const float v = read_scalar_as_float(i_raw, pos, cont->dtype);
                if (mode == 0u)
                {
                    acc *= static_cast<double>(v);
                }
                else if (mode == 1u)
                {
                    acc = (acc != 0.0 && v != 0.0f) ? 1.0 : 0.0;
                }
                else
                {
                    acc = (acc != 0.0 || v != 0.0f) ? 1.0 : 0.0;
                }
            }
            write_scalar_from_double(o_raw, s, target_dt, acc);
        }
        return out;
    }

    auto x_f = (cont->dtype == ScalarType::Float32) ? cont : cont->cast(ScalarType::Float32);
    auto out_f = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
    if (total_slices > 0)
    {
        struct
        {
            uint32_t total_slices;
            uint32_t dim_size;
            uint32_t inner_size;
            uint32_t mode;
            uint32_t out_k;
            uint32_t src_offset;
            uint32_t pad_pow2;
            uint32_t _pad;
        } pc{total_slices, dim_sz, inner, mode, 1u, static_cast<uint32_t>(x_f->storage_offset),
             1u,           0u};
        VulkanPipelineManager::instance().dispatch(
            device_id, "prod_all_any_kernel", {x_f->gpu_buf(), out_f->gpu_buf(), out_f->gpu_buf()},
            &pc, sizeof(pc), total_slices);
    }
    return (target_dt == ScalarType::Float32) ? out_f : out_f->cast(target_dt);
}

} // namespace vultorch
