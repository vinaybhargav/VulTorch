#include "tensor/tensor_internal.hpp"

namespace vultorch
{

std::shared_ptr<NativeTensor>
NativeTensor::cat_dim0(const std::vector<std::shared_ptr<NativeTensor>> &tensors,
                       int target_device_id)
{
    if (tensors.empty())
    {
        throw std::runtime_error("cat_dim0 expects a non-empty list of tensors");
    }
    const auto &first = tensors.front();
    if (first->shape.empty())
    {
        throw std::runtime_error("cat_dim0 requires at least 1D tensors");
    }

    int64_t total_dim0 = 0;
    ScalarType out_dt = first->dtype;
    for (const auto &t : tensors)
    {
        if (t->shape.size() != first->shape.size())
        {
            throw std::runtime_error("cat_dim0: rank mismatch");
        }
        for (size_t d = 1; d < first->shape.size(); ++d)
        {
            if (t->shape[d] != first->shape[d])
            {
                throw std::runtime_error("cat_dim0: non-concat dimension mismatch");
            }
        }
        out_dt = promote_dtype(out_dt, t->dtype);
        total_dim0 += t->shape[0];
    }

    std::vector<int64_t> out_shape = first->shape;
    out_shape[0] = total_dim0;
    auto out = std::make_shared<NativeTensor>(out_shape, target_device_id, out_dt);

    const size_t elem_sz = element_size(out_dt);
    size_t offset_elems = 0;
    for (const auto &t_raw : tensors)
    {
        if (t_raw->numel == 0)
            continue;
        auto t_cast = (t_raw->dtype == out_dt) ? t_raw : t_raw->cast(out_dt);
        auto t = t_cast->force_contiguous();

        const size_t bytes = t->numel * elem_sz;
        const size_t dst_off_bytes = offset_elems * elem_sz;

        if (target_device_id >= 0 && t->device_id >= 0)
        {
            if (t->device_id == target_device_id)
            {
                VulkanMemoryManager::instance().copy_buffer_to_buffer(
                    target_device_id, t->gpu_buf(), out->gpu_buf(), bytes, 0, dst_off_bytes);
            }
            else
            {
                VulkanMemoryManager::instance().copy_cross_device(t->device_id, t->gpu_buf(),
                                                                  target_device_id, out->gpu_buf(),
                                                                  bytes, 0, dst_off_bytes);
            }
        }
        else if (target_device_id == -1 && t->device_id == -1)
        {
            std::memcpy(out->cpu_raw_ptr() + dst_off_bytes, t->cpu_raw_ptr(), bytes);
        }
        else if (target_device_id == -1 && t->device_id >= 0)
        {
            t->to_raw_bytes(out->cpu_raw_ptr() + dst_off_bytes, bytes);
        }
        else
        {
            auto stage_t = t->to_device(target_device_id);
            VulkanMemoryManager::instance().copy_buffer_to_buffer(
                target_device_id, stage_t->gpu_buf(), out->gpu_buf(), bytes, 0, dst_off_bytes);
        }
        offset_elems += t->numel;
    }
    return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::cat(const std::vector<std::shared_ptr<NativeTensor>> &tensors, int64_t dim,
                  int target_device_id)
{
    if (tensors.empty())
    {
        throw std::runtime_error("cat expects at least one tensor");
    }
    const int64_t ndim = static_cast<int64_t>(tensors[0]->shape.size());
    const int64_t nd = normalize_dim(dim, ndim);
    if (ndim > 6)
    {
        throw std::runtime_error("cat supports up to 6 dimensions");
    }
    if (nd == 0 && target_device_id >= 0)
    {
        bool all_dense = true;
        const ScalarType dt0 = tensors[0]->dtype;
        for (const auto &t : tensors)
        {
            if (t->dtype != dt0 || !t->is_dense_zero_offset())
            {
                all_dense = false;
                break;
            }
        }
        if (all_dense)
        {
            return cat_dim0(tensors, target_device_id);
        }
    }

    ScalarType out_dt = tensors[0]->dtype;
    std::vector<int64_t> out_shape = tensors[0]->shape;
    out_shape[nd] = 0;
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        if (static_cast<int64_t>(tensors[i]->shape.size()) != ndim)
        {
            throw std::runtime_error("cat: all tensors must have the same number of dimensions");
        }
        for (int64_t d = 0; d < ndim; ++d)
        {
            if (d != nd && tensors[i]->shape[d] != out_shape[d])
            {
                throw std::runtime_error("cat: non-concatenated dimensions must match");
            }
        }
        out_shape[nd] += tensors[i]->shape[nd];
        out_dt = promote_dtype(out_dt, tensors[i]->dtype);
    }
    (void)0;
    auto out = std::make_shared<NativeTensor>(out_shape, target_device_id, out_dt);
    if (out->numel == 0)
        return out;

    if (target_device_id == -1)
    {
        uint8_t *dst_raw = out->cpu_raw_ptr();
        int64_t dim_off = 0;
        for (const auto &raw_t : tensors)
        {
            auto t = (raw_t->device_id == -1) ? raw_t : raw_t->to_device(-1);
            if (t->numel == 0)
                continue;
            const uint8_t *src_raw = t->cpu_raw_ptr();
            for (size_t n = 0; n < t->numel; ++n)
            {
                size_t rem = n;
                size_t s_pos = t->storage_offset;
                size_t d_pos = static_cast<size_t>(dim_off) * static_cast<size_t>(out->strides[nd]);
                for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
                {
                    const size_t sz = static_cast<size_t>(t->shape[d]);
                    const size_t coord = rem % sz;
                    rem /= sz;
                    s_pos += coord * static_cast<size_t>(t->strides[d]);
                    d_pos += coord * static_cast<size_t>(out->strides[d]);
                }
                const double v =
                    static_cast<double>(read_scalar_as_float(src_raw, s_pos, t->dtype));
                write_scalar_from_double(dst_raw, d_pos, out_dt, v);
            }
            dim_off += t->shape[nd];
        }
        return out;
    }

    int64_t dim_off = 0;
    const uint32_t eb = static_cast<uint32_t>(element_size(out_dt));
    for (const auto &raw_t : tensors)
    {
        auto t =
            (raw_t->device_id == target_device_id) ? raw_t : raw_t->to_device(target_device_id);
        if (t->dtype != out_dt)
        {
            t = t->cast(out_dt);
        }
        if (t->numel == 0)
            continue;

        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t elem_bytes;
            uint32_t src_offset;
            uint32_t dst_offset;
            uint32_t src_shape[6];
            uint32_t src_stride[6];
            uint32_t dst_stride[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(t->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.elem_bytes = eb;
        pc.src_offset = static_cast<uint32_t>(t->storage_offset);
        pc.dst_offset = static_cast<uint32_t>(dim_off * out->strides[nd]);
        for (int64_t d = 0; d < ndim; ++d)
        {
            pc.src_shape[d] = static_cast<uint32_t>(t->shape[d]);
            pc.src_stride[d] = static_cast<uint32_t>(t->strides[d]);
            pc.dst_stride[d] = static_cast<uint32_t>(out->strides[d]);
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(target_device_id, "copy_into_slice",
                                                   {t->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
                                                   groups);
        dim_off += t->shape[nd];
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::pad_nd(const std::vector<int64_t> &pad_before,
                                                   const std::vector<int64_t> &pad_after,
                                                   uint32_t mode, float pad_value) const
{
    const size_t ndim = shape.size();
    if (ndim > 6 || pad_before.size() != ndim || pad_after.size() != ndim)
    {
        throw std::runtime_error("pad_nd: invalid pad vector size or ndim > 6");
    }
    std::vector<int64_t> out_shape(ndim);
    for (size_t d = 0; d < ndim; ++d)
    {
        out_shape[d] = shape[d] + pad_before[d] + pad_after[d];
        if (out_shape[d] < 0)
        {
            throw std::runtime_error("pad_nd: resulting dimension is negative");
        }
    }

    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(out_shape, -1, dtype);
        if (out->numel == 0)
            return out;
        const uint8_t *in_raw = cpu_raw_ptr();
        uint8_t *out_raw = out->cpu_raw_ptr();
        for (size_t n = 0; n < out->numel; ++n)
        {
            size_t rem = n;
            size_t in_pos = storage_offset;
            bool in_bounds = true;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
            {
                const size_t osz = static_cast<size_t>(out_shape[d]);
                const size_t coord = rem % osz;
                rem /= osz;
                int64_t c = static_cast<int64_t>(coord) - pad_before[d];
                const int64_t isz = shape[d];
                if (c < 0 || c >= isz)
                {
                    if (mode == 0u)
                    {
                        in_bounds = false;
                    }
                    else if (mode == 1u)
                    {
                        c = reflect_index_cpu(c, isz);
                        in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
                    }
                    else
                    {
                        c = std::max<int64_t>(0, std::min<int64_t>(c, isz - 1));
                        in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
                    }
                }
                else
                {
                    in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
                }
            }
            const double v = in_bounds
                                 ? static_cast<double>(read_scalar_as_float(in_raw, in_pos, dtype))
                                 : static_cast<double>(pad_value);
            write_scalar_from_double(out_raw, n, dtype, v);
        }
        return out;
    }

    auto x_f = (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
    auto out_f = std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
    if (out_f->numel > 0)
    {
        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t mode;
            float pad_value;
            uint32_t src_offset;
            uint32_t in_shape[6];
            uint32_t out_shape[6];
            uint32_t in_stride[6];
            uint32_t pad_before[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(out_f->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.mode = mode;
        pc.pad_value = pad_value;
        pc.src_offset = static_cast<uint32_t>(x_f->storage_offset);
        for (size_t d = 0; d < ndim; ++d)
        {
            pc.in_shape[d] = static_cast<uint32_t>(x_f->shape[d]);
            pc.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
            pc.in_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
            pc.pad_before[d] = static_cast<uint32_t>(pad_before[d]);
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "pad_nd_kernel",
                                                   {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
                                                   sizeof(pc), groups);
    }
    return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::pad_nd_backward(const std::vector<int64_t> &in_shape,
                                                            const std::vector<int64_t> &pad_before,
                                                            uint32_t mode) const
{
    const size_t ndim = shape.size();
    if (ndim > 6 || in_shape.size() != ndim || pad_before.size() != ndim)
    {
        throw std::runtime_error("pad_nd_backward: invalid shape or ndim > 6");
    }
    if (device_id == -1)
    {
        auto grad_in = std::make_shared<NativeTensor>(in_shape, -1, dtype);
        grad_in->zero_();
        if (numel == 0)
            return grad_in;
        const uint8_t *gout_raw = cpu_raw_ptr();
        uint8_t *gin_raw = grad_in->cpu_raw_ptr();
        for (size_t n = 0; n < numel; ++n)
        {
            size_t rem = n;
            size_t in_pos = 0;
            bool in_bounds = true;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
            {
                const size_t osz = static_cast<size_t>(shape[d]);
                const size_t coord = rem % osz;
                rem /= osz;
                int64_t c = static_cast<int64_t>(coord) - pad_before[d];
                const int64_t isz = in_shape[d];
                if (c < 0 || c >= isz)
                {
                    if (mode == 0u)
                    {
                        in_bounds = false;
                    }
                    else if (mode == 1u)
                    {
                        c = reflect_index_cpu(c, isz);
                        in_pos += static_cast<size_t>(c) * static_cast<size_t>(grad_in->strides[d]);
                    }
                    else
                    {
                        c = std::max<int64_t>(0, std::min<int64_t>(c, isz - 1));
                        in_pos += static_cast<size_t>(c) * static_cast<size_t>(grad_in->strides[d]);
                    }
                }
                else
                {
                    in_pos += static_cast<size_t>(c) * static_cast<size_t>(grad_in->strides[d]);
                }
            }
            if (in_bounds)
            {
                const double cur =
                    static_cast<double>(read_scalar_as_float(gin_raw, in_pos, dtype));
                const double addv =
                    static_cast<double>(read_scalar_as_float(gout_raw, storage_offset + n, dtype));
                write_scalar_from_double(gin_raw, in_pos, dtype, cur + addv);
            }
        }
        return grad_in;
    }

    auto gout_f = (dtype == ScalarType::Float32) ? force_contiguous()
                                                 : cast(ScalarType::Float32)->force_contiguous();
    auto gin_f = std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);
    gin_f->zero_();
    if (gout_f->numel > 0)
    {
        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t mode;
            float pad_value;
            uint32_t src_offset;
            uint32_t in_shape[6];
            uint32_t out_shape[6];
            uint32_t in_stride[6];
            uint32_t pad_before[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(gout_f->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.mode = mode;
        pc.src_offset = static_cast<uint32_t>(gout_f->storage_offset);
        for (size_t d = 0; d < ndim; ++d)
        {
            pc.in_shape[d] = static_cast<uint32_t>(in_shape[d]);
            pc.out_shape[d] = static_cast<uint32_t>(gout_f->shape[d]);
            pc.in_stride[d] = static_cast<uint32_t>(gin_f->strides[d]);
            pc.pad_before[d] = static_cast<uint32_t>(pad_before[d]);
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "pad_nd_backward_kernel",
                                                   {gout_f->gpu_buf(), gin_f->gpu_buf()}, &pc,
                                                   sizeof(pc), groups);
    }
    return (dtype == ScalarType::Float32) ? gin_f : gin_f->cast(dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::tril_triu(bool is_upper, int64_t diagonal) const
{
    const size_t ndim = shape.size();
    if (ndim < 2 || ndim > 6)
    {
        throw std::runtime_error("tril/triu requires 2 <= ndim <= 6");
    }
    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
        if (out->numel == 0)
            return out;
        const uint8_t *in_raw = cpu_raw_ptr();
        uint8_t *out_raw = out->cpu_raw_ptr();
        for (size_t n = 0; n < out->numel; ++n)
        {
            size_t rem = n;
            size_t src_pos = storage_offset;
            int64_t row = 0;
            int64_t col = 0;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
            {
                const size_t sz = static_cast<size_t>(shape[d]);
                const size_t coord = rem % sz;
                rem /= sz;
                src_pos += coord * static_cast<size_t>(strides[d]);
                if (d == static_cast<int>(ndim) - 1)
                    col = static_cast<int64_t>(coord);
                if (d == static_cast<int>(ndim) - 2)
                    row = static_cast<int64_t>(coord);
            }
            const int64_t diff = col - row;
            const bool keep = is_upper ? (diff >= diagonal) : (diff <= diagonal);
            const double v =
                keep ? static_cast<double>(read_scalar_as_float(in_raw, src_pos, dtype)) : 0.0;
            write_scalar_from_double(out_raw, n, dtype, v);
        }
        return out;
    }

    auto x_f = (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
    auto out_f = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    if (out_f->numel > 0)
    {
        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t mode;
            int32_t param;
            uint32_t src_offset;
            uint32_t shape[6];
            uint32_t src_stride[6];
            int32_t shifts[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(out_f->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.mode = is_upper ? 1u : 0u;
        pc.param = static_cast<int32_t>(diagonal);
        pc.src_offset = static_cast<uint32_t>(x_f->storage_offset);
        for (size_t d = 0; d < ndim; ++d)
        {
            pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
            pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "tril_triu_kernel",
                                                   {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
                                                   sizeof(pc), groups);
    }
    return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::flip(const std::vector<int64_t> &dims) const
{
    const size_t ndim = shape.size();
    if (ndim > 6)
        throw std::runtime_error("flip supports up to 6 dimensions");
    uint32_t mask = 0;
    for (int64_t raw_d : dims)
    {
        int64_t d = normalize_dim(raw_d, static_cast<int64_t>(ndim));
        mask |= (1u << static_cast<uint32_t>(d));
    }
    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
        if (out->numel == 0)
            return out;
        const uint8_t *in_raw = cpu_raw_ptr();
        uint8_t *out_raw = out->cpu_raw_ptr();
        for (size_t n = 0; n < out->numel; ++n)
        {
            size_t rem = n;
            size_t src_pos = storage_offset;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
            {
                const size_t sz = static_cast<size_t>(shape[d]);
                size_t coord = rem % sz;
                rem /= sz;
                if ((mask & (1u << static_cast<uint32_t>(d))) != 0u)
                {
                    coord = sz - 1 - coord;
                }
                src_pos += coord * static_cast<size_t>(strides[d]);
            }
            const double v = static_cast<double>(read_scalar_as_float(in_raw, src_pos, dtype));
            write_scalar_from_double(out_raw, n, dtype, v);
        }
        return out;
    }

    auto x_f = (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
    auto out_f = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    if (out_f->numel > 0)
    {
        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t mode;
            int32_t param;
            uint32_t src_offset;
            uint32_t shape[6];
            uint32_t src_stride[6];
            int32_t shifts[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(out_f->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.mode = 2u;
        pc.param = static_cast<int32_t>(mask);
        pc.src_offset = static_cast<uint32_t>(x_f->storage_offset);
        for (size_t d = 0; d < ndim; ++d)
        {
            pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
            pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "flip_roll_kernel",
                                                   {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
                                                   sizeof(pc), groups);
    }
    return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::roll(const std::vector<int64_t> &shifts,
                                                 const std::vector<int64_t> &dims) const
{
    const size_t ndim = shape.size();
    if (ndim > 6 || shifts.size() != dims.size())
    {
        throw std::runtime_error("roll: invalid dims/shifts or ndim > 6");
    }
    std::vector<int32_t> dim_shifts(ndim, 0);
    for (size_t i = 0; i < dims.size(); ++i)
    {
        int64_t d = normalize_dim(dims[i], static_cast<int64_t>(ndim));
        dim_shifts[static_cast<size_t>(d)] += static_cast<int32_t>(shifts[i]);
    }
    if (device_id == -1)
    {
        auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
        if (out->numel == 0)
            return out;
        const uint8_t *in_raw = cpu_raw_ptr();
        uint8_t *out_raw = out->cpu_raw_ptr();
        for (size_t n = 0; n < out->numel; ++n)
        {
            size_t rem = n;
            size_t src_pos = storage_offset;
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d)
            {
                const int64_t sz = shape[d];
                const int64_t coord = static_cast<int64_t>(rem % static_cast<size_t>(sz));
                rem /= static_cast<size_t>(sz);
                const int64_t sh = dim_shifts[static_cast<size_t>(d)] % sz;
                const int64_t sc = (coord - sh + sz) % sz;
                src_pos += static_cast<size_t>(sc) * static_cast<size_t>(strides[d]);
            }
            const double v = static_cast<double>(read_scalar_as_float(in_raw, src_pos, dtype));
            write_scalar_from_double(out_raw, n, dtype, v);
        }
        return out;
    }

    auto x_f = (dtype == ScalarType::Float32) ? shared_from_this() : cast(ScalarType::Float32);
    auto out_f = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    if (out_f->numel > 0)
    {
        struct
        {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t mode;
            int32_t param;
            uint32_t src_offset;
            uint32_t shape[6];
            uint32_t src_stride[6];
            int32_t shifts[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(out_f->numel);
        pc.ndim = static_cast<uint32_t>(ndim);
        pc.mode = 3u;
        pc.src_offset = static_cast<uint32_t>(x_f->storage_offset);
        for (size_t d = 0; d < ndim; ++d)
        {
            pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
            pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
            pc.shifts[d] = dim_shifts[d];
        }
        uint32_t groups = (pc.total_elements + 255u) / 256u;
        VulkanPipelineManager::instance().dispatch(device_id, "flip_roll_kernel",
                                                   {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
                                                   sizeof(pc), groups);
    }
    return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

} // namespace vultorch
