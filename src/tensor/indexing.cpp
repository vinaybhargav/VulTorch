#include "tensor/tensor_internal.hpp"

namespace vultorch {

std::shared_ptr<NativeTensor>
NativeTensor::where_op(const std::shared_ptr<NativeTensor> &cond,
                       const std::shared_ptr<NativeTensor> &a,
                       const std::shared_ptr<NativeTensor> &b) {
  if (cond->device_id != a->device_id || a->device_id != b->device_id) {
    throw std::runtime_error("where: all tensors must be on the same device");
  }
  const int dev = a->device_id;
  const ScalarType common_dt = promote_dtype(a->dtype, b->dtype);

  const BroadcastPlan p_ab =
      make_broadcast_plan(a->shape, a->strides, b->shape, b->strides);
  const BroadcastPlan p_abc = make_broadcast_plan(
      cond->shape, cond->strides, p_ab.out_shape, p_ab.strides_a);
  const std::vector<int64_t> &out_shape = p_abc.out_shape;
  const size_t ndim = out_shape.size();
  if (ndim > 6) {
    throw std::runtime_error("where supports up to 6 dimensions");
  }

  const auto st_c =
      align_strides_for_broadcast(cond->shape, cond->strides, out_shape);
  const auto st_a =
      align_strides_for_broadcast(a->shape, a->strides, out_shape);
  const auto st_b =
      align_strides_for_broadcast(b->shape, b->strides, out_shape);

  if (dev == -1) {
    auto out = std::make_shared<NativeTensor>(out_shape, -1, common_dt);
    if (out->numel == 0)
      return out;
    const uint8_t *c_raw = cond->cpu_raw_ptr();
    const uint8_t *a_raw = a->cpu_raw_ptr();
    const uint8_t *b_raw = b->cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t ic = cond->storage_offset;
      size_t ia = a->storage_offset;
      size_t ib = b->storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t dim_sz = static_cast<size_t>(out_shape[d]);
        const size_t coord = rem % dim_sz;
        rem /= dim_sz;
        ic += coord * static_cast<size_t>(st_c[d]);
        ia += coord * static_cast<size_t>(st_a[d]);
        ib += coord * static_cast<size_t>(st_b[d]);
      }
      const bool c_val = read_scalar_as_float(c_raw, ic, cond->dtype) != 0.0f;
      const double val =
          c_val
              ? static_cast<double>(read_scalar_as_float(a_raw, ia, a->dtype))
              : static_cast<double>(read_scalar_as_float(b_raw, ib, b->dtype));
      write_scalar_from_double(o_raw, n, common_dt, val);
    }
    return out;
  }

  auto cond_f = (cond->dtype == ScalarType::Float32)
                    ? cond
                    : cond->cast(ScalarType::Float32);
  auto a_f =
      (a->dtype == ScalarType::Float32) ? a : a->cast(ScalarType::Float32);
  auto b_f =
      (b->dtype == ScalarType::Float32) ? b : b->cast(ScalarType::Float32);

  const auto st_cf =
      align_strides_for_broadcast(cond_f->shape, cond_f->strides, out_shape);
  const auto st_af =
      align_strides_for_broadcast(a_f->shape, a_f->strides, out_shape);
  const auto st_bf =
      align_strides_for_broadcast(b_f->shape, b_f->strides, out_shape);

  auto out_f =
      std::make_shared<NativeTensor>(out_shape, dev, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t offsetC;
      uint32_t offsetA;
      uint32_t offsetB;
      uint32_t shape[6];
      uint32_t strideC[6];
      uint32_t strideA[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out_f->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.offsetC = static_cast<uint32_t>(cond_f->storage_offset);
    pc.offsetA = static_cast<uint32_t>(a_f->storage_offset);
    pc.offsetB = static_cast<uint32_t>(b_f->storage_offset);
    for (size_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(out_shape[d]);
      pc.strideC[d] = static_cast<uint32_t>(st_cf[d]);
      pc.strideA[d] = static_cast<uint32_t>(st_af[d]);
      pc.strideB[d] = static_cast<uint32_t>(st_bf[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        dev, "where_kernel",
        {cond_f->gpu_buf(), a_f->gpu_buf(), b_f->gpu_buf(), out_f->gpu_buf()},
        &pc, sizeof(pc), groups);
  }
  return (common_dt == ScalarType::Float32) ? out_f : out_f->cast(common_dt);
}

std::shared_ptr<NativeTensor>
NativeTensor::masked_fill(const std::shared_ptr<NativeTensor> &mask,
                          float value) const {
  if (device_id != mask->device_id) {
    throw std::runtime_error(
        "masked_fill: tensor and mask must be on the same device");
  }
  const BroadcastPlan plan =
      make_broadcast_plan(shape, strides, mask->shape, mask->strides);
  if (plan.out_shape != shape) {
    throw std::runtime_error(
        "masked_fill: mask shape must be broadcastable to tensor shape");
  }
  const size_t ndim = shape.size();
  if (ndim > 6) {
    throw std::runtime_error("masked_fill supports up to 6 dimensions");
  }

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const uint8_t *x_raw = cpu_raw_ptr();
    const uint8_t *m_raw = mask->cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t ix = storage_offset;
      size_t im = mask->storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t dim_sz = static_cast<size_t>(shape[d]);
        const size_t coord = rem % dim_sz;
        rem /= dim_sz;
        ix += coord * static_cast<size_t>(plan.strides_a[d]);
        im += coord * static_cast<size_t>(plan.strides_b[d]);
      }
      const bool m_val = read_scalar_as_float(m_raw, im, mask->dtype) != 0.0f;
      const double val =
          m_val ? static_cast<double>(value)
                : static_cast<double>(read_scalar_as_float(x_raw, ix, dtype));
      write_scalar_from_double(o_raw, n, dtype, val);
    }
    return out;
  }

  auto x_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                            : cast(ScalarType::Float32);
  auto m_f = (mask->dtype == ScalarType::Float32)
                 ? mask
                 : mask->cast(ScalarType::Float32);
  const BroadcastPlan plan_f =
      make_broadcast_plan(x_f->shape, x_f->strides, m_f->shape, m_f->strides);

  auto out_f =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      float fill_value;
      uint32_t offsetX;
      uint32_t offsetM;
      uint32_t shape[6];
      uint32_t strideX[6];
      uint32_t strideM[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out_f->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.fill_value = value;
    pc.offsetX = static_cast<uint32_t>(x_f->storage_offset);
    pc.offsetM = static_cast<uint32_t>(m_f->storage_offset);
    for (size_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(shape[d]);
      pc.strideX[d] = static_cast<uint32_t>(plan_f.strides_a[d]);
      pc.strideM[d] = static_cast<uint32_t>(plan_f.strides_b[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "masked_fill_kernel",
        {x_f->gpu_buf(), m_f->gpu_buf(), out_f->gpu_buf()}, &pc, sizeof(pc),
        groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::gather(int64_t dim,
                     const std::shared_ptr<NativeTensor> &index) const {
  if (device_id != index->device_id) {
    throw std::runtime_error(
        "gather: tensor and index must be on the same device");
  }
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  if (static_cast<int64_t>(index->shape.size()) != ndim || ndim > 6) {
    throw std::runtime_error(
        "gather: index must have same ndim (<= 6) as input");
  }
  for (int64_t d = 0; d < ndim; ++d) {
    if (d != nd && index->shape[d] != shape[d]) {
      throw std::runtime_error(
          "gather: non-gather dimensions of index and input must match");
    }
  }

  auto idx_i32 = ensure_int32_for_index(index);

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(index->shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const int32_t *i_ptr =
        reinterpret_cast<const int32_t *>(idx_i32->cpu_raw_ptr());
    const uint8_t *s_raw = cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    const int64_t dim_sz = shape[nd];
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t src_pos = storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t sz = static_cast<size_t>(out->shape[d]);
        const size_t coord = rem % sz;
        rem /= sz;
        if (d != nd)
          src_pos += coord * static_cast<size_t>(strides[d]);
      }
      int64_t k = i_ptr[idx_i32->storage_offset + n];
      if (k < 0)
        k += dim_sz;
      if (k < 0 || k >= dim_sz) {
        throw std::out_of_range("gather: index out of bounds");
      }
      src_pos += static_cast<size_t>(k) * static_cast<size_t>(strides[nd]);
      const double val =
          static_cast<double>(read_scalar_as_float(s_raw, src_pos, dtype));
      write_scalar_from_double(o_raw, n, dtype, val);
    }
    return out;
  }

  auto src_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                              : cast(ScalarType::Float32);
  auto out_f = std::make_shared<NativeTensor>(index->shape, device_id,
                                              ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t dim;
      uint32_t dim_size_target;
      uint32_t offsetA;
      uint32_t offsetIdx;
      uint32_t offsetB;
      float scalar_val;
      uint32_t shape[6];
      uint32_t strideA[6];
      uint32_t strideIdx[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out_f->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.dim = static_cast<uint32_t>(nd);
    pc.dim_size_target = static_cast<uint32_t>(src_f->shape[nd]);
    pc.offsetA = static_cast<uint32_t>(src_f->storage_offset);
    pc.offsetIdx = static_cast<uint32_t>(idx_i32->storage_offset);
    pc.offsetB = 0;
    for (int64_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(out_f->shape[d]);
      pc.strideA[d] = static_cast<uint32_t>(src_f->strides[d]);
      pc.strideIdx[d] = static_cast<uint32_t>(idx_i32->strides[d]);
      pc.strideB[d] = static_cast<uint32_t>(out_f->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "gather_kernel",
        {src_f->gpu_buf(), idx_i32->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::scatter(int64_t dim, const std::shared_ptr<NativeTensor> &index,
                      const std::shared_ptr<NativeTensor> &src,
                      float scalar_val, bool use_scalar,
                      bool accumulate) const {
  if (device_id != index->device_id ||
      (!use_scalar && (!src || device_id != src->device_id))) {
    throw std::runtime_error("scatter: all tensors must be on the same device");
  }
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  if (static_cast<int64_t>(index->shape.size()) != ndim || ndim > 6) {
    throw std::runtime_error(
        "scatter: index must have same ndim (<= 6) as target");
  }

  auto idx_i32 = ensure_int32_for_index(index);

  if (device_id == -1) {
    auto out = clone();
    if (idx_i32->numel == 0)
      return out;
    const int32_t *i_ptr =
        reinterpret_cast<const int32_t *>(idx_i32->cpu_raw_ptr());
    uint8_t *o_raw = out->cpu_raw_ptr();
    const int64_t dim_sz = out->shape[nd];
    for (size_t n = 0; n < idx_i32->numel; ++n) {
      size_t rem = n;
      size_t dst_pos = 0;
      size_t src_pos = use_scalar ? 0 : src->storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t sz = static_cast<size_t>(idx_i32->shape[d]);
        const size_t coord = rem % sz;
        rem /= sz;
        if (d != nd)
          dst_pos += coord * static_cast<size_t>(out->strides[d]);
        if (!use_scalar)
          src_pos += coord * static_cast<size_t>(src->strides[d]);
      }
      int64_t k = i_ptr[idx_i32->storage_offset + n];
      if (k < 0)
        k += dim_sz;
      if (k < 0 || k >= dim_sz) {
        throw std::out_of_range("scatter: index out of bounds");
      }
      dst_pos += static_cast<size_t>(k) * static_cast<size_t>(out->strides[nd]);
      const double v = use_scalar
                           ? static_cast<double>(scalar_val)
                           : static_cast<double>(read_scalar_as_float(
                                 src->cpu_raw_ptr(), src_pos, src->dtype));
      if (accumulate) {
        const double cur = static_cast<double>(
            read_scalar_as_float(o_raw, dst_pos, out->dtype));
        write_scalar_from_double(o_raw, dst_pos, out->dtype, cur + v);
      } else {
        write_scalar_from_double(o_raw, dst_pos, out->dtype, v);
      }
    }
    return out;
  }

  auto dst_f = (dtype == ScalarType::Float32) ? force_contiguous()
                                              : cast(ScalarType::Float32);
  std::shared_ptr<NativeTensor> src_f;
  if (use_scalar) {
    src_f = dst_f; // dummy valid buffer binding
  } else {
    src_f = (src->dtype == ScalarType::Float32)
                ? src
                : src->cast(ScalarType::Float32);
  }

  if (idx_i32->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t dim;
      uint32_t dim_size_target;
      uint32_t offsetA;
      uint32_t offsetIdx;
      uint32_t offsetB;
      float scalar_val;
      uint32_t shape[6];
      uint32_t strideA[6];
      uint32_t strideIdx[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(idx_i32->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.dim = static_cast<uint32_t>(nd);
    pc.dim_size_target = static_cast<uint32_t>(dst_f->shape[nd]);
    pc.offsetA =
        use_scalar ? 0xFFFFFFFFu : static_cast<uint32_t>(src_f->storage_offset);
    pc.offsetIdx = static_cast<uint32_t>(idx_i32->storage_offset);
    pc.offsetB = static_cast<uint32_t>(dst_f->storage_offset);
    pc.scalar_val = scalar_val;
    for (int64_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(idx_i32->shape[d]);
      pc.strideA[d] =
          use_scalar ? 0u : static_cast<uint32_t>(src_f->strides[d]);
      pc.strideIdx[d] = static_cast<uint32_t>(idx_i32->strides[d]);
      pc.strideB[d] = static_cast<uint32_t>(dst_f->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, accumulate ? "scatter_add_kernel" : "scatter_kernel",
        {src_f->gpu_buf(), idx_i32->gpu_buf(), dst_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? dst_f : dst_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::index_select(int64_t dim,
                           const std::shared_ptr<NativeTensor> &index) const {
  if (device_id != index->device_id) {
    throw std::runtime_error(
        "index_select: tensor and index must be on the same device");
  }
  if (index->shape.size() != 1) {
    throw std::runtime_error("index_select: index must be a 1D tensor");
  }
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  if (ndim > 6) {
    throw std::runtime_error("index_select supports up to 6 dimensions");
  }

  auto idx_i32 = ensure_int32_for_index(index);
  std::vector<int64_t> out_shape = shape;
  out_shape[nd] = idx_i32->shape[0];

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(out_shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const int32_t *i_ptr =
        reinterpret_cast<const int32_t *>(idx_i32->cpu_raw_ptr());
    const uint8_t *s_raw = cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    const int64_t dim_sz = shape[nd];
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t src_pos = storage_offset;
      size_t dim_coord = 0;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t sz = static_cast<size_t>(out_shape[d]);
        const size_t coord = rem % sz;
        rem /= sz;
        if (d == nd) {
          dim_coord = coord;
        } else {
          src_pos += coord * static_cast<size_t>(strides[d]);
        }
      }
      int64_t k = i_ptr[idx_i32->storage_offset + dim_coord];
      if (k < 0)
        k += dim_sz;
      if (k < 0 || k >= dim_sz) {
        throw std::out_of_range("index_select: index out of bounds");
      }
      src_pos += static_cast<size_t>(k) * static_cast<size_t>(strides[nd]);
      const double val =
          static_cast<double>(read_scalar_as_float(s_raw, src_pos, dtype));
      write_scalar_from_double(o_raw, n, dtype, val);
    }
    return out;
  }

  auto src_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                              : cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t dim;
      uint32_t dim_size_target;
      uint32_t offsetA;
      uint32_t offsetIdx;
      uint32_t offsetB;
      float scalar_val;
      uint32_t shape[6];
      uint32_t strideA[6];
      uint32_t strideIdx[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out_f->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.dim = static_cast<uint32_t>(nd);
    pc.dim_size_target = static_cast<uint32_t>(src_f->shape[nd]);
    pc.offsetA = static_cast<uint32_t>(src_f->storage_offset);
    pc.offsetIdx = static_cast<uint32_t>(idx_i32->storage_offset);
    pc.strideIdx[0] = static_cast<uint32_t>(idx_i32->strides[0]);
    for (int64_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(out_f->shape[d]);
      pc.strideA[d] = static_cast<uint32_t>(src_f->strides[d]);
      pc.strideB[d] = static_cast<uint32_t>(out_f->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "index_select_kernel",
        {src_f->gpu_buf(), idx_i32->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

void NativeTensor::index_add_(int64_t dim,
                              const std::shared_ptr<NativeTensor> &index,
                              const std::shared_ptr<NativeTensor> &src,
                              float alpha) {
  if (device_id != index->device_id || device_id != src->device_id) {
    throw std::runtime_error(
        "index_add_: all tensors must be on the same device");
  }
  if (index->shape.size() != 1) {
    throw std::runtime_error("index_add_: index must be a 1D tensor");
  }
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  if (static_cast<int64_t>(src->shape.size()) != ndim || ndim > 6 ||
      src->shape[nd] != index->shape[0]) {
    throw std::runtime_error(
        "index_add_: src shape does not match target/index");
  }

  auto idx_i32 = ensure_int32_for_index(index);

  if (device_id == -1) {
    if (src->numel == 0)
      return;
    const int32_t *i_ptr =
        reinterpret_cast<const int32_t *>(idx_i32->cpu_raw_ptr());
    const uint8_t *s_raw = src->cpu_raw_ptr();
    uint8_t *d_raw = cpu_raw_ptr();
    const int64_t dim_sz = shape[nd];
    for (size_t n = 0; n < src->numel; ++n) {
      size_t rem = n;
      size_t src_pos = src->storage_offset;
      size_t dst_pos = storage_offset;
      size_t dim_coord = 0;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t sz = static_cast<size_t>(src->shape[d]);
        const size_t coord = rem % sz;
        rem /= sz;
        src_pos += coord * static_cast<size_t>(src->strides[d]);
        if (d == nd) {
          dim_coord = coord;
        } else {
          dst_pos += coord * static_cast<size_t>(strides[d]);
        }
      }
      int64_t k = i_ptr[idx_i32->storage_offset + dim_coord];
      if (k < 0)
        k += dim_sz;
      if (k < 0 || k >= dim_sz) {
        throw std::out_of_range("index_add_: index out of bounds");
      }
      dst_pos += static_cast<size_t>(k) * static_cast<size_t>(strides[nd]);
      const double cur =
          static_cast<double>(read_scalar_as_float(d_raw, dst_pos, dtype));
      const double addv = static_cast<double>(read_scalar_as_float(
                              s_raw, src_pos, src->dtype)) *
                          alpha;
      write_scalar_from_double(d_raw, dst_pos, dtype, cur + addv);
    }
    return;
  }

  if (dtype != ScalarType::Float32) {
    throw std::runtime_error(
        "index_add_ on GPU requires Float32 destination tensor");
  }
  auto src_f = (src->dtype == ScalarType::Float32)
                   ? src
                   : src->cast(ScalarType::Float32);
  if (src_f->numel > 0) {
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t dim;
      uint32_t dim_size_target;
      uint32_t offsetA;
      uint32_t offsetIdx;
      uint32_t offsetB;
      float scalar_val;
      uint32_t shape[6];
      uint32_t strideA[6];
      uint32_t strideIdx[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(src_f->numel);
    pc.ndim = static_cast<uint32_t>(ndim);
    pc.dim = static_cast<uint32_t>(nd);
    pc.dim_size_target = static_cast<uint32_t>(shape[nd]);
    pc.offsetA = static_cast<uint32_t>(src_f->storage_offset);
    pc.offsetIdx = static_cast<uint32_t>(idx_i32->storage_offset);
    pc.offsetB = static_cast<uint32_t>(storage_offset);
    pc.scalar_val = alpha;
    pc.strideIdx[0] = static_cast<uint32_t>(idx_i32->strides[0]);
    for (int64_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(src_f->shape[d]);
      pc.strideA[d] = static_cast<uint32_t>(src_f->strides[d]);
      pc.strideB[d] = static_cast<uint32_t>(strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "index_add_kernel",
        {src_f->gpu_buf(), idx_i32->gpu_buf(), gpu_buf()}, &pc, sizeof(pc),
        groups);
  }
}


} // namespace vultorch
