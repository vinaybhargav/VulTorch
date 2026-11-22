#include "tensor/tensor_internal.hpp"

namespace vultorch {

std::shared_ptr<NativeTensor> NativeTensor::scan_op(int64_t dim,
                                                    uint32_t mode) const {
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  auto cont = is_contiguous() ? shared_from_this() : contiguous();

  uint32_t outer = 1, inner = 1;
  for (int64_t d = 0; d < nd; ++d)
    outer *= static_cast<uint32_t>(cont->shape[d]);
  const uint32_t dim_sz = static_cast<uint32_t>(cont->shape[nd]);
  for (int64_t d = nd + 1; d < ndim; ++d)
    inner *= static_cast<uint32_t>(cont->shape[d]);
  const uint32_t total_slices = outer * inner;

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(cont->shape, -1, dtype);
    const uint8_t *i_raw = cont->cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    for (uint32_t s = 0; s < total_slices; ++s) {
      const uint32_t o = s / inner;
      const uint32_t in_r = s % inner;
      double acc = (mode == 1u) ? 1.0 : 0.0;
      for (uint32_t j = 0; j < dim_sz; ++j) {
        const size_t pos = (o * dim_sz + j) * inner + in_r;
        const double v = static_cast<double>(read_scalar_as_float(
            i_raw, cont->storage_offset + pos, cont->dtype));
        acc = (mode == 1u) ? (acc * v) : (acc + v);
        write_scalar_from_double(o_raw, pos, dtype, acc);
      }
    }
    return out;
  }

  auto x_f = (cont->dtype == ScalarType::Float32)
                 ? cont
                 : cont->cast(ScalarType::Float32);
  auto out_f = std::make_shared<NativeTensor>(cont->shape, device_id,
                                              ScalarType::Float32);
  if (total_slices > 0 && dim_sz > 0) {
    struct {
      uint32_t total_slices;
      uint32_t dim_size;
      uint32_t inner_size;
      uint32_t mode;
      uint32_t out_k;
      uint32_t src_offset;
      uint32_t pad_pow2;
      uint32_t _pad;
    } pc{total_slices, dim_sz, inner,
         mode,         dim_sz, static_cast<uint32_t>(x_f->storage_offset),
         1u,           0u};
    VulkanPipelineManager::instance().dispatch(
        device_id, "scan_kernel",
        {x_f->gpu_buf(), out_f->gpu_buf(), out_f->gpu_buf()}, &pc, sizeof(pc),
        total_slices);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::sort_topk(int64_t dim, int64_t k, bool descending) const {
  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  auto cont = is_contiguous() ? shared_from_this() : contiguous();

  const int64_t dim_sz = cont->shape[nd];
  const int64_t out_k = (k < 0) ? dim_sz : k;
  if (out_k < 0 || out_k > dim_sz) {
    throw std::out_of_range("topk/sort: k out of range");
  }

  uint32_t outer = 1, inner = 1;
  for (int64_t d = 0; d < nd; ++d)
    outer *= static_cast<uint32_t>(cont->shape[d]);
  for (int64_t d = nd + 1; d < ndim; ++d)
    inner *= static_cast<uint32_t>(cont->shape[d]);
  const uint32_t total_slices = outer * inner;

  std::vector<int64_t> out_shape = cont->shape;
  out_shape[nd] = out_k;

  if (device_id == -1) {
    auto out_v = std::make_shared<NativeTensor>(out_shape, -1, dtype);
    auto out_i =
        std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Int64);
    const uint8_t *i_raw = cont->cpu_raw_ptr();
    uint8_t *ov_raw = out_v->cpu_raw_ptr();
    int64_t *oi_ptr = reinterpret_cast<int64_t *>(out_i->cpu_raw_ptr());

    std::vector<std::pair<float, int64_t>> buf(static_cast<size_t>(dim_sz));
    for (uint32_t s = 0; s < total_slices; ++s) {
      const uint32_t o = s / inner;
      const uint32_t in_r = s % inner;
      for (int64_t j = 0; j < dim_sz; ++j) {
        const size_t pos =
            cont->storage_offset + (o * dim_sz + j) * inner + in_r;
        buf[static_cast<size_t>(j)] = {
            read_scalar_as_float(i_raw, pos, cont->dtype), j};
      }
      std::stable_sort(
          buf.begin(), buf.end(), [descending](const auto &a, const auto &b) {
            return descending ? (a.first > b.first) : (a.first < b.first);
          });
      for (int64_t j = 0; j < out_k; ++j) {
        const size_t opos =
            (o * static_cast<size_t>(out_k) + static_cast<size_t>(j)) * inner +
            in_r;
        write_scalar_from_double(
            ov_raw, opos, dtype,
            static_cast<double>(buf[static_cast<size_t>(j)].first));
        oi_ptr[opos] = buf[static_cast<size_t>(j)].second;
      }
    }
    return {out_v, out_i};
  }

  if (dim_sz > 1024) {
    throw std::runtime_error(
        "GPU sort/topk currently supports dim_size <= 1024");
  }
  uint32_t pow2 = 1;
  while (pow2 < static_cast<uint32_t>(dim_sz))
    pow2 <<= 1u;

  auto x_f = (cont->dtype == ScalarType::Float32)
                 ? cont
                 : cont->cast(ScalarType::Float32);
  auto out_vf =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  auto out_i32 =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Int32);
  if (total_slices > 0 && out_k > 0) {
    struct {
      uint32_t total_slices;
      uint32_t dim_size;
      uint32_t inner_size;
      uint32_t mode;
      uint32_t out_k;
      uint32_t src_offset;
      uint32_t pad_pow2;
      uint32_t _pad;
    } pc{total_slices,
         static_cast<uint32_t>(dim_sz),
         inner,
         descending ? 1u : 0u,
         static_cast<uint32_t>(out_k),
         static_cast<uint32_t>(x_f->storage_offset),
         pow2,
         0u};
    VulkanPipelineManager::instance().dispatch(
        device_id, "sort_topk_kernel",
        {x_f->gpu_buf(), out_vf->gpu_buf(), out_i32->gpu_buf()}, &pc,
        sizeof(pc), total_slices);
  }
  auto out_v = (dtype == ScalarType::Float32) ? out_vf : out_vf->cast(dtype);
  auto out_i = out_i32->cast(ScalarType::Int64);
  return {out_v, out_i};
}


} // namespace vultorch
