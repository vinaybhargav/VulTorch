#include "tensor/tensor_internal.hpp"

namespace vultorch {

std::shared_ptr<NativeTensor> NativeTensor::softmax(int64_t dim, bool is_log) {
  if (shape.empty()) {
    throw std::runtime_error(
        "softmax requires a tensor with at least 1 dimension");
  }
  if (dim < 0)
    dim += static_cast<int64_t>(shape.size());
  if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
    throw std::runtime_error("softmax: dim " + std::to_string(dim) +
                             " out of range for rank " +
                             std::to_string(shape.size()));
  }
  if (dtype != ScalarType::Float32) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("softmax requires floating-point tensor");
    }
    return cast(ScalarType::Float32)->softmax(dim, is_log)->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->softmax(dim, is_log);
  }

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  const bool last_dim = (dim == static_cast<int64_t>(shape.size()) - 1);

  if (device_id >= 0 && last_dim) {
    uint32_t cols = static_cast<uint32_t>(shape.back());
    uint32_t rows = static_cast<uint32_t>(numel / cols);

    struct {
      uint32_t num_rows;
      uint32_t cols;
      uint32_t is_log_softmax;
    } pc{rows, cols, is_log ? 1u : 0u};

    VulkanPipelineManager::instance().dispatch(device_id, "softmax",
                                               {gpu_buf(), out->gpu_buf()}, &pc,
                                               sizeof(pc), 1, rows, 1);
    return out;
  }

  // Dimension-aware path for any axis (and for CPU tensors).
  auto cpu_in = (device_id >= 0) ? to_device(-1) : shared_from_this();
  auto cpu_out = std::make_shared<NativeTensor>(shape, -1, ScalarType::Float32);
  const float *in_ptr = cpu_in->cpu_f32();
  float *out_ptr = cpu_out->cpu_f32();

  int64_t inner = 1;
  for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  const int64_t dim_size = shape[dim];
  const int64_t outer = static_cast<int64_t>(numel) / (dim_size * inner);

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      const int64_t base = o * dim_size * inner + i;

      float m = -1e30f;
      for (int64_t d = 0; d < dim_size; ++d) {
        m = std::max(m, in_ptr[base + d * inner]);
      }

      float s = 0.0f;
      for (int64_t d = 0; d < dim_size; ++d) {
        s += std::exp(in_ptr[base + d * inner] - m);
      }

      const float log_s = std::log(s);
      const float inv_s = 1.0f / s;
      for (int64_t d = 0; d < dim_size; ++d) {
        const float v = in_ptr[base + d * inner] - m;
        out_ptr[base + d * inner] =
            is_log ? (v - log_s) : (std::exp(v) * inv_s);
      }
    }
  }

  return (device_id >= 0) ? cpu_out->to_device(device_id) : cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::cross_entropy_per_row(
    const std::shared_ptr<NativeTensor> &targets) const {
  if (shape.size() != 2) {
    throw std::runtime_error(
        "cross_entropy requires a 2-D logits tensor [rows, cols]");
  }
  if (!targets || targets->numel != static_cast<size_t>(shape[0])) {
    throw std::runtime_error(
        "cross_entropy: expected exactly one target per row");
  }
  if (targets->device_id != device_id) {
    throw std::runtime_error(
        "cross_entropy: targets must be on the same device as logits");
  }
  if (dtype != ScalarType::Float32) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("cross_entropy requires floating-point logits");
    }
    return cast(ScalarType::Float32)
        ->cross_entropy_per_row(targets)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->cross_entropy_per_row(targets);
  }
  auto t_f32 = (targets->dtype == ScalarType::Float32)
                   ? targets->force_contiguous()
                   : targets->cast(ScalarType::Float32);

  const uint32_t rows = static_cast<uint32_t>(shape[0]);
  const uint32_t cols = static_cast<uint32_t>(shape[1]);
  auto out = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(rows)}, device_id,
      ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t rows;
      uint32_t cols;
      uint32_t mode; // 0 = per-row loss
      uint32_t has_grad_buf;
      uint32_t g_is_scalar;
      float scale;
    } pc{rows, cols, 0u, 0u, 0u, 1.0f};

    VulkanPipelineManager::instance().dispatch(
        device_id, "softmax_cross_entropy",
        {gpu_buf(), t_f32->gpu_buf(), out->gpu_buf(), out->gpu_buf()}, &pc,
        sizeof(pc), rows);
    return out;
  }

  const float *logits_ptr = cpu_f32();
  const float *t_ptr = t_f32->cpu_f32();
  float *out_ptr = out->cpu_f32();
  for (uint32_t r = 0; r < rows; ++r) {
    const float *row = &logits_ptr[static_cast<size_t>(r) * cols];
    const int64_t target = static_cast<int64_t>(t_ptr[r]);
    if (target < 0 || target >= static_cast<int64_t>(cols)) {
      throw std::runtime_error("cross_entropy: target index out of range");
    }

    float m = -1e30f;
    for (uint32_t c = 0; c < cols; ++c)
      m = std::max(m, row[c]);
    float s = 0.0f;
    for (uint32_t c = 0; c < cols; ++c)
      s += std::exp(row[c] - m);

    out_ptr[r] = (m + std::log(s)) - row[target];
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::cross_entropy_grad(const std::shared_ptr<NativeTensor> &targets,
                                 const std::shared_ptr<NativeTensor> &grad_out,
                                 float scale) const {
  if (shape.size() != 2) {
    throw std::runtime_error(
        "cross_entropy requires a 2-D logits tensor [rows, cols]");
  }
  if (!targets || targets->numel != static_cast<size_t>(shape[0])) {
    throw std::runtime_error(
        "cross_entropy: expected exactly one target per row");
  }
  if (targets->device_id != device_id) {
    throw std::runtime_error(
        "cross_entropy: targets must be on the same device as logits");
  }

  const uint32_t rows = static_cast<uint32_t>(shape[0]);
  const uint32_t cols = static_cast<uint32_t>(shape[1]);

  if (grad_out != nullptr) {
    if (grad_out->device_id != device_id) {
      throw std::runtime_error(
          "cross_entropy_grad: grad_out must be on the same device");
    }
    if (grad_out->numel != 1 && grad_out->numel != rows) {
      throw std::runtime_error(
          "cross_entropy_grad: grad_out must have 1 or rows elements");
    }
  }
  if (dtype != ScalarType::Float32) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error(
          "cross_entropy_grad requires floating-point logits");
    }
    return cast(ScalarType::Float32)
        ->cross_entropy_grad(targets, grad_out, scale)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->cross_entropy_grad(targets, grad_out, scale);
  }
  auto t_f32 = (targets->dtype == ScalarType::Float32)
                   ? targets->force_contiguous()
                   : targets->cast(ScalarType::Float32);
  std::shared_ptr<NativeTensor> g_f32 = nullptr;
  if (grad_out != nullptr) {
    g_f32 = (grad_out->dtype == ScalarType::Float32)
                ? grad_out->force_contiguous()
                : grad_out->cast(ScalarType::Float32);
  }

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t rows;
      uint32_t cols;
      uint32_t mode; // 1 = gradient
      uint32_t has_grad_buf;
      uint32_t g_is_scalar;
      float scale;
    } pc{rows,
         cols,
         1u,
         (g_f32 != nullptr) ? 1u : 0u,
         (g_f32 != nullptr && g_f32->numel == 1) ? 1u : 0u,
         scale};

    std::shared_ptr<VulkanBuffer> g_buf =
        g_f32 ? g_f32->gpu_buf() : out->gpu_buf();

    VulkanPipelineManager::instance().dispatch(
        device_id, "softmax_cross_entropy",
        {gpu_buf(), t_f32->gpu_buf(), out->gpu_buf(), g_buf}, &pc, sizeof(pc),
        rows);
    return out;
  }

  const float *logits_ptr = cpu_f32();
  const float *t_ptr = t_f32->cpu_f32();
  const float *g_ptr = g_f32 ? g_f32->cpu_f32() : nullptr;
  float *out_ptr = out->cpu_f32();
  for (uint32_t r = 0; r < rows; ++r) {
    const float *row = &logits_ptr[static_cast<size_t>(r) * cols];
    const int64_t target = static_cast<int64_t>(t_ptr[r]);
    if (target < 0 || target >= static_cast<int64_t>(cols)) {
      throw std::runtime_error("cross_entropy: target index out of range");
    }

    float m = -1e30f;
    for (uint32_t c = 0; c < cols; ++c)
      m = std::max(m, row[c]);
    float s = 0.0f;
    for (uint32_t c = 0; c < cols; ++c)
      s += std::exp(row[c] - m);
    const float inv_s = 1.0f / s;

    float row_scale = scale;
    if (g_f32 != nullptr) {
      row_scale *= g_ptr[(g_f32->numel == 1) ? 0 : r];
    }

    for (uint32_t c = 0; c < cols; ++c) {
      const float probability = std::exp(row[c] - m) * inv_s;
      const float onehot = (static_cast<int64_t>(c) == target) ? 1.0f : 0.0f;
      out_ptr[static_cast<size_t>(r) * cols + c] =
          (probability - onehot) * row_scale;
    }
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::embedding(std::shared_ptr<NativeTensor> weight,
                        int64_t padding_idx) {
  if (weight->shape.size() != 2) {
    throw std::runtime_error(
        "embedding weight must be 2D [num_embeddings, embedding_dim]");
  }
  if (weight->device_id != device_id) {
    throw std::runtime_error(
        "embedding indices and weight must be on the same device");
  }

  const ScalarType out_dtype = weight->dtype;
  if (dtype != ScalarType::Float32 || !is_dense_zero_offset() ||
      weight->dtype != ScalarType::Float32 || !weight->is_dense_zero_offset()) {
    auto idx_f32 = (dtype == ScalarType::Float32) ? force_contiguous()
                                                  : cast(ScalarType::Float32);
    auto w_f32 = (weight->dtype == ScalarType::Float32)
                     ? weight->force_contiguous()
                     : weight->cast(ScalarType::Float32);
    auto res = idx_f32->embedding(w_f32, padding_idx);
    return (out_dtype == ScalarType::Float32) ? res : res->cast(out_dtype);
  }

  const uint32_t num_embeddings = static_cast<uint32_t>(weight->shape[0]);
  const uint32_t embedding_dim = static_cast<uint32_t>(weight->shape[1]);
  const uint32_t num_indices = static_cast<uint32_t>(numel);

  std::vector<int64_t> out_shape = shape;
  out_shape.push_back(static_cast<int64_t>(embedding_dim));
  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (out->numel == 0)
    return out;

  if (device_id >= 0) {
    struct {
      uint32_t num_indices;
      uint32_t embedding_dim;
      uint32_t num_embeddings;
      int32_t padding_idx;
    } pc{num_indices, embedding_dim, num_embeddings,
         static_cast<int32_t>(padding_idx)};

    uint32_t groups = (static_cast<uint32_t>(out->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "embedding_forward",
        {gpu_buf(), weight->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), groups,
        1, 1);
  } else {
    const float *idx_ptr = cpu_f32();
    const float *w_ptr = weight->cpu_f32();
    float *out_ptr = out->cpu_f32();
    for (uint32_t i = 0; i < num_indices; ++i) {
      int tok = static_cast<int>(idx_ptr[i]);
      for (uint32_t c = 0; c < embedding_dim; ++c) {
        if (tok == static_cast<int>(padding_idx) || tok < 0 ||
            static_cast<uint32_t>(tok) >= num_embeddings) {
          out_ptr[i * embedding_dim + c] = 0.0f;
        } else {
          out_ptr[i * embedding_dim + c] =
              w_ptr[static_cast<uint32_t>(tok) * embedding_dim + c];
        }
      }
    }
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::embedding_backward(std::shared_ptr<NativeTensor> indices,
                                 int64_t num_embeddings, int64_t padding_idx) {
  if (indices->device_id != device_id) {
    throw std::runtime_error("embedding_backward: device mismatch");
  }
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)
        ->embedding_backward(indices, num_embeddings, padding_idx)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->embedding_backward(indices, num_embeddings,
                                                  padding_idx);
  }
  auto idx_f32 = (indices->dtype == ScalarType::Float32)
                     ? indices->force_contiguous()
                     : indices->cast(ScalarType::Float32);

  const uint32_t embedding_dim = static_cast<uint32_t>(shape.back());
  const uint32_t num_indices = static_cast<uint32_t>(idx_f32->numel);
  const uint32_t num_emb = static_cast<uint32_t>(num_embeddings);

  auto grad_weight = std::make_shared<NativeTensor>(
      std::vector<int64_t>{num_embeddings, static_cast<int64_t>(embedding_dim)},
      device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t num_indices;
      uint32_t embedding_dim;
      uint32_t num_embeddings;
      int32_t padding_idx;
    } pc{num_indices, embedding_dim, num_emb,
         static_cast<int32_t>(padding_idx)};

    uint32_t gx = (embedding_dim + 31u) / 32u;
    uint32_t gy = (num_emb + 7u) / 8u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "embedding_backward",
        {idx_f32->gpu_buf(), gpu_buf(), grad_weight->gpu_buf()}, &pc,
        sizeof(pc), gx, gy, 1);
  } else {
    grad_weight->zero_();
    const float *idx_ptr = idx_f32->cpu_f32();
    const float *gout_ptr = cpu_f32();
    float *gw_ptr = grad_weight->cpu_f32();
    for (uint32_t i = 0; i < num_indices; ++i) {
      int tok = static_cast<int>(idx_ptr[i]);
      if (tok == static_cast<int>(padding_idx) || tok < 0 ||
          static_cast<uint32_t>(tok) >= num_emb) {
        continue;
      }
      for (uint32_t c = 0; c < embedding_dim; ++c) {
        gw_ptr[static_cast<uint32_t>(tok) * embedding_dim + c] +=
            gout_ptr[i * embedding_dim + c];
      }
    }
  }
  return grad_weight;
}


} // namespace vultorch
