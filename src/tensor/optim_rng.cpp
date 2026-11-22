#include "tensor/tensor_internal.hpp"

namespace vultorch {

void NativeTensor::adam_step_(const std::shared_ptr<NativeTensor> &grad,
                              const std::shared_ptr<NativeTensor> &exp_avg,
                              const std::shared_ptr<NativeTensor> &exp_avg_sq,
                              float lr, float beta1, float beta2, float eps,
                              float weight_decay, bool decoupled,
                              int64_t step) {
  if (dtype != ScalarType::Float32 || !is_dense_zero_offset()) {
    throw std::runtime_error("adam_step_: parameter must be dense float32");
  }
  for (const auto *t : {grad.get(), exp_avg.get(), exp_avg_sq.get()}) {
    if (t->device_id != device_id) {
      throw std::runtime_error(
          "adam_step_: all tensors must be on the same device");
    }
    if (t->numel != numel || t->dtype != ScalarType::Float32) {
      throw std::runtime_error(
          "adam_step_: tensors must have matching size and float32 dtype");
    }
    if (!t->is_dense_zero_offset()) {
      throw std::runtime_error(
          "adam_step_: tensors must be contiguous zero-offset");
    }
  }
  if (step < 1) {
    throw std::runtime_error("adam_step_: step must be >= 1");
  }

  // Bias corrections in double: beta^t underflows float precision for large t.
  const float bias_c1 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step)));
  const float bias_c2 = static_cast<float>(
      1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step)));

  if (device_id < 0) {
    float *p_ptr = cpu_f32();
    const float *g_ptr = grad->cpu_f32();
    float *m_ptr = exp_avg->cpu_f32();
    float *v_ptr = exp_avg_sq->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      float p = p_ptr[i];
      float g = g_ptr[i];
      if (decoupled)
        p *= 1.0f - lr * weight_decay;
      else
        g += weight_decay * p;
      const float m = beta1 * m_ptr[i] + (1.0f - beta1) * g;
      const float v = beta2 * v_ptr[i] + (1.0f - beta2) * g * g;
      m_ptr[i] = m;
      v_ptr[i] = v;
      p_ptr[i] = p - (lr / bias_c1) * m / (std::sqrt(v / bias_c2) + eps);
    }
    return;
  }

  const uint32_t total_groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  // Keep each dispatch dimension well under the 65535 limit guaranteed by
  // Vulkan.
  const uint32_t groups_x = std::min<uint32_t>(total_groups, 32768u);
  const uint32_t groups_y = (total_groups + groups_x - 1u) / groups_x;

  struct {
    uint32_t numel;
    uint32_t groups_x;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    uint32_t decoupled;
    float bias_c1;
    float bias_c2;
  } pc{static_cast<uint32_t>(numel),
       groups_x,
       lr,
       beta1,
       beta2,
       eps,
       weight_decay,
       decoupled ? 1u : 0u,
       bias_c1,
       bias_c2};

  VulkanPipelineManager::instance().dispatch(
      device_id, "adam_step",
      {gpu_buf(), grad->gpu_buf(), exp_avg->gpu_buf(), exp_avg_sq->gpu_buf()},
      &pc, sizeof(pc), groups_x, groups_y);
}

void NativeTensor::sgd_step_(const std::shared_ptr<NativeTensor> &grad,
                             const std::shared_ptr<NativeTensor> &momentum_buf,
                             float lr, float momentum, float dampening,
                             float weight_decay, bool nesterov,
                             bool first_step) {
  if (grad->device_id != device_id) {
    throw std::runtime_error("sgd_step_: grad must be on the same device");
  }
  if (grad->numel != numel || !is_dense_zero_offset()) {
    throw std::runtime_error("sgd_step_: requires contiguous matching tensors");
  }
  if (!is_floating_point(dtype)) {
    throw std::runtime_error("sgd_step_: parameter must be floating-point");
  }
  if (dtype != ScalarType::Float32) {
    auto p_f32 = cast(ScalarType::Float32);
    auto g_f32 = (grad->dtype == ScalarType::Float32)
                     ? grad->force_contiguous()
                     : grad->cast(ScalarType::Float32);
    std::shared_ptr<NativeTensor> m_f32 = nullptr;
    if (momentum != 0.0f) {
      if (!momentum_buf || momentum_buf->device_id != device_id ||
          momentum_buf->numel != numel) {
        throw std::runtime_error("sgd_step_: invalid momentum buffer");
      }
      m_f32 = (momentum_buf->dtype == ScalarType::Float32)
                  ? momentum_buf->force_contiguous()
                  : momentum_buf->cast(ScalarType::Float32);
    }
    p_f32->sgd_step_(g_f32, m_f32, lr, momentum, dampening, weight_decay,
                     nesterov, first_step);
    copy_from_native(p_f32->cast(dtype));
    if (m_f32 && momentum_buf->dtype != ScalarType::Float32) {
      momentum_buf->copy_from_native(m_f32->cast(momentum_buf->dtype));
    }
    return;
  }
  auto grad_t = grad->force_contiguous();
  std::shared_ptr<NativeTensor> mom_t = nullptr;
  if (momentum != 0.0f) {
    if (!momentum_buf || momentum_buf->device_id != device_id ||
        momentum_buf->numel != numel || !momentum_buf->is_dense_zero_offset()) {
      throw std::runtime_error("sgd_step_: invalid momentum buffer");
    }
    mom_t = momentum_buf;
  }

  if (device_id < 0) {
    float *p_ptr = cpu_f32();
    const float *g_ptr = grad_t->cpu_f32();
    float *m_ptr = mom_t ? mom_t->cpu_f32() : nullptr;
    for (size_t i = 0; i < numel; ++i) {
      float p = p_ptr[i];
      float g = g_ptr[i];
      if (weight_decay != 0.0f)
        g += weight_decay * p;
      float d_p = g;
      if (momentum != 0.0f) {
        float buf =
            first_step ? g : (momentum * m_ptr[i] + (1.0f - dampening) * g);
        m_ptr[i] = buf;
        d_p = nesterov ? (g + momentum * buf) : buf;
      }
      p_ptr[i] = p - lr * d_p;
    }
    return;
  }

  const uint32_t total_groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  const uint32_t groups_x = std::min<uint32_t>(total_groups, 32768u);
  const uint32_t groups_y = (total_groups + groups_x - 1u) / groups_x;

  struct {
    uint32_t numel;
    uint32_t groups_x;
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    uint32_t nesterov;
    uint32_t first_step;
  } pc{static_cast<uint32_t>(numel),
       groups_x,
       lr,
       momentum,
       dampening,
       weight_decay,
       nesterov ? 1u : 0u,
       first_step ? 1u : 0u};

  std::shared_ptr<VulkanBuffer> b_mom = mom_t ? mom_t->gpu_buf() : gpu_buf();
  VulkanPipelineManager::instance().dispatch(
      device_id, "sgd_step", {gpu_buf(), grad_t->gpu_buf(), b_mom}, &pc,
      sizeof(pc), groups_x, groups_y);
}

namespace {
inline void philox4x32_7_cpu(uint32_t ctr[4], uint32_t key[2]) {
  constexpr uint32_t M0 = 0xD2511F53u;
  constexpr uint32_t M1 = 0xCD9E8D57u;
  constexpr uint32_t W0 = 0x9E3779B9u;
  constexpr uint32_t W1 = 0xBB67AE85u;

  for (int r = 0; r < 7; ++r) {
    uint64_t p0 = static_cast<uint64_t>(M0) * ctr[0];
    uint64_t p1 = static_cast<uint64_t>(M1) * ctr[2];
    uint32_t hi0 = static_cast<uint32_t>(p0 >> 32u);
    uint32_t lo0 = static_cast<uint32_t>(p0);
    uint32_t hi1 = static_cast<uint32_t>(p1 >> 32u);
    uint32_t lo1 = static_cast<uint32_t>(p1);

    uint32_t n0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t n1 = lo1;
    uint32_t n2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t n3 = lo0;
    ctr[0] = n0;
    ctr[1] = n1;
    ctr[2] = n2;
    ctr[3] = n3;
    key[0] += W0;
    key[1] += W1;
  }
}

inline float uint_to_uniform01_cpu(uint32_t u) {
  return (static_cast<float>(u >> 8u) + 0.5f) * (1.0f / 16777216.0f);
}
} // namespace

void NativeTensor::random_philox_(uint32_t mode, float p1, float p2,
                                  uint64_t seed, uint32_t offset,
                                  const std::shared_ptr<NativeTensor> &src) {
  if (numel == 0)
    return;
  if (!is_floating_point(dtype)) {
    throw std::runtime_error("random_philox_: requires floating-point tensor");
  }
  if (dtype != ScalarType::Float32 || !is_dense_zero_offset() ||
      (src &&
       (src->dtype != ScalarType::Float32 || !src->is_dense_zero_offset()))) {
    auto tmp =
        std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    std::shared_ptr<NativeTensor> src_f32 = nullptr;
    if (src) {
      src_f32 = (src->dtype == ScalarType::Float32)
                    ? src->force_contiguous()
                    : src->cast(ScalarType::Float32);
    }
    tmp->random_philox_(mode, p1, p2, seed, offset, src_f32);
    copy_from_native(dtype == ScalarType::Float32 ? tmp : tmp->cast(dtype));
    return;
  }

  const uint32_t seed_lo = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
  const uint32_t seed_hi = static_cast<uint32_t>(seed >> 32ull);

  if (device_id >= 0) {
    struct {
      uint32_t numel;
      uint32_t seed_lo;
      uint32_t seed_hi;
      uint32_t offset;
      float p1;
      float p2;
      uint32_t mode;
    } pc{static_cast<uint32_t>(numel), seed_lo, seed_hi, offset, p1, p2, mode};

    uint32_t num_threads = (static_cast<uint32_t>(numel) + 3u) / 4u;
    uint32_t groups = (num_threads + 255u) / 256u;
    std::shared_ptr<VulkanBuffer> b_in = src ? src->gpu_buf() : gpu_buf();

    VulkanPipelineManager::instance().dispatch(device_id, "philox_rng",
                                               {b_in, gpu_buf()}, &pc,
                                               sizeof(pc), groups, 1, 1);
  } else {
    float *out_ptr = cpu_f32();
    const float *in_ptr = src ? src->cpu_f32() : out_ptr;
    const size_t num_quads = (numel + 3u) / 4u;
    for (size_t q = 0; q < num_quads; ++q) {
      uint32_t ctr[4] = {static_cast<uint32_t>(q), offset, 0u, 0u};
      uint32_t key[2] = {seed_lo, seed_hi};
      philox4x32_7_cpu(ctr, key);

      float us[4] = {
          uint_to_uniform01_cpu(ctr[0]),
          uint_to_uniform01_cpu(ctr[1]),
          uint_to_uniform01_cpu(ctr[2]),
          uint_to_uniform01_cpu(ctr[3]),
      };

      float vals[4];
      if (mode == 0u) {
        float span = p2 - p1;
        for (int k = 0; k < 4; ++k)
          vals[k] = p1 + us[k] * span;
      } else if (mode == 1u) {
        constexpr float kTwoPi = 6.28318530718f;
        float r0 = std::sqrt(-2.0f * std::log(us[0]));
        float r1 = std::sqrt(-2.0f * std::log(us[2]));
        vals[0] = p1 + p2 * (r0 * std::cos(kTwoPi * us[1]));
        vals[1] = p1 + p2 * (r0 * std::sin(kTwoPi * us[1]));
        vals[2] = p1 + p2 * (r1 * std::cos(kTwoPi * us[3]));
        vals[3] = p1 + p2 * (r1 * std::sin(kTwoPi * us[3]));
      } else {
        for (int k = 0; k < 4; ++k) {
          size_t idx = q * 4u + static_cast<size_t>(k);
          vals[k] = (idx < numel && us[k] >= p1) ? (in_ptr[idx] * p2) : 0.0f;
        }
      }

      for (int k = 0; k < 4; ++k) {
        size_t idx = q * 4u + static_cast<size_t>(k);
        if (idx < numel)
          out_ptr[idx] = vals[k];
      }
    }
  }
}


} // namespace vultorch
