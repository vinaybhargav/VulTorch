#include "tensor/tensor_internal.hpp"

namespace vultorch {

std::shared_ptr<NativeTensor>
NativeTensor::layernorm(std::shared_ptr<NativeTensor> gamma,
                        std::shared_ptr<NativeTensor> beta, float eps,
                        uint32_t norm_dims) {
  if (norm_dims == 0 || norm_dims > shape.size()) {
    throw std::runtime_error("layernorm: invalid norm_dims");
  }
  if (dtype != ScalarType::Float32 ||
      (gamma && gamma->dtype != ScalarType::Float32) ||
      (beta && beta->dtype != ScalarType::Float32)) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("layernorm requires floating-point tensor");
    }
    auto in_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                 : cast(ScalarType::Float32);
    auto g_f32 = (gamma && gamma->dtype != ScalarType::Float32)
                     ? gamma->cast(ScalarType::Float32)
                     : gamma;
    auto b_f32 = (beta && beta->dtype != ScalarType::Float32)
                     ? beta->cast(ScalarType::Float32)
                     : beta;
    return in_f32->layernorm(g_f32, b_f32, eps, norm_dims)->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->layernorm(gamma, beta, eps, norm_dims);
  }
  auto gamma_t = gamma ? gamma->force_contiguous() : nullptr;
  auto beta_t = beta ? beta->force_contiguous() : nullptr;

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  uint32_t cols = 1;
  for (size_t i = shape.size() - norm_dims; i < shape.size(); ++i) {
    cols *= static_cast<uint32_t>(shape[i]);
  }
  uint32_t rows = static_cast<uint32_t>(numel / cols);

  if (device_id >= 0) {
    struct {
      uint32_t num_rows;
      uint32_t cols;
      float eps;
      uint32_t has_affine;
    } pc{rows, cols, eps, (gamma_t != nullptr) ? 1u : 0u};

    std::shared_ptr<VulkanBuffer> b_gamma =
        gamma_t ? gamma_t->gpu_buf() : out->gpu_buf();
    std::shared_ptr<VulkanBuffer> b_beta =
        beta_t ? beta_t->gpu_buf() : out->gpu_buf();

    VulkanPipelineManager::instance().dispatch(
        device_id, "layernorm", {gpu_buf(), b_gamma, b_beta, out->gpu_buf()},
        &pc, sizeof(pc), 1, rows, 1);
  } else {
    const float *in_ptr = cpu_f32();
    const float *gam_ptr = gamma_t ? gamma_t->cpu_f32() : nullptr;
    const float *bet_ptr = beta_t ? beta_t->cpu_f32() : nullptr;
    float *out_ptr = out->cpu_f32();
    for (uint32_t r = 0; r < rows; ++r) {
      float mean = 0.0f;
      for (uint32_t c = 0; c < cols; ++c)
        mean += in_ptr[r * cols + c];
      mean /= cols;

      float var = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float diff = in_ptr[r * cols + c] - mean;
        var += diff * diff;
      }
      var /= cols;
      float inv_std = 1.0f / std::sqrt(var + eps);

      for (uint32_t c = 0; c < cols; ++c) {
        float norm = (in_ptr[r * cols + c] - mean) * inv_std;
        if (gam_ptr)
          norm = norm * gam_ptr[c] + (bet_ptr ? bet_ptr[c] : 0.0f);
        out_ptr[r * cols + c] = norm;
      }
    }
  }
  return out;
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>>
NativeTensor::layernorm_backward(std::shared_ptr<NativeTensor> grad_out,
                                 std::shared_ptr<NativeTensor> gamma, float eps,
                                 uint32_t norm_dims, bool need_dx,
                                 bool need_dgamma, bool need_dbeta) {
  if (dtype != ScalarType::Float32 || grad_out->dtype != ScalarType::Float32 ||
      (gamma && gamma->dtype != ScalarType::Float32)) {
    auto in_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                 : cast(ScalarType::Float32);
    auto go_f32 = (grad_out->dtype == ScalarType::Float32)
                      ? grad_out
                      : grad_out->cast(ScalarType::Float32);
    auto ga_f32 = (gamma && gamma->dtype != ScalarType::Float32)
                      ? gamma->cast(ScalarType::Float32)
                      : gamma;
    auto [dx_f, dgam_f, dbe_f] = in_f32->layernorm_backward(
        go_f32, ga_f32, eps, norm_dims, need_dx, need_dgamma, need_dbeta);
    return {
        dx_f ? dx_f->cast(dtype) : nullptr,
        dgam_f ? dgam_f->cast(gamma ? gamma->dtype : dtype) : nullptr,
        dbe_f ? dbe_f->cast(dtype) : nullptr,
    };
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->layernorm_backward(
        grad_out, gamma, eps, norm_dims, need_dx, need_dgamma, need_dbeta);
  }
  auto go_t = grad_out->force_contiguous();
  auto gamma_t = gamma ? gamma->force_contiguous() : nullptr;

  uint32_t cols = 1;
  std::vector<int64_t> param_shape;
  for (size_t i = shape.size() - norm_dims; i < shape.size(); ++i) {
    cols *= static_cast<uint32_t>(shape[i]);
    param_shape.push_back(shape[i]);
  }
  uint32_t rows = static_cast<uint32_t>(numel / cols);

  std::shared_ptr<NativeTensor> dx = nullptr;
  std::shared_ptr<NativeTensor> dgamma = nullptr;
  std::shared_ptr<NativeTensor> dbeta = nullptr;

  if (device_id >= 0) {
    uint32_t flags = (gamma_t != nullptr ? 1u : 0u) | (need_dx ? 2u : 0u) |
                     (need_dgamma ? 4u : 0u);
    struct {
      uint32_t num_rows;
      uint32_t cols;
      float eps;
      uint32_t has_affine;
    } pc{rows, cols, eps, flags};

    std::shared_ptr<VulkanBuffer> b_gamma =
        gamma_t ? gamma_t->gpu_buf() : gpu_buf();
    std::shared_ptr<NativeTensor> gy_xhat_2d = nullptr;

    if (need_dx) {
      dx =
          std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    }
    if (need_dgamma) {
      gy_xhat_2d = std::make_shared<NativeTensor>(
          std::vector<int64_t>{static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols)},
          device_id, ScalarType::Float32);
    }

    if (need_dx || need_dgamma) {
      std::shared_ptr<VulkanBuffer> b_dx = dx ? dx->gpu_buf() : gpu_buf();
      std::shared_ptr<VulkanBuffer> b_aux =
          gy_xhat_2d ? gy_xhat_2d->gpu_buf() : gpu_buf();
      VulkanPipelineManager::instance().dispatch(
          device_id, "layernorm_backward_dx",
          {gpu_buf(), go_t->gpu_buf(), b_gamma, b_dx, b_aux}, &pc, sizeof(pc),
          1, rows, 1);
    }

    if (need_dgamma) {
      dgamma = gy_xhat_2d->reduce_sum(0, false)->reshape_view(param_shape);
    }

    if (need_dbeta) {
      auto g_2d = go_t->reshape_view(std::vector<int64_t>{
          static_cast<int64_t>(rows), static_cast<int64_t>(cols)});
      dbeta = g_2d->reduce_sum(0, false)->reshape_view(param_shape);
    }
  } else {
    if (need_dx)
      dx = std::make_shared<NativeTensor>(shape, -1, ScalarType::Float32);
    if (need_dgamma) {
      dgamma =
          std::make_shared<NativeTensor>(param_shape, -1, ScalarType::Float32);
      dgamma->zero_();
    }
    if (need_dbeta) {
      dbeta =
          std::make_shared<NativeTensor>(param_shape, -1, ScalarType::Float32);
      dbeta->zero_();
    }

    const float *in_ptr = cpu_f32();
    const float *go_ptr = go_t->cpu_f32();
    const float *gam_ptr = gamma_t ? gamma_t->cpu_f32() : nullptr;
    float *dx_ptr = dx ? dx->cpu_f32() : nullptr;
    float *dgam_ptr = dgamma ? dgamma->cpu_f32() : nullptr;
    float *dbe_ptr = dbeta ? dbeta->cpu_f32() : nullptr;

    const float inv_cols = 1.0f / static_cast<float>(cols);
    for (uint32_t r = 0; r < rows; ++r) {
      float mean = 0.0f;
      for (uint32_t c = 0; c < cols; ++c)
        mean += in_ptr[r * cols + c];
      mean *= inv_cols;

      float var = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float diff = in_ptr[r * cols + c] - mean;
        var += diff * diff;
      }
      float inv_std = 1.0f / std::sqrt(var * inv_cols + eps);

      float sum_gy = 0.0f;
      float sum_gy_xhat = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float xhat = (in_ptr[r * cols + c] - mean) * inv_std;
        float g = go_ptr[r * cols + c];
        float w = gam_ptr ? gam_ptr[c] : 1.0f;
        float gy = g * w;
        sum_gy += gy;
        sum_gy_xhat += gy * xhat;
        if (dgam_ptr)
          dgam_ptr[c] += g * xhat;
        if (dbe_ptr)
          dbe_ptr[c] += g;
      }
      if (dx_ptr) {
        for (uint32_t c = 0; c < cols; ++c) {
          float xhat = (in_ptr[r * cols + c] - mean) * inv_std;
          float w = gam_ptr ? gam_ptr[c] : 1.0f;
          float gy = go_ptr[r * cols + c] * w;
          dx_ptr[r * cols + c] =
              (gy - sum_gy * inv_cols - xhat * sum_gy_xhat * inv_cols) *
              inv_std;
        }
      }
    }
  }
  return {dx, dgamma, dbeta};
}

std::shared_ptr<NativeTensor>
NativeTensor::rmsnorm(std::shared_ptr<NativeTensor> gamma, float eps,
                      uint32_t norm_dims) {
  if (norm_dims == 0 || norm_dims > shape.size()) {
    throw std::runtime_error("rmsnorm: invalid norm_dims");
  }
  if (dtype != ScalarType::Float32 ||
      (gamma && gamma->dtype != ScalarType::Float32)) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("rmsnorm requires floating-point tensor");
    }
    auto in_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                 : cast(ScalarType::Float32);
    auto g_f32 = (gamma && gamma->dtype != ScalarType::Float32)
                     ? gamma->cast(ScalarType::Float32)
                     : gamma;
    return in_f32->rmsnorm(g_f32, eps, norm_dims)->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->rmsnorm(gamma, eps, norm_dims);
  }
  auto gamma_t = gamma ? gamma->force_contiguous() : nullptr;

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  uint32_t cols = 1;
  for (size_t i = shape.size() - norm_dims; i < shape.size(); ++i) {
    cols *= static_cast<uint32_t>(shape[i]);
  }
  uint32_t rows = static_cast<uint32_t>(numel / cols);

  if (device_id >= 0) {
    struct {
      uint32_t num_rows;
      uint32_t cols;
      float eps;
      uint32_t has_affine;
    } pc{rows, cols, eps, (gamma_t != nullptr) ? 1u : 0u};

    std::shared_ptr<VulkanBuffer> b_gamma =
        gamma_t ? gamma_t->gpu_buf() : out->gpu_buf();
    VulkanPipelineManager::instance().dispatch(
        device_id, "rmsnorm", {gpu_buf(), b_gamma, b_gamma, out->gpu_buf()},
        &pc, sizeof(pc), 1, rows, 1);
  } else {
    const float *in_ptr = cpu_f32();
    const float *gam_ptr = gamma_t ? gamma_t->cpu_f32() : nullptr;
    float *out_ptr = out->cpu_f32();
    for (uint32_t r = 0; r < rows; ++r) {
      float sq_sum = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float v = in_ptr[r * cols + c];
        sq_sum += v * v;
      }
      float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(cols) + eps);
      for (uint32_t c = 0; c < cols; ++c) {
        float norm = in_ptr[r * cols + c] * inv_rms;
        if (gam_ptr)
          norm *= gam_ptr[c];
        out_ptr[r * cols + c] = norm;
      }
    }
  }
  return out;
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::rmsnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                               std::shared_ptr<NativeTensor> gamma, float eps,
                               uint32_t norm_dims, bool need_dx,
                               bool need_dgamma) {
  if (dtype != ScalarType::Float32 || grad_out->dtype != ScalarType::Float32 ||
      (gamma && gamma->dtype != ScalarType::Float32)) {
    auto in_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                 : cast(ScalarType::Float32);
    auto go_f32 = (grad_out->dtype == ScalarType::Float32)
                      ? grad_out
                      : grad_out->cast(ScalarType::Float32);
    auto ga_f32 = (gamma && gamma->dtype != ScalarType::Float32)
                      ? gamma->cast(ScalarType::Float32)
                      : gamma;
    auto [dx_f, dgam_f] = in_f32->rmsnorm_backward(
        go_f32, ga_f32, eps, norm_dims, need_dx, need_dgamma);
    return {
        dx_f ? dx_f->cast(dtype) : nullptr,
        dgam_f ? dgam_f->cast(gamma ? gamma->dtype : dtype) : nullptr,
    };
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->rmsnorm_backward(grad_out, gamma, eps, norm_dims,
                                                need_dx, need_dgamma);
  }
  auto go_t = grad_out->force_contiguous();
  auto gamma_t = gamma ? gamma->force_contiguous() : nullptr;

  uint32_t cols = 1;
  std::vector<int64_t> param_shape;
  for (size_t i = shape.size() - norm_dims; i < shape.size(); ++i) {
    cols *= static_cast<uint32_t>(shape[i]);
    param_shape.push_back(shape[i]);
  }
  uint32_t rows = static_cast<uint32_t>(numel / cols);

  std::shared_ptr<NativeTensor> dx = nullptr;
  std::shared_ptr<NativeTensor> dgamma = nullptr;

  if (device_id >= 0) {
    uint32_t flags = (gamma_t != nullptr ? 1u : 0u) | (need_dx ? 2u : 0u) |
                     (need_dgamma ? 4u : 0u);
    struct {
      uint32_t num_rows;
      uint32_t cols;
      float eps;
      uint32_t has_affine;
    } pc{rows, cols, eps, flags};

    std::shared_ptr<VulkanBuffer> b_gamma =
        gamma_t ? gamma_t->gpu_buf() : gpu_buf();
    std::shared_ptr<NativeTensor> gy_xhat_2d = nullptr;

    if (need_dx) {
      dx =
          std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    }
    if (need_dgamma) {
      gy_xhat_2d = std::make_shared<NativeTensor>(
          std::vector<int64_t>{static_cast<int64_t>(rows),
                               static_cast<int64_t>(cols)},
          device_id, ScalarType::Float32);
    }

    if (need_dx || need_dgamma) {
      std::shared_ptr<VulkanBuffer> b_dx = dx ? dx->gpu_buf() : gpu_buf();
      std::shared_ptr<VulkanBuffer> b_aux =
          gy_xhat_2d ? gy_xhat_2d->gpu_buf() : gpu_buf();
      VulkanPipelineManager::instance().dispatch(
          device_id, "rmsnorm_backward_dx",
          {gpu_buf(), go_t->gpu_buf(), b_gamma, b_dx, b_aux}, &pc, sizeof(pc),
          1, rows, 1);
    }

    if (need_dgamma) {
      dgamma = gy_xhat_2d->reduce_sum(0, false)->reshape_view(param_shape);
    }
  } else {
    if (need_dx)
      dx = std::make_shared<NativeTensor>(shape, -1, ScalarType::Float32);
    if (need_dgamma) {
      dgamma =
          std::make_shared<NativeTensor>(param_shape, -1, ScalarType::Float32);
      dgamma->zero_();
    }

    const float *in_ptr = cpu_f32();
    const float *go_ptr = go_t->cpu_f32();
    const float *gam_ptr = gamma_t ? gamma_t->cpu_f32() : nullptr;
    float *dx_ptr = dx ? dx->cpu_f32() : nullptr;
    float *dgam_ptr = dgamma ? dgamma->cpu_f32() : nullptr;

    const float inv_cols = 1.0f / static_cast<float>(cols);
    for (uint32_t r = 0; r < rows; ++r) {
      float sq_sum = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float v = in_ptr[r * cols + c];
        sq_sum += v * v;
      }
      float inv_rms = 1.0f / std::sqrt(sq_sum * inv_cols + eps);

      float sum_gy_xhat = 0.0f;
      for (uint32_t c = 0; c < cols; ++c) {
        float xhat = in_ptr[r * cols + c] * inv_rms;
        float g = go_ptr[r * cols + c];
        float w = gam_ptr ? gam_ptr[c] : 1.0f;
        float gy = g * w;
        sum_gy_xhat += gy * xhat;
        if (dgam_ptr)
          dgam_ptr[c] += g * xhat;
      }
      if (dx_ptr) {
        for (uint32_t c = 0; c < cols; ++c) {
          float xhat = in_ptr[r * cols + c] * inv_rms;
          float w = gam_ptr ? gam_ptr[c] : 1.0f;
          float gy = go_ptr[r * cols + c] * w;
          dx_ptr[r * cols + c] = (gy - xhat * sum_gy_xhat * inv_cols) * inv_rms;
        }
      }
    }
  }
  return {dx, dgamma};
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>>
NativeTensor::batchnorm(std::shared_ptr<NativeTensor> running_mean,
                        std::shared_ptr<NativeTensor> running_var,
                        std::shared_ptr<NativeTensor> gamma,
                        std::shared_ptr<NativeTensor> beta, bool training,
                        float momentum, float eps) {
  if (shape.size() < 2) {
    throw std::runtime_error(
        "batchnorm requires at least 2D input [N, C, ...]");
  }
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  const uint32_t N = static_cast<uint32_t>(x_c->shape[0]);
  const uint32_t C = static_cast<uint32_t>(x_c->shape[1]);
  uint32_t L = 1;
  for (size_t d = 2; d < x_c->shape.size(); ++d) {
    L *= static_cast<uint32_t>(x_c->shape[d]);
  }
  const uint32_t NL = N * L;

  auto g_t = gamma ? gamma->force_contiguous() : nullptr;
  auto b_t = beta ? beta->force_contiguous() : nullptr;
  auto y = std::make_shared<NativeTensor>(x_c->shape, device_id,
                                          ScalarType::Float32);
  auto save_mean = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(C)}, device_id,
      ScalarType::Float32);
  auto save_invstd = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(C)}, device_id,
      ScalarType::Float32);

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    float *yp = y->cpu_f32();
    float *smp = save_mean->cpu_f32();
    float *sip = save_invstd->cpu_f32();
    float *rmp = running_mean ? running_mean->cpu_f32() : nullptr;
    float *rvp = running_var ? running_var->cpu_f32() : nullptr;
    const float *gp = g_t ? g_t->cpu_f32() : nullptr;
    const float *bp = b_t ? b_t->cpu_f32() : nullptr;

    for (uint32_t c = 0; c < C; ++c) {
      float m = 0.0f, invstd = 1.0f;
      if (training) {
        double s = 0.0, s2 = 0.0;
        for (uint32_t n = 0; n < N; ++n) {
          for (uint32_t l = 0; l < L; ++l) {
            double v = xp[(n * C + c) * L + l];
            s += v;
            s2 += v * v;
          }
        }
        m = static_cast<float>(s / NL);
        double var = std::max(0.0, (s2 / NL) - static_cast<double>(m) * m);
        invstd = static_cast<float>(1.0 / std::sqrt(var + eps));
        if (rmp && rvp) {
          rmp[c] = (1.0f - momentum) * rmp[c] + momentum * m;
          double unbiased_var = (NL > 1) ? (var * NL / (NL - 1)) : var;
          rvp[c] = (1.0f - momentum) * rvp[c] +
                   momentum * static_cast<float>(unbiased_var);
        }
      } else {
        m = rmp ? rmp[c] : 0.0f;
        float v = rvp ? rvp[c] : 1.0f;
        invstd = 1.0f / std::sqrt(std::max(0.0f, v) + eps);
      }
      smp[c] = m;
      sip[c] = invstd;
      const float gam = gp ? gp[c] : 1.0f;
      const float bet = bp ? bp[c] : 0.0f;
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t l = 0; l < L; ++l) {
          const size_t idx = (n * C + c) * L + l;
          yp[idx] = ((xp[idx] - m) * invstd) * gam + bet;
        }
      }
    }
    return {(dtype == ScalarType::Float32) ? y : y->cast(dtype), save_mean,
            save_invstd};
  }

  struct {
    uint32_t N, C, L;
    float eps, momentum;
    uint32_t training, has_affine, _pad;
  } pc{N, C, L, eps, momentum, training ? 1u : 0u, (g_t && b_t) ? 1u : 0u, 0u};

  if (training) {
    VulkanPipelineManager::instance().dispatch(
        device_id, "batchnorm_stats",
        {x_c->gpu_buf(), x_c->gpu_buf(), x_c->gpu_buf(), x_c->gpu_buf(),
         save_mean->gpu_buf(), save_invstd->gpu_buf()},
        &pc, sizeof(pc), C);

    if (running_mean && running_var) {
      // Update running_mean = (1 - momentum) * running_mean + momentum *
      // save_mean
      auto rm_new =
          running_mean->binary_op_scalar(2, 1.0f - momentum)
              ->binary_op(0, save_mean->binary_op_scalar(2, momentum));
      running_mean->copy_from_native(rm_new);

      // running_var uses Bessel-corrected variance: var = (1 / invstd^2) - eps
      auto inv2 = save_invstd->binary_op(2, save_invstd);
      auto var_b =
          inv2->binary_op_scalar(3, 1.0f, true)->binary_op_scalar(1, eps);
      float bessel = (NL > 1)
                         ? (static_cast<float>(NL) / static_cast<float>(NL - 1))
                         : 1.0f;
      auto var_unbiased = var_b->binary_op_scalar(2, bessel);
      auto rv_new =
          running_var->binary_op_scalar(2, 1.0f - momentum)
              ->binary_op(0, var_unbiased->binary_op_scalar(2, momentum));
      running_var->copy_from_native(rv_new);
    }
  } else {
    if (running_mean)
      save_mean->copy_from_native(running_mean);
    else
      save_mean->zero_();
    if (running_var)
      save_invstd->copy_from_native(running_var);
    else
      save_invstd->fill_(1.0f);
  }

  auto gam_buf = g_t ? g_t->gpu_buf() : save_mean->gpu_buf();
  auto bet_buf = b_t ? b_t->gpu_buf() : save_mean->gpu_buf();
  uint32_t gx = (static_cast<uint32_t>(y->numel) + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "batchnorm_forward",
      {x_c->gpu_buf(), save_mean->gpu_buf(), save_invstd->gpu_buf(), gam_buf,
       bet_buf, y->gpu_buf()},
      &pc, sizeof(pc), gx);

  return {(dtype == ScalarType::Float32) ? y : y->cast(dtype), save_mean,
          save_invstd};
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>>
NativeTensor::batchnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                                 std::shared_ptr<NativeTensor> save_mean,
                                 std::shared_ptr<NativeTensor> save_invstd,
                                 std::shared_ptr<NativeTensor> gamma, float eps,
                                 bool need_dx, bool need_dgamma,
                                 bool need_dbeta) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  const uint32_t N = static_cast<uint32_t>(x_c->shape[0]);
  const uint32_t C = static_cast<uint32_t>(x_c->shape[1]);
  uint32_t L = 1;
  for (size_t d = 2; d < x_c->shape.size(); ++d) {
    L *= static_cast<uint32_t>(x_c->shape[d]);
  }

  auto dx = need_dx ? std::make_shared<NativeTensor>(x_c->shape, device_id,
                                                     ScalarType::Float32)
                    : nullptr;
  auto dgamma = need_dgamma ? std::make_shared<NativeTensor>(
                                  std::vector<int64_t>{static_cast<int64_t>(C)},
                                  device_id, ScalarType::Float32)
                            : nullptr;
  auto dbeta = need_dbeta ? std::make_shared<NativeTensor>(
                                std::vector<int64_t>{static_cast<int64_t>(C)},
                                device_id, ScalarType::Float32)
                          : nullptr;

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const float *dyp = g_c->cpu_f32();
    const float *smp = save_mean->cpu_f32();
    const float *sip = save_invstd->cpu_f32();
    const float *gp = gamma ? gamma->cpu_f32() : nullptr;
    float *dxp = dx ? dx->cpu_f32() : nullptr;
    float *dgp = dgamma ? dgamma->cpu_f32() : nullptr;
    float *dbp = dbeta ? dbeta->cpu_f32() : nullptr;
    const float NL = static_cast<float>(N * L);

    for (uint32_t c = 0; c < C; ++c) {
      const float m = smp[c];
      const float invstd = sip[c];
      double s_dy = 0.0, s_dy_xhat = 0.0;
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t l = 0; l < L; ++l) {
          const size_t idx = (n * C + c) * L + l;
          const float dy = dyp[idx];
          const float xhat = (xp[idx] - m) * invstd;
          s_dy += dy;
          s_dy_xhat += dy * xhat;
        }
      }
      if (dbp)
        dbp[c] = static_cast<float>(s_dy);
      if (dgp)
        dgp[c] = static_cast<float>(s_dy_xhat);
      if (dxp) {
        const float gam = gp ? gp[c] : 1.0f;
        for (uint32_t n = 0; n < N; ++n) {
          for (uint32_t l = 0; l < L; ++l) {
            const size_t idx = (n * C + c) * L + l;
            const float xhat = (xp[idx] - m) * invstd;
            dxp[idx] = (gam * invstd / NL) *
                       (NL * dyp[idx] - static_cast<float>(s_dy) -
                        xhat * static_cast<float>(s_dy_xhat));
          }
        }
      }
    }
    return {dx, dgamma, dbeta};
  }

  struct {
    uint32_t N, C, L;
    float eps, momentum;
    uint32_t training, has_affine, _pad;
  } pc{N, C, L, eps, 0.1f, 1u, gamma ? 1u : 0u, 0u};

  auto sum_dy = dbeta ? dbeta
                      : std::make_shared<NativeTensor>(
                            std::vector<int64_t>{static_cast<int64_t>(C)},
                            device_id, ScalarType::Float32);
  auto sum_dy_xhat = dgamma ? dgamma
                            : std::make_shared<NativeTensor>(
                                  std::vector<int64_t>{static_cast<int64_t>(C)},
                                  device_id, ScalarType::Float32);

  VulkanPipelineManager::instance().dispatch(
      device_id, "batchnorm_backward_reduce",
      {g_c->gpu_buf(), x_c->gpu_buf(), save_mean->gpu_buf(),
       save_invstd->gpu_buf(), sum_dy->gpu_buf(), sum_dy_xhat->gpu_buf()},
      &pc, sizeof(pc), C);

  if (need_dx) {
    auto stats_packed =
        cat_dim0({save_mean, save_invstd, sum_dy, sum_dy_xhat}, device_id);
    auto gam_buf =
        gamma ? gamma->force_contiguous()->gpu_buf() : save_mean->gpu_buf();
    uint32_t gx = (static_cast<uint32_t>(dx->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "batchnorm_backward_dx",
        {g_c->gpu_buf(), x_c->gpu_buf(), stats_packed->gpu_buf(), gam_buf,
         dx->gpu_buf(), dx->gpu_buf()},
        &pc, sizeof(pc), gx);
  }

  return {dx, dgamma, dbeta};
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::groupnorm(uint32_t num_groups,
                        std::shared_ptr<NativeTensor> gamma,
                        std::shared_ptr<NativeTensor> beta, float eps) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  const uint32_t N = static_cast<uint32_t>(x_c->shape[0]);
  const uint32_t C = static_cast<uint32_t>(x_c->shape[1]);
  if (C % num_groups != 0) {
    throw std::runtime_error(
        "GroupNorm: num_channels must be divisible by num_groups");
  }
  uint32_t L = 1;
  for (size_t d = 2; d < x_c->shape.size(); ++d) {
    L *= static_cast<uint32_t>(x_c->shape[d]);
  }

  auto y = std::make_shared<NativeTensor>(x_c->shape, device_id,
                                          ScalarType::Float32);
  auto mean_invstd = std::make_shared<NativeTensor>(
      std::vector<int64_t>{2 * static_cast<int64_t>(N * num_groups)}, device_id,
      ScalarType::Float32);

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    float *yp = y->cpu_f32();
    float *mip = mean_invstd->cpu_f32();
    const float *gp = gamma ? gamma->cpu_f32() : nullptr;
    const float *bp = beta ? beta->cpu_f32() : nullptr;
    const uint32_t c_per_g = C / num_groups;
    const uint32_t group_elems = c_per_g * L;

    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t g = 0; g < num_groups; ++g) {
        const uint32_t ng = n * num_groups + g;
        const size_t base = (n * C + g * c_per_g) * L;
        double s = 0.0, s2 = 0.0;
        for (uint32_t i = 0; i < group_elems; ++i) {
          double v = xp[base + i];
          s += v;
          s2 += v * v;
        }
        const float mean = static_cast<float>(s / group_elems);
        const double var = std::max(0.0, (s2 / group_elems) -
                                             static_cast<double>(mean) * mean);
        const float invstd = static_cast<float>(1.0 / std::sqrt(var + eps));
        mip[ng] = mean;
        mip[N * num_groups + ng] = invstd;
        for (uint32_t i = 0; i < group_elems; ++i) {
          const uint32_t c = g * c_per_g + (i / L);
          const float xhat = (xp[base + i] - mean) * invstd;
          yp[base + i] = (gp && bp) ? (xhat * gp[c] + bp[c]) : xhat;
        }
      }
    }
    return {(dtype == ScalarType::Float32) ? y : y->cast(dtype), mean_invstd};
  }

  struct {
    uint32_t N, C, G, L;
    float eps;
    uint32_t has_affine;
  } pc{N, C, num_groups, L, eps, (gamma && beta) ? 1u : 0u};

  auto gam_buf = gamma ? gamma->force_contiguous()->gpu_buf() : y->gpu_buf();
  auto bet_buf = beta ? beta->force_contiguous()->gpu_buf() : y->gpu_buf();
  VulkanPipelineManager::instance().dispatch(device_id, "groupnorm_forward",
                                             {x_c->gpu_buf(), gam_buf, bet_buf,
                                              gam_buf, y->gpu_buf(),
                                              mean_invstd->gpu_buf()},
                                             &pc, sizeof(pc), N * num_groups);

  return {(dtype == ScalarType::Float32) ? y : y->cast(dtype), mean_invstd};
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>>
NativeTensor::groupnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                                 std::shared_ptr<NativeTensor> mean_invstd,
                                 uint32_t num_groups,
                                 std::shared_ptr<NativeTensor> gamma,
                                 bool need_dx, bool need_dgamma,
                                 bool need_dbeta) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  const uint32_t N = static_cast<uint32_t>(x_c->shape[0]);
  const uint32_t C = static_cast<uint32_t>(x_c->shape[1]);
  uint32_t L = 1;
  for (size_t d = 2; d < x_c->shape.size(); ++d) {
    L *= static_cast<uint32_t>(x_c->shape[d]);
  }

  auto dx = need_dx ? std::make_shared<NativeTensor>(x_c->shape, device_id,
                                                     ScalarType::Float32)
                    : nullptr;
  auto dgamma = need_dgamma ? std::make_shared<NativeTensor>(
                                  std::vector<int64_t>{static_cast<int64_t>(C)},
                                  device_id, ScalarType::Float32)
                            : nullptr;
  auto dbeta = need_dbeta ? std::make_shared<NativeTensor>(
                                std::vector<int64_t>{static_cast<int64_t>(C)},
                                device_id, ScalarType::Float32)
                          : nullptr;

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const float *dyp = g_c->cpu_f32();
    const float *mip = mean_invstd->cpu_f32();
    const float *gp = gamma ? gamma->cpu_f32() : nullptr;
    float *dxp = dx ? dx->cpu_f32() : nullptr;
    float *dgp = dgamma ? dgamma->cpu_f32() : nullptr;
    float *dbp = dbeta ? dbeta->cpu_f32() : nullptr;
    const uint32_t c_per_g = C / num_groups;
    const uint32_t group_elems = c_per_g * L;

    if (dgp)
      std::fill_n(dgp, C, 0.0f);
    if (dbp)
      std::fill_n(dbp, C, 0.0f);

    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t g = 0; g < num_groups; ++g) {
        const uint32_t ng = n * num_groups + g;
        const float mean = mip[ng];
        const float invstd = mip[N * num_groups + ng];
        const size_t base = (n * C + g * c_per_g) * L;
        double s_dyg = 0.0, s_dyg_xhat = 0.0;
        for (uint32_t i = 0; i < group_elems; ++i) {
          const uint32_t c = g * c_per_g + (i / L);
          const float gam = gp ? gp[c] : 1.0f;
          const float dy = dyp[base + i];
          const float xhat = (xp[base + i] - mean) * invstd;
          if (dgp)
            dgp[c] += dy * xhat;
          if (dbp)
            dbp[c] += dy;
          s_dyg += dy * gam;
          s_dyg_xhat += dy * gam * xhat;
        }
        if (dxp) {
          const float M = static_cast<float>(group_elems);
          for (uint32_t i = 0; i < group_elems; ++i) {
            const uint32_t c = g * c_per_g + (i / L);
            const float gam = gp ? gp[c] : 1.0f;
            const float xhat = (xp[base + i] - mean) * invstd;
            dxp[base + i] =
                invstd * (dyp[base + i] * gam - static_cast<float>(s_dyg) / M -
                          xhat * static_cast<float>(s_dyg_xhat) / M);
          }
        }
      }
    }
    return {dx, dgamma, dbeta};
  }

  struct {
    uint32_t N, C, G, L;
    float eps;
    uint32_t has_affine;
  } pc{N, C, num_groups, L, 1e-5f, gamma ? 1u : 0u};

  if (need_dx) {
    auto gam_buf =
        gamma ? gamma->force_contiguous()->gpu_buf() : x_c->gpu_buf();
    VulkanPipelineManager::instance().dispatch(
        device_id, "groupnorm_backward_dx",
        {g_c->gpu_buf(), x_c->gpu_buf(), gam_buf, mean_invstd->gpu_buf(),
         dx->gpu_buf(), dx->gpu_buf()},
        &pc, sizeof(pc), N * num_groups);
  }

  if (need_dgamma || need_dbeta) {
    auto dg_buf = dgamma ? dgamma->gpu_buf() : g_c->gpu_buf();
    auto db_buf = dbeta ? dbeta->gpu_buf() : g_c->gpu_buf();
    VulkanPipelineManager::instance().dispatch(
        device_id, "groupnorm_backward_gamma_beta",
        {g_c->gpu_buf(), x_c->gpu_buf(), mean_invstd->gpu_buf(), x_c->gpu_buf(),
         dg_buf, db_buf},
        &pc, sizeof(pc), C);
  }

  return {dx, dgamma, dbeta};
}


} // namespace vultorch
