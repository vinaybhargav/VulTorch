#include "tensor/tensor_internal.hpp"

namespace vultorch {

// ============================================================================
// Snake Family Periodic Activations & Mamba Selective State Space Scan (S6)
// ============================================================================

namespace {

inline float snake_safe_inv_cpu(float b, float eps) {
  float s = (b >= 0.0f) ? 1.0f : -1.0f;
  return (std::abs(b) < eps) ? (s / eps) : (1.0f / b);
}

inline void eval_periodic_basis_cpu(uint32_t mode, float ax, float g_phase,
                                    float &P, float &dP_du) {
  if (mode == 0u || mode == 1u) {
    const float u = ax + ((mode == 1u) ? g_phase : 0.0f);
    const float s = std::sin(u);
    P = s * s;
    dP_du = std::sin(2.0f * u);
  } else if (mode == 2u) {
    const float c = std::cos(ax);
    P = c * c;
    dP_du = -std::sin(2.0f * ax);
  } else if (mode == 3u) {
    const float u = ax;
    const float s = std::sin(u);
    const float s2 = s * s;
    const float denom = 1.0f + u * u;
    const float inv_d = 1.0f / denom;
    P = s2 * inv_d;
    dP_du = (std::sin(2.0f * u) * denom - 2.0f * u * s2) * (inv_d * inv_d);
  } else {
    const float u = ax;
    const float u2 = u * u;
    if (u2 < 1e-4f) {
      P = 1.0f - u2 / 3.0f + (2.0f * u2 * u2) / 45.0f;
      dP_du = (-2.0f / 3.0f) * u + (8.0f / 45.0f) * (u * u2);
    } else {
      const float s = std::sin(u);
      const float c = std::cos(u);
      const float inv_u = 1.0f / u;
      const float sinc_u = s * inv_u;
      P = sinc_u * sinc_u;
      dP_du = 2.0f * sinc_u * (u * c - s) * (inv_u * inv_u);
    }
  }
}

} // namespace

std::shared_ptr<NativeTensor>
NativeTensor::snake(std::shared_ptr<NativeTensor> alpha,
                    std::shared_ptr<NativeTensor> beta, int64_t channel_dim,
                    bool log_scale, float scalar_a, float eps, uint32_t mode,
                    std::shared_ptr<NativeTensor> gamma, float scalar_gamma) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  const int64_t ndim = static_cast<int64_t>(x_c->shape.size());
  const bool is_scalar = (alpha == nullptr);

  uint32_t C = 1, inner = 1;
  if (is_scalar || ndim == 0) {
    C = 1;
    inner = static_cast<uint32_t>(std::max<size_t>(1, x_c->numel));
  } else {
    int64_t cd = normalize_dim(channel_dim, ndim);
    C = static_cast<uint32_t>(x_c->shape[cd]);
    for (int64_t d = cd + 1; d < ndim; ++d)
      inner *= static_cast<uint32_t>(x_c->shape[d]);
    if (alpha->numel != C || (beta && beta->numel != C) ||
        (gamma && gamma->numel != C)) {
      throw std::runtime_error(
          "Snake: alpha/beta/gamma size must match channel dimension");
    }
  }

  auto a_c = alpha
                 ? ((alpha->dtype == ScalarType::Float32)
                        ? alpha->force_contiguous()
                        : alpha->cast(ScalarType::Float32)->force_contiguous())
                 : nullptr;
  auto b_c = beta ? ((beta->dtype == ScalarType::Float32)
                         ? beta->force_contiguous()
                         : beta->cast(ScalarType::Float32)->force_contiguous())
                  : nullptr;
  auto g_c = gamma
                 ? ((gamma->dtype == ScalarType::Float32)
                        ? gamma->force_contiguous()
                        : gamma->cast(ScalarType::Float32)->force_contiguous())
                 : nullptr;
  auto y = std::make_shared<NativeTensor>(x_c->shape, device_id,
                                          ScalarType::Float32);
  if (y->numel == 0)
    return (dtype == ScalarType::Float32) ? y : y->cast(dtype);

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const float *ap = a_c ? a_c->cpu_f32() : nullptr;
    const float *bp = b_c ? b_c->cpu_f32() : nullptr;
    const float *gp = g_c ? g_c->cpu_f32() : nullptr;
    float *yp = y->cpu_f32();
    for (size_t idx = 0; idx < x_c->numel; ++idx) {
      const uint32_t c = static_cast<uint32_t>((idx / inner) % C);
      float a = scalar_a;
      float b = scalar_a;
      float g_phase = scalar_gamma;
      if (!is_scalar) {
        a = log_scale ? std::exp(ap[c]) : ap[c];
        b = bp ? (log_scale ? std::exp(bp[c]) : bp[c]) : a;
        if (gp)
          g_phase = gp[c];
      }
      const float x = xp[idx];
      float P = 0.0f, dP_du = 0.0f;
      eval_periodic_basis_cpu(mode, a * x, g_phase, P, dP_du);
      const float inv_b = snake_safe_inv_cpu(b, eps);
      yp[idx] = x + P * inv_b;
    }
    return (dtype == ScalarType::Float32) ? y : y->cast(dtype);
  }

  // If both beta and gamma are present, pack [beta, gamma] into a [2*C] buffer
  // for snIn2
  std::shared_ptr<NativeTensor> bg_pack = b_c;
  if (b_c && g_c) {
    bg_pack = cat({b_c, g_c}, 0, device_id);
  } else if (g_c) {
    bg_pack = g_c;
  }

  struct {
    uint32_t total_elements;
    uint32_t channels;
    uint32_t inner_size;
    uint32_t has_beta;
    uint32_t log_scale;
    uint32_t is_scalar_a;
    uint32_t mode;
    uint32_t has_gamma;
    float scalar_a;
    float scalar_gamma;
    float eps;
  } pc{static_cast<uint32_t>(x_c->numel),
       C,
       inner,
       (b_c != nullptr) ? 1u : 0u,
       log_scale ? 1u : 0u,
       is_scalar ? 1u : 0u,
       mode,
       (g_c != nullptr) ? 1u : 0u,
       scalar_a,
       scalar_gamma,
       eps};

  auto abuf = a_c ? a_c->gpu_buf() : x_c->gpu_buf();
  auto bbuf = bg_pack ? bg_pack->gpu_buf() : abuf;
  uint32_t gx = (pc.total_elements + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "snake_forward",
      {x_c->gpu_buf(), abuf, bbuf, x_c->gpu_buf(), y->gpu_buf(), y->gpu_buf()},
      &pc, sizeof(pc), gx);

  return (dtype == ScalarType::Float32) ? y : y->cast(dtype);
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::snake_backward(std::shared_ptr<NativeTensor> grad_out,
                             std::shared_ptr<NativeTensor> alpha,
                             std::shared_ptr<NativeTensor> beta,
                             int64_t channel_dim, bool log_scale,
                             float scalar_a, float eps, bool need_dx,
                             bool need_dalpha, bool need_dbeta, uint32_t mode,
                             std::shared_ptr<NativeTensor> gamma,
                             float scalar_gamma, bool need_dgamma) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  const int64_t ndim = static_cast<int64_t>(x_c->shape.size());
  const bool is_scalar = (alpha == nullptr);

  uint32_t C = 1, inner = 1;
  if (is_scalar || ndim == 0) {
    C = 1;
    inner = static_cast<uint32_t>(std::max<size_t>(1, x_c->numel));
  } else {
    int64_t cd = normalize_dim(channel_dim, ndim);
    C = static_cast<uint32_t>(x_c->shape[cd]);
    for (int64_t d = cd + 1; d < ndim; ++d)
      inner *= static_cast<uint32_t>(x_c->shape[d]);
  }

  auto a_c = alpha
                 ? ((alpha->dtype == ScalarType::Float32)
                        ? alpha->force_contiguous()
                        : alpha->cast(ScalarType::Float32)->force_contiguous())
                 : nullptr;
  auto b_c = beta ? ((beta->dtype == ScalarType::Float32)
                         ? beta->force_contiguous()
                         : beta->cast(ScalarType::Float32)->force_contiguous())
                  : nullptr;
  auto gam_c =
      gamma ? ((gamma->dtype == ScalarType::Float32)
                   ? gamma->force_contiguous()
                   : gamma->cast(ScalarType::Float32)->force_contiguous())
            : nullptr;

  auto dx = need_dx ? std::make_shared<NativeTensor>(x_c->shape, device_id,
                                                     ScalarType::Float32)
                    : nullptr;
  auto dalpha = (need_dalpha && a_c)
                    ? std::make_shared<NativeTensor>(a_c->shape, device_id,
                                                     ScalarType::Float32)
                    : nullptr;
  auto dbeta = (need_dbeta && b_c)
                   ? std::make_shared<NativeTensor>(b_c->shape, device_id,
                                                    ScalarType::Float32)
                   : nullptr;
  auto dgamma = (need_dgamma && gam_c)
                    ? std::make_shared<NativeTensor>(gam_c->shape, device_id,
                                                     ScalarType::Float32)
                    : nullptr;

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const float *dyp = g_c->cpu_f32();
    const float *ap = a_c ? a_c->cpu_f32() : nullptr;
    const float *bp = b_c ? b_c->cpu_f32() : nullptr;
    const float *gp = gam_c ? gam_c->cpu_f32() : nullptr;
    float *dxp = dx ? dx->cpu_f32() : nullptr;
    float *dap = dalpha ? dalpha->cpu_f32() : nullptr;
    float *dbp = dbeta ? dbeta->cpu_f32() : nullptr;
    float *dgp = dgamma ? dgamma->cpu_f32() : nullptr;
    if (dap)
      std::fill_n(dap, C, 0.0f);
    if (dbp)
      std::fill_n(dbp, C, 0.0f);
    if (dgp)
      std::fill_n(dgp, C, 0.0f);

    for (size_t idx = 0; idx < x_c->numel; ++idx) {
      const uint32_t c = static_cast<uint32_t>((idx / inner) % C);
      float a = scalar_a;
      float b = scalar_a;
      float g_phase = scalar_gamma;
      if (!is_scalar) {
        a = log_scale ? std::exp(ap[c]) : ap[c];
        b = bp ? (log_scale ? std::exp(bp[c]) : bp[c]) : a;
        if (gp)
          g_phase = gp[c];
      }
      const float x = xp[idx];
      const float dy = dyp[idx];
      const float inv_b = snake_safe_inv_cpu(b, eps);
      float P = 0.0f, dP_du = 0.0f;
      eval_periodic_basis_cpu(mode, a * x, g_phase, P, dP_du);

      if (dxp) {
        dxp[idx] = dy * (1.0f + a * inv_b * dP_du);
      }
      if (dap || dbp || dgp) {
        float da_elem = 0.0f;
        float db_elem = 0.0f;
        if (bp) {
          da_elem = dy * (inv_b * x * dP_du);
          db_elem = dy * (-P * inv_b * inv_b);
        } else {
          da_elem = dy * inv_b * (x * dP_du - P * inv_b);
        }
        if (log_scale) {
          da_elem *= a;
          db_elem *= b;
        }
        if (dap)
          dap[c] += da_elem;
        if (dbp)
          dbp[c] += db_elem;
        if (dgp && mode == 1u) {
          dgp[c] += dy * (inv_b * dP_du);
        }
      }
    }
    return {dx, dalpha, dbeta, dgamma};
  }

  std::shared_ptr<NativeTensor> bg_pack = b_c;
  if (b_c && gam_c) {
    bg_pack = cat({b_c, gam_c}, 0, device_id);
  } else if (gam_c) {
    bg_pack = gam_c;
  }

  struct {
    uint32_t total_elements;
    uint32_t channels;
    uint32_t inner_size;
    uint32_t has_beta;
    uint32_t log_scale;
    uint32_t is_scalar_a;
    uint32_t mode;
    uint32_t has_gamma;
    float scalar_a;
    float scalar_gamma;
    float eps;
  } pc{static_cast<uint32_t>(x_c->numel),
       C,
       inner,
       (b_c != nullptr) ? 1u : 0u,
       log_scale ? 1u : 0u,
       is_scalar ? 1u : 0u,
       mode,
       (gam_c != nullptr) ? 1u : 0u,
       scalar_a,
       scalar_gamma,
       eps};

  auto abuf = a_c ? a_c->gpu_buf() : x_c->gpu_buf();
  auto bbuf = bg_pack ? bg_pack->gpu_buf() : abuf;

  if (need_dx && pc.total_elements > 0) {
    uint32_t gx = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(device_id, "snake_backward_dx",
                                               {g_c->gpu_buf(), abuf, bbuf,
                                                x_c->gpu_buf(), dx->gpu_buf(),
                                                dx->gpu_buf()},
                                               &pc, sizeof(pc), gx);
  }

  if ((dalpha || dbeta || dgamma) && C > 0) {
    auto da_buf = dalpha ? dalpha->gpu_buf() : g_c->gpu_buf();
    std::shared_ptr<NativeTensor> dbg_out = dbeta;
    if (b_c && gam_c) {
      dbg_out = std::make_shared<NativeTensor>(
          std::vector<int64_t>{static_cast<int64_t>(2 * C)}, device_id,
          ScalarType::Float32);
    } else if (dgamma) {
      dbg_out = dgamma;
    }
    auto db_buf = dbg_out ? dbg_out->gpu_buf() : da_buf;
    VulkanPipelineManager::instance().dispatch(
        device_id, "snake_backward_params",
        {g_c->gpu_buf(), abuf, bbuf, x_c->gpu_buf(), da_buf, db_buf}, &pc,
        sizeof(pc), C);
    if (b_c && gam_c && dbg_out) {
      if (dbeta)
        dbeta = dbg_out->slice_dim0(0, C);
      if (dgamma)
        dgamma = dbg_out->slice_dim0(C, 2 * C);
    }
  }

  return {dx, dalpha, dbeta, dgamma};
}

// ============================================================================
// Mamba Selective State Space Model (S6) Parallel Scan
// ============================================================================

std::shared_ptr<NativeTensor> NativeTensor::selective_scan(
    std::shared_ptr<NativeTensor> delta, std::shared_ptr<NativeTensor> A,
    std::shared_ptr<NativeTensor> B_mat, std::shared_ptr<NativeTensor> C_mat,
    std::shared_ptr<NativeTensor> D_vec) {
  auto u_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto dt_c = (delta->dtype == ScalarType::Float32)
                  ? delta->force_contiguous()
                  : delta->cast(ScalarType::Float32)->force_contiguous();
  auto A_c = (A->dtype == ScalarType::Float32)
                 ? A->force_contiguous()
                 : A->cast(ScalarType::Float32)->force_contiguous();
  auto B_c = (B_mat->dtype == ScalarType::Float32)
                 ? B_mat->force_contiguous()
                 : B_mat->cast(ScalarType::Float32)->force_contiguous();
  auto C_c = (C_mat->dtype == ScalarType::Float32)
                 ? C_mat->force_contiguous()
                 : C_mat->cast(ScalarType::Float32)->force_contiguous();
  auto D_c = D_vec
                 ? ((D_vec->dtype == ScalarType::Float32)
                        ? D_vec->force_contiguous()
                        : D_vec->cast(ScalarType::Float32)->force_contiguous())
                 : nullptr;

  if (u_c->shape.size() != 3 || dt_c->shape != u_c->shape ||
      A_c->shape.size() != 2 || B_c->shape.size() != 3 ||
      C_c->shape != B_c->shape) {
    throw std::runtime_error("selective_scan: expected u, delta [B, D, L], A "
                             "[D, N], B, C [B, N, L]");
  }
  const uint32_t B_sz = static_cast<uint32_t>(u_c->shape[0]);
  const uint32_t D_sz = static_cast<uint32_t>(u_c->shape[1]);
  const uint32_t L_sz = static_cast<uint32_t>(u_c->shape[2]);
  const uint32_t N_sz = static_cast<uint32_t>(A_c->shape[1]);
  if (N_sz > 32) {
    throw std::runtime_error(
        "selective_scan: d_state (N) up to 32 supported in register scan");
  }

  auto y = std::make_shared<NativeTensor>(u_c->shape, device_id,
                                          ScalarType::Float32);
  if (y->numel == 0)
    return (dtype == ScalarType::Float32) ? y : y->cast(dtype);

  if (device_id == -1) {
    const float *up = u_c->cpu_f32();
    const float *dtp = dt_c->cpu_f32();
    const float *Ap = A_c->cpu_f32();
    const float *Bp = B_c->cpu_f32();
    const float *Cp = C_c->cpu_f32();
    const float *Dp = D_c ? D_c->cpu_f32() : nullptr;
    float *yp = y->cpu_f32();

    for (uint32_t b = 0; b < B_sz; ++b) {
      for (uint32_t d = 0; d < D_sz; ++d) {
        float h[32] = {0.0f};
        const float d_skip = Dp ? Dp[d] : 0.0f;
        const size_t u_base = (b * D_sz + d) * L_sz;
        const size_t bc_base = b * N_sz * L_sz;
        for (uint32_t t = 0; t < L_sz; ++t) {
          const float u_t = up[u_base + t];
          const float dt_t = dtp[u_base + t];
          float y_t = d_skip * u_t;
          for (uint32_t n = 0; n < N_sz; ++n) {
            const float a_bar = std::exp(dt_t * Ap[d * N_sz + n]);
            const float b_nt = Bp[bc_base + n * L_sz + t];
            const float c_nt = Cp[bc_base + n * L_sz + t];
            h[n] = a_bar * h[n] + dt_t * b_nt * u_t;
            y_t += c_nt * h[n];
          }
          yp[u_base + t] = y_t;
        }
      }
    }
    return (dtype == ScalarType::Float32) ? y : y->cast(dtype);
  }

  struct {
    uint32_t B, D, L, N, has_D, has_z;
  } pc{B_sz, D_sz, L_sz, N_sz, D_c ? 1u : 0u, 0u};

  auto dbuf = D_c ? D_c->gpu_buf() : u_c->gpu_buf();
  uint32_t gx = (B_sz * D_sz + 63u) / 64u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "ssm_scan_forward",
      {u_c->gpu_buf(), dt_c->gpu_buf(), A_c->gpu_buf(), B_c->gpu_buf(),
       C_c->gpu_buf(), dbuf, y->gpu_buf(), y->gpu_buf()},
      &pc, sizeof(pc), gx);

  return (dtype == ScalarType::Float32) ? y : y->cast(dtype);
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::selective_scan_backward(std::shared_ptr<NativeTensor> grad_out,
                                      std::shared_ptr<NativeTensor> delta,
                                      std::shared_ptr<NativeTensor> A,
                                      std::shared_ptr<NativeTensor> B_mat,
                                      std::shared_ptr<NativeTensor> C_mat,
                                      std::shared_ptr<NativeTensor> D_vec) {
  auto u_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  auto dt_c = (delta->dtype == ScalarType::Float32)
                  ? delta->force_contiguous()
                  : delta->cast(ScalarType::Float32)->force_contiguous();
  auto A_c = (A->dtype == ScalarType::Float32)
                 ? A->force_contiguous()
                 : A->cast(ScalarType::Float32)->force_contiguous();
  auto B_c = (B_mat->dtype == ScalarType::Float32)
                 ? B_mat->force_contiguous()
                 : B_mat->cast(ScalarType::Float32)->force_contiguous();
  auto C_c = (C_mat->dtype == ScalarType::Float32)
                 ? C_mat->force_contiguous()
                 : C_mat->cast(ScalarType::Float32)->force_contiguous();
  auto D_c = D_vec
                 ? ((D_vec->dtype == ScalarType::Float32)
                        ? D_vec->force_contiguous()
                        : D_vec->cast(ScalarType::Float32)->force_contiguous())
                 : nullptr;

  const uint32_t B_sz = static_cast<uint32_t>(u_c->shape[0]);
  const uint32_t D_sz = static_cast<uint32_t>(u_c->shape[1]);
  const uint32_t L_sz = static_cast<uint32_t>(u_c->shape[2]);
  const uint32_t N_sz = static_cast<uint32_t>(A_c->shape[1]);
  const uint32_t BD = B_sz * D_sz;

  auto du = std::make_shared<NativeTensor>(u_c->shape, device_id,
                                           ScalarType::Float32);

  if (device_id == -1) {
    auto ddelta =
        std::make_shared<NativeTensor>(dt_c->shape, -1, ScalarType::Float32);
    auto dA =
        std::make_shared<NativeTensor>(A_c->shape, -1, ScalarType::Float32);
    auto dB =
        std::make_shared<NativeTensor>(B_c->shape, -1, ScalarType::Float32);
    auto dC =
        std::make_shared<NativeTensor>(C_c->shape, -1, ScalarType::Float32);
    auto dD = D_c ? std::make_shared<NativeTensor>(D_c->shape, -1,
                                                   ScalarType::Float32)
                  : nullptr;

    const float *up = u_c->cpu_f32();
    const float *dyp = g_c->cpu_f32();
    const float *dtp = dt_c->cpu_f32();
    const float *Ap = A_c->cpu_f32();
    const float *Bp = B_c->cpu_f32();
    const float *Cp = C_c->cpu_f32();
    const float *Dp = D_c ? D_c->cpu_f32() : nullptr;

    float *dup = du->cpu_f32();
    float *ddtp = ddelta->cpu_f32();
    float *dAp = dA->cpu_f32();
    float *dBp = dB->cpu_f32();
    float *dCp = dC->cpu_f32();
    float *dDp = dD ? dD->cpu_f32() : nullptr;

    std::fill_n(dAp, A_c->numel, 0.0f);
    std::fill_n(dBp, B_c->numel, 0.0f);
    std::fill_n(dCp, C_c->numel, 0.0f);
    if (dDp)
      std::fill_n(dDp, D_c->numel, 0.0f);

    std::vector<float> h_hist(N_sz * L_sz, 0.0f);

    for (uint32_t b = 0; b < B_sz; ++b) {
      for (uint32_t d = 0; d < D_sz; ++d) {
        float h[32] = {0.0f};
        const size_t u_base = (b * D_sz + d) * L_sz;
        const size_t bc_base = b * N_sz * L_sz;
        for (uint32_t t = 0; t < L_sz; ++t) {
          const float u_t = up[u_base + t];
          const float dt_t = dtp[u_base + t];
          for (uint32_t n = 0; n < N_sz; ++n) {
            const float a_bar = std::exp(dt_t * Ap[d * N_sz + n]);
            const float b_nt = Bp[bc_base + n * L_sz + t];
            h[n] = a_bar * h[n] + dt_t * b_nt * u_t;
            h_hist[n * L_sz + t] = h[n];
          }
        }

        float dh[32] = {0.0f};
        const float d_skip = Dp ? Dp[d] : 0.0f;
        for (int ti = static_cast<int>(L_sz) - 1; ti >= 0; --ti) {
          const uint32_t t = static_cast<uint32_t>(ti);
          const float u_t = up[u_base + t];
          const float dt_t = dtp[u_base + t];
          const float dy_t = dyp[u_base + t];

          if (dDp)
            dDp[d] += dy_t * u_t;
          float du_t = dy_t * d_skip;
          float ddt_t = 0.0f;

          for (uint32_t n = 0; n < N_sz; ++n) {
            const float h_t = h_hist[n * L_sz + t];
            const float h_prev = (t > 0) ? h_hist[n * L_sz + (t - 1)] : 0.0f;
            const float b_nt = Bp[bc_base + n * L_sz + t];
            const float c_nt = Cp[bc_base + n * L_sz + t];
            const float a_val = Ap[d * N_sz + n];
            const float a_bar = std::exp(dt_t * a_val);

            dCp[bc_base + n * L_sz + t] += dy_t * h_t;
            const float dh_cur = dh[n] + dy_t * c_nt;
            const float da_bar = dh_cur * h_prev;

            du_t += dh_cur * dt_t * b_nt;
            ddt_t += dh_cur * b_nt * u_t + da_bar * a_bar * a_val;
            dAp[d * N_sz + n] += da_bar * a_bar * dt_t;
            dBp[bc_base + n * L_sz + t] += dh_cur * dt_t * u_t;
            dh[n] = dh_cur * a_bar;
          }
          dup[u_base + t] = du_t;
          ddtp[u_base + t] = ddt_t;
        }
      }
    }
    return {du, ddelta, dA, dB, dC, dD};
  }

  // GPU backward: allocate packed workspace ssmOut1 of size:
  // BD*L (dDelta) + BD*N (dA_bd) + BD*N*L (dB_bd) + BD*N*L (dC_bd) + BD (dD_bd)
  const int64_t sz_ddelta = static_cast<int64_t>(BD * L_sz);
  const int64_t sz_dA = static_cast<int64_t>(BD * N_sz);
  const int64_t sz_dBC = static_cast<int64_t>(BD * N_sz * L_sz);
  const int64_t sz_dD = static_cast<int64_t>(BD);
  const int64_t total_ws = sz_ddelta + sz_dA + 2 * sz_dBC + sz_dD;

  auto ws = std::make_shared<NativeTensor>(std::vector<int64_t>{total_ws},
                                           device_id, ScalarType::Float32);

  struct {
    uint32_t B, D, L, N, has_D, has_z;
  } pc{B_sz, D_sz, L_sz, N_sz, D_c ? 1u : 0u, 0u};

  uint32_t gx = (BD + 63u) / 64u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "ssm_scan_backward",
      {u_c->gpu_buf(), dt_c->gpu_buf(), A_c->gpu_buf(), B_c->gpu_buf(),
       C_c->gpu_buf(), g_c->gpu_buf(), du->gpu_buf(), ws->gpu_buf()},
      &pc, sizeof(pc), gx);

  if (D_c) {
    // du += dy * D[d]
    auto d_reshaped = D_c->reshape_view({1, static_cast<int64_t>(D_sz), 1});
    auto du_skip = g_c->binary_op(2, d_reshaped); // mul
    du = du->binary_op(0, du_skip);               // add
  }

  int64_t off = 0;
  auto ddelta = ws->slice_dim0(off, off + sz_ddelta)->reshape_view(dt_c->shape);
  off += sz_ddelta;
  auto dA_bd = ws->slice_dim0(off, off + sz_dA)
                   ->reshape_view({static_cast<int64_t>(B_sz),
                                   static_cast<int64_t>(D_sz),
                                   static_cast<int64_t>(N_sz)});
  auto dA = dA_bd->reduce_sum(0, false);
  off += sz_dA;
  auto dB_bd =
      ws->slice_dim0(off, off + sz_dBC)
          ->reshape_view(
              {static_cast<int64_t>(B_sz), static_cast<int64_t>(D_sz),
               static_cast<int64_t>(N_sz), static_cast<int64_t>(L_sz)});
  auto dB = dB_bd->reduce_sum(1, false);
  off += sz_dBC;
  auto dC_bd =
      ws->slice_dim0(off, off + sz_dBC)
          ->reshape_view(
              {static_cast<int64_t>(B_sz), static_cast<int64_t>(D_sz),
               static_cast<int64_t>(N_sz), static_cast<int64_t>(L_sz)});
  auto dC = dC_bd->reduce_sum(1, false);
  off += sz_dBC;
  std::shared_ptr<NativeTensor> dD = nullptr;
  if (D_c) {
    auto dD_bd = ws->slice_dim0(off, off + sz_dD)
                     ->reshape_view({static_cast<int64_t>(B_sz),
                                     static_cast<int64_t>(D_sz)});
    dD = dD_bd->reduce_sum(0, false);
  }

  return {du, ddelta, dA, dB, dC, dD};
}

} // namespace vultorch
