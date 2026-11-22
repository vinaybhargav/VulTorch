#include "tensor/tensor_internal.hpp"

namespace vultorch {

// ============================================================================
// Phase 4: FlashAttention-2, RoPE, SwiGLU/GeGLU, QuantizedLinear, AMP Check
// ============================================================================

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::flash_attention(std::shared_ptr<NativeTensor> K,
                              std::shared_ptr<NativeTensor> V, bool is_causal,
                              float scale) {
  if (shape.size() != 4 || K->shape.size() != 4 || V->shape.size() != 4) {
    throw std::runtime_error(
        "flash_attention: Q, K, V must be 4D [B, H, S, D]");
  }
  const ScalarType orig_dt = dtype;
  auto q_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto k_c = (K->dtype == ScalarType::Float32)
                 ? K->force_contiguous()
                 : K->cast(ScalarType::Float32)->force_contiguous();
  auto v_c = (V->dtype == ScalarType::Float32)
                 ? V->force_contiguous()
                 : V->cast(ScalarType::Float32)->force_contiguous();

  const uint32_t B_sz = static_cast<uint32_t>(q_c->shape[0]);
  const uint32_t H_q = static_cast<uint32_t>(q_c->shape[1]);
  const uint32_t S_q = static_cast<uint32_t>(q_c->shape[2]);
  const uint32_t D_sz = static_cast<uint32_t>(q_c->shape[3]);
  const uint32_t H_kv = static_cast<uint32_t>(k_c->shape[1]);
  const uint32_t S_k = static_cast<uint32_t>(k_c->shape[2]);

  if (k_c->shape[0] != q_c->shape[0] || v_c->shape != k_c->shape ||
      static_cast<uint32_t>(k_c->shape[3]) != D_sz) {
    throw std::runtime_error("flash_attention: mismatched Q, K, V shapes");
  }
  if (H_kv == 0 || (H_q % H_kv) != 0) {
    throw std::runtime_error(
        "flash_attention: H_q must be divisible by H_kv (GQA)");
  }
  if (D_sz > 128) {
    throw std::runtime_error("flash_attention: head_dim D <= 128 supported");
  }

  const float eff_scale =
      (scale > 0.0f) ? scale : (1.0f / std::sqrt(static_cast<float>(D_sz)));

  auto out = std::make_shared<NativeTensor>(q_c->shape, device_id,
                                            ScalarType::Float32);
  auto lse = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(B_sz),
                           static_cast<int64_t>(H_q),
                           static_cast<int64_t>(S_q)},
      device_id, ScalarType::Float32);

  if (out->numel == 0) {
    return {(orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt), lse};
  }

  if (device_id == -1) {
    const float *qp = q_c->cpu_f32();
    const float *kp = k_c->cpu_f32();
    const float *vp = v_c->cpu_f32();
    float *op = out->cpu_f32();
    float *lp = lse->cpu_f32();
    const uint32_t group_ratio = H_q / H_kv;
    const int causal_shift = static_cast<int>(S_k) - static_cast<int>(S_q);

    for (uint32_t bh_q = 0; bh_q < B_sz * H_q; ++bh_q) {
      const uint32_t b = bh_q / H_q;
      const uint32_t h_q = bh_q % H_q;
      const uint32_t h_kv = h_q / group_ratio;
      const uint32_t bh_kv = b * H_kv + h_kv;

      for (uint32_t i = 0; i < S_q; ++i) {
        const float *q_row = qp + (bh_q * S_q + i) * D_sz;
        float *o_row = op + (bh_q * S_q + i) * D_sz;
        std::fill_n(o_row, D_sz, 0.0f);
        float m_i = -1e30f;
        float l_i = 0.0f;

        for (uint32_t j = 0; j < S_k; ++j) {
          if (is_causal &&
              static_cast<int>(j) > static_cast<int>(i) + causal_shift) {
            continue;
          }
          const float *k_row = kp + (bh_kv * S_k + j) * D_sz;
          const float *v_row = vp + (bh_kv * S_k + j) * D_sz;
          float dot = 0.0f;
          for (uint32_t d = 0; d < D_sz; ++d)
            dot += q_row[d] * k_row[d];
          const float s_ij = dot * eff_scale;
          const float m_new = std::max(m_i, s_ij);
          const float alpha = std::exp(m_i - m_new);
          const float p_ij = std::exp(s_ij - m_new);
          l_i = l_i * alpha + p_ij;
          for (uint32_t d = 0; d < D_sz; ++d) {
            o_row[d] = o_row[d] * alpha + p_ij * v_row[d];
          }
          m_i = m_new;
        }
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (uint32_t d = 0; d < D_sz; ++d)
          o_row[d] *= inv_l;
        lp[bh_q * S_q + i] = (l_i > 0.0f) ? (m_i + std::log(l_i)) : -1e30f;
      }
    }
    return {(orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt), lse};
  }

  struct {
    uint32_t B, H_q, H_kv, S_q, S_k, D, is_causal;
    float scale;
  } pc{B_sz, H_q, H_kv, S_q, S_k, D_sz, is_causal ? 1u : 0u, eff_scale};

  const uint32_t gx = (S_q + 63u) / 64u;
  const uint32_t gy = B_sz * H_q;
  VulkanPipelineManager::instance().dispatch(
      device_id, "flash_attention_forward",
      {q_c->gpu_buf(), k_c->gpu_buf(), v_c->gpu_buf(), q_c->gpu_buf(),
       q_c->gpu_buf(), q_c->gpu_buf(), out->gpu_buf(), lse->gpu_buf(),
       lse->gpu_buf()},
      &pc, sizeof(pc), gx, gy, 1u);

  return {(orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt), lse};
}

std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
           std::shared_ptr<NativeTensor>>
NativeTensor::flash_attention_backward(std::shared_ptr<NativeTensor> grad_out,
                                       std::shared_ptr<NativeTensor> K,
                                       std::shared_ptr<NativeTensor> V,
                                       std::shared_ptr<NativeTensor> O,
                                       std::shared_ptr<NativeTensor> L,
                                       bool is_causal, float scale) {
  const ScalarType orig_dt = dtype;
  auto q_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto k_c = (K->dtype == ScalarType::Float32)
                 ? K->force_contiguous()
                 : K->cast(ScalarType::Float32)->force_contiguous();
  auto v_c = (V->dtype == ScalarType::Float32)
                 ? V->force_contiguous()
                 : V->cast(ScalarType::Float32)->force_contiguous();
  auto o_c = (O->dtype == ScalarType::Float32)
                 ? O->force_contiguous()
                 : O->cast(ScalarType::Float32)->force_contiguous();
  auto do_c = (grad_out->dtype == ScalarType::Float32)
                  ? grad_out->force_contiguous()
                  : grad_out->cast(ScalarType::Float32)->force_contiguous();
  auto l_c = (L->dtype == ScalarType::Float32)
                 ? L->force_contiguous()
                 : L->cast(ScalarType::Float32)->force_contiguous();

  const uint32_t B_sz = static_cast<uint32_t>(q_c->shape[0]);
  const uint32_t H_q = static_cast<uint32_t>(q_c->shape[1]);
  const uint32_t S_q = static_cast<uint32_t>(q_c->shape[2]);
  const uint32_t D_sz = static_cast<uint32_t>(q_c->shape[3]);
  const uint32_t H_kv = static_cast<uint32_t>(k_c->shape[1]);
  const uint32_t S_k = static_cast<uint32_t>(k_c->shape[2]);
  const float eff_scale =
      (scale > 0.0f) ? scale : (1.0f / std::sqrt(static_cast<float>(D_sz)));

  auto dQ = std::make_shared<NativeTensor>(q_c->shape, device_id,
                                           ScalarType::Float32);

  if (device_id == -1) {
    auto dK =
        std::make_shared<NativeTensor>(k_c->shape, -1, ScalarType::Float32);
    auto dV =
        std::make_shared<NativeTensor>(v_c->shape, -1, ScalarType::Float32);
    const float *qp = q_c->cpu_f32();
    const float *kp = k_c->cpu_f32();
    const float *vp = v_c->cpu_f32();
    const float *op = o_c->cpu_f32();
    const float *dop = do_c->cpu_f32();
    const float *lp = l_c->cpu_f32();
    float *dqp = dQ->cpu_f32();
    float *dkp = dK->cpu_f32();
    float *dvp = dV->cpu_f32();
    std::fill_n(dqp, dQ->numel, 0.0f);
    std::fill_n(dkp, dK->numel, 0.0f);
    std::fill_n(dvp, dV->numel, 0.0f);

    const uint32_t group_ratio = H_q / H_kv;
    const int causal_shift = static_cast<int>(S_k) - static_cast<int>(S_q);

    for (uint32_t bh_q = 0; bh_q < B_sz * H_q; ++bh_q) {
      const uint32_t b = bh_q / H_q;
      const uint32_t h_q = bh_q % H_q;
      const uint32_t h_kv = h_q / group_ratio;
      const uint32_t bh_kv = b * H_kv + h_kv;

      for (uint32_t i = 0; i < S_q; ++i) {
        const float *q_row = qp + (bh_q * S_q + i) * D_sz;
        const float *o_row = op + (bh_q * S_q + i) * D_sz;
        const float *do_row = dop + (bh_q * S_q + i) * D_sz;
        float *dq_row = dqp + (bh_q * S_q + i) * D_sz;
        const float L_i = lp[bh_q * S_q + i];
        float delta_i = 0.0f;
        for (uint32_t d = 0; d < D_sz; ++d)
          delta_i += do_row[d] * o_row[d];

        for (uint32_t j = 0; j < S_k; ++j) {
          if (is_causal &&
              static_cast<int>(j) > static_cast<int>(i) + causal_shift) {
            continue;
          }
          const float *k_row = kp + (bh_kv * S_k + j) * D_sz;
          const float *v_row = vp + (bh_kv * S_k + j) * D_sz;
          float *dk_row = dkp + (bh_kv * S_k + j) * D_sz;
          float *dv_row = dvp + (bh_kv * S_k + j) * D_sz;

          float dot_qk = 0.0f;
          float dp_ij = 0.0f;
          for (uint32_t d = 0; d < D_sz; ++d) {
            dot_qk += q_row[d] * k_row[d];
            dp_ij += do_row[d] * v_row[d];
          }
          const float s_ij = dot_qk * eff_scale;
          const float p_ij = std::exp(s_ij - L_i);
          const float ds_ij = p_ij * (dp_ij - delta_i) * eff_scale;
          for (uint32_t d = 0; d < D_sz; ++d) {
            dq_row[d] += ds_ij * k_row[d];
            dk_row[d] += ds_ij * q_row[d];
            dv_row[d] += p_ij * do_row[d];
          }
        }
      }
    }
    if (orig_dt != ScalarType::Float32) {
      return {dQ->cast(orig_dt), dK->cast(orig_dt), dV->cast(orig_dt)};
    }
    return {dQ, dK, dV};
  }

  struct {
    uint32_t B, H_q, H_kv, S_q, S_k, D, is_causal;
    float scale;
  } pc{B_sz, H_q, H_kv, S_q, S_k, D_sz, is_causal ? 1u : 0u, eff_scale};

  auto delta = std::make_shared<NativeTensor>(l_c->shape, device_id,
                                              ScalarType::Float32);
  const uint32_t prep_gx = (B_sz * H_q * S_q + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "flash_attention_backward_prep",
      {o_c->gpu_buf(), k_c->gpu_buf(), v_c->gpu_buf(), l_c->gpu_buf(),
       do_c->gpu_buf(), l_c->gpu_buf(), delta->gpu_buf(), delta->gpu_buf(),
       delta->gpu_buf()},
      &pc, sizeof(pc), prep_gx);

  const uint32_t dq_gx = (S_q + 63u) / 64u;
  const uint32_t gy = B_sz * H_q;
  VulkanPipelineManager::instance().dispatch(
      device_id, "flash_attention_backward_dq",
      {q_c->gpu_buf(), k_c->gpu_buf(), v_c->gpu_buf(), l_c->gpu_buf(),
       do_c->gpu_buf(), delta->gpu_buf(), dQ->gpu_buf(), dQ->gpu_buf(),
       dQ->gpu_buf()},
      &pc, sizeof(pc), dq_gx, gy, 1u);

  std::vector<int64_t> dkv_hq_shape = {
      static_cast<int64_t>(B_sz), static_cast<int64_t>(H_q),
      static_cast<int64_t>(S_k), static_cast<int64_t>(D_sz)};
  auto dK_hq = std::make_shared<NativeTensor>(dkv_hq_shape, device_id,
                                              ScalarType::Float32);
  auto dV_hq = std::make_shared<NativeTensor>(dkv_hq_shape, device_id,
                                              ScalarType::Float32);

  const uint32_t dkv_gx = (S_k + 63u) / 64u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "flash_attention_backward_dkv",
      {q_c->gpu_buf(), k_c->gpu_buf(), v_c->gpu_buf(), l_c->gpu_buf(),
       do_c->gpu_buf(), delta->gpu_buf(), dQ->gpu_buf(), dK_hq->gpu_buf(),
       dV_hq->gpu_buf()},
      &pc, sizeof(pc), dkv_gx, gy, 1u);

  std::shared_ptr<NativeTensor> dK = dK_hq;
  std::shared_ptr<NativeTensor> dV = dV_hq;
  if (H_q != H_kv) {
    const int64_t group_ratio = static_cast<int64_t>(H_q / H_kv);
    std::vector<int64_t> grouped_shape = {
        static_cast<int64_t>(B_sz), static_cast<int64_t>(H_kv), group_ratio,
        static_cast<int64_t>(S_k), static_cast<int64_t>(D_sz)};
    dK = dK_hq->reshape_view(grouped_shape)->reduce_sum(2, false);
    dV = dV_hq->reshape_view(grouped_shape)->reduce_sum(2, false);
  }

  if (orig_dt != ScalarType::Float32) {
    return {dQ->cast(orig_dt), dK->cast(orig_dt), dV->cast(orig_dt)};
  }
  return {dQ, dK, dV};
}

std::shared_ptr<NativeTensor>
NativeTensor::rope(std::shared_ptr<NativeTensor> cos_t,
                   std::shared_ptr<NativeTensor> sin_t, uint32_t seq_dim,
                   bool is_backward) {
  if (shape.size() != 4) {
    throw std::runtime_error(
        "rope: expected 4D tensor [B, H, S, D] or [B, S, H, D]");
  }
  const ScalarType orig_dt = dtype;
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto c_c = (cos_t->dtype == ScalarType::Float32)
                 ? cos_t->force_contiguous()
                 : cos_t->cast(ScalarType::Float32)->force_contiguous();
  auto s_c = (sin_t->dtype == ScalarType::Float32)
                 ? sin_t->force_contiguous()
                 : sin_t->cast(ScalarType::Float32)->force_contiguous();

  const uint32_t D_sz = static_cast<uint32_t>(x_c->shape[3]);
  if (D_sz % 2 != 0) {
    throw std::runtime_error("rope: head_dim D must be even");
  }
  const uint32_t half_D = D_sz / 2u;
  const uint32_t H_sz =
      static_cast<uint32_t>((seq_dim == 2u) ? x_c->shape[1] : x_c->shape[2]);
  const uint32_t S_sz =
      static_cast<uint32_t>((seq_dim == 2u) ? x_c->shape[2] : x_c->shape[1]);
  const uint32_t cos_last = static_cast<uint32_t>(c_c->shape.back());
  if (cos_last != half_D && cos_last != D_sz) {
    throw std::runtime_error("rope: cos/sin last dimension must be D/2 or D");
  }

  auto out = std::make_shared<NativeTensor>(x_c->shape, device_id,
                                            ScalarType::Float32);
  if (out->numel == 0)
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);

  const uint32_t total_pairs = static_cast<uint32_t>(x_c->numel / 2);

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const float *cp = c_c->cpu_f32();
    const float *sp = s_c->cpu_f32();
    float *op = out->cpu_f32();
    for (uint32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
      const uint32_t d = pair_idx % half_D;
      const uint32_t rem = pair_idx / half_D;
      uint32_t s = 0;
      if (seq_dim == 2u) {
        s = rem % S_sz;
      } else {
        s = (rem / H_sz) % S_sz;
      }
      const uint32_t row_off = rem * D_sz;
      const uint32_t idx0 = row_off + d;
      const uint32_t idx1 = row_off + d + half_D;
      const float c = cp[s * cos_last + d];
      const float sn = sp[s * cos_last + d];
      const float x0 = xp[idx0];
      const float x1 = xp[idx1];
      if (!is_backward) {
        op[idx0] = x0 * c - x1 * sn;
        op[idx1] = x1 * c + x0 * sn;
      } else {
        op[idx0] = x0 * c + x1 * sn;
        op[idx1] = x1 * c - x0 * sn;
      }
    }
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
  }

  struct {
    uint32_t total_pairs, H, S, half_D, cos_stride_s, seq_dim, is_backward;
  } pc{total_pairs,          H_sz, S_sz, half_D, cos_last, seq_dim,
       is_backward ? 1u : 0u};

  const uint32_t gx = (total_pairs + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "rope_kernel",
      {x_c->gpu_buf(), c_c->gpu_buf(), s_c->gpu_buf(), out->gpu_buf()}, &pc,
      sizeof(pc), gx);

  return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
}

std::shared_ptr<NativeTensor> NativeTensor::glu(std::shared_ptr<NativeTensor> b,
                                                uint32_t mode) {
  if (shape != b->shape) {
    throw std::runtime_error("glu: a and b must have identical shape");
  }
  const ScalarType orig_dt = dtype;
  auto a_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto b_c = (b->dtype == ScalarType::Float32)
                 ? b->force_contiguous()
                 : b->cast(ScalarType::Float32)->force_contiguous();
  auto out = std::make_shared<NativeTensor>(a_c->shape, device_id,
                                            ScalarType::Float32);
  if (out->numel == 0)
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);

  if (device_id == -1) {
    const float *ap = a_c->cpu_f32();
    const float *bp = b_c->cpu_f32();
    float *op = out->cpu_f32();
    for (size_t i = 0; i < a_c->numel; ++i) {
      const float g = (mode == 0u) ? apply_unary_op(12u, ap[i], 0.0f, 0.0f)
                                   : apply_unary_op(11u, ap[i], 0.0f, 0.0f);
      op[i] = g * bp[i];
    }
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
  }

  struct {
    uint32_t total_elements;
    uint32_t mode;
  } pc{static_cast<uint32_t>(a_c->numel), mode};

  const uint32_t gx = (pc.total_elements + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(device_id, "glu_forward",
                                             {a_c->gpu_buf(), b_c->gpu_buf(),
                                              b_c->gpu_buf(), out->gpu_buf(),
                                              out->gpu_buf()},
                                             &pc, sizeof(pc), gx);

  return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::glu_backward(std::shared_ptr<NativeTensor> grad_out,
                           std::shared_ptr<NativeTensor> b, uint32_t mode) {
  const ScalarType orig_dt = dtype;
  auto a_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto b_c = (b->dtype == ScalarType::Float32)
                 ? b->force_contiguous()
                 : b->cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();

  auto da = std::make_shared<NativeTensor>(a_c->shape, device_id,
                                           ScalarType::Float32);
  auto db = std::make_shared<NativeTensor>(b_c->shape, device_id,
                                           ScalarType::Float32);
  if (da->numel == 0) {
    return {(orig_dt == ScalarType::Float32) ? da : da->cast(orig_dt),
            (orig_dt == ScalarType::Float32) ? db : db->cast(orig_dt)};
  }

  if (device_id == -1) {
    const float *ap = a_c->cpu_f32();
    const float *bp = b_c->cpu_f32();
    const float *dyp = g_c->cpu_f32();
    float *dap = da->cpu_f32();
    float *dbp = db->cpu_f32();
    for (size_t i = 0; i < a_c->numel; ++i) {
      const float a_val = ap[i];
      const float b_val = bp[i];
      const float dy = dyp[i];
      const float g = (mode == 0u) ? apply_unary_op(12u, a_val, 0.0f, 0.0f)
                                   : apply_unary_op(11u, a_val, 0.0f, 0.0f);
      const float dy_dg = (mode == 0u) ? apply_binary_op(12u, 1.0f, a_val)
                                       : apply_binary_op(11u, 1.0f, a_val);
      dap[i] = dy * b_val * dy_dg;
      dbp[i] = dy * g;
    }
    return {(orig_dt == ScalarType::Float32) ? da : da->cast(orig_dt),
            (orig_dt == ScalarType::Float32) ? db : db->cast(orig_dt)};
  }

  struct {
    uint32_t total_elements;
    uint32_t mode;
  } pc{static_cast<uint32_t>(a_c->numel), mode};

  const uint32_t gx = (pc.total_elements + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(device_id, "glu_backward",
                                             {g_c->gpu_buf(), a_c->gpu_buf(),
                                              b_c->gpu_buf(), da->gpu_buf(),
                                              db->gpu_buf()},
                                             &pc, sizeof(pc), gx);

  return {(orig_dt == ScalarType::Float32) ? da : da->cast(orig_dt),
          (orig_dt == ScalarType::Float32) ? db : db->cast(orig_dt)};
}

std::shared_ptr<NativeTensor> NativeTensor::gemm_quant(
    std::shared_ptr<NativeTensor> qweight, std::shared_ptr<NativeTensor> scales,
    std::shared_ptr<NativeTensor> bias, uint32_t group_size, uint32_t bits) {
  if (bits != 4u && bits != 8u) {
    throw std::runtime_error("gemm_quant: bits must be 4 or 8");
  }
  const ScalarType orig_dt = dtype;
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto qw_c = qweight->force_contiguous();
  auto sc_c = (scales->dtype == ScalarType::Float32)
                  ? scales->force_contiguous()
                  : scales->cast(ScalarType::Float32)->force_contiguous();
  auto b_c = bias ? ((bias->dtype == ScalarType::Float32)
                         ? bias->force_contiguous()
                         : bias->cast(ScalarType::Float32)->force_contiguous())
                  : nullptr;

  if (x_c->shape.empty() || qw_c->shape.size() != 2) {
    throw std::runtime_error("gemm_quant: invalid input or qweight rank");
  }
  const uint32_t K = static_cast<uint32_t>(x_c->shape.back());
  const uint32_t M = static_cast<uint32_t>(x_c->numel / K);
  const uint32_t N = static_cast<uint32_t>(qw_c->shape[0]);
  const uint32_t pack_factor = (bits == 4u) ? 8u : 4u;
  if (K % pack_factor != 0u ||
      static_cast<uint32_t>(qw_c->shape[1]) != K / pack_factor) {
    throw std::runtime_error(
        "gemm_quant: in_features K must match packed qweight columns");
  }

  std::vector<int64_t> out_shape = x_c->shape;
  out_shape.back() = N;
  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (out->numel == 0)
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);

  if (device_id == -1) {
    const float *xp = x_c->cpu_f32();
    const uint32_t *qwp =
        reinterpret_cast<const uint32_t *>(qw_c->cpu_raw_ptr());
    const float *scp = sc_c->cpu_f32();
    const float *bp = b_c ? b_c->cpu_f32() : nullptr;
    float *yp = out->cpu_f32();
    const uint32_t num_groups = (K + group_size - 1u) / group_size;
    const uint32_t words_per_row = K / pack_factor;

    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        float acc = bp ? bp[n] : 0.0f;
        for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx) {
          const uint32_t k_base = w_idx * pack_factor;
          const float sc = scp[n * num_groups + (k_base / group_size)];
          const uint32_t packed = qwp[n * words_per_row + w_idx];
          if (bits == 4u) {
            for (uint32_t j = 0; j < 8u; ++j) {
              const int nib = static_cast<int>((packed >> (j * 4u)) & 0xFu) - 8;
              acc += xp[m * K + k_base + j] * (static_cast<float>(nib) * sc);
            }
          } else {
            for (uint32_t j = 0; j < 4u; ++j) {
              const uint32_t byte_val = (packed >> (j * 8u)) & 0xFFu;
              const int s8 = (byte_val >= 128u)
                                 ? (static_cast<int>(byte_val) - 256)
                                 : static_cast<int>(byte_val);
              acc += xp[m * K + k_base + j] * (static_cast<float>(s8) * sc);
            }
          }
        }
        yp[m * N + n] = acc;
      }
    }
    return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
  }

  struct {
    uint32_t M, N, K, group_size, bits, has_bias;
  } pc{M, N, K, group_size, bits, b_c ? 1u : 0u};

  auto bias_buf = b_c ? b_c->gpu_buf() : x_c->gpu_buf();
  const uint32_t gx = (N + 15u) / 16u;
  const uint32_t gy = (M + 15u) / 16u;
  VulkanPipelineManager::instance().dispatch(device_id, "gemm_quant_kernel",
                                             {x_c->gpu_buf(), qw_c->gpu_buf(),
                                              sc_c->gpu_buf(), bias_buf,
                                              out->gpu_buf(), out->gpu_buf()},
                                             &pc, sizeof(pc), gx, gy, 1u);

  return (orig_dt == ScalarType::Float32) ? out : out->cast(orig_dt);
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::quantize_weights(uint32_t group_size, uint32_t bits) {
  if (shape.size() != 2 || (bits != 4u && bits != 8u)) {
    throw std::runtime_error(
        "quantize_weights: expected 2D weight [N, K] and bits in {4, 8}");
  }
  auto w_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  const uint32_t N = static_cast<uint32_t>(w_c->shape[0]);
  const uint32_t K = static_cast<uint32_t>(w_c->shape[1]);
  const uint32_t pack_factor = (bits == 4u) ? 8u : 4u;
  if (K % pack_factor != 0u) {
    throw std::runtime_error(
        "quantize_weights: K must be divisible by pack_factor");
  }
  const uint32_t words_per_row = K / pack_factor;
  const uint32_t num_groups = (K + group_size - 1u) / group_size;

  auto qweight = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(N),
                           static_cast<int64_t>(words_per_row)},
      device_id, ScalarType::Int32);
  auto scales = std::make_shared<NativeTensor>(
      std::vector<int64_t>{static_cast<int64_t>(N),
                           static_cast<int64_t>(num_groups)},
      device_id, ScalarType::Float32);

  if (device_id == -1) {
    const float *wp = w_c->cpu_f32();
    uint32_t *qwp = reinterpret_cast<uint32_t *>(qweight->cpu_raw_ptr());
    float *scp = scales->cpu_f32();
    const float max_q = (bits == 4u) ? 7.0f : 127.0f;

    for (uint32_t n = 0; n < N; ++n) {
      const float *row = wp + n * K;
      for (uint32_t g = 0; g < num_groups; ++g) {
        const uint32_t k_start = g * group_size;
        const uint32_t k_end = std::min(k_start + group_size, K);
        float max_abs = 0.0f;
        for (uint32_t k = k_start; k < k_end; ++k) {
          max_abs = std::max(max_abs, std::abs(row[k]));
        }
        scp[n * num_groups + g] = std::max(max_abs / max_q, 1e-8f);
      }
      for (uint32_t w_idx = 0; w_idx < words_per_row; ++w_idx) {
        const uint32_t k_base = w_idx * pack_factor;
        const float sc = scp[n * num_groups + (k_base / group_size)];
        uint32_t packed = 0u;
        if (bits == 4u) {
          for (uint32_t j = 0; j < 8u; ++j) {
            const int qv =
                std::clamp(static_cast<int>(std::round(row[k_base + j] / sc)),
                           -7, 7) +
                8;
            packed |= (static_cast<uint32_t>(qv) & 0xFu) << (j * 4u);
          }
        } else {
          for (uint32_t j = 0; j < 4u; ++j) {
            const int qv = std::clamp(
                static_cast<int>(std::round(row[k_base + j] / sc)), -127, 127);
            const uint32_t u8 = static_cast<uint32_t>(qv) & 0xFFu;
            packed |= u8 << (j * 8u);
          }
        }
        qwp[n * words_per_row + w_idx] = packed;
      }
    }
    return {qweight, scales};
  }

  struct {
    uint32_t N, K, group_size, bits;
  } pc{N, K, group_size, bits};

  const uint32_t gx = (words_per_row + 15u) / 16u;
  const uint32_t gy = (N + 15u) / 16u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "quantize_pack_weights",
      {w_c->gpu_buf(), qweight->gpu_buf(), scales->gpu_buf(), w_c->gpu_buf(),
       scales->gpu_buf(), qweight->gpu_buf()},
      &pc, sizeof(pc), gx, gy, 1u);

  return {qweight, scales};
}

bool NativeTensor::amp_unscale_and_check(
    const std::vector<std::shared_ptr<NativeTensor>> &grads, float inv_scale) {
  if (grads.empty())
    return false;

  int gpu_dev = -1;
  for (const auto &g : grads) {
    if (g && g->device_id >= 0) {
      gpu_dev = g->device_id;
      break;
    }
  }

  if (gpu_dev == -1) {
    bool found_inf = false;
    for (const auto &g : grads) {
      if (!g || g->numel == 0)
        continue;
      if (g->dtype == ScalarType::Float32) {
        float *gp = g->cpu_f32();
        for (size_t i = 0; i < g->numel; ++i) {
          gp[i] *= inv_scale;
          if (!std::isfinite(gp[i]))
            found_inf = true;
        }
      } else {
        auto g32 = g->cast(ScalarType::Float32);
        float *gp = g32->cpu_f32();
        for (size_t i = 0; i < g32->numel; ++i) {
          gp[i] *= inv_scale;
          if (!std::isfinite(gp[i]))
            found_inf = true;
        }
        g->copy_from_native(g32->cast(g->dtype));
      }
    }
    return found_inf;
  }

  auto flag = std::make_shared<NativeTensor>(std::vector<int64_t>{1}, gpu_dev,
                                             ScalarType::Int32);
  flag->zero_();

  for (const auto &g : grads) {
    if (!g || g->numel == 0)
      continue;
    if (g->device_id == gpu_dev && g->dtype == ScalarType::Float32 &&
        g->is_dense_zero_offset()) {
      struct {
        uint32_t total_elements;
        float inv_scale;
      } pc{static_cast<uint32_t>(g->numel), inv_scale};
      const uint32_t gx = (pc.total_elements + 255u) / 256u;
      VulkanPipelineManager::instance().dispatch(
          gpu_dev, "amp_unscale_check",
          {g->gpu_buf(), g->gpu_buf(), g->gpu_buf(), g->gpu_buf(), g->gpu_buf(),
           flag->gpu_buf()},
          &pc, sizeof(pc), gx);
    } else {
      auto g32 = (g->device_id == gpu_dev)
                     ? g->cast(ScalarType::Float32)->force_contiguous()
                     : g->to_device(gpu_dev)
                           ->cast(ScalarType::Float32)
                           ->force_contiguous();
      struct {
        uint32_t total_elements;
        float inv_scale;
      } pc{static_cast<uint32_t>(g32->numel), inv_scale};
      const uint32_t gx = (pc.total_elements + 255u) / 256u;
      VulkanPipelineManager::instance().dispatch(
          gpu_dev, "amp_unscale_check",
          {g32->gpu_buf(), g32->gpu_buf(), g32->gpu_buf(), g32->gpu_buf(),
           g32->gpu_buf(), flag->gpu_buf()},
          &pc, sizeof(pc), gx);
      g->copy_from_native(g32->cast(g->dtype)->to_device(g->device_id));
    }
  }

  int32_t host_flag = 0;
  flag->to_raw_bytes(&host_flag, sizeof(int32_t));
  return host_flag != 0;
}

} // namespace vultorch
