#include "tensor/tensor_internal.hpp"

namespace vultorch {

std::shared_ptr<NativeTensor>
NativeTensor::binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other) {
  if (device_id != other->device_id) {
    throw std::runtime_error(
        "Tensors must be on the same device for binary operations!");
  }

  const bool is_comparison =
      (op_type == 7u || op_type == 8u || op_type == 9u || op_type == 14u ||
       op_type == 15u || op_type == 16u);
  const ScalarType common = promote_dtype(dtype, other->dtype);

  if (device_id == -1) {
    const BroadcastPlan plan =
        make_broadcast_plan(shape, strides, other->shape, other->strides);
    const ScalarType out_dt = is_comparison ? ScalarType::Float32 : common;
    auto out = std::make_shared<NativeTensor>(plan.out_shape, -1, out_dt);
    if (out->numel == 0)
      return out;

    const size_t ndim = plan.out_shape.size();
    std::vector<int64_t> coord(ndim, 0);
    int64_t ia = static_cast<int64_t>(storage_offset);
    int64_t ib = static_cast<int64_t>(other->storage_offset);

    if (dtype == ScalarType::Float32 && other->dtype == ScalarType::Float32) {
      const float *a_ptr = cpu_f32();
      const float *b_ptr = other->cpu_f32();
      float *out_ptr = out->cpu_f32();
      for (size_t n = 0; n < out->numel; ++n) {
        out_ptr[n] = apply_binary_op(op_type, a_ptr[ia], b_ptr[ib]);
        advance_odometer(ndim, plan.out_shape, plan.strides_a, coord, ia);
        // Re-step ib using saved delta: use explicit step for both arrays
        // cleanly. Undo coord update for ib helper? Advance manually instead:
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
          size_t ud = static_cast<size_t>(d);
          ib += plan.strides_b[ud];
          if (coord[ud] != 0)
            break;
          ib -= plan.strides_b[ud] * plan.out_shape[ud];
        }
      }
      return out;
    }

    const uint8_t *a_raw = cpu_raw_ptr();
    const uint8_t *b_raw = other->cpu_raw_ptr();
    uint8_t *out_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      float va = read_scalar_as_float(a_raw, static_cast<size_t>(ia), dtype);
      float vb =
          read_scalar_as_float(b_raw, static_cast<size_t>(ib), other->dtype);
      float res = apply_binary_op(op_type, va, vb);
      write_scalar_from_double(out_raw, n, out_dt, static_cast<double>(res));
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        size_t ud = static_cast<size_t>(d);
        ia += plan.strides_a[ud];
        ib += plan.strides_b[ud];
        if (++coord[ud] < plan.out_shape[ud])
          break;
        ia -= plan.strides_a[ud] * plan.out_shape[ud];
        ib -= plan.strides_b[ud] * plan.out_shape[ud];
        coord[ud] = 0;
      }
    }
    return out;
  }

  if (dtype != ScalarType::Float32 || other->dtype != ScalarType::Float32) {
    auto a32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                              : cast(ScalarType::Float32);
    auto b32 = (other->dtype == ScalarType::Float32)
                   ? other
                   : other->cast(ScalarType::Float32);
    auto res32 = a32->binary_op(op_type, b32);
    return is_comparison ? res32 : res32->cast(common);
  }

  bool same_shape = (shape == other->shape);
  if (same_shape && is_dense_zero_offset() && other->is_dense_zero_offset()) {
    auto out =
        std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    if (out->numel == 0)
      return out;
    uint32_t num_vec4 = static_cast<uint32_t>(numel / 4);

    if (num_vec4 > 0) {
      struct {
        uint32_t num_vec4;
        uint32_t op_type;
        float scalar_b;
        uint32_t is_b_scalar;
      } pc{num_vec4, op_type, 0.0f, 0};

      uint32_t groups = (num_vec4 + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_binary_vec4",
          {gpu_buf(), other->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
          groups);
    }

    uint32_t tail_count = static_cast<uint32_t>(numel % 4);
    if (tail_count > 0) {
      struct {
        uint32_t total_elements;
        uint32_t op_type;
        uint32_t ndim;
        uint32_t is_b_scalar;
        float scalar_b;
        uint32_t offsetA;
        uint32_t offsetB;
        uint32_t shape[6];
        uint32_t strideA[6];
        uint32_t strideB[6];
      } pc{};
      pc.total_elements = static_cast<uint32_t>(numel);
      pc.op_type = op_type;
      pc.ndim = 0;
      pc.is_b_scalar = 0;

      uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_binary_broadcast",
          {gpu_buf(), other->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
          groups);
    }
    return out;
  }

  // Fast path for contiguous shapes with storage_offset > 0 and arbitrary rank
  if (same_shape && is_contiguous() && other->is_contiguous()) {
    auto out =
        std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
    if (out->numel == 0)
      return out;
    struct {
      uint32_t total_elements;
      uint32_t op_type;
      uint32_t ndim;
      uint32_t is_b_scalar;
      float scalar_b;
      uint32_t offsetA;
      uint32_t offsetB;
      uint32_t shape[6];
      uint32_t strideA[6];
      uint32_t strideB[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out->numel);
    pc.op_type = op_type;
    pc.ndim = 0;
    pc.is_b_scalar = 0;
    pc.offsetA = static_cast<uint32_t>(storage_offset);
    pc.offsetB = static_cast<uint32_t>(other->storage_offset);

    uint32_t groups = (static_cast<uint32_t>(out->numel) + 255) / 256;
    VulkanPipelineManager::instance().dispatch(
        device_id, "elementwise_binary_broadcast",
        {gpu_buf(), other->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), groups);
    return out;
  }

  // General broadcasting / strided path.
  std::shared_ptr<NativeTensor> a_ready = all_strides_non_negative(strides)
                                              ? shared_from_this()
                                              : force_contiguous();
  std::shared_ptr<NativeTensor> b_ready =
      all_strides_non_negative(other->strides) ? other
                                               : other->force_contiguous();

  BroadcastPlan plan = make_broadcast_plan(a_ready->shape, a_ready->strides,
                                           b_ready->shape, b_ready->strides);
  if (plan.out_shape.size() > 6) {
    a_ready = a_ready->force_contiguous();
    b_ready = b_ready->force_contiguous();
    plan = make_broadcast_plan(a_ready->shape, a_ready->strides, b_ready->shape,
                               b_ready->strides);
    if (plan.out_shape.size() > 6) {
      throw std::runtime_error(
          "Broadcasting supports up to 6 dimensions on GPU");
    }
  }

  const std::vector<int64_t> &out_shape = plan.out_shape;
  const size_t max_ndim = out_shape.size();
  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (out->numel == 0)
    return out;

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    uint32_t ndim;
    uint32_t is_b_scalar;
    float scalar_b;
    uint32_t offsetA;
    uint32_t offsetB;
    uint32_t shape[6];
    uint32_t strideA[6];
    uint32_t strideB[6];
  } pc{};

  pc.total_elements = static_cast<uint32_t>(out->numel);
  pc.op_type = op_type;
  pc.ndim = static_cast<uint32_t>(max_ndim);
  pc.is_b_scalar = 0;
  pc.offsetA = static_cast<uint32_t>(a_ready->storage_offset);
  pc.offsetB = static_cast<uint32_t>(b_ready->storage_offset);

  for (size_t i = 0; i < max_ndim; ++i) {
    pc.shape[i] = static_cast<uint32_t>(out_shape[i]);
    pc.strideA[i] = static_cast<uint32_t>(plan.strides_a[i]);
    pc.strideB[i] = static_cast<uint32_t>(plan.strides_b[i]);
  }

  uint32_t groups = (static_cast<uint32_t>(out->numel) + 255) / 256;
  VulkanPipelineManager::instance().dispatch(
      device_id, "elementwise_binary_broadcast",
      {a_ready->gpu_buf(), b_ready->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc),
      groups);

  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::binary_op_scalar(uint32_t op_type,
                                                             float scalar_val,
                                                             bool reverse) {
  const bool is_comparison =
      (op_type == 7u || op_type == 8u || op_type == 9u || op_type == 14u ||
       op_type == 15u || op_type == 16u);

  if (device_id == -1) {
    const ScalarType out_dt = is_comparison ? ScalarType::Float32 : dtype;
    auto out = std::make_shared<NativeTensor>(shape, -1, out_dt);
    if (out->numel == 0)
      return out;

    const size_t ndim = shape.size();
    std::vector<int64_t> coord(ndim, 0);
    int64_t ia = static_cast<int64_t>(storage_offset);
    if (dtype == ScalarType::Float32) {
      const float *a_ptr = cpu_f32();
      float *out_ptr = out->cpu_f32();
      for (size_t i = 0; i < numel; ++i) {
        float a = reverse ? scalar_val : a_ptr[ia];
        float b = reverse ? a_ptr[ia] : scalar_val;
        out_ptr[i] = apply_binary_op(op_type, a, b);
        advance_odometer(ndim, shape, strides, coord, ia);
      }
    } else {
      const uint8_t *a_raw = cpu_raw_ptr();
      uint8_t *out_raw = out->cpu_raw_ptr();
      for (size_t i = 0; i < numel; ++i) {
        float v = read_scalar_as_float(a_raw, static_cast<size_t>(ia), dtype);
        float a = reverse ? scalar_val : v;
        float b = reverse ? v : scalar_val;
        float res = apply_binary_op(op_type, a, b);
        write_scalar_from_double(out_raw, i, out_dt, static_cast<double>(res));
        advance_odometer(ndim, shape, strides, coord, ia);
      }
    }
    return out;
  }

  if (dtype != ScalarType::Float32) {
    auto res32 = cast(ScalarType::Float32)
                     ->binary_op_scalar(op_type, scalar_val, reverse);
    return is_comparison ? res32 : res32->cast(dtype);
  }

  if (reverse) {
    if (op_type == 1u) {
      auto src =
          is_dense_zero_offset() ? shared_from_this() : force_contiguous();
      auto out =
          std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
      if (out->numel == 0)
        return out;
      struct {
        uint32_t total_elements;
        uint32_t op_type;
        float p1;
        float p2;
      } pc{static_cast<uint32_t>(numel), 15u, -1.0f, scalar_val};
      uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_unary_scalar",
          {src->gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), groups);
      return out;
    }
    auto scalar_t = NativeTensor::from_float_array(&scalar_val, {1}, device_id);
    return scalar_t->binary_op(op_type, shared_from_this());
  }

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out->numel == 0)
    return out;

  if (is_dense_zero_offset()) {
    uint32_t num_vec4 = static_cast<uint32_t>(numel / 4);
    if (num_vec4 > 0) {
      struct {
        uint32_t num_vec4;
        uint32_t op_type;
        float scalar_b;
        uint32_t is_b_scalar;
      } pc{num_vec4, op_type, scalar_val, 1};

      uint32_t groups = (num_vec4 + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_binary_vec4",
          {gpu_buf(), gpu_buf(), out->gpu_buf()}, &pc, sizeof(pc), groups);
    }

    if ((numel % 4 != 0) || num_vec4 == 0) {
      struct {
        uint32_t total_elements;
        uint32_t op_type;
        uint32_t ndim;
        uint32_t is_b_scalar;
        float scalar_b;
        uint32_t offsetA;
        uint32_t offsetB;
        uint32_t shape[6];
        uint32_t strideA[6];
        uint32_t strideB[6];
      } b_pc{};
      b_pc.total_elements = static_cast<uint32_t>(numel);
      b_pc.op_type = op_type;
      b_pc.ndim = 0;
      b_pc.is_b_scalar = 1;
      b_pc.scalar_b = scalar_val;

      uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_binary_broadcast",
          {gpu_buf(), gpu_buf(), out->gpu_buf()}, &b_pc, sizeof(b_pc), groups);
    }
    return out;
  }

  // Non-contiguous or storage_offset > 0 scalar op via broadcast shader
  std::shared_ptr<NativeTensor> src = shared_from_this();
  if (!is_contiguous() &&
      (shape.size() > 6 || !all_strides_non_negative(strides))) {
    src = force_contiguous();
  }

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    uint32_t ndim;
    uint32_t is_b_scalar;
    float scalar_b;
    uint32_t offsetA;
    uint32_t offsetB;
    uint32_t shape[6];
    uint32_t strideA[6];
    uint32_t strideB[6];
  } b_pc{};
  b_pc.total_elements = static_cast<uint32_t>(numel);
  b_pc.op_type = op_type;
  b_pc.is_b_scalar = 1;
  b_pc.scalar_b = scalar_val;
  b_pc.offsetA = static_cast<uint32_t>(src->storage_offset);

  if (src->is_contiguous()) {
    b_pc.ndim = 0;
  } else {
    b_pc.ndim = static_cast<uint32_t>(src->shape.size());
    for (size_t i = 0; i < src->shape.size(); ++i) {
      b_pc.shape[i] = static_cast<uint32_t>(src->shape[i]);
      b_pc.strideA[i] = static_cast<uint32_t>(src->strides[i]);
    }
  }

  uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
  VulkanPipelineManager::instance().dispatch(
      device_id, "elementwise_binary_broadcast",
      {src->gpu_buf(), src->gpu_buf(), out->gpu_buf()}, &b_pc, sizeof(b_pc),
      groups);
  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::unary_op(uint32_t op_type, float p1,
                                                     float p2) {
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)->unary_op(op_type, p1, p2)->cast(dtype);
  }

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(shape, -1, ScalarType::Float32);
    if (out->numel == 0)
      return out;
    const size_t ndim = shape.size();
    std::vector<int64_t> coord(ndim, 0);
    int64_t ia = static_cast<int64_t>(storage_offset);
    const float *in_ptr = cpu_f32();
    float *out_ptr = out->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      out_ptr[i] = apply_unary_op(op_type, in_ptr[ia], p1, p2);
      advance_odometer(ndim, shape, strides, coord, ia);
    }
    return out;
  }

  if (!is_dense_zero_offset()) {
    return force_contiguous()->unary_op(op_type, p1, p2);
  }

  auto out =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out->numel == 0)
    return out;
  uint32_t num_vec4 = static_cast<uint32_t>(numel / 4);

  if (num_vec4 > 0) {
    struct {
      uint32_t num_vec4;
      uint32_t op_type;
      float param1;
      float param2;
    } pc{num_vec4, op_type, p1, p2};

    uint32_t groups = (num_vec4 + 255) / 256;
    VulkanPipelineManager::instance().dispatch(
        device_id, "elementwise_unary_vec4", {gpu_buf(), out->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }

  uint32_t tail = static_cast<uint32_t>(numel % 4);
  if (tail > 0 || num_vec4 == 0) {
    struct {
      uint32_t total_elements;
      uint32_t op_type;
      float param1;
      float param2;
    } pc{static_cast<uint32_t>(numel), op_type, p1, p2};

    uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
    VulkanPipelineManager::instance().dispatch(
        device_id, "elementwise_unary_scalar", {gpu_buf(), out->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }

  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::autodiff_unary_fwd(uint32_t op_type, float param) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto out = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (numel == 0)
    return (dtype == ScalarType::Float32) ? out : out->cast(dtype);

  if (device_id < 0) {
    const float *xp = x_c->cpu_f32();
    float *yp = out->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      float x = xp[i];
      if (op_type == 0u) {
        float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
        yp[i] = x * std::tanh(sp);
      } else if (op_type == 1u) {
        float c = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
        yp[i] = x * c / 6.0f;
      } else if (op_type == 2u) {
        yp[i] = (x >= 0.0f) ? x : (param * x);
      } else if (op_type == 3u) {
        yp[i] = (x >= 0.0f) ? x : (param * (std::exp(x) - 1.0f));
      } else {
        yp[i] = x;
      }
    }
    return (dtype == ScalarType::Float32) ? out : out->cast(dtype);
  }

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    float param;
    uint32_t _pad;
  } pc{static_cast<uint32_t>(numel), op_type, param, 0u};

  uint32_t groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  auto x_buf = x_c->gpu_buf();
  auto out_buf = out->gpu_buf();
  VulkanPipelineManager::instance().dispatch(
      device_id, "autodiff_unary_fwd",
      {x_buf, x_buf, x_buf, out_buf, out_buf}, &pc, sizeof(pc), groups);
  return (dtype == ScalarType::Float32) ? out : out->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::autodiff_unary_bwd(uint32_t op_type,
                                 std::shared_ptr<NativeTensor> grad_out,
                                 float param) {
  auto x_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  auto dx = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (numel == 0)
    return (dtype == ScalarType::Float32) ? dx : dx->cast(dtype);

  if (device_id < 0) {
    const float *xp = x_c->cpu_f32();
    const float *gp = g_c->cpu_f32();
    float *dxp = dx->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      float x = xp[i];
      float g = gp[i];
      if (op_type == 0u) {
        float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
        float tsp = std::tanh(sp);
        float sig = 1.0f / (1.0f + std::exp(-x));
        dxp[i] = g * (tsp + x * (1.0f - tsp * tsp) * sig);
      } else if (op_type == 1u) {
        if (x <= -3.0f)
          dxp[i] = 0.0f;
        else if (x >= 3.0f)
          dxp[i] = g;
        else
          dxp[i] = g * ((2.0f * x + 3.0f) / 6.0f);
      } else if (op_type == 2u) {
        dxp[i] = (x >= 0.0f) ? g : (param * g);
      } else if (op_type == 3u) {
        dxp[i] = (x >= 0.0f) ? g : (g * param * std::exp(x));
      } else {
        dxp[i] = g;
      }
    }
    return (dtype == ScalarType::Float32) ? dx : dx->cast(dtype);
  }

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    float param;
    uint32_t _pad;
  } pc{static_cast<uint32_t>(numel), op_type, param, 0u};

  uint32_t groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  auto x_buf = x_c->gpu_buf();
  auto g_buf = g_c->gpu_buf();
  auto dx_buf = dx->gpu_buf();
  VulkanPipelineManager::instance().dispatch(
      device_id, "autodiff_unary_bwd",
      {x_buf, x_buf, g_buf, dx_buf, dx_buf}, &pc, sizeof(pc), groups);
  return (dtype == ScalarType::Float32) ? dx : dx->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::autodiff_binary_fwd(uint32_t op_type,
                                  std::shared_ptr<NativeTensor> other,
                                  float param) {
  if (shape != other->shape || device_id != other->device_id) {
    throw std::runtime_error(
        "autodiff_binary_fwd: shapes and devices must match");
  }
  auto a_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto b_c = (other->dtype == ScalarType::Float32)
                 ? other->force_contiguous()
                 : other->cast(ScalarType::Float32)->force_contiguous();
  auto out = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (numel == 0)
    return (dtype == ScalarType::Float32) ? out : out->cast(dtype);

  if (device_id < 0) {
    const float *ap = a_c->cpu_f32();
    const float *bp = b_c->cpu_f32();
    float *yp = out->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      float diff = ap[i] - bp[i];
      float ad = std::abs(diff);
      yp[i] = (ad < param) ? (0.5f * diff * diff / param) : (ad - 0.5f * param);
    }
    return (dtype == ScalarType::Float32) ? out : out->cast(dtype);
  }

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    float param;
    uint32_t _pad;
  } pc{static_cast<uint32_t>(numel), op_type, param, 0u};

  uint32_t groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  auto a_buf = a_c->gpu_buf();
  auto b_buf = b_c->gpu_buf();
  auto out_buf = out->gpu_buf();
  VulkanPipelineManager::instance().dispatch(
      device_id, "autodiff_binary_fwd",
      {a_buf, b_buf, a_buf, out_buf, out_buf}, &pc, sizeof(pc), groups);
  return (dtype == ScalarType::Float32) ? out : out->cast(dtype);
}

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::autodiff_binary_bwd(uint32_t op_type,
                                  std::shared_ptr<NativeTensor> other,
                                  std::shared_ptr<NativeTensor> grad_out,
                                  float param) {
  auto a_c = (dtype == ScalarType::Float32)
                 ? force_contiguous()
                 : cast(ScalarType::Float32)->force_contiguous();
  auto b_c = (other->dtype == ScalarType::Float32)
                 ? other->force_contiguous()
                 : other->cast(ScalarType::Float32)->force_contiguous();
  auto g_c = (grad_out->dtype == ScalarType::Float32)
                 ? grad_out->force_contiguous()
                 : grad_out->cast(ScalarType::Float32)->force_contiguous();
  auto da = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  auto db = std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (numel == 0)
    return {da, db};

  if (device_id < 0) {
    const float *ap = a_c->cpu_f32();
    const float *bp = b_c->cpu_f32();
    const float *gp = g_c->cpu_f32();
    float *dap = da->cpu_f32();
    float *dbp = db->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      float diff = ap[i] - bp[i];
      float ad = std::abs(diff);
      float d = (ad < param) ? (diff / param) : ((diff > 0.0f) ? 1.0f : -1.0f);
      float g = gp[i];
      dap[i] = g * d;
      dbp[i] = -g * d;
    }
    return {(dtype == ScalarType::Float32) ? da : da->cast(dtype),
            (other->dtype == ScalarType::Float32) ? db : db->cast(other->dtype)};
  }

  struct {
    uint32_t total_elements;
    uint32_t op_type;
    float param;
    uint32_t _pad;
  } pc{static_cast<uint32_t>(numel), op_type, param, 0u};

  uint32_t groups = (static_cast<uint32_t>(numel) + 255u) / 256u;
  VulkanPipelineManager::instance().dispatch(
      device_id, "autodiff_binary_bwd",
      {a_c->gpu_buf(), b_c->gpu_buf(), g_c->gpu_buf(), da->gpu_buf(),
       db->gpu_buf()},
      &pc, sizeof(pc), groups);
  return {(dtype == ScalarType::Float32) ? da : da->cast(dtype),
          (other->dtype == ScalarType::Float32) ? db : db->cast(other->dtype)};
}

} // namespace vultorch
