#include "tensor/tensor_internal.hpp"

namespace vultorch {

namespace {

// GEMM variants. Which one is fastest depends on the shape *and* the device
// (tile reuse vs occupancy), so the choice is measured, not hardcoded.
enum class MatmulRequirement {
  None,
  // Needs a 16x16x16 f16 x f16 -> f32 cooperative-matrix configuration.
  Fp16CoopMat,
};

struct MatmulCandidate {
  const char *shader;
  uint32_t tile;
  uint32_t shared_bytes;
  uint32_t workgroup_size;
  MatmulRequirement requirement;
  // Opt-in kernels are selectable by name but excluded from automatic
  // selection, so a variant that loses the measurement never spends
  // autotune time or changes numerics behind the user's back.
  bool auto_candidate;
};

constexpr MatmulCandidate kMatmulCandidates[] = {
    // Slang generic SGEMM (shaders/gemm.slang): one source, one entry point per
    // tile configuration. Shared bytes = 4 * BK * 2 * (TILE + 1).
    {"gemm_t32_k16_r2", 32u, 4u * 16u * 2u * (32u + 1u), 256u,
     MatmulRequirement::None, false},
    {"gemm_t64_k16_r4", 64u, 4u * 16u * 2u * (64u + 1u), 256u,
     MatmulRequirement::None, true},
    {"gemm_t64_k8_r4", 64u, 4u * 8u * 2u * (64u + 1u), 256u,
     MatmulRequirement::None, false},
    {"gemm_t64_k16_r8", 64u, 4u * 16u * 2u * (64u + 1u), 64u,
     MatmulRequirement::None, true},
    {"gemm_t128_k8_r8", 128u, 4u * 8u * 2u * (128u + 1u), 256u,
     MatmulRequirement::None, true},
    {"gemm_t128_k16_r8", 128u, 4u * 16u * 2u * (128u + 1u), 256u,
     MatmulRequirement::None, true},
    // Legacy entry-point aliases kept selectable via VULTORCH_MATMUL_KERNEL.
    {"matmul", 32u, 4u * 16u * 2u * (32u + 1u), 256u, MatmulRequirement::None,
     false},
    {"matmul_tiled64", 64u, 4u * 16u * 2u * (64u + 1u), 256u,
     MatmulRequirement::None, false},
    {"matmul_tiled", 128u, 4u * 16u * 2u * (128u + 1u), 256u,
     MatmulRequirement::None, false},
    // Cooperative-matrix 16x16x16 f16 -> f32 variant in shaders/gemm.slang.
    {"matmul_coopmat", 64u, 64u * 17u * 2u + 16u * 68u * 2u + 64u * 68u * 4u,
     64u, MatmulRequirement::Fp16CoopMat, false},
};

struct MatmulChoice {
  const char *shader;
  uint32_t tile;
};

using MatmulDispatch = std::function<void(const char *, uint32_t)>;
// (device, M, N, K, trans_a, trans_b)
using MatmulShapeKey = std::array<uint32_t, 6>;

std::mutex g_matmul_choice_mutex;
std::map<MatmulShapeKey, MatmulChoice> g_matmul_choice_cache;

bool env_flag_enabled(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr)
    return default_value;
  const std::string text(value);
  return !(text == "0" || text == "false" || text == "off");
}

// Picks the GEMM kernel for one shape.
//
// 1. Filter by what the device reports it can run (shared memory, workgroup
//    size, tile fits the problem).
// 2. Honour VULTORCH_MATMUL_KERNEL=<shader> for benchmarking a specific
// variant.
// 3. Measure the survivors once and cache the winner per (device, M, N, K,
// trans).
//    Training reuses a handful of shapes, so this cost is paid a few times.
// VULTORCH_AUTOTUNE=0 disables step 3 and takes the largest viable tile.
MatmulChoice select_matmul_shader(uint32_t device_id, uint32_t M, uint32_t N,
                                  uint32_t K, bool trans_a, bool trans_b,
                                  const MatmulDispatch &run) {
  const auto dev = VulkanManager::instance_ref().get_device(device_id);
  const DeviceCapabilities &caps = dev->capabilities;

  // 1. Capability filter: only variants this device reports it can run.
  std::vector<MatmulChoice> candidates;      // legal here, including opt-in
  std::vector<MatmulChoice> auto_candidates; // eligible for automatic use
  for (const auto &candidate : kMatmulCandidates) {
    if (candidate.shared_bytes > caps.max_shared_memory_bytes)
      continue;
    if (candidate.workgroup_size > caps.max_workgroup_invocations)
      continue;
    if (candidate.requirement == MatmulRequirement::Fp16CoopMat) {
      if (!caps.fp16_coopmat_16x16)
        continue;
      if (caps.subgroup_size < 16u || caps.subgroup_size % 16u != 0u)
        continue;
      if (64u % caps.subgroup_size != 0u)
        continue;
    }
    candidates.push_back({candidate.shader, candidate.tile});
    if (candidate.auto_candidate) {
      auto_candidates.push_back({candidate.shader, candidate.tile});
    }
  }
  if (candidates.empty())
    return {"gemm_t32_k16_r2", 32u};

  // 2. Explicit override, read every call so it can be changed at runtime
  //    (benchmarks and tests). Not cached: an override must win over the
  //    measurement cache.
  const char *const forced = std::getenv("VULTORCH_MATMUL_KERNEL");
  if (forced != nullptr) {
    const std::string wanted(forced);
    for (const auto &candidate : candidates) {
      if (wanted == candidate.shader)
        return candidate;
    }
    if (wanted == "legacy")
      return {"matmul", 32u};
  }

  // 3. Tiny problems are launch-bound; tiling cannot pay for itself and an
  //    autotune sweep would cost far more than the multiplication.
  if (2ull * M * N * K < 1'000'000ull)
    return {"gemm_t32_k16_r2", 32u};

  if (auto_candidates.empty())
    return {"gemm_t32_k16_r2", 32u};
  if (auto_candidates.size() == 1)
    return auto_candidates.front();

  // 4. Measured selection, cached per (device, shape, trans).
  const MatmulShapeKey key{device_id,        M, N, K, trans_a ? 1u : 0u,
                           trans_b ? 1u : 0u};
  {
    std::lock_guard<std::mutex> lock(g_matmul_choice_mutex);
    auto it = g_matmul_choice_cache.find(key);
    if (it != g_matmul_choice_cache.end())
      return it->second;
  }

  if (!env_flag_enabled("VULTORCH_AUTOTUNE", true))
    return auto_candidates.back();

  MatmulChoice best = auto_candidates.front();
  double best_ms = 0.0;

  for (const auto &candidate : auto_candidates) {
    // Fixed *time* budget of warmup: the first runs of a kernel happen at
    // low clocks while the GPU ramps, and the operands are still cold.
    const auto warm_start = std::chrono::steady_clock::now();
    int warm_iters = 0;
    while (warm_iters < 64 && std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - warm_start)
                                      .count() < 15.0) {
      run(candidate.shader, candidate.tile);
      dev->synchronize();
      ++warm_iters;
    }

    double candidate_ms = std::numeric_limits<double>::max();
    for (int rep = 0; rep < 3; ++rep) {
      const auto start = std::chrono::steady_clock::now();
      // Batch 3 dispatches per synchronize() so host/OS scheduling jitter
      // on a single short dispatch cannot skew the winner.
      for (int sub = 0; sub < 3; ++sub) {
        run(candidate.shader, candidate.tile);
      }
      dev->synchronize();
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count() /
                        3.0;
      candidate_ms = std::min(candidate_ms, ms);
    }

    if (best_ms == 0.0 || candidate_ms < best_ms) {
      best_ms = candidate_ms;
      best = candidate;
    }
  }

  {
    std::lock_guard<std::mutex> lock(g_matmul_choice_mutex);
    g_matmul_choice_cache[key] = best;
  }
  return best;
}

} // namespace

std::shared_ptr<NativeTensor>
NativeTensor::matmul(std::shared_ptr<NativeTensor> other, bool trans_a,
                     bool trans_b, std::shared_ptr<NativeTensor> bias,
                     uint32_t epilogue) {
  if (device_id != other->device_id) {
    throw std::runtime_error(
        "Matmul requires both tensors on the same device!");
  }
  if (shape.size() < 2 || other->shape.size() < 2) {
    throw std::runtime_error("Matmul requires at least 2-dimensional tensors");
  }

  ScalarType common = promote_dtype(dtype, other->dtype);
  if (bias != nullptr) {
    common = promote_dtype(common, bias->dtype);
  }

  if (device_id >= 0 && common != ScalarType::Float32 &&
      common != ScalarType::Float16) {
    auto a32 = cast(ScalarType::Float32);
    auto b32 = other->cast(ScalarType::Float32);
    auto bias32 = bias ? bias->cast(ScalarType::Float32) : nullptr;
    return a32->matmul(b32, trans_a, trans_b, bias32, epilogue)->cast(common);
  }

  const int64_t a_rows = shape[shape.size() - 2];
  const int64_t a_cols = shape[shape.size() - 1];
  const int64_t b_rows = other->shape[other->shape.size() - 2];
  const int64_t b_cols = other->shape[other->shape.size() - 1];

  uint32_t M = static_cast<uint32_t>(trans_a ? a_cols : a_rows);
  uint32_t K1 = static_cast<uint32_t>(trans_a ? a_rows : a_cols);
  uint32_t K2 = static_cast<uint32_t>(trans_b ? b_cols : b_rows);
  uint32_t N = static_cast<uint32_t>(trans_b ? b_rows : b_cols);

  if (K1 != K2) {
    throw std::runtime_error("Inner dimensions must match for matmul: " +
                             std::to_string(K1) + " vs " + std::to_string(K2));
  }

  if (bias != nullptr) {
    if (bias->device_id != device_id) {
      throw std::runtime_error("Matmul bias must be on the same device");
    }
    if (bias->numel != N) {
      throw std::runtime_error("Matmul bias must be a tensor of length N");
    }
  }

  const bool b_broadcast = (other->shape.size() == 2);
  if (!b_broadcast) {
    if (shape.size() != other->shape.size()) {
      throw std::runtime_error(
          "Matmul batch rank mismatch: " + std::to_string(shape.size()) +
          " vs " + std::to_string(other->shape.size()));
    }
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
      if (shape[i] != other->shape[i]) {
        throw std::runtime_error("Matmul batch dimensions must match");
      }
    }
  }

  size_t batch_count = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch_count *= static_cast<size_t>(shape[i]);
  }

  std::vector<int64_t> out_shape = shape;
  out_shape[out_shape.size() - 2] = M;
  out_shape[out_shape.size() - 1] = N;

  auto is_uniform_batch_strides = [](const NativeTensor &t) {
    if (!all_strides_non_negative(t.strides))
      return false;
    if (t.shape.size() <= 3)
      return true;
    for (size_t i = 0; i + 3 < t.shape.size(); ++i) {
      if (t.strides[i] != t.shape[i + 1] * t.strides[i + 1]) {
        return false;
      }
    }
    return true;
  };

  auto a_in = (dtype == common) ? shared_from_this() : cast(common);
  auto b_in = (other->dtype == common) ? other : other->cast(common);
  auto bias_in =
      bias ? ((bias->dtype == common) ? bias : bias->cast(common)) : nullptr;

  if (!is_uniform_batch_strides(*a_in))
    a_in = a_in->force_contiguous();
  if (!is_uniform_batch_strides(*b_in))
    b_in = b_in->force_contiguous();
  if (bias_in && !bias_in->is_dense_zero_offset())
    bias_in = bias_in->force_contiguous();

  const int64_t sa0 = a_in->strides[a_in->shape.size() - 2];
  const int64_t sa1 = a_in->strides[a_in->shape.size() - 1];
  const int64_t sb0 = b_in->strides[b_in->shape.size() - 2];
  const int64_t sb1 = b_in->strides[b_in->shape.size() - 1];

  const uint32_t strideA_m = static_cast<uint32_t>(trans_a ? sa1 : sa0);
  const uint32_t strideA_k = static_cast<uint32_t>(trans_a ? sa0 : sa1);
  const uint32_t strideB_k = static_cast<uint32_t>(trans_b ? sb1 : sb0);
  const uint32_t strideB_n = static_cast<uint32_t>(trans_b ? sb0 : sb1);
  const uint32_t strideA_b =
      (a_in->shape.size() == 2)
          ? 0u
          : static_cast<uint32_t>(a_in->strides[a_in->shape.size() - 3]);
  const uint32_t strideB_b =
      b_broadcast ? 0u
                  : ((b_in->shape.size() == 2)
                         ? 0u
                         : static_cast<uint32_t>(
                               b_in->strides[b_in->shape.size() - 3]));

  if (device_id >= 0) {
    if (common == ScalarType::Float16) {
      auto dev = VulkanManager::instance_ref().get_device(
          static_cast<uint32_t>(device_id));
      if (!dev->capabilities.float16_arithmetic ||
          !dev->capabilities.storage_16bit) {
        auto a32 = a_in->cast(ScalarType::Float32);
        auto b32 = b_in->cast(ScalarType::Float32);
        auto bias32 = bias_in ? bias_in->cast(ScalarType::Float32) : nullptr;
        return a32->matmul(b32, trans_a, trans_b, bias32)
            ->cast(ScalarType::Float16);
      }

      auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                                ScalarType::Float16);
      if (out->numel == 0)
        return out;

      struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t batch_count;
        uint32_t strideA_b;
        uint32_t strideA_m;
        uint32_t strideA_k;
        uint32_t strideB_b;
        uint32_t strideB_k;
        uint32_t strideB_n;
        uint32_t strideC_b;
        uint32_t strideC_m;
        uint32_t strideC_n;
        uint32_t has_bias;
        uint32_t offsetA;
        uint32_t offsetB;
        uint32_t offsetC;
      } pc{};

      pc.M = M;
      pc.N = N;
      pc.K = K1;
      pc.batch_count = static_cast<uint32_t>(batch_count);
      pc.strideA_b = strideA_b;
      pc.strideA_m = strideA_m;
      pc.strideA_k = strideA_k;
      pc.strideB_b = strideB_b;
      pc.strideB_k = strideB_k;
      pc.strideB_n = strideB_n;
      pc.strideC_b = static_cast<uint32_t>(M * N);
      pc.strideC_m = static_cast<uint32_t>(N);
      pc.strideC_n = 1u;
      pc.has_bias = (bias_in != nullptr) ? 1u : 0u;
      pc.offsetA = static_cast<uint32_t>(a_in->storage_offset);
      pc.offsetB = static_cast<uint32_t>(b_in->storage_offset);
      pc.offsetC = 0u;

      std::shared_ptr<VulkanBuffer> bias_buf =
          bias_in ? bias_in->gpu_buf() : out->gpu_buf();
      const bool can_coop = dev->capabilities.fp16_coopmat_16x16 &&
                            (M % 16u == 0u) && (N % 16u == 0u) &&
                            (K1 % 16u == 0u) && strideA_k == 1u &&
                            strideB_n == 1u;

      if (can_coop && epilogue == 0u) {
        VulkanPipelineManager::instance().dispatch(
            device_id, "matmul_coopmat_fp16",
            {a_in->gpu_buf(), b_in->gpu_buf(), out->gpu_buf(), bias_buf}, &pc,
            sizeof(pc), N / 16u, M / 16u, static_cast<uint32_t>(batch_count));
      } else {
        const uint32_t gx = (N + 63u) / 64u;
        const uint32_t gy = (M + 63u) / 64u;
        VulkanPipelineManager::instance().dispatch(
            device_id, "gemm_fp16",
            {a_in->gpu_buf(), b_in->gpu_buf(), out->gpu_buf(), bias_buf}, &pc,
            sizeof(pc), gx, gy, static_cast<uint32_t>(batch_count));
        if (epilogue == 1u) {
          out = out->unary_op(10u);
        } else if (epilogue == 2u) {
          out = out->unary_op(11u);
        } else if (epilogue == 3u) {
          out = out->unary_op(12u);
        }
      }
      return out;
    }

    // Native GPU Float32 GEMM
    auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                              ScalarType::Float32);
    if (out->numel == 0)
      return out;

    struct {
      uint32_t M;
      uint32_t N;
      uint32_t K;
      uint32_t batch_count;
      uint32_t strideA_b;
      uint32_t strideA_m;
      uint32_t strideA_k;
      uint32_t strideB_b;
      uint32_t strideB_k;
      uint32_t strideB_n;
      uint32_t strideC_b;
      uint32_t strideC_m;
      uint32_t strideC_n;
      uint32_t has_bias;
      uint32_t offsetA;
      uint32_t offsetB;
      uint32_t offsetC;
    } pc{};

    pc.M = M;
    pc.N = N;
    pc.K = K1;
    pc.batch_count = static_cast<uint32_t>(batch_count);
    pc.strideA_b = strideA_b;
    pc.strideA_m = strideA_m;
    pc.strideA_k = strideA_k;
    pc.strideB_b = strideB_b;
    pc.strideB_k = strideB_k;
    pc.strideB_n = strideB_n;
    pc.strideC_b = static_cast<uint32_t>(M * N);
    pc.strideC_m = static_cast<uint32_t>(N);
    pc.strideC_n = 1u;
    pc.has_bias = (bias_in != nullptr) ? 1u : 0u;
    pc.offsetA = static_cast<uint32_t>(a_in->storage_offset);
    pc.offsetB = static_cast<uint32_t>(b_in->storage_offset);
    pc.offsetC = 0u;

    std::shared_ptr<VulkanBuffer> bias_buf =
        bias_in ? bias_in->gpu_buf() : out->gpu_buf();

    auto run_variant = [&](const char *shader, uint32_t tile) {
      const uint32_t group_x = (N + tile - 1u) / tile;
      const uint32_t group_y = (M + tile - 1u) / tile;
      VulkanPipelineManager::instance().dispatch(
          device_id, shader,
          {a_in->gpu_buf(), b_in->gpu_buf(), out->gpu_buf(), bias_buf}, &pc,
          sizeof(pc), group_x, group_y, static_cast<uint32_t>(batch_count));
    };

    if (epilogue == 1u) {
      run_variant("gemm_t64_k16_r4_relu", 64u);
      return out;
    } else if (epilogue == 2u) {
      run_variant("gemm_t64_k16_r4_gelu", 64u);
      return out;
    } else if (epilogue == 3u) {
      run_variant("gemm_t64_k16_r4_silu", 64u);
      return out;
    }

    const bool eff_ta = trans_a || (strideA_k != 1u);
    const bool eff_tb = trans_b || (strideB_n != 1u);
    const MatmulChoice choice =
        select_matmul_shader(static_cast<uint32_t>(device_id), M, N, K1, eff_ta,
                             eff_tb, run_variant);
    run_variant(choice.shader, choice.tile);
    return out;
  }

  // CPU MatMul
  auto out = std::make_shared<NativeTensor>(out_shape, -1, common);
  if (out->numel == 0)
    return out;

  auto apply_epilogue_cpu = [&](float val) -> float {
    if (epilogue == 1u)
      return std::max(val, 0.0f);
    if (epilogue == 2u)
      return apply_unary_op(11u, val, 0.0f, 0.0f);
    if (epilogue == 3u)
      return apply_unary_op(12u, val, 0.0f, 0.0f);
    return val;
  };

  if (common == ScalarType::Float32) {
    const float *a_ptr = a_in->cpu_f32();
    const float *b_ptr = b_in->cpu_f32();
    const float *bias_ptr = bias_in ? bias_in->cpu_f32() : nullptr;
    float *c_ptr = out->cpu_f32();
    for (size_t b = 0; b < batch_count; ++b) {
      size_t offA = a_in->storage_offset + b * strideA_b;
      size_t offB = b_in->storage_offset + b * strideB_b;
      size_t offC = b * M * N;
      for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
          float sum = bias_ptr ? bias_ptr[n] : 0.0f;
          for (uint32_t k = 0; k < K1; ++k) {
            sum += a_ptr[offA + m * strideA_m + k * strideA_k] *
                   b_ptr[offB + k * strideB_k + n * strideB_n];
          }
          c_ptr[offC + m * N + n] = apply_epilogue_cpu(sum);
        }
      }
    }
    return out;
  }

  const uint8_t *a_raw = a_in->cpu_raw_ptr();
  const uint8_t *b_raw = b_in->cpu_raw_ptr();
  const uint8_t *bias_raw = bias_in ? bias_in->cpu_raw_ptr() : nullptr;
  uint8_t *c_raw = out->cpu_raw_ptr();
  for (size_t b = 0; b < batch_count; ++b) {
    size_t offA = a_in->storage_offset + b * strideA_b;
    size_t offB = b_in->storage_offset + b * strideB_b;
    size_t offC = b * M * N;
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        double sum =
            bias_raw ? read_scalar_as_double(bias_raw, n, common) : 0.0;
        for (uint32_t k = 0; k < K1; ++k) {
          sum += read_scalar_as_double(
                     a_raw, offA + m * strideA_m + k * strideA_k, common) *
                 read_scalar_as_double(
                     b_raw, offB + k * strideB_k + n * strideB_n, common);
        }
        double res =
            static_cast<double>(apply_epilogue_cpu(static_cast<float>(sum)));
        write_scalar_from_double(c_raw, offC + m * N + n, common, res);
      }
    }
  }
  return out;
}


} // namespace vultorch
