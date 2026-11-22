#include "tensor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

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

// Single source of truth for elementwise op semantics. Both the CPU fallback
// and the shader paths use the same op numbering (see
// shaders/elementwise_binary_broadcast.comp).
inline float apply_binary_op(uint32_t op, float a, float b) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  constexpr float kGeluCoeff = 0.044715f;
  switch (op) {
  case 0:
    return a + b;
  case 1:
    return a - b;
  case 2:
    return a * b;
  case 3:
    return a / b;
  case 4:
    return std::pow(a, b);
  case 5:
    return std::max(a, b);
  case 6:
    return std::min(a, b);
  case 7:
    return (a > b) ? 1.0f : 0.0f;
  case 8:
    return (a < b) ? 1.0f : 0.0f;
  case 9:
    return (a == b) ? 1.0f : 0.0f;
  case 10:
    return (b > 0.0f) ? a : 0.0f;
  case 11: {
    const float x2 = b * b;
    const float inner = kSqrt2OverPi * (b + kGeluCoeff * x2 * b);
    const float t = std::tanh(inner);
    const float dt =
        (1.0f - t * t) * kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x2);
    return a * (0.5f * (1.0f + t) + 0.5f * b * dt);
  }
  case 12: {
    const float s = 1.0f / (1.0f + std::exp(-b));
    return a * s * (1.0f + b * (1.0f - s));
  }
  case 13:
    return (b > 0.0f) ? a : ((b < 0.0f) ? -a : 0.0f);
  case 14:
    return (a == b) ? 1.0f : 0.0f;
  case 15:
    return (a >= b) ? 1.0f : 0.0f;
  case 16:
    return (a <= b) ? 1.0f : 0.0f;
  default:
    throw std::runtime_error("Unsupported binary op_type: " +
                             std::to_string(op));
  }
}

// Same op numbering as shaders/elementwise_unary_scalar.comp. Anything missing
// here silently degrades to identity, which is how `gelu`/`silu` used to be a
// no-op on the CPU backend while working on the GPU.
inline float apply_unary_op(uint32_t op, float x, float p1, float p2) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  constexpr float kGeluCoeff = 0.044715f;

  switch (op) {
  case 0:
    return -x;
  case 1:
    return std::exp(x);
  case 2:
    return std::log(x);
  case 3:
    return std::sqrt(x);
  case 4:
    return 1.0f / std::sqrt(x);
  case 5:
    return std::abs(x);
  case 6:
    return std::sin(x);
  case 7:
    return std::cos(x);
  case 8:
    return std::tanh(x);
  case 9:
    return 1.0f / (1.0f + std::exp(-x));
  case 10:
    return std::max(x, 0.0f);
  case 11: {
    const float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
  }
  case 12:
    return x / (1.0f + std::exp(-x));
  case 13:
    return std::min(std::max(x, p1), p2);
  case 14:
    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
  case 15:
    return x * p1 + p2;
  default:
    throw std::runtime_error("Unsupported unary op_type: " +
                             std::to_string(op));
  }
}

// Broadcasting plan shared by the CPU fallback and the GPU broadcast kernel.
// Dimensions are right-aligned; a broadcast dimension gets a stride of 0 so
// the same plan can be used for index arithmetic and for push constants.
struct BroadcastPlan {
  std::vector<int64_t> out_shape;
  std::vector<int64_t> strides_a;
  std::vector<int64_t> strides_b;
};

BroadcastPlan make_broadcast_plan(const std::vector<int64_t> &a_shape,
                                  const std::vector<int64_t> &a_strides,
                                  const std::vector<int64_t> &b_shape,
                                  const std::vector<int64_t> &b_strides) {
  const size_t max_ndim = std::max(a_shape.size(), b_shape.size());
  if (max_ndim > 6) {
    throw std::runtime_error("Broadcasting supports up to 6 dimensions");
  }

  BroadcastPlan plan;
  plan.out_shape.assign(max_ndim, 1);
  plan.strides_a.assign(max_ndim, 0);
  plan.strides_b.assign(max_ndim, 0);

  for (size_t k = 0; k < max_ndim; ++k) {
    const int ai = static_cast<int>(a_shape.size()) - 1 - static_cast<int>(k);
    const int bi = static_cast<int>(b_shape.size()) - 1 - static_cast<int>(k);
    const size_t oi = max_ndim - 1 - k;

    const int64_t dim_a = (ai >= 0) ? a_shape[ai] : 1;
    const int64_t dim_b = (bi >= 0) ? b_shape[bi] : 1;

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::runtime_error(
          "Cannot broadcast shapes: incompatible dimension " +
          std::to_string(dim_a) + " vs " + std::to_string(dim_b));
    }

    plan.out_shape[oi] = std::max(dim_a, dim_b);
    if (ai >= 0 && dim_a > 1)
      plan.strides_a[oi] = a_strides[ai];
    if (bi >= 0 && dim_b > 1)
      plan.strides_b[oi] = b_strides[bi];
  }
  return plan;
}

} // namespace

namespace {

bool compute_view_strides(const std::vector<int64_t> &old_shape,
                          const std::vector<int64_t> &old_strides, size_t numel,
                          const std::vector<int64_t> &new_shape,
                          std::vector<int64_t> &out_strides) {
  out_strides.assign(new_shape.size(), 0);
  if (numel == 0) {
    int64_t s = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
      out_strides[static_cast<size_t>(i)] = s;
      s *= new_shape[static_cast<size_t>(i)];
    }
    return true;
  }
  if (new_shape.empty()) {
    return numel == 1;
  }

  struct Chunk {
    int64_t size;
    int64_t stride;
  };
  std::vector<Chunk> chunks;
  for (int i = static_cast<int>(old_shape.size()) - 1; i >= 0; --i) {
    int64_t d = old_shape[static_cast<size_t>(i)];
    int64_t st = old_strides[static_cast<size_t>(i)];
    if (d == 1)
      continue;
    if (!chunks.empty() && st == chunks.back().size * chunks.back().stride) {
      chunks.back().size *= d;
    } else {
      chunks.push_back({d, st});
    }
  }
  if (chunks.empty()) {
    chunks.push_back({1, 1});
  }

  size_t chunk_idx = 0;
  int64_t chunk_rem = chunks[0].size;
  int64_t cur_stride = chunks[0].stride;

  for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
    int64_t d = new_shape[static_cast<size_t>(i)];
    if (d == 1) {
      if (i + 1 < static_cast<int>(new_shape.size())) {
        out_strides[static_cast<size_t>(i)] =
            out_strides[static_cast<size_t>(i + 1)] *
            new_shape[static_cast<size_t>(i + 1)];
      } else {
        out_strides[static_cast<size_t>(i)] = 1;
      }
      continue;
    }
    while (chunk_rem == 1) {
      ++chunk_idx;
      if (chunk_idx >= chunks.size())
        return false;
      chunk_rem = chunks[chunk_idx].size;
      cur_stride = chunks[chunk_idx].stride;
    }
    if (chunk_rem % d != 0) {
      return false;
    }
    out_strides[static_cast<size_t>(i)] = cur_stride;
    cur_stride *= d;
    chunk_rem /= d;
  }
  return chunk_rem == 1 && (chunk_idx + 1 == chunks.size());
}

void advance_odometer(size_t ndim, const std::vector<int64_t> &shape,
                      const std::vector<int64_t> &strides,
                      std::vector<int64_t> &coord, int64_t &offset_accum) {
  for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
    size_t ud = static_cast<size_t>(d);
    offset_accum += strides[ud];
    if (++coord[ud] < shape[ud]) {
      return;
    }
    offset_accum -= strides[ud] * shape[ud];
    coord[ud] = 0;
  }
}

uint32_t packed_word_count(size_t num_elems, size_t elem_bytes) {
  if (elem_bytes == 4)
    return static_cast<uint32_t>(num_elems);
  if (elem_bytes == 8)
    return static_cast<uint32_t>(num_elems * 2);
  if (elem_bytes == 2)
    return static_cast<uint32_t>((num_elems + 1) / 2);
  return static_cast<uint32_t>((num_elems + 3) / 4);
}

bool all_strides_non_negative(const std::vector<int64_t> &st) {
  for (int64_t v : st) {
    if (v < 0)
      return false;
  }
  return true;
}

} // namespace

NativeTensor::NativeTensor(const std::vector<int64_t> &s, int dev_id,
                           ScalarType dt)
    : shape(s), device_id(dev_id), dtype(dt), storage_offset(0) {
  numel = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      throw std::runtime_error("Negative dimension in tensor shape");
    }
    numel *= static_cast<size_t>(d);
  }
  compute_default_strides();

  storage = std::make_shared<TensorStorage>();
  storage->id = TensorStorage::next_id();

  const size_t total_bytes = numel * element_size(dtype);
  if (device_id >= 0) {
    VkDeviceSize alloc_bytes =
        total_bytes == 0 ? 4 : static_cast<VkDeviceSize>(total_bytes);
    storage->gpu_buffer = VulkanMemoryManager::instance().allocate(
        static_cast<uint32_t>(device_id), alloc_bytes);
  } else {
    storage->cpu_bytes.resize(total_bytes, 0);
  }
}

NativeTensor::NativeTensor(const std::vector<int64_t> &view_shape,
                           const std::vector<int64_t> &view_strides,
                           size_t offset_elems,
                           std::shared_ptr<TensorStorage> shared_storage,
                           int dev_id, ScalarType dt)
    : shape(view_shape), strides(view_strides), device_id(dev_id), dtype(dt),
      storage_offset(offset_elems), storage(std::move(shared_storage)) {
  numel = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      throw std::runtime_error("Negative dimension in view shape");
    }
    numel *= static_cast<size_t>(d);
  }
}

std::shared_ptr<NativeTensor>
NativeTensor::create(const std::vector<int64_t> &shape, int dev_id,
                     ScalarType dt) {
  return std::make_shared<NativeTensor>(shape, dev_id, dt);
}

// Metadata-only construction: shape/strides are set, storage attached by
// caller.
NativeTensor::NativeTensor(const std::vector<int64_t> &s, int dev_id,
                           MetadataOnly)
    : shape(s), device_id(dev_id) {
  numel = 1;
  for (int64_t d : shape) {
    numel *= static_cast<size_t>(d);
  }
  compute_default_strides();
}

uintptr_t NativeTensor::data_ptr() const {
  const size_t byte_off = storage_offset * itemsize();
  if (device_id < 0) {
    const uint8_t *ptr = cpu_raw_ptr();
    return ptr ? (reinterpret_cast<uintptr_t>(ptr) + byte_off) : 0;
  }
  auto buf = gpu_buf();
  if (!buf)
    return 0;
  return reinterpret_cast<uintptr_t>(buf->buffer) +
         static_cast<uintptr_t>(buf->arena_offset) +
         static_cast<uintptr_t>(byte_off);
}

std::shared_ptr<NativeTensor>
NativeTensor::strided_copy(const std::vector<int64_t> &out_shape,
                           const std::vector<int64_t> &src_strides) const {
  if (out_shape.size() != src_strides.size()) {
    throw std::runtime_error("strided_copy: shape and stride rank mismatch");
  }

  auto out = std::make_shared<NativeTensor>(out_shape, device_id, dtype);
  if (out->numel == 0)
    return out;

  const size_t elem_sz = itemsize();

  if (device_id >= 0 && out_shape.size() <= 6 &&
      all_strides_non_negative(src_strides)) {
    uint32_t ndim = static_cast<uint32_t>(out_shape.size());
    struct {
      uint32_t total_elements;
      uint32_t ndim;
      uint32_t src_offset;
      uint32_t elem_bytes;
      uint32_t shape[6];
      uint32_t src_stride[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(out->numel);
    pc.ndim = ndim;
    pc.src_offset = static_cast<uint32_t>(storage_offset);
    pc.elem_bytes = static_cast<uint32_t>(elem_sz);
    for (uint32_t i = 0; i < ndim; ++i) {
      pc.shape[i] = static_cast<uint32_t>(out_shape[i]);
      pc.src_stride[i] = static_cast<uint32_t>(src_strides[i]);
    }

    uint32_t words = packed_word_count(out->numel, elem_sz);
    uint32_t groups = (words + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(device_id, "copy_strided",
                                               {gpu_buf(), out->gpu_buf()}, &pc,
                                               sizeof(pc), groups);
    return out;
  }

  // CPU strided copy (or GPU fallback when rank > 6 / negative stride)
  if (device_id >= 0) {
    auto cpu_src = std::make_shared<NativeTensor>(out_shape, -1, dtype);
    // Copy whole underlying GPU storage to host once, strided-copy on CPU,
    // upload.
    auto buf = gpu_buf();
    std::vector<uint8_t> raw_storage(static_cast<size_t>(buf->size));
    VulkanMemoryManager::instance().copy_device_to_host(
        device_id, buf, raw_storage.data(), buf->size);
    const size_t ndim = out_shape.size();
    std::vector<int64_t> coord(ndim, 0);
    int64_t src_idx = static_cast<int64_t>(storage_offset);
    uint8_t *dst_raw = cpu_src->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      std::memcpy(dst_raw + n * elem_sz,
                  raw_storage.data() + static_cast<size_t>(src_idx) * elem_sz,
                  elem_sz);
      advance_odometer(ndim, out_shape, src_strides, coord, src_idx);
    }
    return cpu_src->to_device(device_id);
  }

  const size_t ndim = out_shape.size();
  std::vector<int64_t> coord(ndim, 0);
  int64_t src_idx = static_cast<int64_t>(storage_offset);
  const uint8_t *src_raw = cpu_raw_ptr();
  uint8_t *dst_raw = out->cpu_raw_ptr();
  for (size_t n = 0; n < out->numel; ++n) {
    std::memcpy(dst_raw + n * elem_sz,
                src_raw + static_cast<size_t>(src_idx) * elem_sz, elem_sz);
    advance_odometer(ndim, out_shape, src_strides, coord, src_idx);
  }
  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::contiguous() const {
  if (is_contiguous()) {
    return std::const_pointer_cast<NativeTensor>(shared_from_this());
  }
  return strided_copy(shape, strides);
}

std::shared_ptr<NativeTensor> NativeTensor::force_contiguous() const {
  if (is_dense_zero_offset()) {
    return std::const_pointer_cast<NativeTensor>(shared_from_this());
  }
  if (is_contiguous() && device_id >= 0 && numel > 0) {
    auto out = std::make_shared<NativeTensor>(shape, device_id, dtype);
    VulkanMemoryManager::instance().copy_buffer_to_buffer(
        device_id, gpu_buf(), out->gpu_buf(), nbytes(),
        storage_offset * itemsize(), 0);
    return out;
  }
  return strided_copy(shape, strides);
}

std::shared_ptr<NativeTensor>
NativeTensor::cast(ScalarType target_dtype) const {
  if (dtype == target_dtype) {
    return std::const_pointer_cast<NativeTensor>(shared_from_this());
  }

  auto out = std::make_shared<NativeTensor>(shape, device_id, target_dtype);
  if (numel == 0)
    return out;

  if (device_id >= 0) {
    std::shared_ptr<const NativeTensor> src_t = shared_from_this();
    bool flat_contig = is_contiguous();
    if (!flat_contig &&
        (shape.size() > 6 || !all_strides_non_negative(strides))) {
      src_t = force_contiguous();
      flat_contig = true;
    }

    struct {
      uint32_t total_elements;
      uint32_t src_dtype;
      uint32_t dst_dtype;
      uint32_t src_offset;
      uint32_t ndim;
      uint32_t shape[6];
      uint32_t src_stride[6];
    } pc{};
    pc.total_elements = static_cast<uint32_t>(numel);
    pc.src_dtype = static_cast<uint32_t>(src_t->dtype);
    pc.dst_dtype = static_cast<uint32_t>(target_dtype);
    pc.src_offset = static_cast<uint32_t>(src_t->storage_offset);

    if (flat_contig) {
      pc.ndim = 1u;
      pc.shape[0] = static_cast<uint32_t>(numel);
      pc.src_stride[0] = 1u;
    } else {
      pc.ndim = static_cast<uint32_t>(src_t->shape.size());
      for (uint32_t i = 0; i < pc.ndim; ++i) {
        pc.shape[i] = static_cast<uint32_t>(src_t->shape[i]);
        pc.src_stride[i] = static_cast<uint32_t>(src_t->strides[i]);
      }
    }

    uint32_t words = packed_word_count(numel, element_size(target_dtype));
    uint32_t groups = (words + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "cast_dtype", {src_t->gpu_buf(), out->gpu_buf()}, &pc,
        sizeof(pc), groups);
    return out;
  }

  // CPU cast (exact integer path for Int32/Int64/Bool/UInt8 ->
  // Int64/Int32/Bool/UInt8)
  const size_t ndim = shape.size();
  std::vector<int64_t> coord(ndim, 0);
  int64_t src_idx = static_cast<int64_t>(storage_offset);
  const uint8_t *src_raw = cpu_raw_ptr();
  uint8_t *dst_raw = out->cpu_raw_ptr();

  for (size_t n = 0; n < numel; ++n) {
    size_t s = static_cast<size_t>(src_idx);
    if (target_dtype == ScalarType::Int64 &&
        (dtype == ScalarType::Int32 || dtype == ScalarType::Bool ||
         dtype == ScalarType::UInt8)) {
      int64_t iv = 0;
      if (dtype == ScalarType::Int32) {
        int32_t v32 = 0;
        std::memcpy(&v32, src_raw + s * 4, 4);
        iv = static_cast<int64_t>(v32);
      } else if (dtype == ScalarType::Bool) {
        iv = (src_raw[s] != 0) ? 1 : 0;
      } else {
        iv = static_cast<int64_t>(src_raw[s]);
      }
      std::memcpy(dst_raw + n * 8, &iv, 8);
    } else if (dtype == ScalarType::Int64 &&
               (target_dtype == ScalarType::Int32 ||
                target_dtype == ScalarType::Bool ||
                target_dtype == ScalarType::UInt8)) {
      int64_t iv = 0;
      std::memcpy(&iv, src_raw + s * 8, 8);
      if (target_dtype == ScalarType::Int32) {
        int32_t v32 = static_cast<int32_t>(iv);
        std::memcpy(dst_raw + n * 4, &v32, 4);
      } else if (target_dtype == ScalarType::Bool) {
        dst_raw[n] = (iv != 0) ? 1u : 0u;
      } else {
        dst_raw[n] = static_cast<uint8_t>(iv);
      }
    } else {
      double val = read_scalar_as_double(src_raw, s, dtype);
      write_scalar_from_double(dst_raw, n, target_dtype, val);
    }
    advance_odometer(ndim, shape, strides, coord, src_idx);
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::as_strided(const std::vector<int64_t> &new_shape,
                         const std::vector<int64_t> &new_strides,
                         int64_t new_offset) const {
  if (new_shape.size() != new_strides.size()) {
    throw std::runtime_error("as_strided: shape and stride rank mismatch");
  }
  size_t eff_offset =
      (new_offset < 0) ? storage_offset : static_cast<size_t>(new_offset);
  size_t view_numel = 1;
  int64_t max_rel = 0;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] < 0) {
      throw std::runtime_error("as_strided: negative shape dimension");
    }
    if (new_strides[i] < 0) {
      throw std::runtime_error("as_strided: negative stride not supported");
    }
    view_numel *= static_cast<size_t>(new_shape[i]);
    if (new_shape[i] > 0) {
      max_rel += (new_shape[i] - 1) * new_strides[i];
    }
  }
  if (view_numel > 0) {
    size_t cap_elems = (device_id >= 0)
                           ? (gpu_buf()->size / itemsize())
                           : (storage->cpu_bytes.size() / itemsize());
    if (eff_offset + static_cast<size_t>(max_rel) >= cap_elems) {
      throw std::runtime_error("as_strided: view extends past end of storage");
    }
  }
  return std::make_shared<NativeTensor>(new_shape, new_strides, eff_offset,
                                        storage, device_id, dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::transpose_view(int64_t dim0,
                                                           int64_t dim1) const {
  const int64_t ndim = static_cast<int64_t>(shape.size());
  if (dim0 < 0)
    dim0 += ndim;
  if (dim1 < 0)
    dim1 += ndim;
  if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
    throw std::runtime_error("transpose: dimension out of range");
  }
  std::vector<int64_t> new_shape = shape;
  std::vector<int64_t> new_strides = strides;
  std::swap(new_shape[static_cast<size_t>(dim0)],
            new_shape[static_cast<size_t>(dim1)]);
  std::swap(new_strides[static_cast<size_t>(dim0)],
            new_strides[static_cast<size_t>(dim1)]);
  return std::make_shared<NativeTensor>(new_shape, new_strides, storage_offset,
                                        storage, device_id, dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::permute_view(const std::vector<int64_t> &dims) const {
  const size_t ndim = shape.size();
  if (dims.size() != ndim) {
    throw std::runtime_error("permute: number of dims must match tensor rank");
  }
  std::vector<bool> seen(ndim, false);
  std::vector<int64_t> new_shape(ndim);
  std::vector<int64_t> new_strides(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t d = dims[i];
    if (d < 0)
      d += static_cast<int64_t>(ndim);
    if (d < 0 || d >= static_cast<int64_t>(ndim) ||
        seen[static_cast<size_t>(d)]) {
      throw std::runtime_error("permute: invalid or duplicate dimension");
    }
    seen[static_cast<size_t>(d)] = true;
    new_shape[i] = shape[static_cast<size_t>(d)];
    new_strides[i] = strides[static_cast<size_t>(d)];
  }
  return std::make_shared<NativeTensor>(new_shape, new_strides, storage_offset,
                                        storage, device_id, dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::expand_view(const std::vector<int64_t> &new_shape) const {
  if (new_shape.size() < shape.size()) {
    throw std::runtime_error(
        "expand: target rank cannot be smaller than tensor rank");
  }
  const size_t out_ndim = new_shape.size();
  const size_t lead = out_ndim - shape.size();
  std::vector<int64_t> resolved_shape(out_ndim);
  std::vector<int64_t> new_strides(out_ndim);

  for (size_t i = 0; i < out_ndim; ++i) {
    int64_t target_d = new_shape[i];
    if (i < lead) {
      if (target_d < 0) {
        throw std::runtime_error(
            "expand: cannot use -1 for new leading dimension");
      }
      resolved_shape[i] = target_d;
      new_strides[i] = 0;
    } else {
      size_t j = i - lead;
      int64_t src_d = shape[j];
      if (target_d == -1)
        target_d = src_d;
      if (target_d == src_d) {
        resolved_shape[i] = src_d;
        new_strides[i] = strides[j];
      } else if (src_d == 1 && target_d >= 0) {
        resolved_shape[i] = target_d;
        new_strides[i] = 0;
      } else {
        throw std::runtime_error(
            "expand: expanded size must match existing size or be 1");
      }
    }
  }
  return std::make_shared<NativeTensor>(
      resolved_shape, new_strides, storage_offset, storage, device_id, dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::slice_view(int64_t dim,
                                                       int64_t start,
                                                       int64_t stop,
                                                       int64_t step) const {
  const int64_t ndim = static_cast<int64_t>(shape.size());
  if (dim < 0)
    dim += ndim;
  if (dim < 0 || dim >= ndim) {
    throw std::runtime_error("slice_view: dimension out of range");
  }
  if (step <= 0) {
    throw std::runtime_error("slice_view: step must be > 0");
  }
  const size_t udim = static_cast<size_t>(dim);
  const int64_t dim_len = shape[udim];
  if (start < 0)
    start += dim_len;
  if (stop < 0)
    stop += dim_len;
  start = std::max<int64_t>(0, std::min<int64_t>(start, dim_len));
  stop = std::max<int64_t>(start, std::min<int64_t>(stop, dim_len));

  const int64_t slice_len =
      (stop > start) ? ((stop - start + step - 1) / step) : 0;
  std::vector<int64_t> new_shape = shape;
  std::vector<int64_t> new_strides = strides;
  new_shape[udim] = slice_len;
  new_strides[udim] *= step;
  const size_t new_offset =
      storage_offset +
      static_cast<size_t>(start) * static_cast<size_t>(strides[udim]);
  return std::make_shared<NativeTensor>(new_shape, new_strides, new_offset,
                                        storage, device_id, dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::reshape_view(const std::vector<int64_t> &new_shape) const {
  std::vector<int64_t> resolved = new_shape;
  int infer_idx = -1;
  size_t known_prod = 1;
  for (size_t i = 0; i < resolved.size(); ++i) {
    if (resolved[i] == -1) {
      if (infer_idx >= 0) {
        throw std::runtime_error("reshape_view: only one -1 dimension allowed");
      }
      infer_idx = static_cast<int>(i);
    } else if (resolved[i] < 0) {
      throw std::runtime_error("reshape_view: negative dimension size");
    } else {
      known_prod *= static_cast<size_t>(resolved[i]);
    }
  }
  if (infer_idx >= 0) {
    if (known_prod == 0 || numel % known_prod != 0) {
      throw std::runtime_error("reshape_view: cannot infer dimension size");
    }
    resolved[static_cast<size_t>(infer_idx)] =
        static_cast<int64_t>(numel / known_prod);
  }

  size_t new_numel = 1;
  for (int64_t d : resolved)
    new_numel *= static_cast<size_t>(d);
  if (new_numel != numel) {
    throw std::runtime_error("reshape_view: element count mismatch");
  }

  std::vector<int64_t> new_strides;
  if (compute_view_strides(shape, strides, numel, resolved, new_strides)) {
    return std::make_shared<NativeTensor>(resolved, new_strides, storage_offset,
                                          storage, device_id, dtype);
  }

  auto dense = force_contiguous();
  std::vector<int64_t> dense_strides(resolved.size(), 1);
  int64_t st = 1;
  for (int i = static_cast<int>(resolved.size()) - 1; i >= 0; --i) {
    dense_strides[static_cast<size_t>(i)] = st;
    st *= resolved[static_cast<size_t>(i)];
  }
  return std::make_shared<NativeTensor>(resolved, dense_strides, 0,
                                        dense->storage, device_id, dtype);
}

void NativeTensor::copy_from_native(const std::shared_ptr<NativeTensor> &src) {
  if (src.get() == this)
    return;
  if (src->numel != numel) {
    throw std::runtime_error("copy_from_native: element count mismatch");
  }
  if (src->device_id != device_id) {
    throw std::runtime_error(
        "copy_from_native requires both tensors on the same device");
  }
  if (!is_dense_zero_offset()) {
    throw std::runtime_error("copy_from_native: destination must be dense "
                             "contiguous with zero offset");
  }
  auto src_ready = (src->dtype == dtype) ? src->force_contiguous()
                                         : src->cast(dtype)->force_contiguous();

  if (device_id >= 0) {
    VulkanMemoryManager::instance().copy_buffer_to_buffer(
        device_id, src_ready->gpu_buf(), gpu_buf(), nbytes());
  } else {
    std::memcpy(cpu_raw_ptr(), src_ready->cpu_raw_ptr(), nbytes());
  }
}

std::shared_ptr<NativeTensor> NativeTensor::slice_dim0(int64_t start,
                                                       int64_t end) const {
  if (shape.empty()) {
    throw std::runtime_error("slice_dim0 requires at least 1D tensor");
  }
  return slice_view(0, start, end, 1);
}

std::shared_ptr<NativeTensor> NativeTensor::cat_dim0(
    const std::vector<std::shared_ptr<NativeTensor>> &tensors,
    int target_device_id) {
  if (tensors.empty()) {
    throw std::runtime_error("cat_dim0 expects a non-empty list of tensors");
  }
  const auto &first = tensors.front();
  if (first->shape.empty()) {
    throw std::runtime_error("cat_dim0 requires at least 1D tensors");
  }

  int64_t total_dim0 = 0;
  ScalarType out_dt = first->dtype;
  for (const auto &t : tensors) {
    if (t->shape.size() != first->shape.size()) {
      throw std::runtime_error("cat_dim0: rank mismatch");
    }
    for (size_t d = 1; d < first->shape.size(); ++d) {
      if (t->shape[d] != first->shape[d]) {
        throw std::runtime_error("cat_dim0: non-concat dimension mismatch");
      }
    }
    out_dt = promote_dtype(out_dt, t->dtype);
    total_dim0 += t->shape[0];
  }

  std::vector<int64_t> out_shape = first->shape;
  out_shape[0] = total_dim0;
  auto out =
      std::make_shared<NativeTensor>(out_shape, target_device_id, out_dt);

  const size_t elem_sz = element_size(out_dt);
  size_t offset_elems = 0;
  for (const auto &t_raw : tensors) {
    if (t_raw->numel == 0)
      continue;
    auto t_cast = (t_raw->dtype == out_dt) ? t_raw : t_raw->cast(out_dt);
    auto t = t_cast->force_contiguous();

    const size_t bytes = t->numel * elem_sz;
    const size_t dst_off_bytes = offset_elems * elem_sz;

    if (target_device_id >= 0 && t->device_id >= 0) {
      if (t->device_id == target_device_id) {
        VulkanMemoryManager::instance().copy_buffer_to_buffer(
            target_device_id, t->gpu_buf(), out->gpu_buf(), bytes, 0,
            dst_off_bytes);
      } else {
        VulkanMemoryManager::instance().copy_cross_device(
            t->device_id, t->gpu_buf(), target_device_id, out->gpu_buf(), bytes,
            0, dst_off_bytes);
      }
    } else if (target_device_id == -1 && t->device_id == -1) {
      std::memcpy(out->cpu_raw_ptr() + dst_off_bytes, t->cpu_raw_ptr(), bytes);
    } else if (target_device_id == -1 && t->device_id >= 0) {
      t->to_raw_bytes(out->cpu_raw_ptr() + dst_off_bytes, bytes);
    } else {
      auto stage_t = t->to_device(target_device_id);
      VulkanMemoryManager::instance().copy_buffer_to_buffer(
          target_device_id, stage_t->gpu_buf(), out->gpu_buf(), bytes, 0,
          dst_off_bytes);
    }
    offset_elems += t->numel;
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::from_float_array(const float *data,
                               const std::vector<int64_t> &shape, int dev_id) {
  auto tensor =
      std::make_shared<NativeTensor>(shape, dev_id, ScalarType::Float32);
  tensor->copy_from_host(data, tensor->numel);
  return tensor;
}

std::shared_ptr<NativeTensor>
NativeTensor::from_raw_bytes(const void *data, size_t byte_count,
                             const std::vector<int64_t> &shape, int dev_id,
                             ScalarType dt) {
  auto tensor = std::make_shared<NativeTensor>(shape, dev_id, dt);
  tensor->copy_from_raw_bytes(data, byte_count);
  return tensor;
}

void NativeTensor::compute_default_strides() {
  strides.resize(shape.size());
  if (shape.empty())
    return;
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
}

bool NativeTensor::is_contiguous() const {
  if (numel == 0 || shape.empty())
    return true;
  int64_t expected = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    int64_t d = shape[static_cast<size_t>(i)];
    if (d != 1) {
      if (strides[static_cast<size_t>(i)] != expected)
        return false;
      expected *= d;
    }
  }
  return true;
}

void NativeTensor::copy_from_raw_bytes(const void *src, size_t byte_count) {
  if (!is_dense_zero_offset()) {
    throw std::runtime_error(
        "copy_from_raw_bytes requires dense zero-offset tensor");
  }
  size_t copy_bytes = std::min(byte_count, nbytes());
  if (copy_bytes == 0)
    return;
  if (device_id >= 0) {
    VulkanMemoryManager::instance().copy_host_to_device(
        device_id, src, gpu_buf(), static_cast<VkDeviceSize>(copy_bytes));
  } else {
    std::memcpy(cpu_raw_ptr(), src, copy_bytes);
  }
}

void NativeTensor::to_raw_bytes(void *dst, size_t byte_count) const {
  auto src = is_dense_zero_offset()
                 ? std::const_pointer_cast<NativeTensor>(shared_from_this())
                 : force_contiguous();
  size_t copy_bytes = std::min(byte_count, src->nbytes());
  if (copy_bytes == 0)
    return;
  if (src->device_id >= 0) {
    VulkanMemoryManager::instance().copy_device_to_host(
        src->device_id, src->gpu_buf(), dst,
        static_cast<VkDeviceSize>(copy_bytes));
  } else {
    std::memcpy(dst, src->cpu_raw_ptr(), copy_bytes);
  }
}

void NativeTensor::copy_from_host(const float *src, size_t count) {
  if (dtype == ScalarType::Float32 && is_dense_zero_offset()) {
    copy_from_raw_bytes(src, std::min(count, numel) * sizeof(float));
    return;
  }
  size_t copy_count = std::min(count, numel);
  auto f32 =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  f32->copy_from_raw_bytes(src, copy_count * sizeof(float));
  auto casted = f32->cast(dtype);
  copy_from_native(casted);
}

void NativeTensor::copy_to_host(float *dst, size_t count) {
  if (dtype == ScalarType::Float32) {
    to_raw_bytes(dst, std::min(count, numel) * sizeof(float));
    return;
  }
  auto f32 = cast(ScalarType::Float32);
  f32->to_raw_bytes(dst, std::min(count, numel) * sizeof(float));
}

std::shared_ptr<NativeTensor> NativeTensor::to_device(int target_device_id) {
  if (device_id == target_device_id) {
    return shared_from_this();
  }
  auto src = force_contiguous();
  auto target = std::make_shared<NativeTensor>(shape, target_device_id, dtype);
  if (numel == 0)
    return target;

  if (src->device_id == -1 && target_device_id >= 0) {
    target->copy_from_raw_bytes(src->cpu_raw_ptr(), src->nbytes());
  } else if (src->device_id >= 0 && target_device_id == -1) {
    src->to_raw_bytes(target->cpu_raw_ptr(), src->nbytes());
  } else {
    VulkanMemoryManager::instance().copy_cross_device(
        src->device_id, src->gpu_buf(), target_device_id, target->gpu_buf(),
        src->nbytes());
  }
  return target;
}

std::shared_ptr<NativeTensor> NativeTensor::clone() const {
  if (!is_dense_zero_offset()) {
    return force_contiguous();
  }
  auto target = std::make_shared<NativeTensor>(shape, device_id, dtype);
  if (numel == 0)
    return target;
  if (device_id >= 0) {
    VulkanMemoryManager::instance().copy_buffer_to_buffer(
        device_id, gpu_buf(), target->gpu_buf(), nbytes());
  } else {
    std::memcpy(target->cpu_raw_ptr(), cpu_raw_ptr(), nbytes());
  }
  return target;
}

void NativeTensor::fill_(float val) {
  if (!is_dense_zero_offset()) {
    throw std::runtime_error("fill_: requires contiguous zero-offset tensor");
  }
  if (numel == 0)
    return;

  if (device_id < 0) {
    if (dtype == ScalarType::Float32) {
      float *ptr = cpu_f32();
      std::fill(ptr, ptr + numel, val);
    } else {
      uint8_t *ptr = cpu_raw_ptr();
      for (size_t i = 0; i < numel; ++i) {
        write_scalar_from_double(ptr, i, dtype, static_cast<double>(val));
      }
    }
    return;
  }

  auto buf = gpu_buf();
  if (dtype == ScalarType::Float32 || val == 0.0f) {
    size_t word_count =
        (dtype == ScalarType::Float32) ? numel : ((nbytes() + 3) / 4);
    uint32_t num_vec4 = static_cast<uint32_t>(word_count / 4);
    if (num_vec4 > 0) {
      struct {
        uint32_t num_vec4;
        uint32_t op_type;
        float param1;
        float param2;
      } pc{num_vec4, 15u, 0.0f, val};

      uint32_t groups = (num_vec4 + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_unary_vec4", {buf, buf}, &pc, sizeof(pc),
          groups);
    }
    uint32_t tail = static_cast<uint32_t>(word_count % 4);
    if (tail > 0 || num_vec4 == 0) {
      struct {
        uint32_t total_elements;
        uint32_t op_type;
        float param1;
        float param2;
      } pc{static_cast<uint32_t>(word_count), 15u, 0.0f, val};

      uint32_t groups = (static_cast<uint32_t>(word_count) + 255) / 256;
      VulkanPipelineManager::instance().dispatch(
          device_id, "elementwise_unary_scalar", {buf, buf}, &pc, sizeof(pc),
          groups);
    }
    return;
  }

  auto tmp =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  tmp->fill_(val);
  auto casted = tmp->cast(dtype);
  copy_from_native(casted);
}

void NativeTensor::zero_() { fill_(0.0f); }

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

// ----------------------------------------------------------------------------
// GPU Compute Dispatches
// ----------------------------------------------------------------------------

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
NativeTensor::matmul(std::shared_ptr<NativeTensor> other, bool trans_a,
                     bool trans_b, std::shared_ptr<NativeTensor> bias) {
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
    return a32->matmul(b32, trans_a, trans_b, bias32)->cast(common);
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

      if (can_coop) {
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
          c_ptr[offC + m * N + n] = sum;
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
        write_scalar_from_double(c_raw, offC + m * N + n, common, sum);
      }
    }
  }
  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_sum(int64_t dim,
                                                       bool keepdim) {
  if (!is_dense_zero_offset()) {
    return force_contiguous()->reduce_sum(dim, keepdim);
  }
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)->reduce_sum(dim, keepdim)->cast(dtype);
  }

  const int64_t TOTAL_DIM = -999999;
  if (dim == TOTAL_DIM) {
    std::vector<int64_t> out_shape = keepdim
                                         ? std::vector<int64_t>(shape.size(), 1)
                                         : std::vector<int64_t>{1};
    auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                              ScalarType::Float32);

    if (device_id >= 0) {
      struct {
        uint32_t num_rows;
        uint32_t cols;
        uint32_t op_type;
      } pc{1, static_cast<uint32_t>(numel), 0};

      VulkanPipelineManager::instance().dispatch(device_id, "reduction",
                                                 {gpu_buf(), out->gpu_buf()},
                                                 &pc, sizeof(pc), 1, 1, 1);
    } else {
      const float *in_ptr = cpu_f32();
      float sum = 0.0f;
      for (size_t i = 0; i < numel; ++i)
        sum += in_ptr[i];
      out->cpu_f32()[0] = sum;
    }
    return out;
  }

  if (dim < 0)
    dim += static_cast<int64_t>(shape.size());

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int64_t>(i) == dim) {
      if (keepdim)
        out_shape.push_back(1);
    } else {
      out_shape.push_back(shape[i]);
    }
  }
  if (out_shape.empty())
    out_shape.push_back(1);

  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

  if (device_id >= 0 && dim == static_cast<int64_t>(shape.size()) - 1) {
    uint32_t cols = static_cast<uint32_t>(shape.back());
    uint32_t rows = static_cast<uint32_t>(numel / cols);

    struct {
      uint32_t num_rows;
      uint32_t cols;
      uint32_t op_type;
    } pc{rows, cols, 0};

    VulkanPipelineManager::instance().dispatch(device_id, "reduction",
                                               {gpu_buf(), out->gpu_buf()}, &pc,
                                               sizeof(pc), 1, rows, 1);
    return out;
  }

  if (device_id >= 0) {
    uint32_t outer = 1;
    uint32_t inner = 1;
    for (int64_t i = 0; i < dim; ++i)
      outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
    for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
      inner *= static_cast<uint32_t>(shape[i]);
    }

    struct {
      uint32_t outer;
      uint32_t dim_size;
      uint32_t inner;
      uint32_t op_type;
    } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner,
         0u};

    if (inner >= 16u && outer <= 65535u) {
      VulkanPipelineManager::instance().dispatch(
          device_id, "reduction_strided", {gpu_buf(), out->gpu_buf()}, &pc,
          sizeof(pc), (inner + 31u) / 32u, outer);
    } else {
      VulkanPipelineManager::instance().dispatch(
          device_id, "reduction_generic", {gpu_buf(), out->gpu_buf()}, &pc,
          sizeof(pc), outer * inner);
    }
    return out;
  }

  // CPU fallback.
  auto cpu_out =
      std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Float32);
  cpu_out->zero_();
  const float *in_ptr = cpu_f32();
  float *out_ptr = cpu_out->cpu_f32();

  for (size_t i = 0; i < numel; ++i) {
    size_t rem_idx = i;
    size_t target_idx = 0;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
      size_t c = rem_idx % static_cast<size_t>(shape[static_cast<size_t>(d)]);
      rem_idx /= static_cast<size_t>(shape[static_cast<size_t>(d)]);
      if (d != dim) {
        int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
        target_idx += c * static_cast<size_t>(
                              cpu_out->strides[static_cast<size_t>(out_d)]);
      }
    }
    out_ptr[target_idx] += in_ptr[i];
  }

  return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_mean(int64_t dim,
                                                        bool keepdim) {
  const int64_t TOTAL_DIM = -999999;
  if (dtype != ScalarType::Float32) {
    auto f32 = cast(ScalarType::Float32)->reduce_mean(dim, keepdim);
    return is_floating_point(dtype) ? f32->cast(dtype) : f32;
  }
  auto sum = reduce_sum(dim, keepdim);
  float divisor =
      (dim == TOTAL_DIM)
          ? static_cast<float>(numel)
          : static_cast<float>(shape[static_cast<size_t>(
                dim < 0 ? dim + static_cast<int64_t>(shape.size()) : dim)]);
  return sum->binary_op_scalar(2, 1.0f / divisor);
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_max(int64_t dim,
                                                       bool keepdim) {
  if (!is_dense_zero_offset()) {
    return force_contiguous()->reduce_max(dim, keepdim);
  }
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)->reduce_max(dim, keepdim)->cast(dtype);
  }

  const int64_t TOTAL_DIM = -999999;
  if (dim == TOTAL_DIM) {
    std::vector<int64_t> out_shape = keepdim
                                         ? std::vector<int64_t>(shape.size(), 1)
                                         : std::vector<int64_t>{1};
    auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                              ScalarType::Float32);
    if (device_id >= 0) {
      struct {
        uint32_t num_rows;
        uint32_t cols;
        uint32_t op_type;
      } pc{1, static_cast<uint32_t>(numel), 1};

      VulkanPipelineManager::instance().dispatch(device_id, "reduction",
                                                 {gpu_buf(), out->gpu_buf()},
                                                 &pc, sizeof(pc), 1, 1, 1);
    } else {
      const float *in_ptr = cpu_f32();
      float max_v = -1e30f;
      for (size_t i = 0; i < numel; ++i)
        max_v = std::max(max_v, in_ptr[i]);
      out->cpu_f32()[0] = max_v;
    }
    return out;
  }

  if (dim < 0)
    dim += static_cast<int64_t>(shape.size());

  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int64_t>(i) == dim) {
      if (keepdim)
        out_shape.push_back(1);
    } else {
      out_shape.push_back(shape[i]);
    }
  }
  if (out_shape.empty())
    out_shape.push_back(1);

  if (device_id >= 0) {
    uint32_t outer = 1;
    uint32_t inner = 1;
    for (int64_t i = 0; i < dim; ++i)
      outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
    for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
      inner *= static_cast<uint32_t>(shape[i]);
    }

    auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                              ScalarType::Float32);

    struct {
      uint32_t outer;
      uint32_t dim_size;
      uint32_t inner;
      uint32_t op_type;
    } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner,
         1u};

    if (inner >= 16u && outer <= 65535u) {
      VulkanPipelineManager::instance().dispatch(
          device_id, "reduction_strided", {gpu_buf(), out->gpu_buf()}, &pc,
          sizeof(pc), (inner + 31u) / 32u, outer);
    } else {
      VulkanPipelineManager::instance().dispatch(
          device_id, "reduction_generic", {gpu_buf(), out->gpu_buf()}, &pc,
          sizeof(pc), outer * inner);
    }
    return out;
  }

  auto cpu_out =
      std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Float32);
  cpu_out->fill_(-1e30f);
  const float *in_ptr = cpu_f32();
  float *out_ptr = cpu_out->cpu_f32();

  for (size_t i = 0; i < numel; ++i) {
    size_t rem_idx = i;
    size_t target_idx = 0;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
      size_t c = rem_idx % static_cast<size_t>(shape[static_cast<size_t>(d)]);
      rem_idx /= static_cast<size_t>(shape[static_cast<size_t>(d)]);
      if (d != dim) {
        int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
        target_idx += c * static_cast<size_t>(
                              cpu_out->strides[static_cast<size_t>(out_d)]);
      }
    }
    out_ptr[target_idx] = std::max(out_ptr[target_idx], in_ptr[i]);
  }

  return cpu_out;
}

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
NativeTensor::conv2d(std::shared_ptr<NativeTensor> weight,
                     std::shared_ptr<NativeTensor> bias, uint32_t stride_h,
                     uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                     uint32_t groups, bool use_im2col) {
  if (shape.size() != 4 || weight->shape.size() != 4) {
    throw std::runtime_error("conv2d requires 4D tensors [N, C, H, W]");
  }
  if (dtype != ScalarType::Float32 || weight->dtype != ScalarType::Float32 ||
      (bias && bias->dtype != ScalarType::Float32)) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("conv2d requires floating-point tensors");
    }
    auto in_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                 : cast(ScalarType::Float32);
    auto w_f32 = (weight->dtype == ScalarType::Float32)
                     ? weight
                     : weight->cast(ScalarType::Float32);
    auto b_f32 = (bias && bias->dtype != ScalarType::Float32)
                     ? bias->cast(ScalarType::Float32)
                     : bias;
    return in_f32
        ->conv2d(w_f32, b_f32, stride_h, stride_w, pad_h, pad_w, groups,
                 use_im2col)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->conv2d(weight, bias, stride_h, stride_w, pad_h,
                                      pad_w, groups, use_im2col);
  }
  auto w = weight->force_contiguous();
  auto bias_t = bias ? bias->force_contiguous() : nullptr;

  uint32_t N = static_cast<uint32_t>(shape[0]);
  uint32_t C_in = static_cast<uint32_t>(shape[1]);
  uint32_t H_in = static_cast<uint32_t>(shape[2]);
  uint32_t W_in = static_cast<uint32_t>(shape[3]);

  uint32_t C_out = static_cast<uint32_t>(w->shape[0]);
  uint32_t kH = static_cast<uint32_t>(w->shape[2]);
  uint32_t kW = static_cast<uint32_t>(w->shape[3]);

  uint32_t H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
  uint32_t W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

  std::vector<int64_t> out_shape = {
      static_cast<int64_t>(N), static_cast<int64_t>(C_out),
      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};

  if (device_id >= 0 && groups > 1u) {
    auto out = std::make_shared<NativeTensor>(out_shape, device_id,
                                              ScalarType::Float32);
    struct {
      uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
    } pc{N,        C_in,  H_in,  W_in,   C_out,
         H_out,    W_out, kH,    kW,     stride_h,
         stride_w, pad_h, pad_w, groups, (bias_t != nullptr) ? 1u : 0u};
    uint32_t gx = (W_out + 7) / 8;
    uint32_t gy = (H_out + 7) / 8;
    uint32_t gz = (N * C_out + 3) / 4;
    std::shared_ptr<VulkanBuffer> b_bias =
        bias_t ? bias_t->gpu_buf() : out->gpu_buf();
    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d_grouped_forward",
        {gpu_buf(), w->gpu_buf(), b_bias, out->gpu_buf()}, &pc, sizeof(pc), gx,
        gy, gz);
    return out;
  }

  if (device_id >= 0 && groups == 1u && use_im2col &&
      (C_in >= 16u || N * H_out * W_out > 4096u)) {
    uint32_t K = C_in * kH * kW;
    uint32_t HW_out = H_out * W_out;
    const size_t per_sample = std::max<size_t>(static_cast<size_t>(K) * HW_out,
                                               static_cast<size_t>(C_out) * HW_out);
    const size_t max_ws_elems = static_cast<size_t>(
        VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() / sizeof(float));
    const uint32_t chunk_n = static_cast<uint32_t>(
        std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
    if (N > chunk_n) {
      std::vector<std::shared_ptr<NativeTensor>> parts;
      for (uint32_t n0 = 0; n0 < N; n0 += chunk_n) {
        uint32_t n1 = std::min(N, n0 + chunk_n);
        auto in_c = slice_dim0(n0, n1);
        parts.push_back(in_c->conv2d(w, bias_t, stride_h, stride_w, pad_h, pad_w, groups, true));
      }
      return cat_dim0(parts, device_id);
    }
    auto col = std::make_shared<NativeTensor>(
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(K),
                             static_cast<int64_t>(HW_out)},
        device_id, ScalarType::Float32);
    struct {
      uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w;
    } pc{N,  C_in, H_in,     W_in,     H_out, W_out,
         kH, kW,   stride_h, stride_w, pad_h, pad_w};
    uint32_t gx = (static_cast<uint32_t>(col->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(device_id, "im2col_kernel",
                                               {gpu_buf(), col->gpu_buf()}, &pc,
                                               sizeof(pc), gx);

    auto w_3d =
        w->reshape_view(
             {1, static_cast<int64_t>(C_out), static_cast<int64_t>(K)})
            ->expand_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                           static_cast<int64_t>(K)});
    auto out_3d = w_3d->matmul(col);
    auto out = out_3d->reshape_view(out_shape);
    if (bias_t) {
      out = out->binary_op(
          0, bias_t->reshape_view({1, static_cast<int64_t>(C_out), 1, 1}));
    }
    return out;
  }

  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t N;
      uint32_t C_in;
      uint32_t H_in;
      uint32_t W_in;
      uint32_t C_out;
      uint32_t H_out;
      uint32_t W_out;
      uint32_t kH;
      uint32_t kW;
      uint32_t stride_h;
      uint32_t stride_w;
      uint32_t pad_h;
      uint32_t pad_w;
      uint32_t has_bias;
    } pc{};

    pc.N = N;
    pc.C_in = C_in;
    pc.H_in = H_in;
    pc.W_in = W_in;
    pc.C_out = C_out;
    pc.H_out = H_out;
    pc.W_out = W_out;
    pc.kH = kH;
    pc.kW = kW;
    pc.stride_h = stride_h;
    pc.stride_w = stride_w;
    pc.pad_h = pad_h;
    pc.pad_w = pad_w;
    pc.has_bias = (bias_t != nullptr) ? 1u : 0u;

    uint32_t gx = (W_out + 7) / 8;
    uint32_t gy = (H_out + 7) / 8;
    uint32_t gz = N * ((C_out + 3) / 4);

    std::shared_ptr<VulkanBuffer> b_bias =
        bias_t ? bias_t->gpu_buf() : out->gpu_buf();

    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d", {gpu_buf(), w->gpu_buf(), b_bias, out->gpu_buf()},
        &pc, sizeof(pc), gx, gy, gz);
  } else {
    const float *in_ptr = cpu_f32();
    const float *w_ptr = w->cpu_f32();
    const float *b_ptr = bias_t ? bias_t->cpu_f32() : nullptr;
    float *out_ptr = out->cpu_f32();
    const uint32_t c_out_per_g = C_out / groups;
    const uint32_t c_in_per_g = C_in / groups;
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t oc = 0; oc < C_out; ++oc) {
        const uint32_t g = oc / c_out_per_g;
        const uint32_t ic_start = g * c_in_per_g;
        float bias_val = b_ptr ? b_ptr[oc] : 0.0f;
        for (uint32_t oh = 0; oh < H_out; ++oh) {
          for (uint32_t ow = 0; ow < W_out; ++ow) {
            float acc = bias_val;
            int in_h_base = int(oh * stride_h) - int(pad_h);
            int in_w_base = int(ow * stride_w) - int(pad_w);

            for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel) {
              uint32_t ic = ic_start + ic_rel;
              for (uint32_t kh = 0; kh < kH; ++kh) {
                int ih = in_h_base + int(kh);
                if (ih >= 0 && ih < int(H_in)) {
                  for (uint32_t kw = 0; kw < kW; ++kw) {
                    int iw = in_w_base + int(kw);
                    if (iw >= 0 && iw < int(W_in)) {
                      acc += in_ptr[n * (C_in * H_in * W_in) +
                                    ic * (H_in * W_in) + ih * W_in + iw] *
                             w_ptr[oc * (c_in_per_g * kH * kW) +
                                   ic_rel * (kH * kW) + kh * kW + kw];
                    }
                  }
                }
              }
            }
            out_ptr[n * (C_out * H_out * W_out) + oc * (H_out * W_out) +
                    oh * W_out + ow] = acc;
          }
        }
      }
    }
  }
  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::conv2d_backward_input(
    std::shared_ptr<NativeTensor> weight, const std::vector<int64_t> &in_shape,
    uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
    uint32_t groups, bool use_im2col) {
  if (dtype != ScalarType::Float32 || weight->dtype != ScalarType::Float32) {
    auto g_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                : cast(ScalarType::Float32);
    auto w_f32 = (weight->dtype == ScalarType::Float32)
                     ? weight
                     : weight->cast(ScalarType::Float32);
    return g_f32
        ->conv2d_backward_input(w_f32, in_shape, stride_h, stride_w, pad_h,
                                pad_w, groups, use_im2col)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->conv2d_backward_input(
        weight, in_shape, stride_h, stride_w, pad_h, pad_w, groups, use_im2col);
  }
  auto w_t = weight->force_contiguous();

  auto grad_in =
      std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);
  grad_in->zero_();

  uint32_t N = static_cast<uint32_t>(in_shape[0]);
  uint32_t C_in = static_cast<uint32_t>(in_shape[1]);
  uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
  uint32_t W_in = static_cast<uint32_t>(in_shape[3]);

  uint32_t C_out = static_cast<uint32_t>(w_t->shape[0]);
  uint32_t kH = static_cast<uint32_t>(w_t->shape[2]);
  uint32_t kW = static_cast<uint32_t>(w_t->shape[3]);

  uint32_t H_out = static_cast<uint32_t>(shape[2]);
  uint32_t W_out = static_cast<uint32_t>(shape[3]);

  if (device_id >= 0 && groups > 1u) {
    struct {
      uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
    } pc{N,  C_in,     H_in,     W_in,  C_out, H_out,  W_out, kH,
         kW, stride_h, stride_w, pad_h, pad_w, groups, 0u};
    uint32_t gx = (W_in + 7) / 8;
    uint32_t gy = (H_in + 7) / 8;
    uint32_t gz = (N * C_in + 3) / 4;
    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d_grouped_backward_input",
        {gpu_buf(), w_t->gpu_buf(), grad_in->gpu_buf(), grad_in->gpu_buf()},
        &pc, sizeof(pc), gx, gy, gz);
    return grad_in;
  }

  if (device_id >= 0 && groups == 1u && use_im2col &&
      (C_in >= 16u || N * H_out * W_out > 4096u)) {
    uint32_t K = C_in * kH * kW;
    uint32_t HW_out = H_out * W_out;
    const size_t per_sample = std::max<size_t>(static_cast<size_t>(K) * HW_out,
                                               static_cast<size_t>(C_out) * HW_out);
    const size_t max_ws_elems = static_cast<size_t>(
        VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() / sizeof(float));
    const uint32_t chunk_n = static_cast<uint32_t>(
        std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
    if (N > chunk_n) {
      std::vector<std::shared_ptr<NativeTensor>> parts;
      for (uint32_t n0 = 0; n0 < N; n0 += chunk_n) {
        uint32_t n1 = std::min(N, n0 + chunk_n);
        std::vector<int64_t> chunk_in = {static_cast<int64_t>(n1 - n0),
                                         static_cast<int64_t>(C_in),
                                         static_cast<int64_t>(H_in),
                                         static_cast<int64_t>(W_in)};
        auto g_c = slice_dim0(n0, n1);
        parts.push_back(g_c->conv2d_backward_input(
            w_t, chunk_in, stride_h, stride_w, pad_h, pad_w, groups, true));
      }
      return cat_dim0(parts, device_id);
    }
    auto g_3d =
        reshape_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                      static_cast<int64_t>(HW_out)});
    auto w_3d =
        w_t->reshape_view(
               {1, static_cast<int64_t>(C_out), static_cast<int64_t>(K)})
            ->expand_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                           static_cast<int64_t>(K)});
    auto col_grad = w_3d->matmul(g_3d, true, false);
    struct {
      uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w;
    } pc{N,  C_in, H_in,     W_in,     H_out, W_out,
         kH, kW,   stride_h, stride_w, pad_h, pad_w};
    uint32_t gx = (static_cast<uint32_t>(grad_in->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "col2im_kernel", {col_grad->gpu_buf(), grad_in->gpu_buf()},
        &pc, sizeof(pc), gx);
    return grad_in;
  }

  if (device_id >= 0) {
    struct {
      uint32_t N;
      uint32_t C_in;
      uint32_t H_in;
      uint32_t W_in;
      uint32_t C_out;
      uint32_t H_out;
      uint32_t W_out;
      uint32_t kH;
      uint32_t kW;
      uint32_t stride_h;
      uint32_t stride_w;
      uint32_t pad_h;
      uint32_t pad_w;
    } pc{};

    pc.N = N;
    pc.C_in = C_in;
    pc.H_in = H_in;
    pc.W_in = W_in;
    pc.C_out = C_out;
    pc.H_out = H_out;
    pc.W_out = W_out;
    pc.kH = kH;
    pc.kW = kW;
    pc.stride_h = stride_h;
    pc.stride_w = stride_w;
    pc.pad_h = pad_h;
    pc.pad_w = pad_w;

    uint32_t gx = (W_in + 7) / 8;
    uint32_t gy = (H_in + 7) / 8;
    uint32_t gz = N * ((C_in + 3) / 4);

    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d_backward_input",
        {gpu_buf(), w_t->gpu_buf(), grad_in->gpu_buf()}, &pc, sizeof(pc), gx,
        gy, gz);
  } else {
    const float *gout_ptr = cpu_f32();
    const float *w_ptr = w_t->cpu_f32();
    float *gin_ptr = grad_in->cpu_f32();
    const uint32_t c_out_per_g = C_out / groups;
    const uint32_t c_in_per_g = C_in / groups;
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t oc = 0; oc < C_out; ++oc) {
        const uint32_t g_idx = oc / c_out_per_g;
        const uint32_t ic_start = g_idx * c_in_per_g;
        for (uint32_t oh = 0; oh < H_out; ++oh) {
          for (uint32_t ow = 0; ow < W_out; ++ow) {
            const float g =
                gout_ptr[((n * C_out + oc) * H_out + oh) * W_out + ow];
            const int in_h_base =
                static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
            const int in_w_base =
                static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

            for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel) {
              const uint32_t ic = ic_start + ic_rel;
              for (uint32_t kh = 0; kh < kH; ++kh) {
                const int ih = in_h_base + static_cast<int>(kh);
                if (ih < 0 || ih >= static_cast<int>(H_in))
                  continue;
                for (uint32_t kw = 0; kw < kW; ++kw) {
                  const int iw = in_w_base + static_cast<int>(kw);
                  if (iw < 0 || iw >= static_cast<int>(W_in))
                    continue;
                  const float wv =
                      w_ptr[((oc * c_in_per_g + ic_rel) * kH + kh) * kW + kw];
                  gin_ptr[((n * C_in + ic) * H_in + ih) * W_in + iw] += g * wv;
                }
              }
            }
          }
        }
      }
    }
  }
  return grad_in;
}

std::shared_ptr<NativeTensor>
NativeTensor::conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                     const std::vector<int64_t> &weight_shape,
                                     uint32_t stride_h, uint32_t stride_w,
                                     uint32_t pad_h, uint32_t pad_w,
                                     uint32_t groups, bool use_im2col) {
  if (dtype != ScalarType::Float32 || input->dtype != ScalarType::Float32) {
    auto g_f32 = (dtype == ScalarType::Float32) ? shared_from_this()
                                                : cast(ScalarType::Float32);
    auto in_f32 = (input->dtype == ScalarType::Float32)
                      ? input
                      : input->cast(ScalarType::Float32);
    return g_f32
        ->conv2d_backward_weight(in_f32, weight_shape, stride_h, stride_w,
                                 pad_h, pad_w, groups, use_im2col)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->conv2d_backward_weight(
        input, weight_shape, stride_h, stride_w, pad_h, pad_w, groups,
        use_im2col);
  }
  auto in_t = input->force_contiguous();

  auto grad_weight = std::make_shared<NativeTensor>(weight_shape, device_id,
                                                    ScalarType::Float32);
  grad_weight->zero_();

  uint32_t N = static_cast<uint32_t>(in_t->shape[0]);
  uint32_t C_in = static_cast<uint32_t>(in_t->shape[1]);
  uint32_t H_in = static_cast<uint32_t>(in_t->shape[2]);
  uint32_t W_in = static_cast<uint32_t>(in_t->shape[3]);

  uint32_t C_out = static_cast<uint32_t>(weight_shape[0]);
  uint32_t kH = static_cast<uint32_t>(weight_shape[2]);
  uint32_t kW = static_cast<uint32_t>(weight_shape[3]);

  uint32_t H_out = static_cast<uint32_t>(shape[2]);
  uint32_t W_out = static_cast<uint32_t>(shape[3]);

  if (device_id >= 0 && groups > 1u) {
    struct {
      uint32_t N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w, groups, has_bias;
    } pc{N,  C_in,     H_in,     W_in,  C_out, H_out,  W_out, kH,
         kW, stride_h, stride_w, pad_h, pad_w, groups, 0u};
    uint32_t gx = (static_cast<uint32_t>(grad_weight->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d_grouped_backward_weight",
        {gpu_buf(), in_t->gpu_buf(), grad_weight->gpu_buf(),
         grad_weight->gpu_buf()},
        &pc, sizeof(pc), gx);
    return grad_weight;
  }

  if (device_id >= 0 && groups == 1u && use_im2col &&
      (C_in >= 16u || N * H_out * W_out > 4096u)) {
    uint32_t K = C_in * kH * kW;
    uint32_t HW_out = H_out * W_out;
    const size_t per_sample = std::max<size_t>(static_cast<size_t>(K) * HW_out,
                                               static_cast<size_t>(C_out) * K);
    const size_t max_ws_elems = static_cast<size_t>(
        VulkanManager::instance_ref().get_device(device_id)->max_workspace_bytes() / sizeof(float));
    const uint32_t chunk_n = static_cast<uint32_t>(
        std::max<size_t>(1ULL, max_ws_elems / std::max<size_t>(1ULL, per_sample)));
    if (N > chunk_n) {
      std::shared_ptr<NativeTensor> dw_acc = nullptr;
      for (uint32_t n0 = 0; n0 < N; n0 += chunk_n) {
        uint32_t n1 = std::min(N, n0 + chunk_n);
        auto in_c = in_t->slice_dim0(n0, n1);
        auto g_c = slice_dim0(n0, n1);
        auto dw_c = g_c->conv2d_backward_weight(
            in_c, weight_shape, stride_h, stride_w, pad_h, pad_w, groups, true);
        dw_acc = dw_acc ? dw_acc->binary_op(0, dw_c) : dw_c;
      }
      return dw_acc;
    }
    auto col = std::make_shared<NativeTensor>(
        std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(K),
                             static_cast<int64_t>(HW_out)},
        device_id, ScalarType::Float32);
    struct {
      uint32_t N, C_in, H_in, W_in, H_out, W_out, kH, kW;
      uint32_t stride_h, stride_w, pad_h, pad_w;
    } pc{N,  C_in, H_in,     W_in,     H_out, W_out,
         kH, kW,   stride_h, stride_w, pad_h, pad_w};
    uint32_t gx = (static_cast<uint32_t>(col->numel) + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "im2col_kernel", {in_t->gpu_buf(), col->gpu_buf()}, &pc,
        sizeof(pc), gx);

    auto g_3d =
        reshape_view({static_cast<int64_t>(N), static_cast<int64_t>(C_out),
                      static_cast<int64_t>(HW_out)});
    auto dw_3d = g_3d->matmul(col, false, true);
    return dw_3d->reduce_sum(0, false)->reshape_view(weight_shape);
  }

  if (device_id >= 0) {
    struct {
      uint32_t N;
      uint32_t C_in;
      uint32_t H_in;
      uint32_t W_in;
      uint32_t C_out;
      uint32_t H_out;
      uint32_t W_out;
      uint32_t kH;
      uint32_t kW;
      uint32_t stride_h;
      uint32_t stride_w;
      uint32_t pad_h;
      uint32_t pad_w;
    } pc{};

    pc.N = N;
    pc.C_in = C_in;
    pc.H_in = H_in;
    pc.W_in = W_in;
    pc.C_out = C_out;
    pc.H_out = H_out;
    pc.W_out = W_out;
    pc.kH = kH;
    pc.kW = kW;
    pc.stride_h = stride_h;
    pc.stride_w = stride_w;
    pc.pad_h = pad_h;
    pc.pad_w = pad_w;

    uint32_t total_pairs = C_out * C_in;
    VulkanPipelineManager::instance().dispatch(
        device_id, "conv2d_backward_weight",
        {gpu_buf(), in_t->gpu_buf(), grad_weight->gpu_buf()}, &pc, sizeof(pc),
        total_pairs, 1, 1);
  } else {
    const float *gout_ptr = cpu_f32();
    const float *in_ptr = in_t->cpu_f32();
    float *gw_ptr = grad_weight->cpu_f32();
    const uint32_t c_out_per_g = C_out / groups;
    const uint32_t c_in_per_g = C_in / groups;
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t oc = 0; oc < C_out; ++oc) {
        const uint32_t g_idx = oc / c_out_per_g;
        const uint32_t ic_start = g_idx * c_in_per_g;
        for (uint32_t oh = 0; oh < H_out; ++oh) {
          for (uint32_t ow = 0; ow < W_out; ++ow) {
            const float g =
                gout_ptr[((n * C_out + oc) * H_out + oh) * W_out + ow];
            const int in_h_base =
                static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
            const int in_w_base =
                static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

            for (uint32_t ic_rel = 0; ic_rel < c_in_per_g; ++ic_rel) {
              const uint32_t ic = ic_start + ic_rel;
              for (uint32_t kh = 0; kh < kH; ++kh) {
                const int ih = in_h_base + static_cast<int>(kh);
                if (ih < 0 || ih >= static_cast<int>(H_in))
                  continue;
                for (uint32_t kw = 0; kw < kW; ++kw) {
                  const int iw = in_w_base + static_cast<int>(kw);
                  if (iw < 0 || iw >= static_cast<int>(W_in))
                    continue;
                  const float x =
                      in_ptr[((n * C_in + ic) * H_in + ih) * W_in + iw];
                  gw_ptr[((oc * c_in_per_g + ic_rel) * kH + kh) * kW + kw] +=
                      g * x;
                }
              }
            }
          }
        }
      }
    }
  }
  return grad_weight;
}

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

std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
NativeTensor::maxpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h,
                        uint32_t stride_w, uint32_t pad_h, uint32_t pad_w) {
  if (shape.size() != 4) {
    throw std::runtime_error("maxpool2d requires 4D input [N, C, H, W]");
  }
  if (dtype != ScalarType::Float32) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("maxpool2d requires floating-point tensor");
    }
    auto [out_f32, idx_f32] =
        cast(ScalarType::Float32)
            ->maxpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w);
    return {out_f32->cast(dtype), idx_f32};
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->maxpool2d(kH, kW, stride_h, stride_w, pad_h,
                                         pad_w);
  }

  uint32_t N = static_cast<uint32_t>(shape[0]);
  uint32_t C = static_cast<uint32_t>(shape[1]);
  uint32_t H_in = static_cast<uint32_t>(shape[2]);
  uint32_t W_in = static_cast<uint32_t>(shape[3]);

  uint32_t H_out = (H_in + 2u * pad_h - kH) / stride_h + 1u;
  uint32_t W_out = (W_in + 2u * pad_w - kW) / stride_w + 1u;

  std::vector<int64_t> out_shape = {
      static_cast<int64_t>(N), static_cast<int64_t>(C),
      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};
  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  auto indices =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w,
          pad_h, pad_w;
    } pc{N,  C,  H_in,     W_in,     H_out, W_out,
         kH, kW, stride_h, stride_w, pad_h, pad_w};

    uint32_t gx = (W_out + 15u) / 16u;
    uint32_t gy = (H_out + 15u) / 16u;
    uint32_t gz = N * C;

    VulkanPipelineManager::instance().dispatch(
        device_id, "maxpool2d", {gpu_buf(), out->gpu_buf(), indices->gpu_buf()},
        &pc, sizeof(pc), gx, gy, gz);
  } else {
    const float *in_ptr = cpu_f32();
    float *out_ptr = out->cpu_f32();
    float *idx_ptr = indices->cpu_f32();
    for (uint32_t nc = 0; nc < N * C; ++nc) {
      uint32_t in_nc_off = nc * H_in * W_in;
      uint32_t out_nc_off = nc * H_out * W_out;
      for (uint32_t oh = 0; oh < H_out; ++oh) {
        for (uint32_t ow = 0; ow < W_out; ++ow) {
          int ih_base =
              static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
          int iw_base =
              static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
          float max_val = -3.402823466e+38f;
          uint32_t max_idx = 0u;
          for (uint32_t kh = 0; kh < kH; ++kh) {
            int ih = ih_base + static_cast<int>(kh);
            if (ih >= 0 && ih < static_cast<int>(H_in)) {
              for (uint32_t kw = 0; kw < kW; ++kw) {
                int iw = iw_base + static_cast<int>(kw);
                if (iw >= 0 && iw < static_cast<int>(W_in)) {
                  uint32_t cur = in_nc_off + static_cast<uint32_t>(ih) * W_in +
                                 static_cast<uint32_t>(iw);
                  float v = in_ptr[cur];
                  if (v > max_val) {
                    max_val = v;
                    max_idx = cur;
                  }
                }
              }
            }
          }
          uint32_t out_idx = out_nc_off + oh * W_out + ow;
          out_ptr[out_idx] = max_val;
          float idx_bits = 0.0f;
          std::memcpy(&idx_bits, &max_idx, sizeof(float));
          idx_ptr[out_idx] = idx_bits;
        }
      }
    }
  }
  return {out, indices};
}

std::shared_ptr<NativeTensor> NativeTensor::maxpool2d_backward(
    std::shared_ptr<NativeTensor> indices, const std::vector<int64_t> &in_shape,
    uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)
        ->maxpool2d_backward(indices, in_shape, kH, kW, stride_h, stride_w,
                             pad_h, pad_w)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->maxpool2d_backward(
        indices, in_shape, kH, kW, stride_h, stride_w, pad_h, pad_w);
  }
  auto idx_t = indices->force_contiguous();

  uint32_t N = static_cast<uint32_t>(in_shape[0]);
  uint32_t C = static_cast<uint32_t>(in_shape[1]);
  uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
  uint32_t W_in = static_cast<uint32_t>(in_shape[3]);
  uint32_t H_out = static_cast<uint32_t>(shape[2]);
  uint32_t W_out = static_cast<uint32_t>(shape[3]);

  auto grad_in =
      std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w,
          pad_h, pad_w;
    } pc{N,  C,  H_in,     W_in,     H_out, W_out,
         kH, kW, stride_h, stride_w, pad_h, pad_w};

    uint32_t gx = (W_in + 15u) / 16u;
    uint32_t gy = (H_in + 15u) / 16u;
    uint32_t gz = N * C;

    VulkanPipelineManager::instance().dispatch(
        device_id, "maxpool2d_backward",
        {gpu_buf(), idx_t->gpu_buf(), grad_in->gpu_buf()}, &pc, sizeof(pc), gx,
        gy, gz);
  } else {
    grad_in->zero_();
    const float *gout_ptr = cpu_f32();
    const float *idx_ptr = idx_t->cpu_f32();
    float *gin_ptr = grad_in->cpu_f32();
    for (size_t i = 0; i < numel; ++i) {
      uint32_t max_idx = 0u;
      std::memcpy(&max_idx, &idx_ptr[i], sizeof(uint32_t));
      if (max_idx < grad_in->numel) {
        gin_ptr[max_idx] += gout_ptr[i];
      }
    }
  }
  return grad_in;
}

std::shared_ptr<NativeTensor>
NativeTensor::avgpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h,
                        uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                        bool count_include_pad, bool is_adaptive,
                        uint32_t adaptive_out_h, uint32_t adaptive_out_w) {
  if (shape.size() != 4) {
    throw std::runtime_error("avgpool2d requires 4D input [N, C, H, W]");
  }
  if (dtype != ScalarType::Float32) {
    if (!is_floating_point(dtype)) {
      throw std::runtime_error("avgpool2d requires floating-point tensor");
    }
    return cast(ScalarType::Float32)
        ->avgpool2d(kH, kW, stride_h, stride_w, pad_h, pad_w, count_include_pad,
                    is_adaptive, adaptive_out_h, adaptive_out_w)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->avgpool2d(kH, kW, stride_h, stride_w, pad_h,
                                         pad_w, count_include_pad, is_adaptive,
                                         adaptive_out_h, adaptive_out_w);
  }

  uint32_t N = static_cast<uint32_t>(shape[0]);
  uint32_t C = static_cast<uint32_t>(shape[1]);
  uint32_t H_in = static_cast<uint32_t>(shape[2]);
  uint32_t W_in = static_cast<uint32_t>(shape[3]);

  uint32_t H_out =
      is_adaptive ? adaptive_out_h : ((H_in + 2u * pad_h - kH) / stride_h + 1u);
  uint32_t W_out =
      is_adaptive ? adaptive_out_w : ((W_in + 2u * pad_w - kW) / stride_w + 1u);

  std::vector<int64_t> out_shape = {
      static_cast<int64_t>(N), static_cast<int64_t>(C),
      static_cast<int64_t>(H_out), static_cast<int64_t>(W_out)};
  auto out =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w,
          pad_h, pad_w;
      uint32_t count_include_pad, is_adaptive;
    } pc{N,
         C,
         H_in,
         W_in,
         H_out,
         W_out,
         kH,
         kW,
         stride_h,
         stride_w,
         pad_h,
         pad_w,
         count_include_pad ? 1u : 0u,
         is_adaptive ? 1u : 0u};

    uint32_t gx = (W_out + 15u) / 16u;
    uint32_t gy = (H_out + 15u) / 16u;
    uint32_t gz = N * C;

    VulkanPipelineManager::instance().dispatch(device_id, "avgpool2d",
                                               {gpu_buf(), out->gpu_buf()}, &pc,
                                               sizeof(pc), gx, gy, gz);
  } else {
    const float *in_ptr = cpu_f32();
    float *out_ptr = out->cpu_f32();
    for (uint32_t nc = 0; nc < N * C; ++nc) {
      uint32_t in_nc_off = nc * H_in * W_in;
      uint32_t out_nc_off = nc * H_out * W_out;
      for (uint32_t oh = 0; oh < H_out; ++oh) {
        for (uint32_t ow = 0; ow < W_out; ++ow) {
          int ih_s, ih_e, iw_s, iw_e;
          if (is_adaptive) {
            ih_s = static_cast<int>((oh * H_in) / H_out);
            ih_e = static_cast<int>(((oh + 1u) * H_in + H_out - 1u) / H_out);
            iw_s = static_cast<int>((ow * W_in) / W_out);
            iw_e = static_cast<int>(((ow + 1u) * W_in + W_out - 1u) / W_out);
          } else {
            ih_s = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
            iw_s = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
            ih_e = ih_s + static_cast<int>(kH);
            iw_e = iw_s + static_cast<int>(kW);
          }
          int ih_cs = std::max(ih_s, 0);
          int iw_cs = std::max(iw_s, 0);
          int ih_ce = std::min(ih_e, static_cast<int>(H_in));
          int iw_ce = std::min(iw_e, static_cast<int>(W_in));

          float sum = 0.0f;
          for (int ih = ih_cs; ih < ih_ce; ++ih) {
            for (int iw = iw_cs; iw < iw_ce; ++iw) {
              sum += in_ptr[in_nc_off + static_cast<uint32_t>(ih) * W_in +
                            static_cast<uint32_t>(iw)];
            }
          }
          int denom = (is_adaptive || !count_include_pad)
                          ? std::max(1, (ih_ce - ih_cs) * (iw_ce - iw_cs))
                          : static_cast<int>(kH * kW);
          out_ptr[out_nc_off + oh * W_out + ow] =
              sum / static_cast<float>(denom);
        }
      }
    }
  }
  return out;
}

std::shared_ptr<NativeTensor> NativeTensor::avgpool2d_backward(
    const std::vector<int64_t> &in_shape, uint32_t kH, uint32_t kW,
    uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
    bool count_include_pad, bool is_adaptive) {
  if (dtype != ScalarType::Float32) {
    return cast(ScalarType::Float32)
        ->avgpool2d_backward(in_shape, kH, kW, stride_h, stride_w, pad_h, pad_w,
                             count_include_pad, is_adaptive)
        ->cast(dtype);
  }
  if (!is_dense_zero_offset()) {
    return force_contiguous()->avgpool2d_backward(
        in_shape, kH, kW, stride_h, stride_w, pad_h, pad_w, count_include_pad,
        is_adaptive);
  }

  uint32_t N = static_cast<uint32_t>(in_shape[0]);
  uint32_t C = static_cast<uint32_t>(in_shape[1]);
  uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
  uint32_t W_in = static_cast<uint32_t>(in_shape[3]);
  uint32_t H_out = static_cast<uint32_t>(shape[2]);
  uint32_t W_out = static_cast<uint32_t>(shape[3]);

  auto grad_in =
      std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);

  if (device_id >= 0) {
    struct {
      uint32_t N, C, H_in, W_in, H_out, W_out, kH, kW, stride_h, stride_w,
          pad_h, pad_w;
      uint32_t count_include_pad, is_adaptive;
    } pc{N,
         C,
         H_in,
         W_in,
         H_out,
         W_out,
         kH,
         kW,
         stride_h,
         stride_w,
         pad_h,
         pad_w,
         count_include_pad ? 1u : 0u,
         is_adaptive ? 1u : 0u};

    uint32_t gx = (W_in + 15u) / 16u;
    uint32_t gy = (H_in + 15u) / 16u;
    uint32_t gz = N * C;

    VulkanPipelineManager::instance().dispatch(device_id, "avgpool2d_backward",
                                               {gpu_buf(), grad_in->gpu_buf()},
                                               &pc, sizeof(pc), gx, gy, gz);
  } else {
    grad_in->zero_();
    const float *gout_ptr = cpu_f32();
    float *gin_ptr = grad_in->cpu_f32();
    for (uint32_t nc = 0; nc < N * C; ++nc) {
      uint32_t in_nc_off = nc * H_in * W_in;
      uint32_t out_nc_off = nc * H_out * W_out;
      for (uint32_t oh = 0; oh < H_out; ++oh) {
        for (uint32_t ow = 0; ow < W_out; ++ow) {
          int ih_s, ih_e, iw_s, iw_e;
          if (is_adaptive) {
            ih_s = static_cast<int>((oh * H_in) / H_out);
            ih_e = static_cast<int>(((oh + 1u) * H_in + H_out - 1u) / H_out);
            iw_s = static_cast<int>((ow * W_in) / W_out);
            iw_e = static_cast<int>(((ow + 1u) * W_in + W_out - 1u) / W_out);
          } else {
            ih_s = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
            iw_s = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
            ih_e = ih_s + static_cast<int>(kH);
            iw_e = iw_s + static_cast<int>(kW);
          }
          int ih_cs = std::max(ih_s, 0);
          int iw_cs = std::max(iw_s, 0);
          int ih_ce = std::min(ih_e, static_cast<int>(H_in));
          int iw_ce = std::min(iw_e, static_cast<int>(W_in));

          int denom = (is_adaptive || !count_include_pad)
                          ? std::max(1, (ih_ce - ih_cs) * (iw_ce - iw_cs))
                          : static_cast<int>(kH * kW);
          float g = gout_ptr[out_nc_off + oh * W_out + ow] /
                    static_cast<float>(denom);
          for (int ih = ih_cs; ih < ih_ce; ++ih) {
            for (int iw = iw_cs; iw < iw_ce; ++iw) {
              gin_ptr[in_nc_off + static_cast<uint32_t>(ih) * W_in +
                      static_cast<uint32_t>(iw)] += g;
            }
          }
        }
      }
    }
  }
  return grad_in;
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

// ============================================================================
// Phase 3.1 & 3.2: Indexing, Conditional, Multi-Axis Cat, Pad, Tril/Triu,
// Flip/Roll
// ============================================================================

namespace {

int64_t normalize_dim(int64_t dim, int64_t ndim) {
  int64_t d = (dim < 0) ? (dim + ndim) : dim;
  if (d < 0 || d >= ndim) {
    throw std::out_of_range("Dimension out of range: " + std::to_string(dim));
  }
  return d;
}

std::vector<int64_t>
align_strides_for_broadcast(const std::vector<int64_t> &src_shape,
                            const std::vector<int64_t> &src_strides,
                            const std::vector<int64_t> &target_shape) {
  const size_t ndim = target_shape.size();
  std::vector<int64_t> out(ndim, 0);
  for (size_t k = 0; k < src_shape.size(); ++k) {
    const size_t si = src_shape.size() - 1 - k;
    const size_t ti = ndim - 1 - k;
    if (src_shape[si] > 1) {
      out[ti] = src_strides[si];
    }
  }
  return out;
}

int reflect_index_cpu(int64_t c, int64_t len) {
  if (len <= 1)
    return 0;
  while (c < 0 || c >= len) {
    if (c < 0)
      c = -c;
    if (c >= len)
      c = 2 * (len - 1) - c;
  }
  return static_cast<int>(c);
}

std::shared_ptr<NativeTensor>
ensure_int32_for_index(const std::shared_ptr<NativeTensor> &idx) {
  auto cont = idx->is_contiguous() ? idx : idx->contiguous();
  if (cont->dtype == ScalarType::Int32)
    return cont;
  return cont->cast(ScalarType::Int32);
}

} // namespace

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

std::shared_ptr<NativeTensor>
NativeTensor::cat(const std::vector<std::shared_ptr<NativeTensor>> &tensors,
                  int64_t dim, int target_device_id) {
  if (tensors.empty()) {
    throw std::runtime_error("cat expects at least one tensor");
  }
  const int64_t ndim = static_cast<int64_t>(tensors[0]->shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  if (ndim > 6) {
    throw std::runtime_error("cat supports up to 6 dimensions");
  }
  if (nd == 0 && target_device_id >= 0) {
    bool all_dense = true;
    const ScalarType dt0 = tensors[0]->dtype;
    for (const auto &t : tensors) {
      if (t->dtype != dt0 || !t->is_dense_zero_offset()) {
        all_dense = false;
        break;
      }
    }
    if (all_dense) {
      return cat_dim0(tensors, target_device_id);
    }
  }

  ScalarType out_dt = tensors[0]->dtype;
  std::vector<int64_t> out_shape = tensors[0]->shape;
  out_shape[nd] = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (static_cast<int64_t>(tensors[i]->shape.size()) != ndim) {
      throw std::runtime_error(
          "cat: all tensors must have the same number of dimensions");
    }
    for (int64_t d = 0; d < ndim; ++d) {
      if (d != nd && tensors[i]->shape[d] != out_shape[d]) {
        throw std::runtime_error("cat: non-concatenated dimensions must match");
      }
    }
    out_shape[nd] += tensors[i]->shape[nd];
    out_dt = promote_dtype(out_dt, tensors[i]->dtype);
  }
  (void)0;
  auto out =
      std::make_shared<NativeTensor>(out_shape, target_device_id, out_dt);
  if (out->numel == 0)
    return out;

  if (target_device_id == -1) {
    uint8_t *dst_raw = out->cpu_raw_ptr();
    int64_t dim_off = 0;
    for (const auto &raw_t : tensors) {
      auto t = (raw_t->device_id == -1) ? raw_t : raw_t->to_device(-1);
      if (t->numel == 0)
        continue;
      const uint8_t *src_raw = t->cpu_raw_ptr();
      for (size_t n = 0; n < t->numel; ++n) {
        size_t rem = n;
        size_t s_pos = t->storage_offset;
        size_t d_pos = static_cast<size_t>(dim_off) *
                       static_cast<size_t>(out->strides[nd]);
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
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
  for (const auto &raw_t : tensors) {
    auto t = (raw_t->device_id == target_device_id)
                 ? raw_t
                 : raw_t->to_device(target_device_id);
    if (t->dtype != out_dt) {
      t = t->cast(out_dt);
    }
    if (t->numel == 0)
      continue;

    struct {
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
    for (int64_t d = 0; d < ndim; ++d) {
      pc.src_shape[d] = static_cast<uint32_t>(t->shape[d]);
      pc.src_stride[d] = static_cast<uint32_t>(t->strides[d]);
      pc.dst_stride[d] = static_cast<uint32_t>(out->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        target_device_id, "copy_into_slice", {t->gpu_buf(), out->gpu_buf()},
        &pc, sizeof(pc), groups);
    dim_off += t->shape[nd];
  }
  return out;
}

std::shared_ptr<NativeTensor>
NativeTensor::pad_nd(const std::vector<int64_t> &pad_before,
                     const std::vector<int64_t> &pad_after, uint32_t mode,
                     float pad_value) const {
  const size_t ndim = shape.size();
  if (ndim > 6 || pad_before.size() != ndim || pad_after.size() != ndim) {
    throw std::runtime_error("pad_nd: invalid pad vector size or ndim > 6");
  }
  std::vector<int64_t> out_shape(ndim);
  for (size_t d = 0; d < ndim; ++d) {
    out_shape[d] = shape[d] + pad_before[d] + pad_after[d];
    if (out_shape[d] < 0) {
      throw std::runtime_error("pad_nd: resulting dimension is negative");
    }
  }

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(out_shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const uint8_t *in_raw = cpu_raw_ptr();
    uint8_t *out_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t in_pos = storage_offset;
      bool in_bounds = true;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t osz = static_cast<size_t>(out_shape[d]);
        const size_t coord = rem % osz;
        rem /= osz;
        int64_t c = static_cast<int64_t>(coord) - pad_before[d];
        const int64_t isz = shape[d];
        if (c < 0 || c >= isz) {
          if (mode == 0u) {
            in_bounds = false;
          } else if (mode == 1u) {
            c = reflect_index_cpu(c, isz);
            in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
          } else {
            c = std::max<int64_t>(0, std::min<int64_t>(c, isz - 1));
            in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
          }
        } else {
          in_pos += static_cast<size_t>(c) * static_cast<size_t>(strides[d]);
        }
      }
      const double v =
          in_bounds
              ? static_cast<double>(read_scalar_as_float(in_raw, in_pos, dtype))
              : static_cast<double>(pad_value);
      write_scalar_from_double(out_raw, n, dtype, v);
    }
    return out;
  }

  auto x_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                            : cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
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
    for (size_t d = 0; d < ndim; ++d) {
      pc.in_shape[d] = static_cast<uint32_t>(x_f->shape[d]);
      pc.out_shape[d] = static_cast<uint32_t>(out_shape[d]);
      pc.in_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
      pc.pad_before[d] = static_cast<uint32_t>(pad_before[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "pad_nd_kernel", {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::pad_nd_backward(const std::vector<int64_t> &in_shape,
                              const std::vector<int64_t> &pad_before,
                              uint32_t mode) const {
  const size_t ndim = shape.size();
  if (ndim > 6 || in_shape.size() != ndim || pad_before.size() != ndim) {
    throw std::runtime_error("pad_nd_backward: invalid shape or ndim > 6");
  }
  if (device_id == -1) {
    auto grad_in = std::make_shared<NativeTensor>(in_shape, -1, dtype);
    grad_in->zero_();
    if (numel == 0)
      return grad_in;
    const uint8_t *gout_raw = cpu_raw_ptr();
    uint8_t *gin_raw = grad_in->cpu_raw_ptr();
    for (size_t n = 0; n < numel; ++n) {
      size_t rem = n;
      size_t in_pos = 0;
      bool in_bounds = true;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t osz = static_cast<size_t>(shape[d]);
        const size_t coord = rem % osz;
        rem /= osz;
        int64_t c = static_cast<int64_t>(coord) - pad_before[d];
        const int64_t isz = in_shape[d];
        if (c < 0 || c >= isz) {
          if (mode == 0u) {
            in_bounds = false;
          } else if (mode == 1u) {
            c = reflect_index_cpu(c, isz);
            in_pos += static_cast<size_t>(c) *
                      static_cast<size_t>(grad_in->strides[d]);
          } else {
            c = std::max<int64_t>(0, std::min<int64_t>(c, isz - 1));
            in_pos += static_cast<size_t>(c) *
                      static_cast<size_t>(grad_in->strides[d]);
          }
        } else {
          in_pos +=
              static_cast<size_t>(c) * static_cast<size_t>(grad_in->strides[d]);
        }
      }
      if (in_bounds) {
        const double cur =
            static_cast<double>(read_scalar_as_float(gin_raw, in_pos, dtype));
        const double addv = static_cast<double>(
            read_scalar_as_float(gout_raw, storage_offset + n, dtype));
        write_scalar_from_double(gin_raw, in_pos, dtype, cur + addv);
      }
    }
    return grad_in;
  }

  auto gout_f = (dtype == ScalarType::Float32)
                    ? force_contiguous()
                    : cast(ScalarType::Float32)->force_contiguous();
  auto gin_f =
      std::make_shared<NativeTensor>(in_shape, device_id, ScalarType::Float32);
  gin_f->zero_();
  if (gout_f->numel > 0) {
    struct {
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
    for (size_t d = 0; d < ndim; ++d) {
      pc.in_shape[d] = static_cast<uint32_t>(in_shape[d]);
      pc.out_shape[d] = static_cast<uint32_t>(gout_f->shape[d]);
      pc.in_stride[d] = static_cast<uint32_t>(gin_f->strides[d]);
      pc.pad_before[d] = static_cast<uint32_t>(pad_before[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "pad_nd_backward_kernel",
        {gout_f->gpu_buf(), gin_f->gpu_buf()}, &pc, sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? gin_f : gin_f->cast(dtype);
}

std::shared_ptr<NativeTensor> NativeTensor::tril_triu(bool is_upper,
                                                      int64_t diagonal) const {
  const size_t ndim = shape.size();
  if (ndim < 2 || ndim > 6) {
    throw std::runtime_error("tril/triu requires 2 <= ndim <= 6");
  }
  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const uint8_t *in_raw = cpu_raw_ptr();
    uint8_t *out_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t src_pos = storage_offset;
      int64_t row = 0;
      int64_t col = 0;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
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
      const double v = keep ? static_cast<double>(
                                  read_scalar_as_float(in_raw, src_pos, dtype))
                            : 0.0;
      write_scalar_from_double(out_raw, n, dtype, v);
    }
    return out;
  }

  auto x_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                            : cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
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
    for (size_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
      pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "tril_triu_kernel", {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::flip(const std::vector<int64_t> &dims) const {
  const size_t ndim = shape.size();
  if (ndim > 6)
    throw std::runtime_error("flip supports up to 6 dimensions");
  uint32_t mask = 0;
  for (int64_t raw_d : dims) {
    int64_t d = normalize_dim(raw_d, static_cast<int64_t>(ndim));
    mask |= (1u << static_cast<uint32_t>(d));
  }
  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const uint8_t *in_raw = cpu_raw_ptr();
    uint8_t *out_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t src_pos = storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const size_t sz = static_cast<size_t>(shape[d]);
        size_t coord = rem % sz;
        rem /= sz;
        if ((mask & (1u << static_cast<uint32_t>(d))) != 0u) {
          coord = sz - 1 - coord;
        }
        src_pos += coord * static_cast<size_t>(strides[d]);
      }
      const double v =
          static_cast<double>(read_scalar_as_float(in_raw, src_pos, dtype));
      write_scalar_from_double(out_raw, n, dtype, v);
    }
    return out;
  }

  auto x_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                            : cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
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
    for (size_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
      pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "flip_roll_kernel", {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

std::shared_ptr<NativeTensor>
NativeTensor::roll(const std::vector<int64_t> &shifts,
                   const std::vector<int64_t> &dims) const {
  const size_t ndim = shape.size();
  if (ndim > 6 || shifts.size() != dims.size()) {
    throw std::runtime_error("roll: invalid dims/shifts or ndim > 6");
  }
  std::vector<int32_t> dim_shifts(ndim, 0);
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t d = normalize_dim(dims[i], static_cast<int64_t>(ndim));
    dim_shifts[static_cast<size_t>(d)] += static_cast<int32_t>(shifts[i]);
  }
  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(shape, -1, dtype);
    if (out->numel == 0)
      return out;
    const uint8_t *in_raw = cpu_raw_ptr();
    uint8_t *out_raw = out->cpu_raw_ptr();
    for (size_t n = 0; n < out->numel; ++n) {
      size_t rem = n;
      size_t src_pos = storage_offset;
      for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
        const int64_t sz = shape[d];
        const int64_t coord =
            static_cast<int64_t>(rem % static_cast<size_t>(sz));
        rem /= static_cast<size_t>(sz);
        const int64_t sh = dim_shifts[static_cast<size_t>(d)] % sz;
        const int64_t sc = (coord - sh + sz) % sz;
        src_pos += static_cast<size_t>(sc) * static_cast<size_t>(strides[d]);
      }
      const double v =
          static_cast<double>(read_scalar_as_float(in_raw, src_pos, dtype));
      write_scalar_from_double(out_raw, n, dtype, v);
    }
    return out;
  }

  auto x_f = (dtype == ScalarType::Float32) ? shared_from_this()
                                            : cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(shape, device_id, ScalarType::Float32);
  if (out_f->numel > 0) {
    struct {
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
    for (size_t d = 0; d < ndim; ++d) {
      pc.shape[d] = static_cast<uint32_t>(x_f->shape[d]);
      pc.src_stride[d] = static_cast<uint32_t>(x_f->strides[d]);
      pc.shifts[d] = dim_shifts[d];
    }
    uint32_t groups = (pc.total_elements + 255u) / 256u;
    VulkanPipelineManager::instance().dispatch(
        device_id, "flip_roll_kernel", {x_f->gpu_buf(), out_f->gpu_buf()}, &pc,
        sizeof(pc), groups);
  }
  return (dtype == ScalarType::Float32) ? out_f : out_f->cast(dtype);
}

// ============================================================================
// Phase 3.3: Extended Reductions, Prefix Scans, Sort / Top-K
// ============================================================================

std::shared_ptr<NativeTensor>
NativeTensor::argmax_argmin(int64_t dim, bool keepdim, bool is_min) const {
  if (dim == -999999) {
    auto flat = reshape_view({static_cast<int64_t>(numel)});
    auto res = flat->argmax_argmin(0, false, is_min);
    if (keepdim) {
      std::vector<int64_t> kd(shape.size(), 1);
      return res->reshape_view(kd);
    }
    return res;
  }

  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  auto cont = is_contiguous() ? shared_from_this() : contiguous();

  uint32_t outer = 1, inner = 1;
  for (int64_t d = 0; d < nd; ++d)
    outer *= static_cast<uint32_t>(cont->shape[d]);
  const uint32_t dim_sz = static_cast<uint32_t>(cont->shape[nd]);
  for (int64_t d = nd + 1; d < ndim; ++d)
    inner *= static_cast<uint32_t>(cont->shape[d]);

  std::vector<int64_t> out_shape;
  for (int64_t d = 0; d < ndim; ++d) {
    if (d == nd) {
      if (keepdim)
        out_shape.push_back(1);
    } else {
      out_shape.push_back(cont->shape[d]);
    }
  }

  const uint32_t total_slices = outer * inner;
  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(out_shape, -1, ScalarType::Int64);
    int64_t *o_ptr = reinterpret_cast<int64_t *>(out->cpu_raw_ptr());
    const uint8_t *i_raw = cont->cpu_raw_ptr();
    for (uint32_t s = 0; s < total_slices; ++s) {
      const uint32_t o = s / inner;
      const uint32_t in_r = s % inner;
      float best_v = 0.0f;
      int64_t best_i = 0;
      for (uint32_t j = 0; j < dim_sz; ++j) {
        const size_t pos =
            cont->storage_offset + (o * dim_sz + j) * inner + in_r;
        const float v = read_scalar_as_float(i_raw, pos, cont->dtype);
        if (j == 0 || (is_min ? (v < best_v) : (v > best_v))) {
          best_v = v;
          best_i = static_cast<int64_t>(j);
        }
      }
      o_ptr[s] = best_i;
    }
    return out;
  }

  auto x_f = (cont->dtype == ScalarType::Float32)
                 ? cont
                 : cont->cast(ScalarType::Float32);
  auto idx_i32 =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Int32);
  if (total_slices > 0) {
    struct {
      uint32_t total_slices;
      uint32_t dim_size;
      uint32_t inner_size;
      uint32_t mode;
      uint32_t out_k;
      uint32_t src_offset;
      uint32_t pad_pow2;
      uint32_t _pad;
    } pc{total_slices, dim_sz,
         inner,        is_min ? 1u : 0u,
         1u,           static_cast<uint32_t>(x_f->storage_offset),
         1u,           0u};
    VulkanPipelineManager::instance().dispatch(
        device_id, "argmax_argmin_kernel",
        {x_f->gpu_buf(), idx_i32->gpu_buf(), idx_i32->gpu_buf()}, &pc,
        sizeof(pc), total_slices);
  }
  return idx_i32->cast(ScalarType::Int64);
}

std::shared_ptr<NativeTensor>
NativeTensor::reduce_prod_all_any(int64_t dim, bool keepdim,
                                  uint32_t mode) const {
  if (dim == -999999) {
    auto flat = reshape_view({static_cast<int64_t>(numel)});
    auto res = flat->reduce_prod_all_any(0, false, mode);
    if (keepdim) {
      std::vector<int64_t> kd(shape.size(), 1);
      return res->reshape_view(kd);
    }
    return res;
  }

  const int64_t ndim = static_cast<int64_t>(shape.size());
  const int64_t nd = normalize_dim(dim, ndim);
  auto cont = is_contiguous() ? shared_from_this() : contiguous();

  uint32_t outer = 1, inner = 1;
  for (int64_t d = 0; d < nd; ++d)
    outer *= static_cast<uint32_t>(cont->shape[d]);
  const uint32_t dim_sz = static_cast<uint32_t>(cont->shape[nd]);
  for (int64_t d = nd + 1; d < ndim; ++d)
    inner *= static_cast<uint32_t>(cont->shape[d]);

  std::vector<int64_t> out_shape;
  for (int64_t d = 0; d < ndim; ++d) {
    if (d == nd) {
      if (keepdim)
        out_shape.push_back(1);
    } else {
      out_shape.push_back(cont->shape[d]);
    }
  }

  const ScalarType target_dt = (mode == 0u) ? dtype : ScalarType::Bool;
  const uint32_t total_slices = outer * inner;

  if (device_id == -1) {
    auto out = std::make_shared<NativeTensor>(out_shape, -1, target_dt);
    const uint8_t *i_raw = cont->cpu_raw_ptr();
    uint8_t *o_raw = out->cpu_raw_ptr();
    for (uint32_t s = 0; s < total_slices; ++s) {
      const uint32_t o = s / inner;
      const uint32_t in_r = s % inner;
      double acc = (mode == 2u) ? 0.0 : 1.0;
      for (uint32_t j = 0; j < dim_sz; ++j) {
        const size_t pos =
            cont->storage_offset + (o * dim_sz + j) * inner + in_r;
        const float v = read_scalar_as_float(i_raw, pos, cont->dtype);
        if (mode == 0u) {
          acc *= static_cast<double>(v);
        } else if (mode == 1u) {
          acc = (acc != 0.0 && v != 0.0f) ? 1.0 : 0.0;
        } else {
          acc = (acc != 0.0 || v != 0.0f) ? 1.0 : 0.0;
        }
      }
      write_scalar_from_double(o_raw, s, target_dt, acc);
    }
    return out;
  }

  auto x_f = (cont->dtype == ScalarType::Float32)
                 ? cont
                 : cont->cast(ScalarType::Float32);
  auto out_f =
      std::make_shared<NativeTensor>(out_shape, device_id, ScalarType::Float32);
  if (total_slices > 0) {
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
         mode,         1u,     static_cast<uint32_t>(x_f->storage_offset),
         1u,           0u};
    VulkanPipelineManager::instance().dispatch(
        device_id, "prod_all_any_kernel",
        {x_f->gpu_buf(), out_f->gpu_buf(), out_f->gpu_buf()}, &pc, sizeof(pc),
        total_slices);
  }
  return (target_dt == ScalarType::Float32) ? out_f : out_f->cast(target_dt);
}

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

// ============================================================================
// Phase 3.4: BatchNorm & GroupNorm
// ============================================================================

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
