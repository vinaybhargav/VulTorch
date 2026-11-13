#include "tensor.hpp"
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

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
    const char* shader;
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
    // Purely register/shared-free baseline; always available.
    {"matmul", 32u, 0u, 256u, MatmulRequirement::None, true},
    // TK * (TILE + PAD) * sizeof(float) * 2
    {"matmul_tiled64", 64u, 16u * (64u + 1u) * 4u * 2u, 256u, MatmulRequirement::None, true},
    {"matmul_tiled", 128u, 16u * (128u + 1u) * 4u * 2u, 256u, MatmulRequirement::None, true},
    // sA + sB (f16, padded) + output staging (f32, padded).
    // Measured 5x slower than matmul_tiled on the RX 7900 XTX (2.9 vs 15.6
    // TFLOPS at 2048^3): one subgroup per workgroup is too little parallelism
    // for the fp32->f16 staging loop. Kept selectable for further work; see
    // docs/kernels/gemm.md.
    {"matmul_coopmat", 64u,
     64u * 17u * 2u + 16u * 68u * 2u + 64u * 68u * 4u, 64u,
     MatmulRequirement::Fp16CoopMat, false},
};

struct MatmulChoice {
    const char* shader;
    uint32_t tile;
};

using MatmulDispatch = std::function<void(const char*, uint32_t)>;
// (device, M, N, K)
using MatmulShapeKey = std::array<uint32_t, 4>;

std::mutex g_matmul_choice_mutex;
std::map<MatmulShapeKey, MatmulChoice> g_matmul_choice_cache;

bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    const std::string text(value);
    return !(text == "0" || text == "false" || text == "off");
}

// Picks the GEMM kernel for one shape.
//
// 1. Filter by what the device reports it can run (shared memory, workgroup
//    size, tile fits the problem).
// 2. Honour VULTORCH_MATMUL_KERNEL=<shader> for benchmarking a specific variant.
// 3. Measure the survivors once and cache the winner per (device, M, N, K).
//    Training reuses a handful of shapes, so this cost is paid a few times.
// VULTORCH_AUTOTUNE=0 disables step 3 and takes the largest viable tile.
MatmulChoice select_matmul_shader(uint32_t device_id, uint32_t M, uint32_t N, uint32_t K,
                                  const MatmulDispatch& run) {
    const auto dev = VulkanManager::instance_ref().get_device(device_id);
    const DeviceCapabilities& caps = dev->capabilities;

    // 1. Capability filter: only variants this device reports it can run.
    std::vector<MatmulChoice> candidates;      // legal here, including opt-in
    std::vector<MatmulChoice> auto_candidates; // eligible for automatic use
    for (const auto& candidate : kMatmulCandidates) {
        if (candidate.shared_bytes > caps.max_shared_memory_bytes) continue;
        if (candidate.workgroup_size > caps.max_workgroup_invocations) continue;
        if (candidate.requirement == MatmulRequirement::Fp16CoopMat) {
            // The kernel splits 64 columns across gl_NumSubgroups, so each
            // subgroup must own at least one 16-wide tile.
            if (!caps.fp16_coopmat_16x16) continue;
            if (caps.subgroup_size < 16u || caps.subgroup_size % 16u != 0u) continue;
            if (64u % caps.subgroup_size != 0u) continue;
        }
        candidates.push_back({candidate.shader, candidate.tile});
        if (candidate.auto_candidate) {
            auto_candidates.push_back({candidate.shader, candidate.tile});
        }
    }
    if (candidates.empty()) return {"matmul", 32u};

    // 2. Explicit override, read every call so it can be changed at runtime
    //    (benchmarks and tests). Not cached: an override must win over the
    //    measurement cache.
    const char* const forced = std::getenv("VULTORCH_MATMUL_KERNEL");
    if (forced != nullptr) {
        const std::string wanted(forced);
        for (const auto& candidate : candidates) {
            if (wanted == candidate.shader) return candidate;
        }
        if (wanted == "legacy") return {"matmul", 32u};
    }

    // 3. Tiny problems are launch-bound; tiling cannot pay for itself and an
    //    autotune sweep would cost far more than the multiplication.
    if (2ull * M * N * K < 1'000'000ull) return {"matmul", 32u};

    if (auto_candidates.empty()) return {"matmul", 32u};
    if (auto_candidates.size() == 1) return auto_candidates.front();

    // 4. Measured selection, cached per (device, shape). Training reuses a
    //    handful of shapes, so this cost is paid a few times per process.
    const MatmulShapeKey key{device_id, M, N, K};
    {
        std::lock_guard<std::mutex> lock(g_matmul_choice_mutex);
        auto it = g_matmul_choice_cache.find(key);
        if (it != g_matmul_choice_cache.end()) return it->second;
    }

    if (!env_flag_enabled("VULTORCH_AUTOTUNE", true)) return auto_candidates.back();

    MatmulChoice best = auto_candidates.front();
    double best_ms = 0.0;

    for (const auto& candidate : auto_candidates) {
        // Fixed *time* budget of warmup: the first runs of a kernel happen at
        // low clocks while the GPU ramps, and the operands are still cold. A
        // count-based warmup would let power management decide the contest.
        const auto warm_start = std::chrono::steady_clock::now();
        int warm_iters = 0;
        while (warm_iters < 128 &&
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_start)
                       .count() < 20.0) {
            run(candidate.shader, candidate.tile);
            dev->synchronize();
            ++warm_iters;
        }

        double candidate_ms = std::numeric_limits<double>::max();
        for (int rep = 0; rep < 5; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            run(candidate.shader, candidate.tile);
            dev->synchronize();
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
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
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return std::pow(a, b);
        case 5: return std::max(a, b);
        case 6: return std::min(a, b);
        case 7: return (a > b) ? 1.0f : 0.0f;
        case 8: return (a < b) ? 1.0f : 0.0f;
        case 9: return (a == b) ? 1.0f : 0.0f;
        default:
            throw std::runtime_error("Unsupported binary op_type: " + std::to_string(op));
    }
}

// Same op numbering as shaders/elementwise_unary_scalar.comp. Anything missing
// here silently degrades to identity, which is how `gelu`/`silu` used to be a
// no-op on the CPU backend while working on the GPU.
inline float apply_unary_op(uint32_t op, float x, float p1, float p2) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kGeluCoeff = 0.044715f;

    switch (op) {
        case 0: return -x;
        case 1: return std::exp(x);
        case 2: return std::log(x);
        case 3: return std::sqrt(x);
        case 4: return 1.0f / std::sqrt(x);
        case 5: return std::abs(x);
        case 6: return std::sin(x);
        case 7: return std::cos(x);
        case 8: return std::tanh(x);
        case 9: return 1.0f / (1.0f + std::exp(-x));
        case 10: return std::max(x, 0.0f);
        case 11: {
            const float inner = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
            return 0.5f * x * (1.0f + std::tanh(inner));
        }
        case 12: return x / (1.0f + std::exp(-x));
        case 13: return std::min(std::max(x, p1), p2);
        case 14: return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        case 15: return x * p1 + p2;
        default:
            throw std::runtime_error("Unsupported unary op_type: " + std::to_string(op));
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

BroadcastPlan make_broadcast_plan(const std::vector<int64_t>& a_shape,
                                  const std::vector<int64_t>& a_strides,
                                  const std::vector<int64_t>& b_shape,
                                  const std::vector<int64_t>& b_strides) {
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
            throw std::runtime_error("Cannot broadcast shapes: incompatible dimension " +
                                     std::to_string(dim_a) + " vs " + std::to_string(dim_b));
        }

        plan.out_shape[oi] = std::max(dim_a, dim_b);
        if (ai >= 0 && dim_a > 1) plan.strides_a[oi] = a_strides[ai];
        if (bi >= 0 && dim_b > 1) plan.strides_b[oi] = b_strides[bi];
    }
    return plan;
}

} // namespace

NativeTensor::NativeTensor(const std::vector<int64_t>& s, int dev_id)
    : device_id(dev_id), shape(s) {
    numel = 1;
    for (auto d : shape) {
        numel *= static_cast<size_t>(d);
    }
    compute_default_strides();

    if (device_id >= 0) {
        gpu_buffer = VulkanMemoryManager::instance().allocate(device_id, numel * sizeof(float));
    } else {
        cpu_data.resize(numel, 0.0f);
    }
}

std::shared_ptr<NativeTensor> NativeTensor::create(const std::vector<int64_t>& shape, int dev_id) {
    return std::make_shared<NativeTensor>(shape, dev_id);
}

// Metadata-only construction: shape/strides are set, no storage is allocated.
// The caller is responsible for attaching a buffer.
NativeTensor::NativeTensor(const std::vector<int64_t>& s, int dev_id, MetadataOnly)
    : device_id(dev_id), shape(s) {
    numel = 1;
    for (auto d : shape) {
        numel *= static_cast<size_t>(d);
    }
    compute_default_strides();
}

std::shared_ptr<NativeTensor> NativeTensor::strided_copy(const std::vector<int64_t>& out_shape,
                                                         const std::vector<int64_t>& src_strides) const {
    if (out_shape.size() != src_strides.size()) {
        throw std::runtime_error("strided_copy: shape and stride rank mismatch");
    }
    if (out_shape.size() > 6) {
        throw std::runtime_error("strided_copy supports up to 6 dimensions");
    }

    auto out = std::make_shared<NativeTensor>(out_shape, device_id);

    if (device_id >= 0) {
        uint32_t ndim = static_cast<uint32_t>(out_shape.size());
        struct {
            uint32_t total_elements;
            uint32_t ndim;
            uint32_t shape[6];
            uint32_t src_stride[6];
        } pc{};
        pc.total_elements = static_cast<uint32_t>(out->numel);
        pc.ndim = ndim;
        for (uint32_t i = 0; i < ndim; ++i) {
            pc.shape[i] = static_cast<uint32_t>(out_shape[i]);
            if (src_strides[i] < 0) {
                throw std::runtime_error("strided_copy: negative strides are not supported");
            }
            pc.src_stride[i] = static_cast<uint32_t>(src_strides[i]);
        }

        uint32_t groups = (static_cast<uint32_t>(out->numel) + 255) / 256;
        VulkanPipelineManager::instance().dispatch(
            device_id, "copy_strided",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), groups);
        return out;
    }

    // CPU fallback.
    const size_t ndim = out_shape.size();
    std::vector<int64_t> coord(ndim, 0);
    int64_t src_idx = 0;
    for (size_t n = 0; n < out->numel; ++n) {
        out->cpu_data[n] = cpu_data[src_idx];
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            src_idx += src_strides[d];
            if (++coord[d] < out_shape[d]) {
                break;
            }
            src_idx -= src_strides[d] * out_shape[d];
            coord[d] = 0;
        }
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::reshape_view(const std::vector<int64_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("reshape_view requires a contiguous tensor");
    }

    size_t new_numel = 1;
    for (auto d : new_shape) {
        new_numel *= static_cast<size_t>(d);
    }
    if (new_numel != numel) {
        throw std::runtime_error("reshape_view: element count mismatch");
    }

    auto view = std::make_shared<NativeTensor>(new_shape, device_id, MetadataOnly{});
    // GPU: share the buffer (no copy). CPU: the CPU path stores data in a
    // std::vector, so sharing is expressed as a copy.
    view->gpu_buffer = gpu_buffer;
    view->cpu_data = cpu_data;
    return view;
}

void NativeTensor::copy_from_native(const std::shared_ptr<NativeTensor>& src) {
    if (src.get() == this) return;
    if (src->numel != numel) {
        throw std::runtime_error("copy_from_native: element count mismatch");
    }
    if (src->device_id != device_id) {
        throw std::runtime_error("copy_from_native requires both tensors on the same device");
    }

    if (device_id >= 0) {
        VulkanMemoryManager::instance().copy_buffer_to_buffer(
            device_id, src->gpu_buffer, gpu_buffer, numel * sizeof(float));
    } else {
        std::copy(src->cpu_data.begin(), src->cpu_data.begin() + numel, cpu_data.begin());
    }
}

std::shared_ptr<NativeTensor> NativeTensor::from_float_array(const float* data, const std::vector<int64_t>& shape, int dev_id) {
    auto tensor = std::make_shared<NativeTensor>(shape, dev_id);
    tensor->copy_from_host(data, tensor->numel);
    return tensor;
}

void NativeTensor::compute_default_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

bool NativeTensor::is_contiguous() const {
    if (shape.empty()) return true;
    int64_t expected = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

void NativeTensor::copy_from_host(const float* src, size_t count) {
    size_t copy_count = std::min(count, numel);
    if (device_id >= 0) {
        VulkanMemoryManager::instance().copy_host_to_device(device_id, src, gpu_buffer, copy_count * sizeof(float));
    } else {
        std::copy(src, src + copy_count, cpu_data.data());
    }
}

void NativeTensor::copy_to_host(float* dst, size_t count) {
    size_t copy_count = std::min(count, numel);
    if (device_id >= 0) {
        VulkanMemoryManager::instance().copy_device_to_host(device_id, gpu_buffer, dst, copy_count * sizeof(float));
    } else {
        std::copy(cpu_data.data(), cpu_data.data() + copy_count, dst);
    }
}

std::shared_ptr<NativeTensor> NativeTensor::to_device(int target_device_id) {
    if (device_id == target_device_id) {
        return shared_from_this();
    }
    auto target = std::make_shared<NativeTensor>(shape, target_device_id);
    if (device_id == -1 && target_device_id >= 0) {
        // CPU -> GPU
        target->copy_from_host(cpu_data.data(), numel);
    } else if (device_id >= 0 && target_device_id == -1) {
        // GPU -> CPU
        copy_to_host(target->cpu_data.data(), numel);
    } else {
        // GPU -> GPU (Cross-GPU Vulkan transfer, e.g. vk:0 to vk:1)
        VulkanMemoryManager::instance().copy_cross_device(device_id, gpu_buffer,
                                                         target_device_id, target->gpu_buffer,
                                                         numel * sizeof(float));
    }
    return target;
}

std::shared_ptr<NativeTensor> NativeTensor::clone() {
    auto target = std::make_shared<NativeTensor>(shape, device_id);
    if (device_id >= 0) {
        VulkanMemoryManager::instance().copy_buffer_to_buffer(device_id, gpu_buffer, target->gpu_buffer, numel * sizeof(float));
    } else {
        target->cpu_data = cpu_data;
    }
    return target;
}

void NativeTensor::fill_(float val) {
    if (device_id >= 0) {
        unary_op(15, 0.0f, val); // scale by 0, add val
    } else {
        std::fill(cpu_data.begin(), cpu_data.end(), val);
    }
}

void NativeTensor::zero_() {
    fill_(0.0f);
}

// ----------------------------------------------------------------------------
// GPU Compute Dispatches
// ----------------------------------------------------------------------------

std::shared_ptr<NativeTensor> NativeTensor::binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other) {
    if (device_id != other->device_id) {
        throw std::runtime_error("Tensors must be on the same device for binary operations!");
    }

    if (device_id == -1) {
        // CPU fallback. Uses the same broadcast plan as the GPU path so the two
        // backends cannot disagree (previously this indexed `other` with the
        // output linear index and read out of bounds on broadcast shapes).
        const BroadcastPlan plan = make_broadcast_plan(shape, strides, other->shape, other->strides);
        auto out = std::make_shared<NativeTensor>(plan.out_shape, -1);

        const size_t ndim = plan.out_shape.size();
        std::vector<int64_t> coord(ndim, 0);
        int64_t ia = 0;
        int64_t ib = 0;

        for (size_t n = 0; n < out->numel; ++n) {
            out->cpu_data[n] = apply_binary_op(op_type, cpu_data[ia], other->cpu_data[ib]);

            // Odometer increment, last dimension fastest.
            for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
                ia += plan.strides_a[d];
                ib += plan.strides_b[d];
                if (++coord[d] < plan.out_shape[d]) {
                    break;
                }
                ia -= plan.strides_a[d] * plan.out_shape[d];
                ib -= plan.strides_b[d] * plan.out_shape[d];
                coord[d] = 0;
            }
        }
        return out;
    }

    // Determine output shape (handle broadcasting)
    bool same_shape = (shape == other->shape);
    if (same_shape && is_contiguous() && other->is_contiguous()) {
        auto out = std::make_shared<NativeTensor>(shape, device_id);
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
                {gpu_buffer, other->gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), groups);
        }

        // Tail elements
        uint32_t tail_count = static_cast<uint32_t>(numel % 4);
        if (tail_count > 0) {
            struct {
                uint32_t total_elements;
                uint32_t op_type;
                uint32_t ndim;
                uint32_t is_b_scalar;
                float scalar_b;
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
                {gpu_buffer, other->gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), groups);
        }
        return out;
    } else {
        // Multi-dimensional broadcasting, driven by the shared plan.
        const BroadcastPlan plan = make_broadcast_plan(shape, strides, other->shape, other->strides);
        const std::vector<int64_t>& out_shape = plan.out_shape;
        const size_t max_ndim = out_shape.size();

        auto out = std::make_shared<NativeTensor>(out_shape, device_id);

        struct {
            uint32_t total_elements;
            uint32_t op_type;
            uint32_t ndim;
            uint32_t is_b_scalar;
            float scalar_b;
            uint32_t shape[6];
            uint32_t strideA[6];
            uint32_t strideB[6];
        } pc{};

        pc.total_elements = static_cast<uint32_t>(out->numel);
        pc.op_type = op_type;
        pc.ndim = static_cast<uint32_t>(max_ndim);
        pc.is_b_scalar = 0;

        if (max_ndim > 6) {
            throw std::runtime_error("Broadcasting supports up to 6 dimensions");
        }
        for (size_t i = 0; i < max_ndim; ++i) {
            pc.shape[i] = static_cast<uint32_t>(out_shape[i]);
            pc.strideA[i] = static_cast<uint32_t>(plan.strides_a[i]);
            pc.strideB[i] = static_cast<uint32_t>(plan.strides_b[i]);
        }

        uint32_t groups = (static_cast<uint32_t>(out->numel) + 255) / 256;
        VulkanPipelineManager::instance().dispatch(
            device_id, "elementwise_binary_broadcast",
            {gpu_buffer, other->gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), groups);

        return out;
    }
}

std::shared_ptr<NativeTensor> NativeTensor::binary_op_scalar(uint32_t op_type, float scalar_val, bool reverse) {
    if (device_id == -1) {
        auto out = std::make_shared<NativeTensor>(shape, -1);
        for (size_t i = 0; i < numel; ++i) {
            float a = reverse ? scalar_val : cpu_data[i];
            float b = reverse ? cpu_data[i] : scalar_val;
            out->cpu_data[i] = apply_binary_op(op_type, a, b);
        }
        return out;
    }

    auto out = std::make_shared<NativeTensor>(shape, device_id);
    uint32_t num_vec4 = static_cast<uint32_t>(numel / 4);

    if (num_vec4 > 0 && !reverse) {
        struct {
            uint32_t num_vec4;
            uint32_t op_type;
            float scalar_b;
            uint32_t is_b_scalar;
        } pc{num_vec4, op_type, scalar_val, 1};

        uint32_t groups = (num_vec4 + 255) / 256;
        VulkanPipelineManager::instance().dispatch(
            device_id, "elementwise_binary_vec4",
            {gpu_buffer, gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), groups);
    }

    // Tails or reverse scalar ops
    if ((numel % 4 != 0) || reverse || num_vec4 == 0) {
        struct {
            uint32_t total_elements;
            uint32_t op_type;
            float p1;
            float p2;
        } pc{static_cast<uint32_t>(numel), 15, 1.0f, 0.0f};

        // If reverse sub: out = scalar_val - x => (-1)*x + scalar_val
        if (reverse && op_type == 1) {
            pc.p1 = -1.0f;
            pc.p2 = scalar_val;
            uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
            VulkanPipelineManager::instance().dispatch(
                device_id, "elementwise_unary_scalar",
                {gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), groups);
            return out;
        }

        struct {
            uint32_t total_elements;
            uint32_t op_type;
            uint32_t ndim;
            uint32_t is_b_scalar;
            float scalar_b;
            uint32_t shape[6];
            uint32_t strideA[6];
            uint32_t strideB[6];
        } b_pc{};
        b_pc.total_elements = static_cast<uint32_t>(numel);
        b_pc.op_type = op_type;
        b_pc.is_b_scalar = 1;
        b_pc.scalar_b = scalar_val;

        uint32_t groups = (static_cast<uint32_t>(numel) + 255) / 256;
        VulkanPipelineManager::instance().dispatch(
            device_id, "elementwise_binary_broadcast",
            {gpu_buffer, gpu_buffer, out->gpu_buffer},
            &b_pc, sizeof(b_pc), groups);
    }

    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::unary_op(uint32_t op_type, float p1, float p2) {
    if (device_id == -1) {
        auto out = std::make_shared<NativeTensor>(shape, -1);
        for (size_t i = 0; i < numel; ++i) {
            out->cpu_data[i] = apply_unary_op(op_type, cpu_data[i], p1, p2);
        }
        return out;
    }

    auto out = std::make_shared<NativeTensor>(shape, device_id);
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
            device_id, "elementwise_unary_vec4",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), groups);
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
            device_id, "elementwise_unary_scalar",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), groups);
    }

    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::matmul(std::shared_ptr<NativeTensor> other) {
    if (device_id != other->device_id) {
        throw std::runtime_error("Matmul requires both tensors on the same device!");
    }

    if (shape.size() < 2 || other->shape.size() < 2) {
        throw std::runtime_error("Matmul requires at least 2-dimensional tensors");
    }
    if (!is_contiguous() || !other->is_contiguous()) {
        throw std::runtime_error("Matmul requires contiguous tensors");
    }

    uint32_t M = static_cast<uint32_t>(shape[shape.size() - 2]);
    uint32_t K1 = static_cast<uint32_t>(shape[shape.size() - 1]);
    uint32_t K2 = static_cast<uint32_t>(other->shape[other->shape.size() - 2]);
    uint32_t N = static_cast<uint32_t>(other->shape[other->shape.size() - 1]);

    if (K1 != K2) {
        throw std::runtime_error("Inner dimensions must match for matmul: " + std::to_string(K1) + " vs " + std::to_string(K2));
    }

    // Batch handling: B may either be unbatched (shared across every batch item,
    // the `Linear`-on-3D case) or have exactly the same leading dimensions. Any
    // other combination used to produce a wrong output shape and read past the
    // end of B, so it is rejected explicitly until general broadcasting lands.
    const bool b_broadcast = (other->shape.size() == 2);
    if (!b_broadcast) {
        if (shape.size() != other->shape.size()) {
            throw std::runtime_error("Matmul batch rank mismatch: " + std::to_string(shape.size()) +
                                     " vs " + std::to_string(other->shape.size()));
        }
        for (size_t i = 0; i + 2 < shape.size(); ++i) {
            if (shape[i] != other->shape[i]) {
                throw std::runtime_error("Matmul batch dimensions must match (general batch broadcasting is not implemented)");
            }
        }
    }

    // Calculate batch count
    size_t batch_count = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_count *= shape[i];
    }

    std::vector<int64_t> out_shape = shape;
    out_shape[out_shape.size() - 1] = N;
    auto out = std::make_shared<NativeTensor>(out_shape, device_id);

    if (device_id >= 0) {
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
        } pc{};

        pc.M = M;
        pc.N = N;
        pc.K = K1;
        pc.batch_count = static_cast<uint32_t>(batch_count);
        pc.strideA_b = static_cast<uint32_t>(M * K1);
        pc.strideA_m = static_cast<uint32_t>(K1);
        pc.strideA_k = 1;
        pc.strideB_b = b_broadcast ? 0u : static_cast<uint32_t>(K1 * N);
        pc.strideB_k = static_cast<uint32_t>(N);
        pc.strideB_n = 1;
        pc.strideC_b = static_cast<uint32_t>(M * N);
        pc.strideC_m = static_cast<uint32_t>(N);
        pc.strideC_n = 1;

        auto run_variant = [&](const char* shader, uint32_t tile) {
            const uint32_t group_x = (N + tile - 1u) / tile;
            const uint32_t group_y = (M + tile - 1u) / tile;
            VulkanPipelineManager::instance().dispatch(
                device_id, shader,
                {gpu_buffer, other->gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), group_x, group_y, static_cast<uint32_t>(batch_count));
        };

        const MatmulChoice choice = select_matmul_shader(
            static_cast<uint32_t>(device_id), M, N, K1, run_variant);
        run_variant(choice.shader, choice.tile);
    } else {
        // CPU MatMul fallback
        const size_t batch_stride_b = b_broadcast ? 0 : static_cast<size_t>(K1) * N;
        for (size_t b = 0; b < batch_count; ++b) {
            size_t offA = b * M * K1;
            size_t offB = b * batch_stride_b;
            size_t offC = b * M * N;

            for (uint32_t m = 0; m < M; ++m) {
                for (uint32_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < K1; ++k) {
                        sum += cpu_data[offA + m * K1 + k] * other->cpu_data[offB + k * N + n];
                    }
                    out->cpu_data[offC + m * N + n] = sum;
                }
            }
        }
    }

    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_sum(int64_t dim, bool keepdim) {
    const int64_t TOTAL_DIM = -999999;
    if (dim == TOTAL_DIM) {
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        auto out = std::make_shared<NativeTensor>(out_shape, device_id);

        if (device_id >= 0) {
            struct {
                uint32_t num_rows;
                uint32_t cols;
                uint32_t op_type;
            } pc{1, static_cast<uint32_t>(numel), 0};

            VulkanPipelineManager::instance().dispatch(
                device_id, "reduction",
                {gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), 1, 1, 1);
        } else {
            float sum = 0.0f;
            for (float v : cpu_data) sum += v;
            out->cpu_data[0] = sum;
        }
        return out;
    }

    if (dim < 0) dim += shape.size();

    // Determine output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto out = std::make_shared<NativeTensor>(out_shape, device_id);

    // Fast path: reduction along last dimension on GPU
    if (device_id >= 0 && dim == static_cast<int64_t>(shape.size()) - 1 && is_contiguous()) {
        uint32_t cols = static_cast<uint32_t>(shape.back());
        uint32_t rows = static_cast<uint32_t>(numel / cols);

        struct {
            uint32_t num_rows;
            uint32_t cols;
            uint32_t op_type;
        } pc{rows, cols, 0};

        VulkanPipelineManager::instance().dispatch(
            device_id, "reduction",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), 1, rows, 1);
        return out;
    }

    // General reduction on GPU: contiguous tensor viewed as [outer, dim, inner].
    if (device_id >= 0) {
        uint32_t outer = 1;
        uint32_t inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
        for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
            inner *= static_cast<uint32_t>(shape[i]);
        }

        struct {
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t op_type;
        } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner, 0u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "reduction_generic",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), outer * inner);
        return out;
    }

    // CPU fallback.
    auto cpu_in = shared_from_this();
    auto cpu_out = std::make_shared<NativeTensor>(out_shape, -1);
    cpu_out->zero_();

    for (size_t i = 0; i < numel; ++i) {
        size_t rem_idx = i;
        size_t target_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            size_t c = rem_idx % shape[d];
            rem_idx /= shape[d];
            if (d != dim) {
                int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
                target_idx += c * cpu_out->strides[out_d];
            }
        }
        cpu_out->cpu_data[target_idx] += cpu_in->cpu_data[i];
    }

    return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_mean(int64_t dim, bool keepdim) {
    const int64_t TOTAL_DIM = -999999;
    auto sum = reduce_sum(dim, keepdim);
    float divisor = (dim == TOTAL_DIM) ? static_cast<float>(numel) : static_cast<float>(shape[dim < 0 ? dim + shape.size() : dim]);
    return sum->binary_op_scalar(2, 1.0f / divisor);
}

std::shared_ptr<NativeTensor> NativeTensor::reduce_max(int64_t dim, bool keepdim) {
    const int64_t TOTAL_DIM = -999999;
    if (dim == TOTAL_DIM) {
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        auto out = std::make_shared<NativeTensor>(out_shape, device_id);
        if (device_id >= 0) {
            struct {
                uint32_t num_rows;
                uint32_t cols;
                uint32_t op_type;
            } pc{1, static_cast<uint32_t>(numel), 1};

            VulkanPipelineManager::instance().dispatch(
                device_id, "reduction",
                {gpu_buffer, out->gpu_buffer},
                &pc, sizeof(pc), 1, 1, 1);
        } else {
            float max_v = -1e30f;
            for (float v : cpu_data) max_v = std::max(max_v, v);
            out->cpu_data[0] = max_v;
        }
        return out;
    }

    if (dim < 0) dim += shape.size();

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    // General reduction on GPU: contiguous tensor viewed as [outer, dim, inner].
    if (device_id >= 0) {
        uint32_t outer = 1;
        uint32_t inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= static_cast<uint32_t>(shape[static_cast<size_t>(i)]);
        for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) {
            inner *= static_cast<uint32_t>(shape[i]);
        }

        auto out = std::make_shared<NativeTensor>(out_shape, device_id);

        struct {
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t op_type;
        } pc{outer, static_cast<uint32_t>(shape[static_cast<size_t>(dim)]), inner, 1u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "reduction_generic",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), outer * inner);
        return out;
    }

    // CPU fallback.
    auto cpu_in = shared_from_this();
    auto cpu_out = std::make_shared<NativeTensor>(out_shape, -1);
    cpu_out->fill_(-1e30f);

    for (size_t i = 0; i < numel; ++i) {
        size_t rem_idx = i;
        size_t target_idx = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            size_t c = rem_idx % shape[d];
            rem_idx /= shape[d];
            if (d != dim) {
                int out_d = (d < dim) ? d : (keepdim ? d : d - 1);
                target_idx += c * cpu_out->strides[out_d];
            }
        }
        cpu_out->cpu_data[target_idx] = std::max(cpu_out->cpu_data[target_idx], cpu_in->cpu_data[i]);
    }

    return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::softmax(int64_t dim, bool is_log) {
    if (shape.empty()) {
        throw std::runtime_error("softmax requires a tensor with at least 1 dimension");
    }
    if (dim < 0) dim += static_cast<int64_t>(shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::runtime_error("softmax: dim " + std::to_string(dim) + " out of range for rank " +
                                 std::to_string(shape.size()));
    }

    auto out = std::make_shared<NativeTensor>(shape, device_id);
    const bool last_dim = (dim == static_cast<int64_t>(shape.size()) - 1);

    if (device_id >= 0 && last_dim) {
        uint32_t cols = static_cast<uint32_t>(shape.back());
        uint32_t rows = static_cast<uint32_t>(numel / cols);

        struct {
            uint32_t num_rows;
            uint32_t cols;
            uint32_t is_log_softmax;
        } pc{rows, cols, is_log ? 1u : 0u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "softmax",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), 1, rows, 1);
        return out;
    }

    // Dimension-aware path for any axis (and for CPU tensors). The storage is
    // contiguous, so consecutive elements of one reduction group are `inner`
    // apart. Kept simple and obviously correct; the fast kernel above only
    // covers the last axis.
    auto cpu_in = (device_id >= 0) ? to_device(-1) : shared_from_this();
    auto cpu_out = std::make_shared<NativeTensor>(shape, -1);

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
                m = std::max(m, cpu_in->cpu_data[base + d * inner]);
            }

            float s = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                s += std::exp(cpu_in->cpu_data[base + d * inner] - m);
            }

            const float log_s = std::log(s);
            const float inv_s = 1.0f / s;
            for (int64_t d = 0; d < dim_size; ++d) {
                const float v = cpu_in->cpu_data[base + d * inner] - m;
                cpu_out->cpu_data[base + d * inner] = is_log ? (v - log_s) : (std::exp(v) * inv_s);
            }
        }
    }

    return (device_id >= 0) ? cpu_out->to_device(device_id) : cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::cross_entropy_per_row(
    const std::shared_ptr<NativeTensor>& targets) const {
    if (shape.size() != 2) {
        throw std::runtime_error("cross_entropy requires a 2-D logits tensor [rows, cols]");
    }
    if (!targets || targets->numel != static_cast<size_t>(shape[0])) {
        throw std::runtime_error("cross_entropy: expected exactly one target per row");
    }
    if (targets->device_id != device_id) {
        throw std::runtime_error("cross_entropy: targets must be on the same device as logits");
    }

    const uint32_t rows = static_cast<uint32_t>(shape[0]);
    const uint32_t cols = static_cast<uint32_t>(shape[1]);
    auto out = std::make_shared<NativeTensor>(
        std::vector<int64_t>{static_cast<int64_t>(rows)}, device_id);

    if (device_id >= 0) {
        struct {
            uint32_t rows;
            uint32_t cols;
            uint32_t mode; // 0 = per-row loss
        } pc{rows, cols, 0u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "softmax_cross_entropy",
            {gpu_buffer, targets->gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), rows);
        return out;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = &cpu_data[static_cast<size_t>(r) * cols];
        const int64_t target = static_cast<int64_t>(targets->cpu_data[r]);
        if (target < 0 || target >= static_cast<int64_t>(cols)) {
            throw std::runtime_error("cross_entropy: target index out of range");
        }

        float m = -1e30f;
        for (uint32_t c = 0; c < cols; ++c) m = std::max(m, row[c]);
        float s = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) s += std::exp(row[c] - m);

        out->cpu_data[r] = (m + std::log(s)) - row[target];
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::cross_entropy_grad(
    const std::shared_ptr<NativeTensor>& targets) const {
    if (shape.size() != 2) {
        throw std::runtime_error("cross_entropy requires a 2-D logits tensor [rows, cols]");
    }
    if (!targets || targets->numel != static_cast<size_t>(shape[0])) {
        throw std::runtime_error("cross_entropy: expected exactly one target per row");
    }
    if (targets->device_id != device_id) {
        throw std::runtime_error("cross_entropy: targets must be on the same device as logits");
    }

    const uint32_t rows = static_cast<uint32_t>(shape[0]);
    const uint32_t cols = static_cast<uint32_t>(shape[1]);
    auto out = std::make_shared<NativeTensor>(shape, device_id);

    if (device_id >= 0) {
        struct {
            uint32_t rows;
            uint32_t cols;
            uint32_t mode; // 1 = gradient
        } pc{rows, cols, 1u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "softmax_cross_entropy",
            {gpu_buffer, targets->gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), rows);
        return out;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row = &cpu_data[static_cast<size_t>(r) * cols];
        const int64_t target = static_cast<int64_t>(targets->cpu_data[r]);
        if (target < 0 || target >= static_cast<int64_t>(cols)) {
            throw std::runtime_error("cross_entropy: target index out of range");
        }

        float m = -1e30f;
        for (uint32_t c = 0; c < cols; ++c) m = std::max(m, row[c]);
        float s = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) s += std::exp(row[c] - m);
        const float inv_s = 1.0f / s;

        for (uint32_t c = 0; c < cols; ++c) {
            const float probability = std::exp(row[c] - m) * inv_s;
            const float onehot = (static_cast<int64_t>(c) == target) ? 1.0f : 0.0f;
            out->cpu_data[static_cast<size_t>(r) * cols + c] = probability - onehot;
        }
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::conv2d(std::shared_ptr<NativeTensor> weight,
                                                  std::shared_ptr<NativeTensor> bias,
                                                  uint32_t stride_h, uint32_t stride_w,
                                                  uint32_t pad_h, uint32_t pad_w) {
    if (shape.size() != 4 || weight->shape.size() != 4) {
        throw std::runtime_error("conv2d requires 4D tensors [N, C, H, W]");
    }

    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t C_in = static_cast<uint32_t>(shape[1]);
    uint32_t H_in = static_cast<uint32_t>(shape[2]);
    uint32_t W_in = static_cast<uint32_t>(shape[3]);

    uint32_t C_out = static_cast<uint32_t>(weight->shape[0]);
    uint32_t kH = static_cast<uint32_t>(weight->shape[2]);
    uint32_t kW = static_cast<uint32_t>(weight->shape[3]);

    uint32_t H_out = (H_in + 2 * pad_h - kH) / stride_h + 1;
    uint32_t W_out = (W_in + 2 * pad_w - kW) / stride_w + 1;

    std::vector<int64_t> out_shape = {static_cast<int64_t>(N),
                                      static_cast<int64_t>(C_out),
                                      static_cast<int64_t>(H_out),
                                      static_cast<int64_t>(W_out)};
    auto out = std::make_shared<NativeTensor>(out_shape, device_id);

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

        pc.N = N; pc.C_in = C_in; pc.H_in = H_in; pc.W_in = W_in;
        pc.C_out = C_out; pc.H_out = H_out; pc.W_out = W_out;
        pc.kH = kH; pc.kW = kW;
        pc.stride_h = stride_h; pc.stride_w = stride_w;
        pc.pad_h = pad_h; pc.pad_w = pad_w;
        pc.has_bias = (bias != nullptr) ? 1u : 0u;

        uint32_t gx = (W_out + 7) / 8;
        uint32_t gy = (H_out + 7) / 8;
        uint32_t gz = N * ((C_out + 3) / 4);

        std::shared_ptr<VulkanBuffer> b_bias = bias ? bias->gpu_buffer : out->gpu_buffer;

        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d",
            {gpu_buffer, weight->gpu_buffer, b_bias, out->gpu_buffer},
            &pc, sizeof(pc), gx, gy, gz);
    } else {
        // CPU conv2d
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t oc = 0; oc < C_out; ++oc) {
                float b = (bias != nullptr) ? bias->cpu_data[oc] : 0.0f;
                for (uint32_t oh = 0; oh < H_out; ++oh) {
                    for (uint32_t ow = 0; ow < W_out; ++ow) {
                        float acc = b;
                        int in_h_base = int(oh * stride_h) - int(pad_h);
                        int in_w_base = int(ow * stride_w) - int(pad_w);

                        for (uint32_t ic = 0; ic < C_in; ++ic) {
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                int ih = in_h_base + int(kh);
                                if (ih >= 0 && ih < int(H_in)) {
                                    for (uint32_t kw = 0; kw < kW; ++kw) {
                                        int iw = in_w_base + int(kw);
                                        if (iw >= 0 && iw < int(W_in)) {
                                            acc += cpu_data[n * (C_in*H_in*W_in) + ic * (H_in*W_in) + ih * W_in + iw] *
                                                   weight->cpu_data[oc * (C_in*kH*kW) + ic * (kH*kW) + kh * kW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        out->cpu_data[n * (C_out*H_out*W_out) + oc * (H_out*W_out) + oh * W_out + ow] = acc;
                    }
                }
            }
        }
    }
    return out;
}

std::shared_ptr<NativeTensor> NativeTensor::conv2d_backward_input(std::shared_ptr<NativeTensor> weight,
                                                                 const std::vector<int64_t>& in_shape,
                                                                 uint32_t stride_h, uint32_t stride_w,
                                                                 uint32_t pad_h, uint32_t pad_w) {
    auto grad_in = std::make_shared<NativeTensor>(in_shape, device_id);
    grad_in->zero_();

    uint32_t N = static_cast<uint32_t>(in_shape[0]);
    uint32_t C_in = static_cast<uint32_t>(in_shape[1]);
    uint32_t H_in = static_cast<uint32_t>(in_shape[2]);
    uint32_t W_in = static_cast<uint32_t>(in_shape[3]);

    uint32_t C_out = static_cast<uint32_t>(weight->shape[0]);
    uint32_t kH = static_cast<uint32_t>(weight->shape[2]);
    uint32_t kW = static_cast<uint32_t>(weight->shape[3]);

    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

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

        pc.N = N; pc.C_in = C_in; pc.H_in = H_in; pc.W_in = W_in;
        pc.C_out = C_out; pc.H_out = H_out; pc.W_out = W_out;
        pc.kH = kH; pc.kW = kW;
        pc.stride_h = stride_h; pc.stride_w = stride_w;
        pc.pad_h = pad_h; pc.pad_w = pad_w;

        uint32_t gx = (W_in + 7) / 8;
        uint32_t gy = (H_in + 7) / 8;
        uint32_t gz = N * ((C_in + 3) / 4);

        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_backward_input",
            {gpu_buffer, weight->gpu_buffer, grad_in->gpu_buffer},
            &pc, sizeof(pc), gx, gy, gz);
    } else {
        // CPU fallback. Previously this branch was missing entirely and the
        // function returned an all-zero gradient without raising.
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t oc = 0; oc < C_out; ++oc) {
                for (uint32_t oh = 0; oh < H_out; ++oh) {
                    for (uint32_t ow = 0; ow < W_out; ++ow) {
                        const float g = cpu_data[((n * C_out + oc) * H_out + oh) * W_out + ow];
                        const int in_h_base = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        const int in_w_base = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

                        for (uint32_t ic = 0; ic < C_in; ++ic) {
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                const int ih = in_h_base + static_cast<int>(kh);
                                if (ih < 0 || ih >= static_cast<int>(H_in)) continue;
                                for (uint32_t kw = 0; kw < kW; ++kw) {
                                    const int iw = in_w_base + static_cast<int>(kw);
                                    if (iw < 0 || iw >= static_cast<int>(W_in)) continue;
                                    const float w = weight->cpu_data[((oc * C_in + ic) * kH + kh) * kW + kw];
                                    grad_in->cpu_data[((n * C_in + ic) * H_in + ih) * W_in + iw] += g * w;
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

std::shared_ptr<NativeTensor> NativeTensor::conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                                                  const std::vector<int64_t>& weight_shape,
                                                                  uint32_t stride_h, uint32_t stride_w,
                                                                  uint32_t pad_h, uint32_t pad_w) {
    auto grad_weight = std::make_shared<NativeTensor>(weight_shape, device_id);
    grad_weight->zero_();

    uint32_t N = static_cast<uint32_t>(input->shape[0]);
    uint32_t C_in = static_cast<uint32_t>(input->shape[1]);
    uint32_t H_in = static_cast<uint32_t>(input->shape[2]);
    uint32_t W_in = static_cast<uint32_t>(input->shape[3]);

    uint32_t C_out = static_cast<uint32_t>(weight_shape[0]);
    uint32_t kH = static_cast<uint32_t>(weight_shape[2]);
    uint32_t kW = static_cast<uint32_t>(weight_shape[3]);

    uint32_t H_out = static_cast<uint32_t>(shape[2]);
    uint32_t W_out = static_cast<uint32_t>(shape[3]);

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

        pc.N = N; pc.C_in = C_in; pc.H_in = H_in; pc.W_in = W_in;
        pc.C_out = C_out; pc.H_out = H_out; pc.W_out = W_out;
        pc.kH = kH; pc.kW = kW;
        pc.stride_h = stride_h; pc.stride_w = stride_w;
        pc.pad_h = pad_h; pc.pad_w = pad_w;

        uint32_t total_pairs = C_out * C_in;
        VulkanPipelineManager::instance().dispatch(
            device_id, "conv2d_backward_weight",
            {gpu_buffer, input->gpu_buffer, grad_weight->gpu_buffer},
            &pc, sizeof(pc), total_pairs, 1, 1);
    } else {
        // CPU fallback: dL/dweight for a 4D NCHW convolution.
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t oc = 0; oc < C_out; ++oc) {
                for (uint32_t oh = 0; oh < H_out; ++oh) {
                    for (uint32_t ow = 0; ow < W_out; ++ow) {
                        const float g = cpu_data[((n * C_out + oc) * H_out + oh) * W_out + ow];
                        const int in_h_base = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
                        const int in_w_base = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

                        for (uint32_t ic = 0; ic < C_in; ++ic) {
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                const int ih = in_h_base + static_cast<int>(kh);
                                if (ih < 0 || ih >= static_cast<int>(H_in)) continue;
                                for (uint32_t kw = 0; kw < kW; ++kw) {
                                    const int iw = in_w_base + static_cast<int>(kw);
                                    if (iw < 0 || iw >= static_cast<int>(W_in)) continue;
                                    const float x = input->cpu_data[((n * C_in + ic) * H_in + ih) * W_in + iw];
                                    grad_weight->cpu_data[((oc * C_in + ic) * kH + kh) * kW + kw] += g * x;
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

std::shared_ptr<NativeTensor> NativeTensor::layernorm(std::shared_ptr<NativeTensor> gamma,
                                                     std::shared_ptr<NativeTensor> beta, float eps) {
    auto out = std::make_shared<NativeTensor>(shape, device_id);
    uint32_t cols = static_cast<uint32_t>(shape.back());
    uint32_t rows = static_cast<uint32_t>(numel / cols);

    if (device_id >= 0) {
        struct {
            uint32_t num_rows;
            uint32_t cols;
            float eps;
            uint32_t has_affine;
        } pc{rows, cols, eps, (gamma != nullptr) ? 1u : 0u};

        std::shared_ptr<VulkanBuffer> b_gamma = gamma ? gamma->gpu_buffer : out->gpu_buffer;
        std::shared_ptr<VulkanBuffer> b_beta = beta ? beta->gpu_buffer : out->gpu_buffer;

        VulkanPipelineManager::instance().dispatch(
            device_id, "layernorm",
            {gpu_buffer, b_gamma, b_beta, out->gpu_buffer},
            &pc, sizeof(pc), 1, rows, 1);
    } else {
        // CPU fallback
        for (uint32_t r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) mean += cpu_data[r * cols + c];
            mean /= cols;

            float var = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) {
                float diff = cpu_data[r * cols + c] - mean;
                var += diff * diff;
            }
            var /= cols;
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (uint32_t c = 0; c < cols; ++c) {
                float norm = (cpu_data[r * cols + c] - mean) * inv_std;
                if (gamma) norm = norm * gamma->cpu_data[c] + (beta ? beta->cpu_data[c] : 0.0f);
                out->cpu_data[r * cols + c] = norm;
            }
        }
    }
    return out;
}

} // namespace vultorch
