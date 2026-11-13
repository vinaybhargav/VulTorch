#include "tensor.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace vultorch {

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
        // CPU fallback
        auto out = std::make_shared<NativeTensor>(shape, -1);
        for (size_t i = 0; i < numel; ++i) {
            float a = cpu_data[i];
            float b = other->cpu_data[i];
            switch (op_type) {
                case 0: out->cpu_data[i] = a + b; break;
                case 1: out->cpu_data[i] = a - b; break;
                case 2: out->cpu_data[i] = a * b; break;
                case 3: out->cpu_data[i] = a / b; break;
                case 4: out->cpu_data[i] = std::pow(a, b); break;
                case 5: out->cpu_data[i] = std::max(a, b); break;
                case 6: out->cpu_data[i] = std::min(a, b); break;
                default: out->cpu_data[i] = a + b; break;
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
        // Multi-dimensional Broadcasting
        size_t max_ndim = std::max(shape.size(), other->shape.size());
        if (max_ndim > 6) throw std::runtime_error("Broadcasting supports up to 6 dimensions");

        std::vector<int64_t> out_shape(max_ndim);
        std::vector<uint32_t> strideA_expanded(6, 0);
        std::vector<uint32_t> strideB_expanded(6, 0);

        for (int i = 0; i < static_cast<int>(max_ndim); ++i) {
            int a_idx = static_cast<int>(shape.size()) - 1 - i;
            int b_idx = static_cast<int>(other->shape.size()) - 1 - i;
            int out_idx = static_cast<int>(max_ndim) - 1 - i;

            int64_t dim_a = (a_idx >= 0) ? shape[a_idx] : 1;
            int64_t dim_b = (b_idx >= 0) ? other->shape[b_idx] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::runtime_error("Cannot broadcast shapes!");
            }
            out_shape[out_idx] = std::max(dim_a, dim_b);

            if (a_idx >= 0 && dim_a > 1) strideA_expanded[out_idx] = static_cast<uint32_t>(strides[a_idx]);
            if (b_idx >= 0 && dim_b > 1) strideB_expanded[out_idx] = static_cast<uint32_t>(other->strides[b_idx]);
        }

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

        for (size_t i = 0; i < max_ndim; ++i) {
            pc.shape[i] = static_cast<uint32_t>(out_shape[i]);
            pc.strideA[i] = strideA_expanded[i];
            pc.strideB[i] = strideB_expanded[i];
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
            switch (op_type) {
                case 0: out->cpu_data[i] = a + b; break;
                case 1: out->cpu_data[i] = a - b; break;
                case 2: out->cpu_data[i] = a * b; break;
                case 3: out->cpu_data[i] = a / b; break;
                default: out->cpu_data[i] = a + b; break;
            }
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
            float x = cpu_data[i];
            switch (op_type) {
                case 0: out->cpu_data[i] = -x; break;
                case 1: out->cpu_data[i] = std::exp(x); break;
                case 2: out->cpu_data[i] = std::log(x); break;
                case 3: out->cpu_data[i] = std::sqrt(x); break;
                case 8: out->cpu_data[i] = std::tanh(x); break;
                case 9: out->cpu_data[i] = 1.0f / (1.0f + std::exp(-x)); break;
                case 10: out->cpu_data[i] = std::max(x, 0.0f); break;
                case 15: out->cpu_data[i] = x * p1 + p2; break;
                default: out->cpu_data[i] = x; break;
            }
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

    uint32_t M = static_cast<uint32_t>(shape[shape.size() - 2]);
    uint32_t K1 = static_cast<uint32_t>(shape[shape.size() - 1]);
    uint32_t K2 = static_cast<uint32_t>(other->shape[other->shape.size() - 2]);
    uint32_t N = static_cast<uint32_t>(other->shape[other->shape.size() - 1]);

    if (K1 != K2) {
        throw std::runtime_error("Inner dimensions must match for matmul: " + std::to_string(K1) + " vs " + std::to_string(K2));
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
        pc.strideB_b = static_cast<uint32_t>(K1 * N);
        pc.strideB_k = static_cast<uint32_t>(N);
        pc.strideB_n = 1;
        pc.strideC_b = static_cast<uint32_t>(M * N);
        pc.strideC_m = static_cast<uint32_t>(N);
        pc.strideC_n = 1;

        uint32_t group_x = (N + 31) / 32;
        uint32_t group_y = (M + 31) / 32;
        uint32_t group_z = static_cast<uint32_t>(batch_count);

        VulkanPipelineManager::instance().dispatch(
            device_id, "matmul",
            {gpu_buffer, other->gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), group_x, group_y, group_z);
    } else {
        // CPU MatMul fallback
        for (size_t b = 0; b < batch_count; ++b) {
            size_t offA = b * M * K1;
            size_t offB = b * K1 * N;
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

    // General reduction on CPU (handles arbitrary dimensions without infinite recursion)
    auto cpu_in = (device_id >= 0) ? to_device(-1) : shared_from_this();
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

    if (device_id >= 0) {
        return cpu_out->to_device(device_id);
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

    auto cpu_in = (device_id >= 0) ? to_device(-1) : shared_from_this();
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

    if (device_id >= 0) {
        return cpu_out->to_device(device_id);
    }
    return cpu_out;
}

std::shared_ptr<NativeTensor> NativeTensor::softmax(int64_t dim, bool is_log) {
    if (dim < 0) dim += shape.size();
    uint32_t cols = static_cast<uint32_t>(shape.back());
    uint32_t rows = static_cast<uint32_t>(numel / cols);

    auto out = std::make_shared<NativeTensor>(shape, device_id);

    if (device_id >= 0 && dim == static_cast<int64_t>(shape.size()) - 1) {
        struct {
            uint32_t num_rows;
            uint32_t cols;
            uint32_t is_log_softmax;
        } pc{rows, cols, is_log ? 1u : 0u};

        VulkanPipelineManager::instance().dispatch(
            device_id, "softmax",
            {gpu_buffer, out->gpu_buffer},
            &pc, sizeof(pc), 1, rows, 1);
    } else {
        // CPU fallback for arbitrary dimension
        auto cpu_t = to_device(-1);
        auto out_cpu = std::make_shared<NativeTensor>(shape, -1);
        for (uint32_t r = 0; r < rows; ++r) {
            float m = -1e30f;
            for (uint32_t c = 0; c < cols; ++c) m = std::max(m, cpu_t->cpu_data[r * cols + c]);
            float s = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) s += std::exp(cpu_t->cpu_data[r * cols + c] - m);
            float inv_s = 1.0f / s;
            float log_s = std::log(s);
            for (uint32_t c = 0; c < cols; ++c) {
                float val = cpu_t->cpu_data[r * cols + c];
                out_cpu->cpu_data[r * cols + c] = is_log ? ((val - m) - log_s) : (std::exp(val - m) * inv_s);
            }
        }
        return out_cpu->to_device(device_id);
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
