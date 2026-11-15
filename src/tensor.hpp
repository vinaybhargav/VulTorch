#pragma once

#include "dtype.hpp"
#include "runtime/vk_context.hpp"
#include "runtime/vk_memory.hpp"
#include "runtime/vk_pipeline.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace vultorch {

struct TensorStorage {
    std::vector<uint8_t> cpu_bytes;
    std::shared_ptr<VulkanBuffer> gpu_buffer;
    uint64_t id = 0;

    static uint64_t next_id() {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
};

class NativeTensor : public std::enable_shared_from_this<NativeTensor> {
public:
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    size_t numel = 0;
    int device_id = -1; // -1 = CPU, >= 0 = Vulkan GPU ID
    ScalarType dtype = ScalarType::Float32;
    size_t storage_offset = 0; // in elements
    std::shared_ptr<TensorStorage> storage;

    NativeTensor(const std::vector<int64_t>& shape, int dev_id = -1,
                 ScalarType dt = ScalarType::Float32);

    // Zero-copy view constructor sharing an existing TensorStorage.
    NativeTensor(const std::vector<int64_t>& view_shape,
                 const std::vector<int64_t>& view_strides,
                 size_t offset_elems,
                 std::shared_ptr<TensorStorage> shared_storage,
                 int dev_id,
                 ScalarType dt);

    static std::shared_ptr<NativeTensor> create(const std::vector<int64_t>& shape,
                                                int dev_id = -1,
                                                ScalarType dt = ScalarType::Float32);

    static std::shared_ptr<NativeTensor> from_float_array(const float* data,
                                                          const std::vector<int64_t>& shape,
                                                          int dev_id = -1);

    static std::shared_ptr<NativeTensor> from_raw_bytes(const void* data,
                                                        size_t byte_count,
                                                        const std::vector<int64_t>& shape,
                                                        int dev_id,
                                                        ScalarType dt);

    void compute_default_strides();
    bool is_contiguous() const;
    bool is_dense_zero_offset() const { return is_contiguous() && storage_offset == 0; }

    size_t itemsize() const { return element_size(dtype); }
    size_t nbytes() const { return numel * itemsize(); }
    uint64_t storage_id() const { return storage ? storage->id : 0; }
    uintptr_t data_ptr() const;
    std::string dtype_str() const { return dtype_name(dtype); }

    std::shared_ptr<VulkanBuffer> gpu_buf() const { return storage ? storage->gpu_buffer : nullptr; }
    uint8_t* cpu_raw_ptr() { return (storage && !storage->cpu_bytes.empty()) ? storage->cpu_bytes.data() : nullptr; }
    const uint8_t* cpu_raw_ptr() const { return (storage && !storage->cpu_bytes.empty()) ? storage->cpu_bytes.data() : nullptr; }
    float* cpu_f32() { return reinterpret_cast<float*>(cpu_raw_ptr()); }
    const float* cpu_f32() const { return reinterpret_cast<const float*>(cpu_raw_ptr()); }

    void copy_from_host(const float* src, size_t count);
    void copy_to_host(float* dst, size_t count);
    void copy_from_raw_bytes(const void* src, size_t byte_count);
    void to_raw_bytes(void* dst, size_t byte_count) const;

    std::shared_ptr<NativeTensor> to_device(int target_device_id);
    std::shared_ptr<NativeTensor> clone() const;
    std::shared_ptr<NativeTensor> contiguous() const;
    std::shared_ptr<NativeTensor> force_contiguous() const;
    std::shared_ptr<NativeTensor> cast(ScalarType target_dtype) const;

    std::shared_ptr<NativeTensor> strided_copy(const std::vector<int64_t>& out_shape,
                                               const std::vector<int64_t>& src_strides) const;

    // Zero-copy view ops (all return views sharing storage with 0 GPU dispatches)
    std::shared_ptr<NativeTensor> as_strided(const std::vector<int64_t>& new_shape,
                                             const std::vector<int64_t>& new_strides,
                                             int64_t new_offset = -1) const;
    std::shared_ptr<NativeTensor> transpose_view(int64_t dim0, int64_t dim1) const;
    std::shared_ptr<NativeTensor> permute_view(const std::vector<int64_t>& dims) const;
    std::shared_ptr<NativeTensor> expand_view(const std::vector<int64_t>& new_shape) const;
    std::shared_ptr<NativeTensor> slice_view(int64_t dim, int64_t start, int64_t stop, int64_t step = 1) const;
    std::shared_ptr<NativeTensor> reshape_view(const std::vector<int64_t>& new_shape) const;

    void copy_from_native(const std::shared_ptr<NativeTensor>& src);
    std::shared_ptr<NativeTensor> slice_dim0(int64_t start, int64_t end) const;
    static std::shared_ptr<NativeTensor> cat_dim0(
        const std::vector<std::shared_ptr<NativeTensor>>& tensors, int target_device_id);

    void fill_(float val);
    void zero_();
    void adam_step_(const std::shared_ptr<NativeTensor>& grad,
                    const std::shared_ptr<NativeTensor>& exp_avg,
                    const std::shared_ptr<NativeTensor>& exp_avg_sq,
                    float lr, float beta1, float beta2, float eps,
                    float weight_decay, bool decoupled, int64_t step);
    void sgd_step_(const std::shared_ptr<NativeTensor>& grad,
                   const std::shared_ptr<NativeTensor>& momentum_buf,
                   float lr, float momentum, float dampening,
                   float weight_decay, bool nesterov, bool first_step);
    void random_philox_(uint32_t mode, float p1, float p2,
                        uint64_t seed, uint32_t offset,
                        const std::shared_ptr<NativeTensor>& src = nullptr);

    std::shared_ptr<NativeTensor> binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other);
    std::shared_ptr<NativeTensor> binary_op_scalar(uint32_t op_type, float scalar_val, bool reverse = false);
    std::shared_ptr<NativeTensor> unary_op(uint32_t op_type, float p1 = 0.0f, float p2 = 0.0f);
    std::shared_ptr<NativeTensor> matmul(std::shared_ptr<NativeTensor> other,
                                         bool trans_a = false, bool trans_b = false,
                                         std::shared_ptr<NativeTensor> bias = nullptr);

    std::shared_ptr<NativeTensor> reduce_sum(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_mean(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_max(int64_t dim = -1, bool keepdim = false);

    std::shared_ptr<NativeTensor> softmax(int64_t dim = -1, bool is_log = false);
    std::shared_ptr<NativeTensor> cross_entropy_per_row(const std::shared_ptr<NativeTensor>& targets) const;
    std::shared_ptr<NativeTensor> cross_entropy_grad(const std::shared_ptr<NativeTensor>& targets,
                                                     const std::shared_ptr<NativeTensor>& grad_out = nullptr,
                                                     float scale = 1.0f) const;

    std::shared_ptr<NativeTensor> conv2d(std::shared_ptr<NativeTensor> weight,
                                         std::shared_ptr<NativeTensor> bias = nullptr,
                                         uint32_t stride_h = 1, uint32_t stride_w = 1,
                                         uint32_t pad_h = 0, uint32_t pad_w = 0);
    std::shared_ptr<NativeTensor> conv2d_backward_input(std::shared_ptr<NativeTensor> weight,
                                                        const std::vector<int64_t>& in_shape,
                                                        uint32_t stride_h = 1, uint32_t stride_w = 1,
                                                        uint32_t pad_h = 0, uint32_t pad_w = 0);
    std::shared_ptr<NativeTensor> conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                                         const std::vector<int64_t>& weight_shape,
                                                         uint32_t stride_h = 1, uint32_t stride_w = 1,
                                                         uint32_t pad_h = 0, uint32_t pad_w = 0);

    std::shared_ptr<NativeTensor> layernorm(std::shared_ptr<NativeTensor> gamma = nullptr,
                                            std::shared_ptr<NativeTensor> beta = nullptr,
                                            float eps = 1e-5f, uint32_t norm_dims = 1);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    layernorm_backward(std::shared_ptr<NativeTensor> grad_out,
                       std::shared_ptr<NativeTensor> gamma,
                       float eps, uint32_t norm_dims,
                       bool need_dx = true, bool need_dgamma = true, bool need_dbeta = true);

    std::shared_ptr<NativeTensor> rmsnorm(std::shared_ptr<NativeTensor> gamma = nullptr,
                                          float eps = 1e-6f, uint32_t norm_dims = 1);
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    rmsnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                     std::shared_ptr<NativeTensor> gamma,
                     float eps, uint32_t norm_dims,
                     bool need_dx = true, bool need_dgamma = true);

    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    maxpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
              uint32_t pad_h, uint32_t pad_w);
    std::shared_ptr<NativeTensor> maxpool2d_backward(
        std::shared_ptr<NativeTensor> indices,
        const std::vector<int64_t>& in_shape,
        uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
        uint32_t pad_h, uint32_t pad_w);

    std::shared_ptr<NativeTensor> avgpool2d(
        uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
        uint32_t pad_h, uint32_t pad_w, bool count_include_pad = true,
        bool is_adaptive = false, uint32_t adaptive_out_h = 1, uint32_t adaptive_out_w = 1);
    std::shared_ptr<NativeTensor> avgpool2d_backward(
        const std::vector<int64_t>& in_shape,
        uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w,
        uint32_t pad_h, uint32_t pad_w, bool count_include_pad = true,
        bool is_adaptive = false);

    std::shared_ptr<NativeTensor> embedding(
        std::shared_ptr<NativeTensor> weight, int64_t padding_idx = -1);
    std::shared_ptr<NativeTensor> embedding_backward(
        std::shared_ptr<NativeTensor> indices, int64_t num_embeddings,
        int64_t padding_idx = -1);

private:
    struct MetadataOnly {};
    NativeTensor(const std::vector<int64_t>& s, int dev_id, MetadataOnly);

    std::shared_ptr<NativeTensor> broadcast_binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other);
};

} // namespace vultorch
