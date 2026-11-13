#pragma once

#include "runtime/vk_context.hpp"
#include "runtime/vk_memory.hpp"
#include "runtime/vk_pipeline.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace vultorch {

class NativeTensor : public std::enable_shared_from_this<NativeTensor> {
public:
    int device_id; // -1 for CPU, >= 0 for Vulkan GPU device
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    size_t numel;

    std::vector<float> cpu_data;
    std::shared_ptr<VulkanBuffer> gpu_buffer;

    // Constructs metadata only. Used by reshape_view(), which then shares
    // storage with an existing tensor; no buffer is allocated.
    struct MetadataOnly {};

    NativeTensor(const std::vector<int64_t>& shape, int dev_id = 0);
    NativeTensor(const std::vector<int64_t>& shape, int dev_id, MetadataOnly);
    ~NativeTensor() = default;

    static std::shared_ptr<NativeTensor> create(const std::vector<int64_t>& shape, int dev_id = 0);
    static std::shared_ptr<NativeTensor> from_float_array(const float* data, const std::vector<int64_t>& shape, int dev_id = 0);

    bool is_contiguous() const;
    void compute_default_strides();

    void copy_from_host(const float* src, size_t count);
    void copy_to_host(float* dst, size_t count);

    std::shared_ptr<NativeTensor> to_device(int target_device_id);
    std::shared_ptr<NativeTensor> clone();
    void fill_(float val);
    void zero_();

    // Materialises a strided view into a new contiguous tensor. `src_strides`
    // are the source strides expressed in the output's coordinate order, which
    // makes transpose, permute and expand the same operation.
    std::shared_ptr<NativeTensor> strided_copy(const std::vector<int64_t>& out_shape,
                                              const std::vector<int64_t>& src_strides) const;

    // Shares this tensor's storage with a new shape. Only valid when this
    // tensor is contiguous and the element count matches. On GPU nothing is
    // copied; on CPU the buffer is copied (the CPU path has no shared storage).
    std::shared_ptr<NativeTensor> reshape_view(const std::vector<int64_t>& new_shape) const;

    // Copies from another tensor on the same device without a host round trip.
    void copy_from_native(const std::shared_ptr<NativeTensor>& src);

    // High performance GPU Compute Operations
    std::shared_ptr<NativeTensor> binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other);
    std::shared_ptr<NativeTensor> binary_op_scalar(uint32_t op_type, float scalar_val, bool reverse = false);
    std::shared_ptr<NativeTensor> unary_op(uint32_t op_type, float p1 = 0.0f, float p2 = 0.0f);
    std::shared_ptr<NativeTensor> matmul(std::shared_ptr<NativeTensor> other);
    std::shared_ptr<NativeTensor> reduce_sum(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_mean(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_max(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> softmax(int64_t dim = -1, bool is_log = false);

    // Fused softmax cross-entropy over the last axis of a 2-D [rows, cols]
    // tensor. `targets` is a [rows] tensor of class indices stored as floats.
    // per_row() returns [rows]; grad() returns [rows, cols] holding
    // (softmax - onehot), unscaled so the caller can apply the reduction.
    std::shared_ptr<NativeTensor> cross_entropy_per_row(const std::shared_ptr<NativeTensor>& targets) const;
    std::shared_ptr<NativeTensor> cross_entropy_grad(const std::shared_ptr<NativeTensor>& targets) const;
    std::shared_ptr<NativeTensor> conv2d(std::shared_ptr<NativeTensor> weight, std::shared_ptr<NativeTensor> bias,
                                        uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w);
    std::shared_ptr<NativeTensor> conv2d_backward_input(std::shared_ptr<NativeTensor> weight,
                                                        const std::vector<int64_t>& in_shape,
                                                        uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w);
    std::shared_ptr<NativeTensor> conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                                         const std::vector<int64_t>& weight_shape,
                                                         uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w);
    std::shared_ptr<NativeTensor> layernorm(std::shared_ptr<NativeTensor> gamma, std::shared_ptr<NativeTensor> beta, float eps);
};

} // namespace vultorch
