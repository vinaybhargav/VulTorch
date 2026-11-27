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

namespace vultorch
{

struct TensorStorage
{
    std::vector<uint8_t> cpu_bytes;
    std::shared_ptr<VulkanBuffer> gpu_buffer;
    uint64_t id = 0;
    uint32_t version = 0;

    static uint64_t next_id()
    {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
};

class NativeTensor : public std::enable_shared_from_this<NativeTensor>
{
  public:
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    size_t numel = 0;
    int device_id = -1; // -1 = CPU, >= 0 = Vulkan GPU ID
    ScalarType dtype = ScalarType::Float32;
    size_t storage_offset = 0; // in elements
    std::shared_ptr<TensorStorage> storage;

    NativeTensor(const std::vector<int64_t> &shape, int dev_id = -1,
                 ScalarType dt = ScalarType::Float32);

    // Zero-copy view constructor sharing an existing TensorStorage.
    NativeTensor(const std::vector<int64_t> &view_shape, const std::vector<int64_t> &view_strides,
                 size_t offset_elems, std::shared_ptr<TensorStorage> shared_storage, int dev_id,
                 ScalarType dt);

    static std::shared_ptr<NativeTensor> create(const std::vector<int64_t> &shape, int dev_id = -1,
                                                ScalarType dt = ScalarType::Float32);

    static std::shared_ptr<NativeTensor>
    from_float_array(const float *data, const std::vector<int64_t> &shape, int dev_id = -1);

    static std::shared_ptr<NativeTensor> from_raw_bytes(const void *data, size_t byte_count,
                                                        const std::vector<int64_t> &shape,
                                                        int dev_id, ScalarType dt);

    void compute_default_strides();
    bool is_contiguous() const;
    bool is_dense_zero_offset() const
    {
        return is_contiguous() && storage_offset == 0;
    }

    size_t itemsize() const
    {
        return element_size(dtype);
    }
    size_t nbytes() const
    {
        return numel * itemsize();
    }
    uint64_t storage_id() const
    {
        return storage ? storage->id : 0;
    }
    uint32_t version() const
    {
        return storage ? storage->version : 0;
    }
    void bump_version()
    {
        if (storage)
            ++storage->version;
    }
    uintptr_t data_ptr() const;
    std::string dtype_str() const
    {
        return dtype_name(dtype);
    }

    std::shared_ptr<VulkanBuffer> gpu_buf() const
    {
        return storage ? storage->gpu_buffer : nullptr;
    }
    uint8_t *cpu_raw_ptr()
    {
        return (storage && !storage->cpu_bytes.empty()) ? storage->cpu_bytes.data() : nullptr;
    }
    const uint8_t *cpu_raw_ptr() const
    {
        return (storage && !storage->cpu_bytes.empty()) ? storage->cpu_bytes.data() : nullptr;
    }
    float *cpu_f32()
    {
        return reinterpret_cast<float *>(cpu_raw_ptr());
    }
    const float *cpu_f32() const
    {
        return reinterpret_cast<const float *>(cpu_raw_ptr());
    }

    void copy_from_host(const float *src, size_t count);
    void copy_to_host(float *dst, size_t count);
    void copy_from_raw_bytes(const void *src, size_t byte_count);
    void to_raw_bytes(void *dst, size_t byte_count) const;

    std::shared_ptr<NativeTensor> to_device(int target_device_id);
    std::shared_ptr<NativeTensor> clone() const;
    std::shared_ptr<NativeTensor> contiguous() const;
    std::shared_ptr<NativeTensor> force_contiguous() const;
    std::shared_ptr<NativeTensor> cast(ScalarType target_dtype) const;

    std::shared_ptr<NativeTensor> strided_copy(const std::vector<int64_t> &out_shape,
                                               const std::vector<int64_t> &src_strides) const;

    // Zero-copy view ops (all return views sharing storage with 0 GPU dispatches)
    std::shared_ptr<NativeTensor> as_strided(const std::vector<int64_t> &new_shape,
                                             const std::vector<int64_t> &new_strides,
                                             int64_t new_offset = -1) const;
    std::shared_ptr<NativeTensor> transpose_view(int64_t dim0, int64_t dim1) const;
    std::shared_ptr<NativeTensor> permute_view(const std::vector<int64_t> &dims) const;
    std::shared_ptr<NativeTensor> expand_view(const std::vector<int64_t> &new_shape) const;
    std::shared_ptr<NativeTensor> slice_view(int64_t dim, int64_t start, int64_t stop,
                                             int64_t step = 1) const;
    std::shared_ptr<NativeTensor> reshape_view(const std::vector<int64_t> &new_shape) const;

    void copy_from_native(const std::shared_ptr<NativeTensor> &src);
    std::shared_ptr<NativeTensor> slice_dim0(int64_t start, int64_t end) const;
    static std::shared_ptr<NativeTensor>
    cat_dim0(const std::vector<std::shared_ptr<NativeTensor>> &tensors, int target_device_id);
    static std::shared_ptr<NativeTensor>
    cat(const std::vector<std::shared_ptr<NativeTensor>> &tensors, int64_t dim,
        int target_device_id);

    // Phase 3.1: Indexing & Conditional
    static std::shared_ptr<NativeTensor> where_op(const std::shared_ptr<NativeTensor> &cond,
                                                  const std::shared_ptr<NativeTensor> &a,
                                                  const std::shared_ptr<NativeTensor> &b);
    std::shared_ptr<NativeTensor> masked_fill(const std::shared_ptr<NativeTensor> &mask,
                                              float value) const;
    std::shared_ptr<NativeTensor> gather(int64_t dim,
                                         const std::shared_ptr<NativeTensor> &index) const;
    std::shared_ptr<NativeTensor> scatter(int64_t dim, const std::shared_ptr<NativeTensor> &index,
                                          const std::shared_ptr<NativeTensor> &src,
                                          float scalar_val = 0.0f, bool use_scalar = false,
                                          bool accumulate = false) const;
    std::shared_ptr<NativeTensor> index_select(int64_t dim,
                                               const std::shared_ptr<NativeTensor> &index) const;
    void index_add_(int64_t dim, const std::shared_ptr<NativeTensor> &index,
                    const std::shared_ptr<NativeTensor> &src, float alpha = 1.0f);

    // Phase 3.2: Manipulation (Pad, Tril/Triu, Flip, Roll)
    std::shared_ptr<NativeTensor> pad_nd(const std::vector<int64_t> &pad_before,
                                         const std::vector<int64_t> &pad_after, uint32_t mode = 0,
                                         float pad_value = 0.0f) const;
    std::shared_ptr<NativeTensor> pad_nd_backward(const std::vector<int64_t> &in_shape,
                                                  const std::vector<int64_t> &pad_before,
                                                  uint32_t mode = 0) const;
    std::shared_ptr<NativeTensor> tril_triu(bool is_upper, int64_t diagonal = 0) const;
    std::shared_ptr<NativeTensor> flip(const std::vector<int64_t> &dims) const;
    std::shared_ptr<NativeTensor> roll(const std::vector<int64_t> &shifts,
                                       const std::vector<int64_t> &dims) const;

    // Phase 3.3: Extended Reductions, Prefix Scans, Sort / Top-K
    std::shared_ptr<NativeTensor> argmax_argmin(int64_t dim = -999999, bool keepdim = false,
                                                bool is_min = false) const;
    std::shared_ptr<NativeTensor> reduce_prod_all_any(int64_t dim = -999999, bool keepdim = false,
                                                      uint32_t mode = 0) const;
    std::shared_ptr<NativeTensor> scan_op(int64_t dim, uint32_t mode) const;
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    sort_topk(int64_t dim, int64_t k = -1, bool descending = true) const;

    void fill_(float val);
    void zero_();
    void adam_step_(const std::shared_ptr<NativeTensor> &grad,
                    const std::shared_ptr<NativeTensor> &exp_avg,
                    const std::shared_ptr<NativeTensor> &exp_avg_sq, float lr, float beta1,
                    float beta2, float eps, float weight_decay, bool decoupled, int64_t step);
    void sgd_step_(const std::shared_ptr<NativeTensor> &grad,
                   const std::shared_ptr<NativeTensor> &momentum_buf, float lr, float momentum,
                   float dampening, float weight_decay, bool nesterov, bool first_step);
    void random_philox_(uint32_t mode, float p1, float p2, uint64_t seed, uint32_t offset,
                        const std::shared_ptr<NativeTensor> &src = nullptr);

    std::shared_ptr<NativeTensor> binary_op(uint32_t op_type, std::shared_ptr<NativeTensor> other);
    std::shared_ptr<NativeTensor> binary_op_scalar(uint32_t op_type, float scalar_val,
                                                   bool reverse = false);
    std::shared_ptr<NativeTensor> unary_op(uint32_t op_type, float p1 = 0.0f, float p2 = 0.0f);
    std::shared_ptr<NativeTensor> matmul(std::shared_ptr<NativeTensor> other, bool trans_a = false,
                                         bool trans_b = false,
                                         std::shared_ptr<NativeTensor> bias = nullptr,
                                         uint32_t epilogue = 0);

    std::shared_ptr<NativeTensor> reduce_sum(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_mean(int64_t dim = -1, bool keepdim = false);
    std::shared_ptr<NativeTensor> reduce_max(int64_t dim = -1, bool keepdim = false);

    std::shared_ptr<NativeTensor> softmax(int64_t dim = -1, bool is_log = false);
    std::shared_ptr<NativeTensor>
    cross_entropy_per_row(const std::shared_ptr<NativeTensor> &targets) const;
    std::shared_ptr<NativeTensor>
    cross_entropy_grad(const std::shared_ptr<NativeTensor> &targets,
                       const std::shared_ptr<NativeTensor> &grad_out = nullptr,
                       float scale = 1.0f) const;

    std::shared_ptr<NativeTensor> conv2d(std::shared_ptr<NativeTensor> weight,
                                         std::shared_ptr<NativeTensor> bias = nullptr,
                                         uint32_t stride_h = 1, uint32_t stride_w = 1,
                                         uint32_t pad_h = 0, uint32_t pad_w = 0,
                                         uint32_t groups = 1, bool use_im2col = true);
    std::shared_ptr<NativeTensor> conv2d_backward_input(std::shared_ptr<NativeTensor> weight,
                                                        const std::vector<int64_t> &in_shape,
                                                        uint32_t stride_h = 1,
                                                        uint32_t stride_w = 1, uint32_t pad_h = 0,
                                                        uint32_t pad_w = 0, uint32_t groups = 1,
                                                        bool use_im2col = true);
    std::shared_ptr<NativeTensor> conv2d_backward_weight(std::shared_ptr<NativeTensor> input,
                                                         const std::vector<int64_t> &weight_shape,
                                                         uint32_t stride_h = 1,
                                                         uint32_t stride_w = 1, uint32_t pad_h = 0,
                                                         uint32_t pad_w = 0, uint32_t groups = 1,
                                                         bool use_im2col = true);

    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>>
    batchnorm(std::shared_ptr<NativeTensor> running_mean, std::shared_ptr<NativeTensor> running_var,
              std::shared_ptr<NativeTensor> gamma = nullptr,
              std::shared_ptr<NativeTensor> beta = nullptr, bool training = true,
              float momentum = 0.1f, float eps = 1e-5f);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>>
    batchnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                       std::shared_ptr<NativeTensor> save_mean,
                       std::shared_ptr<NativeTensor> save_invstd,
                       std::shared_ptr<NativeTensor> gamma = nullptr, float eps = 1e-5f,
                       bool need_dx = true, bool need_dgamma = true, bool need_dbeta = true);

    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    groupnorm(uint32_t num_groups, std::shared_ptr<NativeTensor> gamma = nullptr,
              std::shared_ptr<NativeTensor> beta = nullptr, float eps = 1e-5f);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>>
    groupnorm_backward(std::shared_ptr<NativeTensor> grad_out,
                       std::shared_ptr<NativeTensor> mean_invstd, uint32_t num_groups,
                       std::shared_ptr<NativeTensor> gamma = nullptr, bool need_dx = true,
                       bool need_dgamma = true, bool need_dbeta = true);

    std::shared_ptr<NativeTensor> snake(std::shared_ptr<NativeTensor> alpha = nullptr,
                                        std::shared_ptr<NativeTensor> beta = nullptr,
                                        int64_t channel_dim = 1, bool log_scale = false,
                                        float scalar_a = 1.0f, float eps = 1e-9f, uint32_t mode = 0,
                                        std::shared_ptr<NativeTensor> gamma = nullptr,
                                        float scalar_gamma = 0.0f);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    snake_backward(std::shared_ptr<NativeTensor> grad_out,
                   std::shared_ptr<NativeTensor> alpha = nullptr,
                   std::shared_ptr<NativeTensor> beta = nullptr, int64_t channel_dim = 1,
                   bool log_scale = false, float scalar_a = 1.0f, float eps = 1e-9f,
                   bool need_dx = true, bool need_dalpha = false, bool need_dbeta = false,
                   uint32_t mode = 0, std::shared_ptr<NativeTensor> gamma = nullptr,
                   float scalar_gamma = 0.0f, bool need_dgamma = false);

    std::shared_ptr<NativeTensor> selective_scan(std::shared_ptr<NativeTensor> delta,
                                                 std::shared_ptr<NativeTensor> A,
                                                 std::shared_ptr<NativeTensor> B_mat,
                                                 std::shared_ptr<NativeTensor> C_mat,
                                                 std::shared_ptr<NativeTensor> D_vec = nullptr);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    selective_scan_backward(std::shared_ptr<NativeTensor> grad_out,
                            std::shared_ptr<NativeTensor> delta, std::shared_ptr<NativeTensor> A,
                            std::shared_ptr<NativeTensor> B_mat,
                            std::shared_ptr<NativeTensor> C_mat,
                            std::shared_ptr<NativeTensor> D_vec = nullptr);

    // Phase 4: FlashAttention-2, RoPE, SwiGLU/GeGLU, QuantizedLinear, AMP Check
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    flash_attention(std::shared_ptr<NativeTensor> K, std::shared_ptr<NativeTensor> V,
                    bool is_causal = false, float scale = -1.0f);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>>
    flash_attention_backward(std::shared_ptr<NativeTensor> grad_out,
                             std::shared_ptr<NativeTensor> K, std::shared_ptr<NativeTensor> V,
                             std::shared_ptr<NativeTensor> O, std::shared_ptr<NativeTensor> L,
                             bool is_causal = false, float scale = -1.0f);

    std::shared_ptr<NativeTensor> rope(std::shared_ptr<NativeTensor> cos_t,
                                       std::shared_ptr<NativeTensor> sin_t, uint32_t seq_dim = 2,
                                       bool is_backward = false);

    std::shared_ptr<NativeTensor> glu(std::shared_ptr<NativeTensor> b, uint32_t mode = 0);
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    glu_backward(std::shared_ptr<NativeTensor> grad_out, std::shared_ptr<NativeTensor> b,
                 uint32_t mode = 0);

    std::shared_ptr<NativeTensor> gemm_quant(std::shared_ptr<NativeTensor> qweight,
                                             std::shared_ptr<NativeTensor> scales,
                                             std::shared_ptr<NativeTensor> bias = nullptr,
                                             uint32_t group_size = 64, uint32_t bits = 4);
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    quantize_weights(uint32_t group_size = 64, uint32_t bits = 4);

    static bool amp_unscale_and_check(const std::vector<std::shared_ptr<NativeTensor>> &grads,
                                      float inv_scale);

    std::shared_ptr<NativeTensor> autodiff_unary_fwd(uint32_t op_type, float param = 0.0f);
    std::shared_ptr<NativeTensor> autodiff_unary_bwd(uint32_t op_type,
                                                     std::shared_ptr<NativeTensor> grad_out,
                                                     float param = 0.0f);
    std::shared_ptr<NativeTensor>
    autodiff_binary_fwd(uint32_t op_type, std::shared_ptr<NativeTensor> other, float param = 1.0f);
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    autodiff_binary_bwd(uint32_t op_type, std::shared_ptr<NativeTensor> other,
                        std::shared_ptr<NativeTensor> grad_out, float param = 1.0f);

    std::shared_ptr<NativeTensor> layernorm(std::shared_ptr<NativeTensor> gamma = nullptr,
                                            std::shared_ptr<NativeTensor> beta = nullptr,
                                            float eps = 1e-5f, uint32_t norm_dims = 1);
    std::tuple<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>,
               std::shared_ptr<NativeTensor>>
    layernorm_backward(std::shared_ptr<NativeTensor> grad_out, std::shared_ptr<NativeTensor> gamma,
                       float eps, uint32_t norm_dims, bool need_dx = true, bool need_dgamma = true,
                       bool need_dbeta = true);

    std::shared_ptr<NativeTensor> rmsnorm(std::shared_ptr<NativeTensor> gamma = nullptr,
                                          float eps = 1e-6f, uint32_t norm_dims = 1);
    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    rmsnorm_backward(std::shared_ptr<NativeTensor> grad_out, std::shared_ptr<NativeTensor> gamma,
                     float eps, uint32_t norm_dims, bool need_dx = true, bool need_dgamma = true);

    std::pair<std::shared_ptr<NativeTensor>, std::shared_ptr<NativeTensor>>
    maxpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
              uint32_t pad_w);
    std::shared_ptr<NativeTensor> maxpool2d_backward(std::shared_ptr<NativeTensor> indices,
                                                     const std::vector<int64_t> &in_shape,
                                                     uint32_t kH, uint32_t kW, uint32_t stride_h,
                                                     uint32_t stride_w, uint32_t pad_h,
                                                     uint32_t pad_w);

    std::shared_ptr<NativeTensor> avgpool2d(uint32_t kH, uint32_t kW, uint32_t stride_h,
                                            uint32_t stride_w, uint32_t pad_h, uint32_t pad_w,
                                            bool count_include_pad = true, bool is_adaptive = false,
                                            uint32_t adaptive_out_h = 1,
                                            uint32_t adaptive_out_w = 1);
    std::shared_ptr<NativeTensor> avgpool2d_backward(const std::vector<int64_t> &in_shape,
                                                     uint32_t kH, uint32_t kW, uint32_t stride_h,
                                                     uint32_t stride_w, uint32_t pad_h,
                                                     uint32_t pad_w, bool count_include_pad = true,
                                                     bool is_adaptive = false);

    std::shared_ptr<NativeTensor> embedding(std::shared_ptr<NativeTensor> weight,
                                            int64_t padding_idx = -1);
    std::shared_ptr<NativeTensor> embedding_backward(std::shared_ptr<NativeTensor> indices,
                                                     int64_t num_embeddings,
                                                     int64_t padding_idx = -1);

  private:
    struct MetadataOnly
    {
    };
    NativeTensor(const std::vector<int64_t> &s, int dev_id, MetadataOnly);

    std::shared_ptr<NativeTensor> broadcast_binary_op(uint32_t op_type,
                                                      std::shared_ptr<NativeTensor> other);
};

} // namespace vultorch
