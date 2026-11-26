#include "tensor/tensor_internal.hpp"

namespace vultorch {

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
  bump_version();
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
  bump_version();
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
  bump_version();

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


} // namespace vultorch
