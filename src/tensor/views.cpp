#include "tensor/tensor_internal.hpp"

namespace vultorch {

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

std::shared_ptr<NativeTensor> NativeTensor::slice_dim0(int64_t start,
                                                       int64_t end) const {
  if (shape.empty()) {
    throw std::runtime_error("slice_dim0 requires at least 1D tensor");
  }
  return slice_view(0, start, end, 1);
}


} // namespace vultorch
