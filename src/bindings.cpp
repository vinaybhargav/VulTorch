#include "runtime/vk_context.hpp"
#include "runtime/vk_graph.hpp"
#include "runtime/vk_memory.hpp"
#include "runtime/vk_pipeline.hpp"
#include "tensor.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace vultorch;

namespace
{

py::dtype scalar_type_to_numpy_dtype(ScalarType dt)
{
    switch (dt)
    {
    case ScalarType::Float32:
        return py::dtype("float32");
    case ScalarType::Float16:
        return py::dtype("float16");
    case ScalarType::BFloat16:
        return py::dtype("float32"); // Exposed as fp32 to NumPy
    case ScalarType::Int32:
        return py::dtype("int32");
    case ScalarType::Int64:
        return py::dtype("int64");
    case ScalarType::Bool:
        return py::dtype("bool");
    case ScalarType::UInt8:
        return py::dtype("uint8");
    }
    return py::dtype("float32");
}

ScalarType infer_scalar_type_from_numpy(const py::array &arr)
{
    const py::dtype dt = arr.dtype();
    if (dt.is(py::dtype("float32")) || dt.is(py::dtype("float64")))
        return ScalarType::Float32;
    if (dt.is(py::dtype("float16")))
        return ScalarType::Float16;
    if (dt.is(py::dtype("int32")))
        return ScalarType::Int32;
    if (dt.is(py::dtype("int64")))
        return ScalarType::Int64;
    if (dt.is(py::dtype("bool")))
        return ScalarType::Bool;
    if (dt.is(py::dtype("uint8")))
        return ScalarType::UInt8;
    return ScalarType::Float32;
}

py::array ensure_contiguous_numpy_for_dtype(const py::array &arr, ScalarType target_dt)
{
    py::dtype target_np = scalar_type_to_numpy_dtype(target_dt);
    const bool is_c = (arr.flags() & py::array::c_style) != 0;
    if (is_c && arr.dtype().is(target_np))
    {
        return arr;
    }
    py::module_ np = py::module_::import("numpy");
    return np.attr("ascontiguousarray")(arr, target_np).cast<py::array>();
}

} // namespace

PYBIND11_MODULE(_vultorch_backend, m)
{
    m.doc() = "VulTorch: High-Performance Vulkan-Native Deep Learning Engine";

    // Ensure Vulkan pipelines and pooled buffers are cleaned up before Python
    // process teardown (while GPU drivers and ThreadPoolExecutor are still
    // cleanly shutting down).
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function(
        []()
        {
            py::gil_scoped_release release;
            VulkanManager::instance_ref().synchronize_all();
            VulkanGraph::release_all();
            VulkanMemoryManager::instance().shutdown();
            VulkanPipelineManager::instance().clear();
        }));

    // Vulkan Device & Manager functions
    m.def("get_device_count", []() { return VulkanManager::instance_ref().get_device_count(); });

    m.def("get_device_info",
          [](uint32_t device_id)
          {
              auto info = VulkanManager::instance_ref().get_device_info(device_id);
              py::dict d;
              d["id"] = info.id;
              d["name"] = info.name;
              d["vendor"] = info.vendor;
              d["api_version"] =
                  std::to_string(info.api_major) + "." + std::to_string(info.api_minor);
              d["total_memory"] = info.total_memory;
              d["is_discrete"] = info.is_discrete;
              d["has_cooperative_matrix"] = info.has_cooperative_matrix;
              d["vendor_id"] = info.vendor_id;
              d["device_id"] = info.device_id;

              py::dict caps;
              caps["subgroup_size"] = info.capabilities.subgroup_size;
              caps["max_workgroup_invocations"] = info.capabilities.max_workgroup_invocations;
              caps["max_shared_memory_bytes"] = info.capabilities.max_shared_memory_bytes;
              caps["float16_arithmetic"] = info.capabilities.float16_arithmetic;
              caps["storage_16bit"] = info.capabilities.storage_16bit;
              caps["subgroup_basic"] = info.capabilities.subgroup_basic;
              caps["subgroup_arithmetic"] = info.capabilities.subgroup_arithmetic;
              caps["shader_atomic_float"] = info.capabilities.shader_atomic_float;
              caps["fp16_coopmat_16x16"] = info.capabilities.fp16_coopmat_16x16;
              d["capabilities"] = caps;

              py::list configs;
              for (const auto &c : info.cooperative_matrix_configs)
              {
                  py::dict cd;
                  cd["M"] = c.M;
                  cd["N"] = c.N;
                  cd["K"] = c.K;
                  cd["a_type"] = c.a_type;
                  cd["b_type"] = c.b_type;
                  cd["c_type"] = c.c_type;
                  cd["result_type"] = c.result_type;
                  cd["scope"] = c.scope;
                  configs.append(cd);
              }
              d["cooperative_matrix_configs"] = configs;
              return d;
          });

    m.def("get_all_devices",
          []()
          {
              auto list = VulkanManager::instance_ref().get_all_device_info();
              py::list pylist;
              for (const auto &info : list)
              {
                  py::dict d;
                  d["id"] = info.id;
                  d["name"] = info.name;
                  d["vendor"] = info.vendor;
                  d["api_version"] =
                      std::to_string(info.api_major) + "." + std::to_string(info.api_minor);
                  d["total_memory"] = info.total_memory;
                  d["is_discrete"] = info.is_discrete;
                  d["has_cooperative_matrix"] = info.has_cooperative_matrix;
                  d["cooperative_matrix_configs_count"] = info.cooperative_matrix_configs.size();
                  pylist.append(d);
              }
              return pylist;
          });

    m.def(
        "synchronize",
        [](int device_id)
        {
            if (device_id >= 0)
            {
                VulkanManager::instance_ref().synchronize_device(device_id);
            }
        },
        py::arg("device_id") = 0, py::call_guard<py::gil_scoped_release>());

    m.def(
        "synchronize_all", []() { VulkanManager::instance_ref().synchronize_all(); },
        py::call_guard<py::gil_scoped_release>());

    m.def("set_shader_directory", [](const std::string &dir)
          { VulkanPipelineManager::instance().set_shader_directory(dir); });

    m.def("get_shader_directory",
          []() { return VulkanPipelineManager::instance().get_shader_directory(); });

    m.def(
        "recorded_dispatches", [](uint32_t device_id)
        { return VulkanManager::instance_ref().get_device(device_id)->recorded_dispatches(); },
        py::arg("device_id") = 0);

    m.def(
        "submitted_batches", [](uint32_t device_id)
        { return VulkanManager::instance_ref().get_device(device_id)->submitted_batches(); },
        py::arg("device_id") = 0);

    m.def(
        "driver_allocation_count", [](int device_id)
        { return VulkanMemoryManager::instance().driver_allocation_count(device_id); },
        py::arg("device_id") = -1);

    m.def(
        "active_arena_bytes",
        [](int device_id) { return VulkanMemoryManager::instance().active_arena_bytes(device_id); },
        py::arg("device_id") = -1);

    m.def(
        "reserved_arena_bytes", [](int device_id)
        { return VulkanMemoryManager::instance().reserved_arena_bytes(device_id); },
        py::arg("device_id") = -1);

    m.def("pooled_buffer_count",
          []() { return VulkanMemoryManager::instance().pooled_buffer_count(); });

    m.def("live_allocation_count",
          []() { return VulkanMemoryManager::instance().live_allocation_count(); });

    // NativeTensor Bindings
    py::class_<NativeTensor, std::shared_ptr<NativeTensor>>(m, "NativeTensor")
      .def(py::init([](const std::vector<int64_t> &shape, int dev_id,
                       const std::string &dt) {
             return std::make_shared<NativeTensor>(shape, dev_id,
                                                   parse_dtype(dt));
           }),
           py::arg("shape"), py::arg("device_id") = 0,
           py::arg("dtype") = "float32")
      .def_static(
          "create",
          [](const std::vector<int64_t> &shape, int dev_id,
             const std::string &dt) {
            return NativeTensor::create(shape, dev_id, parse_dtype(dt));
          },
          py::arg("shape"), py::arg("device_id") = 0,
          py::arg("dtype") = "float32")
      .def_readonly("device_id", &NativeTensor::device_id)
      .def_readonly("shape", &NativeTensor::shape)
      .def_readonly("strides", &NativeTensor::strides)
      .def_readonly("numel", &NativeTensor::numel)
      .def_readonly("storage_offset", &NativeTensor::storage_offset)
      .def_property_readonly("dtype", &NativeTensor::dtype_str)
      .def("itemsize", &NativeTensor::itemsize)
      .def("nbytes", &NativeTensor::nbytes)
      .def("storage_id", &NativeTensor::storage_id)
      .def("version", &NativeTensor::version)
      .def("bump_version", &NativeTensor::bump_version)
      .def("data_ptr", &NativeTensor::data_ptr)
      .def("to_device", &NativeTensor::to_device, py::arg("target_device_id"),
           py::call_guard<py::gil_scoped_release>())
      .def("copy_from_numpy",
           [](std::shared_ptr<NativeTensor> self, py::array arr) {
             py::array arr_c =
                 ensure_contiguous_numpy_for_dtype(arr, self->dtype);
             const size_t count =
                 std::min(self->numel, static_cast<size_t>(arr_c.size()));
             if (count == 0)
               return;
             if (self->dtype == ScalarType::BFloat16) {
               const float *fptr = static_cast<const float *>(arr_c.data());
               std::vector<uint16_t> bf_buf(count);
               for (size_t i = 0; i < count; ++i) {
                 bf_buf[i] = fp32_to_bf16(fptr[i]);
               }
               py::gil_scoped_release release;
               self->copy_from_raw_bytes(bf_buf.data(),
                                         count * sizeof(uint16_t));
             } else {
               const void *ptr = arr_c.data();
               const size_t bytes = count * self->itemsize();
               py::gil_scoped_release release;
               self->copy_from_raw_bytes(ptr, bytes);
             }
           })
      .def("clone", &NativeTensor::clone,
           py::call_guard<py::gil_scoped_release>())
      .def("strided_copy", &NativeTensor::strided_copy, py::arg("out_shape"),
           py::arg("src_strides"), py::call_guard<py::gil_scoped_release>())
      .def("contiguous", &NativeTensor::contiguous,
           py::call_guard<py::gil_scoped_release>())
      .def("force_contiguous", &NativeTensor::force_contiguous,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "cast",
          [](std::shared_ptr<NativeTensor> self, const std::string &dt) {
            ScalarType target = parse_dtype(dt);
            py::gil_scoped_release release;
            return self->cast(target);
          },
          py::arg("dtype"))
      .def("as_strided", &NativeTensor::as_strided, py::arg("shape"),
           py::arg("strides"), py::arg("storage_offset") = -1)
      .def("transpose_view", &NativeTensor::transpose_view, py::arg("dim0"),
           py::arg("dim1"))
      .def("permute_view", &NativeTensor::permute_view, py::arg("dims"))
      .def("expand_view", &NativeTensor::expand_view, py::arg("shape"))
      .def("slice_view", &NativeTensor::slice_view, py::arg("dim"),
           py::arg("start"), py::arg("end"), py::arg("step") = 1)
      .def("reshape_view", &NativeTensor::reshape_view, py::arg("new_shape"))
      .def("copy_from_native", &NativeTensor::copy_from_native, py::arg("src"),
           py::call_guard<py::gil_scoped_release>())
      .def("slice_dim0", &NativeTensor::slice_dim0, py::arg("start"),
           py::arg("end"), py::call_guard<py::gil_scoped_release>())
      .def_static("cat_dim0", &NativeTensor::cat_dim0, py::arg("tensors"),
                  py::arg("target_device_id"),
                  py::call_guard<py::gil_scoped_release>())
      .def("fill_", &NativeTensor::fill_, py::arg("val"),
           py::call_guard<py::gil_scoped_release>())
      .def("zero_", &NativeTensor::zero_,
           py::call_guard<py::gil_scoped_release>())
      .def("is_contiguous", &NativeTensor::is_contiguous)
      .def("is_dense_zero_offset", &NativeTensor::is_dense_zero_offset)
      .def("adam_step_", &NativeTensor::adam_step_, py::arg("grad"),
           py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("lr"),
           py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
           py::arg("weight_decay"), py::arg("decoupled"), py::arg("step"),
           py::call_guard<py::gil_scoped_release>())
      .def("sgd_step_", &NativeTensor::sgd_step_, py::arg("grad"),
           py::arg("momentum_buf") = nullptr, py::arg("lr"),
           py::arg("momentum") = 0.0f, py::arg("dampening") = 0.0f,
           py::arg("weight_decay") = 0.0f, py::arg("nesterov") = false,
           py::arg("first_step") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("random_philox_", &NativeTensor::random_philox_, py::arg("mode"),
           py::arg("p1"), py::arg("p2"), py::arg("seed"), py::arg("offset"),
           py::arg("src") = nullptr, py::call_guard<py::gil_scoped_release>())
      .def("binary_op", &NativeTensor::binary_op, py::arg("op_type"),
           py::arg("other"), py::call_guard<py::gil_scoped_release>())
      .def("binary_op_scalar", &NativeTensor::binary_op_scalar,
           py::arg("op_type"), py::arg("scalar_val"),
           py::arg("reverse") = false, py::call_guard<py::gil_scoped_release>())
      .def("unary_op", &NativeTensor::unary_op, py::arg("op_type"),
           py::arg("p1") = 0.0f, py::arg("p2") = 0.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("matmul", &NativeTensor::matmul, py::arg("other"),
           py::arg("trans_a") = false, py::arg("trans_b") = false,
           py::arg("bias") = nullptr, py::arg("epilogue") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("reduce_sum", &NativeTensor::reduce_sum, py::arg("dim") = -1,
           py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>())
      .def("reduce_mean", &NativeTensor::reduce_mean, py::arg("dim") = -1,
           py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>())
      .def("reduce_max", &NativeTensor::reduce_max, py::arg("dim") = -1,
           py::arg("keepdim") = false, py::call_guard<py::gil_scoped_release>())
      .def("softmax", &NativeTensor::softmax, py::arg("dim") = -1,
           py::arg("is_log") = false, py::call_guard<py::gil_scoped_release>())
      .def("cross_entropy_per_row", &NativeTensor::cross_entropy_per_row,
           py::arg("targets"), py::call_guard<py::gil_scoped_release>())
      .def("cross_entropy_grad", &NativeTensor::cross_entropy_grad,
           py::arg("targets"), py::arg("grad_out") = nullptr,
           py::arg("scale") = 1.0f, py::call_guard<py::gil_scoped_release>())
      .def("conv2d", &NativeTensor::conv2d, py::arg("weight"),
           py::arg("bias") = nullptr, py::arg("stride_h") = 1,
           py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::arg("groups") = 1, py::arg("use_im2col") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("conv2d_backward_input", &NativeTensor::conv2d_backward_input,
           py::arg("weight"), py::arg("in_shape"), py::arg("stride_h") = 1,
           py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::arg("groups") = 1, py::arg("use_im2col") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("conv2d_backward_weight", &NativeTensor::conv2d_backward_weight,
           py::arg("input"), py::arg("weight_shape"), py::arg("stride_h") = 1,
           py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::arg("groups") = 1, py::arg("use_im2col") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("batchnorm", &NativeTensor::batchnorm, py::arg("running_mean"),
           py::arg("running_var"), py::arg("gamma") = nullptr,
           py::arg("beta") = nullptr, py::arg("training") = true,
           py::arg("momentum") = 0.1f, py::arg("eps") = 1e-5f,
           py::call_guard<py::gil_scoped_release>())
      .def("batchnorm_backward", &NativeTensor::batchnorm_backward,
           py::arg("grad_out"), py::arg("save_mean"), py::arg("save_invstd"),
           py::arg("gamma") = nullptr, py::arg("eps") = 1e-5f,
           py::arg("need_dx") = true, py::arg("need_dgamma") = true,
           py::arg("need_dbeta") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("groupnorm", &NativeTensor::groupnorm, py::arg("num_groups"),
           py::arg("gamma") = nullptr, py::arg("beta") = nullptr,
           py::arg("eps") = 1e-5f, py::call_guard<py::gil_scoped_release>())
      .def("groupnorm_backward", &NativeTensor::groupnorm_backward,
           py::arg("grad_out"), py::arg("mean_invstd"), py::arg("num_groups"),
           py::arg("gamma") = nullptr, py::arg("need_dx") = true,
           py::arg("need_dgamma") = true, py::arg("need_dbeta") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("snake", &NativeTensor::snake, py::arg("alpha") = nullptr,
           py::arg("beta") = nullptr, py::arg("channel_dim") = 1,
           py::arg("log_scale") = false, py::arg("scalar_a") = 1.0f,
           py::arg("eps") = 1e-9f, py::arg("mode") = 0,
           py::arg("gamma") = nullptr, py::arg("scalar_gamma") = 0.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("snake_backward", &NativeTensor::snake_backward, py::arg("grad_out"),
           py::arg("alpha") = nullptr, py::arg("beta") = nullptr,
           py::arg("channel_dim") = 1, py::arg("log_scale") = false,
           py::arg("scalar_a") = 1.0f, py::arg("eps") = 1e-9f,
           py::arg("need_dx") = true, py::arg("need_dalpha") = false,
           py::arg("need_dbeta") = false, py::arg("mode") = 0,
           py::arg("gamma") = nullptr, py::arg("scalar_gamma") = 0.0f,
           py::arg("need_dgamma") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("selective_scan", &NativeTensor::selective_scan, py::arg("delta"),
           py::arg("A"), py::arg("B_mat"), py::arg("C_mat"),
           py::arg("D_vec") = nullptr, py::call_guard<py::gil_scoped_release>())
      .def("selective_scan_backward", &NativeTensor::selective_scan_backward,
           py::arg("grad_out"), py::arg("delta"), py::arg("A"),
           py::arg("B_mat"), py::arg("C_mat"), py::arg("D_vec") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("flash_attention", &NativeTensor::flash_attention, py::arg("K"),
           py::arg("V"), py::arg("is_causal") = false, py::arg("scale") = -1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("flash_attention_backward", &NativeTensor::flash_attention_backward,
           py::arg("grad_out"), py::arg("K"), py::arg("V"), py::arg("O"),
           py::arg("L"), py::arg("is_causal") = false, py::arg("scale") = -1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("rope", &NativeTensor::rope, py::arg("cos_t"), py::arg("sin_t"),
           py::arg("seq_dim") = 2, py::arg("is_backward") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("glu", &NativeTensor::glu, py::arg("b"), py::arg("mode") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("glu_backward", &NativeTensor::glu_backward, py::arg("grad_out"),
           py::arg("b"), py::arg("mode") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("gemm_quant", &NativeTensor::gemm_quant, py::arg("qweight"),
           py::arg("scales"), py::arg("bias") = nullptr,
           py::arg("group_size") = 64, py::arg("bits") = 4,
           py::call_guard<py::gil_scoped_release>())
      .def("quantize_weights", &NativeTensor::quantize_weights,
           py::arg("group_size") = 64, py::arg("bits") = 4,
           py::call_guard<py::gil_scoped_release>())
      .def_static("amp_unscale_and_check", &NativeTensor::amp_unscale_and_check,
                  py::arg("grads"), py::arg("inv_scale"),
                  py::call_guard<py::gil_scoped_release>())
      .def("autodiff_unary_fwd", &NativeTensor::autodiff_unary_fwd,
           py::arg("op_type"), py::arg("param") = 0.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("autodiff_unary_bwd", &NativeTensor::autodiff_unary_bwd,
           py::arg("op_type"), py::arg("grad_out"), py::arg("param") = 0.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("autodiff_binary_fwd", &NativeTensor::autodiff_binary_fwd,
           py::arg("op_type"), py::arg("other"), py::arg("param") = 1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("autodiff_binary_bwd", &NativeTensor::autodiff_binary_bwd,
           py::arg("op_type"), py::arg("other"), py::arg("grad_out"),
           py::arg("param") = 1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("layernorm", &NativeTensor::layernorm, py::arg("gamma") = nullptr,
           py::arg("beta") = nullptr, py::arg("eps") = 1e-5f,
           py::arg("norm_dims") = 1, py::call_guard<py::gil_scoped_release>())
      .def("layernorm_backward", &NativeTensor::layernorm_backward,
           py::arg("grad_out"), py::arg("gamma") = nullptr,
           py::arg("eps") = 1e-5f, py::arg("norm_dims") = 1,
           py::arg("need_dx") = true, py::arg("need_dgamma") = true,
           py::arg("need_dbeta") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("rmsnorm", &NativeTensor::rmsnorm, py::arg("gamma") = nullptr,
           py::arg("eps") = 1e-6f, py::arg("norm_dims") = 1,
           py::call_guard<py::gil_scoped_release>())
      .def("rmsnorm_backward", &NativeTensor::rmsnorm_backward,
           py::arg("grad_out"), py::arg("gamma") = nullptr,
           py::arg("eps") = 1e-6f, py::arg("norm_dims") = 1,
           py::arg("need_dx") = true, py::arg("need_dgamma") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("maxpool2d", &NativeTensor::maxpool2d, py::arg("kH"), py::arg("kW"),
           py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h") = 0,
           py::arg("pad_w") = 0, py::call_guard<py::gil_scoped_release>())
      .def("maxpool2d_backward", &NativeTensor::maxpool2d_backward,
           py::arg("indices"), py::arg("in_shape"), py::arg("kH"),
           py::arg("kW"), py::arg("stride_h"), py::arg("stride_w"),
           py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("avgpool2d", &NativeTensor::avgpool2d, py::arg("kH"), py::arg("kW"),
           py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h") = 0,
           py::arg("pad_w") = 0, py::arg("count_include_pad") = true,
           py::arg("is_adaptive") = false, py::arg("out_h") = 0,
           py::arg("out_w") = 0, py::call_guard<py::gil_scoped_release>())
      .def("avgpool2d_backward", &NativeTensor::avgpool2d_backward,
           py::arg("in_shape"), py::arg("kH"), py::arg("kW"),
           py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h") = 0,
           py::arg("pad_w") = 0, py::arg("count_include_pad") = true,
           py::arg("is_adaptive") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("embedding", &NativeTensor::embedding, py::arg("weight"),
           py::arg("padding_idx") = -1,
           py::call_guard<py::gil_scoped_release>())
      .def("embedding_backward", &NativeTensor::embedding_backward,
           py::arg("indices"), py::arg("num_embeddings"),
           py::arg("padding_idx") = -1,
           py::call_guard<py::gil_scoped_release>())
      .def_static("cat", &NativeTensor::cat, py::arg("tensors"),
                  py::arg("dim") = 0, py::arg("target_device_id") = 0,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("where_op", &NativeTensor::where_op, py::arg("cond"),
                  py::arg("a"), py::arg("b"),
                  py::call_guard<py::gil_scoped_release>())
      .def("masked_fill", &NativeTensor::masked_fill, py::arg("mask"),
           py::arg("value"), py::call_guard<py::gil_scoped_release>())
      .def("gather", &NativeTensor::gather, py::arg("dim"), py::arg("index"),
           py::call_guard<py::gil_scoped_release>())
      .def("scatter", &NativeTensor::scatter, py::arg("dim"), py::arg("index"),
           py::arg("src") = nullptr, py::arg("scalar_val") = 0.0f,
           py::arg("use_scalar") = false, py::arg("accumulate") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("index_select", &NativeTensor::index_select, py::arg("dim"),
           py::arg("index"), py::call_guard<py::gil_scoped_release>())
      .def("index_add_", &NativeTensor::index_add_, py::arg("dim"),
           py::arg("index"), py::arg("src"), py::arg("alpha") = 1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("pad_nd", &NativeTensor::pad_nd, py::arg("pad_before"),
           py::arg("pad_after"), py::arg("mode") = 0,
           py::arg("pad_value") = 0.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("pad_nd_backward", &NativeTensor::pad_nd_backward,
           py::arg("in_shape"), py::arg("pad_before"), py::arg("mode") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("tril_triu", &NativeTensor::tril_triu, py::arg("is_upper"),
           py::arg("diagonal") = 0, py::call_guard<py::gil_scoped_release>())
      .def("flip", &NativeTensor::flip, py::arg("dims"),
           py::call_guard<py::gil_scoped_release>())
      .def("roll", &NativeTensor::roll, py::arg("shifts"), py::arg("dims"),
           py::call_guard<py::gil_scoped_release>())
      .def("argmax_argmin", &NativeTensor::argmax_argmin,
           py::arg("dim") = -999999, py::arg("keepdim") = false,
           py::arg("is_min") = false, py::call_guard<py::gil_scoped_release>())
      .def("reduce_prod_all_any", &NativeTensor::reduce_prod_all_any,
           py::arg("dim") = -999999, py::arg("keepdim") = false,
           py::arg("mode") = 0, py::call_guard<py::gil_scoped_release>())
      .def("scan_op", &NativeTensor::scan_op, py::arg("dim"),
           py::arg("mode") = 0, py::call_guard<py::gil_scoped_release>())
      .def("sort_topk", &NativeTensor::sort_topk, py::arg("dim"),
           py::arg("k") = -1, py::arg("descending") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("to_numpy",
           [](std::shared_ptr<NativeTensor> self) -> py::array {
             std::vector<ssize_t> py_shape(self->shape.begin(),
                                           self->shape.end());
             if (self->dtype == ScalarType::BFloat16) {
               std::vector<uint16_t> bf_buf(self->numel);
               {
                 py::gil_scoped_release release;
                 self->to_raw_bytes(bf_buf.data(),
                                    self->numel * sizeof(uint16_t));
               }
               py::array_t<float> arr(py_shape);
               float *out_ptr = arr.mutable_data();
               for (size_t i = 0; i < self->numel; ++i) {
                 out_ptr[i] = bf16_to_fp32(bf_buf[i]);
               }
               return arr;
             }
             py::dtype np_dt = scalar_type_to_numpy_dtype(self->dtype);
             py::array arr(np_dt, py_shape);
             void *ptr = arr.mutable_data();
             const size_t bytes = self->nbytes();
             {
               py::gil_scoped_release release;
               self->to_raw_bytes(ptr, bytes);
             }
             return arr;
           })
      .def_static(
          "from_numpy",
          [](py::array arr, int device_id, const std::string &target_dtype) {
            std::vector<int64_t> shape(arr.ndim());
            for (int i = 0; i < arr.ndim(); ++i) {
              shape[i] = arr.shape(i);
            }
            ScalarType dt = target_dtype.empty()
                                ? infer_scalar_type_from_numpy(arr)
                                : parse_dtype(target_dtype);
            py::array arr_c = ensure_contiguous_numpy_for_dtype(arr, dt);
            if (dt == ScalarType::BFloat16) {
              const size_t count = static_cast<size_t>(arr_c.size());
              const float *fptr = static_cast<const float *>(arr_c.data());
              std::vector<uint16_t> bf_buf(count);
              for (size_t i = 0; i < count; ++i) {
                bf_buf[i] = fp32_to_bf16(fptr[i]);
              }
              py::gil_scoped_release release;
              return NativeTensor::from_raw_bytes(bf_buf.data(),
                                                  count * sizeof(uint16_t),
                                                  shape, device_id, dt);
            }
            const void *ptr = arr_c.data();
            const size_t bytes =
                static_cast<size_t>(arr_c.size()) * element_size(dt);
            py::gil_scoped_release release;
            return NativeTensor::from_raw_bytes(ptr, bytes, shape, device_id,
                                                dt);
          },
          py::arg("array"), py::arg("device_id") = 0, py::arg("dtype") = "");

    py::class_<VulkanGraph>(m, "VulkanGraph")
        .def(py::init<uint32_t>(), py::arg("device_id") = 0)
        .def("capture_begin", &VulkanGraph::capture_begin, py::call_guard<py::gil_scoped_release>())
        .def("capture_end", &VulkanGraph::capture_end, py::call_guard<py::gil_scoped_release>())
        .def("replay", &VulkanGraph::replay, py::call_guard<py::gil_scoped_release>())
        .def("release", &VulkanGraph::release, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("device_id", &VulkanGraph::device_id)
        .def_property_readonly("num_dispatches", &VulkanGraph::num_dispatches)
        .def_property_readonly("is_captured", &VulkanGraph::is_captured);
}
