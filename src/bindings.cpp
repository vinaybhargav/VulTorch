#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.hpp"
#include "runtime/vk_context.hpp"
#include "runtime/vk_pipeline.hpp"

namespace py = pybind11;
using namespace vultorch;

PYBIND11_MODULE(_vultorch_backend, m) {
    m.doc() = "VulTorch: High-Performance Vulkan-Native Deep Learning Engine";

    // Vulkan Device & Manager functions
    m.def("get_device_count", []() {
        return VulkanManager::instance_ref().get_device_count();
    });

    m.def("get_device_info", [](uint32_t device_id) {
        auto info = VulkanManager::instance_ref().get_device_info(device_id);
        py::dict d;
        d["id"] = info.id;
        d["name"] = info.name;
        d["vendor"] = info.vendor;
        d["api_version"] = std::to_string(info.api_major) + "." + std::to_string(info.api_minor);
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
        caps["fp16_coopmat_16x16"] = info.capabilities.fp16_coopmat_16x16;
        d["capabilities"] = caps;

        py::list configs;
        for (const auto& c : info.cooperative_matrix_configs) {
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

    m.def("get_all_devices", []() {
        auto list = VulkanManager::instance_ref().get_all_device_info();
        py::list pylist;
        for (const auto& info : list) {
            py::dict d;
            d["id"] = info.id;
            d["name"] = info.name;
            d["vendor"] = info.vendor;
            d["api_version"] = std::to_string(info.api_major) + "." + std::to_string(info.api_minor);
            d["total_memory"] = info.total_memory;
            d["is_discrete"] = info.is_discrete;
            d["has_cooperative_matrix"] = info.has_cooperative_matrix;
            d["cooperative_matrix_configs_count"] = info.cooperative_matrix_configs.size();
            pylist.append(d);
        }
        return pylist;
    });

    m.def("synchronize", [](int device_id) {
        if (device_id >= 0) {
            VulkanManager::instance_ref().synchronize_device(device_id);
        }
    }, py::arg("device_id") = 0);

    m.def("synchronize_all", []() {
        VulkanManager::instance_ref().synchronize_all();
    });

    m.def("set_shader_directory", [](const std::string& dir) {
        VulkanPipelineManager::instance().set_shader_directory(dir);
    });

    m.def("get_shader_directory", []() {
        return VulkanPipelineManager::instance().get_shader_directory();
    });

    // NativeTensor Bindings
    py::class_<NativeTensor, std::shared_ptr<NativeTensor>>(m, "NativeTensor")
        .def(py::init([](const std::vector<int64_t>& shape, int dev_id) {
            return std::make_shared<NativeTensor>(shape, dev_id);
        }), py::arg("shape"), py::arg("device_id") = 0)
        .def_readonly("device_id", &NativeTensor::device_id)
        .def_readonly("shape", &NativeTensor::shape)
        .def_readonly("strides", &NativeTensor::strides)
        .def_readonly("numel", &NativeTensor::numel)
        .def("to_device", &NativeTensor::to_device, py::arg("target_device_id"))
        .def("copy_from_numpy", [](std::shared_ptr<NativeTensor> self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            self->copy_from_host(arr.data(), std::min(self->numel, static_cast<size_t>(arr.size())));
        })
        .def("clone", &NativeTensor::clone)
        .def("strided_copy", &NativeTensor::strided_copy, py::arg("out_shape"), py::arg("src_strides"))
        .def("reshape_view", &NativeTensor::reshape_view, py::arg("new_shape"))
        .def("copy_from_native", &NativeTensor::copy_from_native, py::arg("src"))
        .def("fill_", &NativeTensor::fill_, py::arg("val"))
        .def("zero_", &NativeTensor::zero_)
        .def("is_contiguous", &NativeTensor::is_contiguous)
        .def("adam_step_", &NativeTensor::adam_step_, py::arg("grad"), py::arg("exp_avg"),
             py::arg("exp_avg_sq"), py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
             py::arg("eps"), py::arg("weight_decay"), py::arg("decoupled"), py::arg("step"))
        .def("binary_op", &NativeTensor::binary_op, py::arg("op_type"), py::arg("other"))
        .def("binary_op_scalar", &NativeTensor::binary_op_scalar, py::arg("op_type"), py::arg("scalar_val"), py::arg("reverse") = false)
        .def("unary_op", &NativeTensor::unary_op, py::arg("op_type"), py::arg("p1") = 0.0f, py::arg("p2") = 0.0f)
        .def("matmul", &NativeTensor::matmul, py::arg("other"))
        .def("reduce_sum", &NativeTensor::reduce_sum, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("reduce_mean", &NativeTensor::reduce_mean, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("reduce_max", &NativeTensor::reduce_max, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("softmax", &NativeTensor::softmax, py::arg("dim") = -1, py::arg("is_log") = false)
        .def("cross_entropy_per_row", &NativeTensor::cross_entropy_per_row, py::arg("targets"))
        .def("cross_entropy_grad", &NativeTensor::cross_entropy_grad, py::arg("targets"))
        .def("conv2d", &NativeTensor::conv2d, py::arg("weight"), py::arg("bias") = nullptr,
             py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0)
        .def("conv2d_backward_input", &NativeTensor::conv2d_backward_input, py::arg("weight"),
             py::arg("in_shape"), py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0)
        .def("conv2d_backward_weight", &NativeTensor::conv2d_backward_weight, py::arg("input"),
             py::arg("weight_shape"), py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0)
        .def("layernorm", &NativeTensor::layernorm, py::arg("gamma") = nullptr, py::arg("beta") = nullptr, py::arg("eps") = 1e-5f)
        .def("to_numpy", [](std::shared_ptr<NativeTensor> self) -> py::array_t<float> {
            std::vector<ssize_t> py_shape(self->shape.begin(), self->shape.end());
            std::vector<ssize_t> py_strides;
            for (auto s : self->strides) {
                py_strides.push_back(s * sizeof(float));
            }
            py::array_t<float> arr(py_shape, py_strides);
            self->copy_to_host(arr.mutable_data(), self->numel);
            return arr;
        })
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr, int device_id) {
            std::vector<int64_t> shape(arr.ndim());
            for (int i = 0; i < arr.ndim(); ++i) {
                shape[i] = arr.shape(i);
            }
            auto tensor = NativeTensor::create(shape, device_id);
            tensor->copy_from_host(arr.data(), tensor->numel);
            return tensor;
        }, py::arg("array"), py::arg("device_id") = 0);
}
