cmake_minimum_required(VERSION 3.18)
project(vultorch_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings
#
# Our sources must compile clean. Third-party headers are included as SYSTEM so
# their warnings never mask ours. Set -DVULTORCH_WERROR=ON (CI) to make any
# warning a build failure.
option(VULTORCH_WERROR "Treat compiler warnings as errors" OFF)
option(VULTORCH_DEV_COPY_ARTIFACTS
       "Copy the built extension and shaders into the source tree (in-place development)"
       ON)
if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(VULTORCH_WERROR)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-qual)
    if(VULTORCH_WERROR)
        add_compile_options(-Werror)
    endif()
endif()

# Vulkan Headers and Library
include_directories(SYSTEM third_party/vulkan-headers/include)
include_directories(src)

find_library(VULKAN_LIB vulkan PATHS /usr/lib /usr/local/lib)
if (NOT VULKAN_LIB)
    message(FATAL_ERROR "libvulkan.so not found!")
endif()

# Python and pybind11
#
# No hardcoded interpreter path: the interpreter is whatever the build frontend
# (pip / scikit-build-core) or the user passes via -DPython_EXECUTABLE=...
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11 ships its CMake config inside the Python package. A pip/scikit-build
# driven build already has it on CMAKE_PREFIX_PATH; a plain `cmake -S . -B build`
# does not, so ask the selected interpreter where it lives.
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_rc
        ERROR_QUIET)
    if(_pybind11_rc EQUAL 0 AND _pybind11_cmakedir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# ----------------------------------------------------------------------------
# Compute shaders: .slang -> .spv at build time
#
# All compute shaders are authored in Slang. One source file can hold many
# specialised entry points (e.g. every GEMM tile configuration). Each
# "<file>:<entry>" pair below becomes <entry>.spv, whose entry point slangc
# emits as "main" for the runtime.
find_program(SLANGC NAMES slangc)
if(NOT SLANGC)
    message(FATAL_ERROR
        "slangc not found. Install the Slang compiler (Arch AUR: shader-slang-bin,"
        " or the LunarG Vulkan SDK) to build VulTorch from source.")
endif()

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

set(SLANG_KERNELS
    gemm:gemm_t32_k16_r2
    gemm:gemm_t64_k16_r4
    gemm:gemm_t64_k8_r4
    gemm:gemm_t64_k16_r8
    gemm:gemm_t128_k8_r8
    gemm:gemm_t128_k16_r8
    gemm:matmul
    gemm:matmul_tiled64
    gemm:matmul_tiled
    gemm:matmul_coopmat
    adam:adam_step
    elementwise_unary_vec4:elementwise_unary_vec4
    elementwise_unary_scalar:elementwise_unary_scalar
    elementwise_binary_vec4:elementwise_binary_vec4
    elementwise_binary_broadcast:elementwise_binary_broadcast
    copy_strided:copy_strided
    reduction:reduction
    reduction_generic:reduction_generic
    reduction_generic:reduction_strided
    softmax:softmax
    softmax_cross_entropy:softmax_cross_entropy
    layernorm:layernorm
    layernorm:layernorm_backward_dx
    layernorm:rmsnorm
    layernorm:rmsnorm_backward_dx
    conv2d:conv2d
    conv2d_backward:conv2d_backward_input
    conv2d_backward:conv2d_backward_weight
    maxpool2d:maxpool2d
    maxpool2d_backward:maxpool2d_backward
    avgpool2d:avgpool2d
    avgpool2d:avgpool2d_backward
    embedding:embedding_forward
    embedding:embedding_backward
    random:philox_rng
    sgd:sgd_step
    gemm_fp16:gemm_fp16
    gemm_fp16:matmul_coopmat_fp16
    cast:cast_dtype
    indexing:where_kernel
    indexing:masked_fill_kernel
    indexing:gather_kernel
    indexing:scatter_kernel
    indexing:scatter_add_kernel
    indexing:index_select_kernel
    indexing:index_add_kernel
    manip:copy_into_slice
    manip:pad_nd_kernel
    manip:pad_nd_backward_kernel
    manip:tril_triu_kernel
    manip:flip_roll_kernel
    scan_sort:argmax_argmin_kernel
    scan_sort:prod_all_any_kernel
    scan_sort:scan_kernel
    scan_sort:sort_topk_kernel
    conv2d_im2col:im2col_kernel
    conv2d_im2col:col2im_kernel
    conv2d_depthwise:conv2d_grouped_forward
    conv2d_depthwise:conv2d_grouped_backward_input
    conv2d_depthwise:conv2d_grouped_backward_weight
    batchnorm:batchnorm_stats
    batchnorm:batchnorm_forward
    batchnorm:batchnorm_backward_reduce
    batchnorm:batchnorm_backward_dx
    groupnorm:groupnorm_forward
    groupnorm:groupnorm_backward_dx
    groupnorm:groupnorm_backward_gamma_beta
)

set(SPV_OUTPUTS)
foreach(kernel ${SLANG_KERNELS})
    string(REPLACE ":" ";" parts "${kernel}")
    list(GET parts 0 slang_file)
    list(GET parts 1 entry)
    set(src "${SHADER_SOURCE_DIR}/${slang_file}.slang")
    set(spv "${SHADER_OUTPUT_DIR}/${entry}.spv")
    add_custom_command(
        OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${SLANGC}" "${src}" -target spirv -profile spirv_1_5
                -Wno-41012 -Wno-41017 -entry "${entry}" -O3 -o "${spv}"
        DEPENDS "${src}"
        COMMENT "Compiling Slang kernel ${slang_file}.slang:${entry}"
        VERBATIM)
    list(APPEND SPV_OUTPUTS "${spv}")
endforeach()

add_custom_target(vultorch_shaders ALL DEPENDS ${SPV_OUTPUTS})

# ----------------------------------------------------------------------------
# Extension module
set(SOURCES
    src/runtime/vk_context.cpp
    src/runtime/vk_memory.cpp
    src/runtime/vk_pipeline.cpp
    src/tensor.cpp
    src/bindings.cpp
)

pybind11_add_module(_vultorch_backend ${SOURCES})
target_link_libraries(_vultorch_backend PRIVATE ${VULKAN_LIB})
add_dependencies(_vultorch_backend vultorch_shaders)

# ----------------------------------------------------------------------------
# Install layout: the wheel contains vultorch/_vultorch_backend*.so,
# vultorch/shaders/*.spv and the pure-Python package next to each other.
# Everything is installed from here so the wheel has exactly one definition.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vultorch/"
        DESTINATION vultorch
        FILES_MATCHING PATTERN "*.py")
install(TARGETS _vultorch_backend
        LIBRARY DESTINATION vultorch
        RUNTIME DESTINATION vultorch)
install(FILES ${SPV_OUTPUTS} DESTINATION vultorch/shaders)

# ----------------------------------------------------------------------------
# In-place development: running from a checkout needs the artifacts beside the
# Python sources. Disabled for wheel builds so the source tree never leaks into
# a distribution.
if(VULTORCH_DEV_COPY_ARTIFACTS)
    set(DEV_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vultorch")
    set(DEV_SHADER_DIR "${DEV_PACKAGE_DIR}/shaders")

    # A POST_BUILD command only fires when the target relinks, so a deleted copy
    # would never come back. An always-run target with copy_if_different keeps
    # the checkout importable and adds no work when nothing changed.
    add_custom_target(vultorch_dev_artifacts ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${DEV_SHADER_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${SPV_OUTPUTS} "${DEV_SHADER_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:_vultorch_backend>" "${DEV_PACKAGE_DIR}/"
        DEPENDS _vultorch_backend vultorch_shaders
        COMMENT "Copying build artifacts into the source tree"
        VERBATIM)
endif()
