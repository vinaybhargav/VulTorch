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
# Compute shaders: .comp -> .spv at build time
#
# The SPIR-V is a build product, never a checked-in binary, so a shader fix
# cannot silently fail to reach the binary.
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR
        "glslangValidator not found. Install it (Debian/Ubuntu: glslang-tools,"
        " Arch: glslang, Fedora: glslang) to build VulTorch from source.")
endif()

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${SHADER_SOURCE_DIR}/*.comp")
if(NOT SHADER_SOURCES)
    message(FATAL_ERROR "No compute shaders found in ${SHADER_SOURCE_DIR}")
endif()

set(SPV_OUTPUTS)
foreach(shader ${SHADER_SOURCES})
    get_filename_component(shader_name "${shader}" NAME_WE)
    set(spv "${SHADER_OUTPUT_DIR}/${shader_name}.spv")

    add_custom_command(
        OUTPUT "${spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLANG_VALIDATOR}" --target-env vulkan1.3 -V "${shader}" -o "${spv}"
        DEPENDS "${shader}"
        COMMENT "Compiling shader ${shader_name}.comp"
        VERBATIM)
    list(APPEND SPV_OUTPUTS "${spv}")
endforeach()

# Slang kernels: one source file can hold many specialised entry points (e.g.
# every GEMM tile configuration). Each "<file>:<entry>" pair below becomes
# <entry>.spv, whose entry point slangc renames to "main" — the same contract
# as the GLSL shaders, so the runtime does not care which language produced it.
find_program(SLANGC NAMES slangc)
if(NOT SLANGC)
    message(FATAL_ERROR
        "slangc not found. Install the Slang compiler (Arch AUR: shader-slang-bin,"
        " or the LunarG Vulkan SDK) to build VulTorch from source.")
endif()

set(SLANG_KERNELS
    gemm:gemm_t32_k16_r2
    gemm:gemm_t64_k16_r4
    gemm:gemm_t64_k8_r4
    gemm:gemm_t64_k16_r8
    gemm:gemm_t128_k8_r8
    gemm:gemm_t128_k16_r8
    adam:adam_step
)

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
                -entry "${entry}" -O3 -o "${spv}"
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
