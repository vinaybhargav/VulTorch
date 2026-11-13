cmake_minimum_required(VERSION 3.18)
project(vultorch_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Vulkan Headers and Library
include_directories(third_party/vulkan-headers/include)
include_directories(src)

find_library(VULKAN_LIB vulkan PATHS /usr/lib /usr/local/lib)
if (NOT VULKAN_LIB)
    message(FATAL_ERROR "libvulkan.so not found!")
endif()

# Find Python and Pybind11
execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3" -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Parse include dirs
string(REPLACE "-I" "" PYBIND11_INCLUDES_CLEAN "${PYBIND11_INCLUDES}")
separate_arguments(PYBIND11_INCLUDES_LIST UNIX_COMMAND "${PYBIND11_INCLUDES_CLEAN}")
include_directories(${PYBIND11_INCLUDES_LIST})

set(SOURCES
    src/runtime/vk_context.cpp
    src/runtime/vk_memory.cpp
    src/runtime/vk_pipeline.cpp
    src/tensor.cpp
    src/bindings.cpp
)

add_library(_vultorch_backend SHARED ${SOURCES})
target_link_libraries(_vultorch_backend PRIVATE ${VULKAN_LIB})
set_target_properties(_vultorch_backend PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    OUTPUT_NAME "_vultorch_backend"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/vultorch"
)
