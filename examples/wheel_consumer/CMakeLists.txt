cmake_minimum_required(VERSION 3.26)
project(tensoracle_wheel_consumer LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS
    Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Read installation metadata without importing tensoracle or loading its binding.
# PEP 517 supplies tensoracle as a build dependency, independently of this source.
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "from importlib.metadata import distribution; print(distribution('tensoracle').locate_file(''))"
    OUTPUT_VARIABLE _site OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${_site}/tensoracle/cpp")
find_package(tensoracle 2 CONFIG REQUIRED)

set(TENSORRT_ROOT "$ENV{TENSORACLE_SDK_ROOT}" CACHE PATH "TensorRT 11 SDK headers")
find_path(_tensorrt_include NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED)
if(NOT CUDAToolkit_ROOT)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "from importlib.metadata import distribution; print(distribution('nvidia-cuda-runtime').locate_file('nvidia/cu13'))"
        OUTPUT_VARIABLE CUDAToolkit_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
endif()
tensoracle_find_cuda(OUT_INCLUDE_DIR _cuda_include OUT_DRIVER_LIBRARY _cuda_driver)

nanobind_add_module(_native STABLE_ABI bindings.cpp)
target_include_directories(_native PRIVATE "${_tensorrt_include}")
target_link_libraries(_native PRIVATE tensoracle::tensoracle tensoracle_cuda_driver)

# Both wheels install beside one another in site-packages. Only their relative
# placement survives; the temporary PEP 517 build environment must not survive.
file(RELATIVE_PATH _library_from_module
    "${_site}/tensoracle_consumer" "${_site}/tensoracle/cpp/lib")
set_target_properties(_native PROPERTIES
    BUILD_RPATH_USE_ORIGIN YES
    INSTALL_RPATH "$ORIGIN/${_library_from_module}"
    INSTALL_RPATH_USE_LINK_PATH NO)
install(TARGETS _native LIBRARY DESTINATION tensoracle_consumer)
