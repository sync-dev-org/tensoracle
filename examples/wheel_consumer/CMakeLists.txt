cmake_minimum_required(VERSION 3.26)
if(NOT TENSORRT_ROOT OR NOT CUDAToolkit_ROOT)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cpp/conan/bootstrap.cmake")
endif()
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

if(NOT TENSORRT_ROOT)
    find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
    list(GET TensorRT_INCLUDE_DIRS 0 _tensorrt_include)
    get_filename_component(TENSORRT_ROOT "${_tensorrt_include}" DIRECTORY)
endif()
find_path(_tensorrt_include NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED)
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
