cmake_minimum_required(VERSION 3.26)
project(tensoracle_python LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# The build environment supplies headers and the link library from wheels.
# The standalone C++ project receives paths and never discovers Python itself.
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "from importlib.metadata import distribution; print(distribution('tensorrt-cu13-libs').locate_file('tensorrt_libs')); print(distribution('nvidia-cuda-runtime').locate_file('nvidia/cu13'))"
    OUTPUT_VARIABLE _dependencies OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" _dependencies "${_dependencies}")
list(GET _dependencies 0 _tensorrt_libs)
list(GET _dependencies 1 _cuda_root)
find_library(_nvinfer NAMES nvinfer nvinfer_11 libnvinfer.so.11
    PATHS "${_tensorrt_libs}" NO_DEFAULT_PATH REQUIRED)
set(TENSORACLE_TENSORRT_LIBRARY "${_nvinfer}" CACHE FILEPATH "TensorRT from the build wheel" FORCE)
set(CUDAToolkit_ROOT "${_cuda_root}" CACHE PATH "CUDA headers from the build wheel" FORCE)
set(TENSORACLE_INSTALL_ROOT "tensoracle/cpp" CACHE STRING "Wheel SDK prefix" FORCE)
set(TENSORACLE_INSTALL_RPATH "$ORIGIN/../../../tensorrt_libs" CACHE STRING "TensorRT wheel lookup" FORCE)
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Wheel library directory" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Tests are built separately" FORCE)
set(TENSORACLE_BUILD_CUBIN_HELPER OFF CACHE BOOL "Ship the source-based helper" FORCE)

# A staged sdist has cpp next to CMakeLists.txt; the normal checkout has ../cpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
    set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
else()
    set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
endif()
add_subdirectory("${_cpp}" cpp)
nanobind_add_module(_native STABLE_ABI bindings.cpp cubin_bindings.cpp)
target_link_libraries(_native PRIVATE tensoracle::tensoracle)
# The build tree keeps the module under tensoracle/ so the stub generator imports it by the
# package name that its exception classes record.
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensoracle"
    BUILD_RPATH_USE_ORIGIN YES INSTALL_RPATH "$ORIGIN/cpp/lib")
install(TARGETS _native LIBRARY DESTINATION tensoracle RUNTIME DESTINATION tensoracle)
# The build RPATH finds libtensoracle and the TensorRT wheel while the stub is generated.
# Private names are kept because public signatures return _CacheResult.
nanobind_add_stub(_native_stub MODULE tensoracle._native OUTPUT _native.pyi
    PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}" DEPENDS _native INCLUDE_PRIVATE)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_native.pyi" DESTINATION tensoracle)
