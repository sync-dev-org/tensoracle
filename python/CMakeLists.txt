cmake_minimum_required(VERSION 3.26)
# A staged sdist has cpp next to CMakeLists.txt; the checkout has ../cpp.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
    set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
else()
    set(_cpp "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")
endif()
set(BUILD_TESTING OFF CACHE BOOL "Tests are built separately" FORCE)
include("${_cpp}/conan/bootstrap.cmake")
project(tensoracle_python LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
get_filename_component(_tensorrt_root "${TensorRT_INCLUDE_DIR}" DIRECTORY)
find_library(_stub_nvinfer NAMES nvinfer PATHS "${_tensorrt_root}/lib"
    NO_DEFAULT_PATH REQUIRED NO_CACHE)

set(TENSORACLE_INSTALL_ROOT "tensoracle/cpp" CACHE STRING "Wheel SDK prefix" FORCE)
set(TENSORACLE_INSTALL_RPATH "$ORIGIN/../../../tensorrt_libs" CACHE STRING "TensorRT wheel lookup" FORCE)
set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Wheel library directory" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Tests are built separately" FORCE)
set(TENSORACLE_BUILD_CUBIN_HELPER OFF CACHE BOOL "Ship the source-based helper" FORCE)

add_subdirectory("${_cpp}" cpp)
nanobind_add_module(_native STABLE_ABI bindings.cpp cubin_bindings.cpp)
target_link_libraries(_native PRIVATE tensoracle::tensoracle)
# The build tree keeps the module under tensoracle/ so the stub generator imports it by the
# package name that its exception classes record.
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tensoracle"
    BUILD_RPATH_USE_ORIGIN YES INSTALL_RPATH "$ORIGIN/cpp/lib")
install(TARGETS _native LIBRARY DESTINATION tensoracle RUNTIME DESTINATION tensoracle)
# The build RPATH finds libtensoracle and the Conan TensorRT SDK during stub generation.
# Private names are kept because public signatures return _CacheResult.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_native.pyi"
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/build_backend.py"
        "${_stub_nvinfer}" "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_native.pyi"
    DEPENDS _native "${CMAKE_CURRENT_SOURCE_DIR}/build_backend.py" VERBATIM)
add_custom_target(_native_stub ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/_native.pyi")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_native.pyi" DESTINATION tensoracle)
