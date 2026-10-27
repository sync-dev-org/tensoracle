cmake_minimum_required(VERSION 3.20)
project(tensoracle_cubin LANGUAGES CXX)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 11 SDK root containing include/NvInfer.h")
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT SDK headers are missing; set TENSORRT_ROOT to the TensorRT 11 SDK")
endif()
include(cmake/TensoracleCuda.cmake)
_tensoracle_find_cuda()

add_library(tensoracle_cubin STATIC src/cubin_plugin.cpp)
target_compile_features(tensoracle_cubin PUBLIC cxx_std_17)
target_include_directories(tensoracle_cubin PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${TENSORRT_ROOT}/include")
target_link_libraries(tensoracle_cubin PUBLIC tensoracle_internal_cuda_driver)
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(tensoracle_cubin PUBLIC Threads::Threads)
endif()
target_compile_options(tensoracle_cubin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)

include(CTest)
if(BUILD_TESTING)
    add_executable(tensoracle_core_test tests/core.cpp)
    target_link_libraries(tensoracle_core_test PRIVATE tensoracle_cubin)
    add_test(NAME tensoracle_fields_and_sha256 COMMAND tensoracle_core_test)
endif()
