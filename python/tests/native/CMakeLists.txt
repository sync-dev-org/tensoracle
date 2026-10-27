cmake_minimum_required(VERSION 3.20)
project(tensoracle_test_fixtures LANGUAGES C CXX)

if(TEST_CUBIN_SOURCE)
    include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleCuda.cmake")
    _tensoracle_find_cuda()
    set(_nvcc "${_tensoracle_cuda_root}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
    if(NOT EXISTS "${_nvcc}")
        message(FATAL_ERROR "nvcc is missing at ${_nvcc}; install PyPI nvidia-cuda-nvcc 13.2")
    endif()
    set(_cubin "${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin")
    add_custom_command(OUTPUT "${_cubin}"
        COMMAND "${_nvcc}" --cubin "-arch=sm_${TEST_CUDA_ARCH}"
            "${TEST_CUBIN_SOURCE}" -o "${_cubin}"
        DEPENDS "${TEST_CUBIN_SOURCE}"
        VERBATIM)
    add_custom_target(test_cubin ALL DEPENDS "${_cubin}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/artifact-$<CONFIG>.txt"
        CONTENT "${_cubin}")
    return()
endif()

if(TEST_SHARED)
    add_library(${TEST_TARGET} SHARED ${TEST_SOURCES})
else()
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    # The contract executable uses assert; keep its checks in Release builds too.
    if(MSVC)
        target_compile_options(${TEST_TARGET} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
    endif()
endif()
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
target_include_directories(${TEST_TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" ${TEST_INCLUDE_DIRS})

if(TENSORACLE_CPP_ROOT)
    include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleCuda.cmake")
    _tensoracle_find_cuda()
    target_sources(${TEST_TARGET} PRIVATE "${TENSORACLE_CPP_ROOT}/src/cubin_plugin.cpp")
    target_include_directories(${TEST_TARGET} PRIVATE
        "${TENSORACLE_CPP_ROOT}/include" "${TENSORRT_ROOT}/include")
    target_link_libraries(${TEST_TARGET} PRIVATE tensoracle_internal_cuda_driver)
    target_compile_options(${TEST_TARGET} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)
endif()

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/artifact-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:${TEST_TARGET}>")
