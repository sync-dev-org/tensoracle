include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/TensoracleCuda.cmake")

function(tensoracle_add_cubin_plugin)
    cmake_parse_arguments(ARG "" "TARGET;TENSORRT_ROOT;BUNDLE_ROOT" "SOURCES" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_TENSORRT_ROOT OR NOT ARG_BUNDLE_ROOT OR NOT ARG_SOURCES)
        message(FATAL_ERROR "TARGET, TENSORRT_ROOT, BUNDLE_ROOT, and SOURCES are required")
    endif()

    get_filename_component(_tensoracle_cpp "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    _tensoracle_find_cuda()
    find_path(TENSORACLE_NVINFER_INCLUDE NvInfer.h
        PATHS "${ARG_TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    set(_tensoracle_nvinfer_names nvinfer)
    if(WIN32)
        list(PREPEND _tensoracle_nvinfer_names nvinfer_11)
    endif()
    find_library(TENSORACLE_NVINFER_LIBRARY NAMES ${_tensoracle_nvinfer_names}
        PATHS "${ARG_TENSORRT_ROOT}/lib" "${ARG_TENSORRT_ROOT}/lib64"
        NO_DEFAULT_PATH REQUIRED)

    add_library(${ARG_TARGET} SHARED
        ${ARG_SOURCES}
        "${_tensoracle_cpp}/src/cubin_plugin.cpp"
        "${ARG_BUNDLE_ROOT}/bundle.cpp")
    target_compile_features(${ARG_TARGET} PRIVATE cxx_std_17)
    target_compile_options(${ARG_TARGET} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)
    target_include_directories(${ARG_TARGET} PRIVATE
        "${_tensoracle_cpp}/include" "${TENSORACLE_NVINFER_INCLUDE}")
    target_link_libraries(${ARG_TARGET} PRIVATE
        "${TENSORACLE_NVINFER_LIBRARY}" tensoracle_internal_cuda_driver)
    if(NOT WIN32)
        find_package(Threads REQUIRED)
        target_link_libraries(${ARG_TARGET} PRIVATE Threads::Threads)
    endif()
    set_target_properties(${ARG_TARGET} PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
endfunction()
