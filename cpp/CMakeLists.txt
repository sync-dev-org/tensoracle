cmake_minimum_required(VERSION 3.26)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoracle/tensoracle.h")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoracle/tensoracle.h" _tensoracle_header)
foreach(_component MAJOR MINOR PATCH)
    string(REGEX MATCH "#define TENSORACLE_VERSION_${_component} ([0-9]+)" _match "${_tensoracle_header}")
    set(_tensoracle_${_component} "${CMAKE_MATCH_1}")
endforeach()
project(tensoracle VERSION ${_tensoracle_MAJOR}.${_tensoracle_MINOR}.${_tensoracle_PATCH} LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/TensoracleCuda.cmake)
option(TENSORACLE_BUILD_CUBIN_HELPER "Build the standalone cubin helper archive" ON)
if(TENSORACLE_BUILD_CUBIN_HELPER)
    # Preserve the existing helper's discovery and diagnostics for native builds.
    _tensoracle_find_cuda()
    get_target_property(TENSORACLE_CUDA_INCLUDE tensoracle_cuda_driver INTERFACE_INCLUDE_DIRECTORIES)
else()
    # The shared core only needs headers; wheel builds need no driver link stub.
    find_path(TENSORACLE_CUDA_INCLUDE cuda_runtime_api.h
        HINTS "${CUDAToolkit_ROOT}/include" "$ENV{CUDAToolkit_ROOT}/include"
              /usr/local/cuda/include REQUIRED)
endif()

set(TENSORRT_ROOT "$ENV{TENSORACLE_SDK_ROOT}" CACHE PATH "TensorRT 11 SDK root")
set(TENSORACLE_TENSORRT_LIBRARY "$ENV{TENSORACLE_TENSORRT_LIBRARY}" CACHE FILEPATH
    "TensorRT 11 libnvinfer (may be supplied by the tensorrt_libs wheel)")
find_path(TENSORACLE_TENSORRT_INCLUDE NvInfer.h
    HINTS "${TENSORRT_ROOT}/include" REQUIRED)
if(NOT TENSORACLE_TENSORRT_LIBRARY)
    find_library(_tensoracle_nvinfer NAMES nvinfer nvinfer_11 libnvinfer.so.11
        HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED)
    set(TENSORACLE_TENSORRT_LIBRARY "${_tensoracle_nvinfer}" CACHE FILEPATH "TensorRT 11 libnvinfer" FORCE)
endif()
if(NOT EXISTS "${TENSORACLE_TENSORRT_LIBRARY}")
    message(FATAL_ERROR "TensorRT libnvinfer is missing: ${TENSORACLE_TENSORRT_LIBRARY}")
endif()

include(cmake/TensoracleDependencies.cmake)
add_library(tensoracle SHARED src/abi.cpp src/error.cpp src/bundle.cpp src/producer.cpp src/comparison.cpp)
target_sources(tensoracle PRIVATE src/engine.cpp src/cache.cpp src/plugins.cpp src/cache_open.cpp)
target_sources(tensoracle PRIVATE src/cache_management.cpp src/cache_management_internal.cpp)
target_sources(tensoracle PRIVATE src/builder.cpp src/safetensors.cpp)
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(tensoracle PRIVATE Threads::Threads)
endif()
add_library(tensoracle::tensoracle ALIAS tensoracle)
target_compile_features(tensoracle PUBLIC cxx_std_17)
target_compile_definitions(tensoracle PRIVATE TENSORACLE_BUILDING_LIBRARY)
target_include_directories(tensoracle PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle PRIVATE "${TENSORACLE_TENSORRT_LIBRARY}" tensoracle_json tensoracle_miniz ${CMAKE_DL_LIBS})
set(TENSORACLE_INSTALL_RPATH "$ORIGIN" CACHE STRING "Relative runtime lookup paths for the installed library")
set_target_properties(tensoracle PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH_USE_ORIGIN YES
    INSTALL_RPATH "${TENSORACLE_INSTALL_RPATH}")
if(UNIX AND NOT APPLE)
    target_link_options(tensoracle PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/tensoracle.map")
    set_property(TARGET tensoracle APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/tensoracle.map")
endif()

# Keep the existing source-based plugin helper and its standalone test target.
if(TENSORACLE_BUILD_CUBIN_HELPER)
    add_library(tensoracle_cubin STATIC src/cubin_plugin.cpp)
    target_compile_features(tensoracle_cubin PUBLIC cxx_std_17)
    target_include_directories(tensoracle_cubin PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${TENSORACLE_TENSORRT_INCLUDE}")
    target_link_libraries(tensoracle_cubin PUBLIC tensoracle_internal_cuda_driver)
    if(NOT WIN32)
        target_link_libraries(tensoracle_cubin PUBLIC Threads::Threads)
    endif()
    target_compile_options(tensoracle_cubin PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wno-deprecated-declarations>)
endif()

# Wheels put this relocatable SDK beneath tensoracle/cpp. Native installs use
# the usual prefix/include and prefix/lib layout.
set(TENSORACLE_INSTALL_ROOT "" CACHE STRING "Relative SDK prefix inside the install prefix")
set(_tensoracle_root "${TENSORACLE_INSTALL_ROOT}")
if(_tensoracle_root)
    string(APPEND _tensoracle_root "/")
endif()
set(TENSORACLE_INCLUDE_DIR "${_tensoracle_root}${CMAKE_INSTALL_INCLUDEDIR}")
set(TENSORACLE_LIBRARY_DIR "${_tensoracle_root}${CMAKE_INSTALL_LIBDIR}")
set(TENSORACLE_HELPER_DIR "${_tensoracle_root}cmake")
set(TENSORACLE_PACKAGE_DIR "${TENSORACLE_LIBRARY_DIR}/cmake/tensoracle")
target_include_directories(tensoracle PUBLIC "$<INSTALL_INTERFACE:${TENSORACLE_INCLUDE_DIR}>")
install(TARGETS tensoracle EXPORT tensoracleTargets
    LIBRARY DESTINATION "${TENSORACLE_LIBRARY_DIR}"
    RUNTIME DESTINATION "${_tensoracle_root}${CMAKE_INSTALL_BINDIR}"
    ARCHIVE DESTINATION "${TENSORACLE_LIBRARY_DIR}")
install(DIRECTORY include/ DESTINATION "${TENSORACLE_INCLUDE_DIR}")
install(FILES "${tensoracle_json_SOURCE_DIR}/LICENSE.MIT"
    DESTINATION "${_tensoracle_root}licenses/nlohmann-json")
install(FILES "${tensoracle_miniz_SOURCE_DIR}/LICENSE"
    DESTINATION "${_tensoracle_root}licenses/miniz")
install(FILES src/cubin_plugin.cpp src/sha256.hpp DESTINATION "${_tensoracle_root}src")
install(FILES cmake/TensoracleCuda.cmake cmake/TensoracleCubinPlugin.cmake
    DESTINATION "${TENSORACLE_HELPER_DIR}")
configure_package_config_file(cmake/tensoracleConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tensoracleConfig.cmake"
    INSTALL_DESTINATION "${TENSORACLE_PACKAGE_DIR}" PATH_VARS TENSORACLE_HELPER_DIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/tensoracleConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT tensoracleTargets NAMESPACE tensoracle:: DESTINATION "${TENSORACLE_PACKAGE_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tensoracleConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tensoracleConfigVersion.cmake"
    DESTINATION "${TENSORACLE_PACKAGE_DIR}")

include(CTest)
if(BUILD_TESTING)
    if(TENSORACLE_BUILD_CUBIN_HELPER)
        add_executable(tensoracle_core_test tests/core.cpp)
        target_link_libraries(tensoracle_core_test PRIVATE tensoracle_cubin)
        add_test(NAME tensoracle_fields_and_sha256 COMMAND tensoracle_core_test)
    endif()
    add_subdirectory(tests)
endif()
