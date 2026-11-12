cmake_minimum_required(VERSION 3.26)
option(BUILD_TESTING "Build tensoracle tests" ON)
if(BUILD_TESTING)
    set(TENSORACLE_CONAN_OFFLINE ON CACHE BOOL "Use prepared test dependencies")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/conan/bootstrap.cmake")
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
option(BUILD_SHARED_LIBS "Build the shared tensoracle library" ON)
option(TENSORACLE_BUILD_CUBIN_HELPER "Build the standalone cubin helper archive" ON)
if(TENSORACLE_BUILD_CUBIN_HELPER OR NOT BUILD_SHARED_LIBS)
    # Preserve the existing helper's discovery and diagnostics for native builds.
    _tensoracle_find_cuda()
    get_target_property(TENSORACLE_CUDA_INCLUDE tensoracle_cuda_driver INTERFACE_INCLUDE_DIRECTORIES)
else()
    # The Conan toolchain supplies the same headers without linking the driver.
    find_path(TENSORACLE_CUDA_INCLUDE cuda_runtime_api.h
        PATHS "${CUDAToolkit_ROOT}/include" NO_DEFAULT_PATH REQUIRED NO_CACHE)
endif()

include(cmake/TensoracleDependencies.cmake)
find_path(TENSORACLE_TENSORRT_INCLUDE NvInfer.h
    PATHS ${TensorRT_INCLUDE_DIRS} NO_DEFAULT_PATH REQUIRED NO_CACHE)
get_filename_component(_tensoracle_tensorrt_root "${TENSORACLE_TENSORRT_INCLUDE}" DIRECTORY)
set(TENSORACLE_TENSORRT_LIBRARY "${_tensoracle_tensorrt_root}/lib/libnvinfer.so.11")
message(STATUS "TensorRT SDK from Conan: ${TENSORACLE_TENSORRT_INCLUDE}; ${TENSORACLE_TENSORRT_LIBRARY}")

add_library(tensoracle src/abi.cpp src/error.cpp src/bundle.cpp src/producer.cpp src/comparison.cpp)
target_sources(tensoracle PRIVATE src/engine.cpp src/cache.cpp src/plugins.cpp src/cache_open.cpp)
target_sources(tensoracle PRIVATE src/cache_management.cpp src/cache_management_internal.cpp)
target_sources(tensoracle PRIVATE src/builder.cpp src/safetensors.cpp)
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(tensoracle PRIVATE Threads::Threads)
endif()
add_library(tensoracle::tensoracle ALIAS tensoracle)
target_compile_features(tensoracle PUBLIC cxx_std_17)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(tensoracle PRIVATE TENSORACLE_BUILDING_LIBRARY)
else()
    target_compile_definitions(tensoracle PUBLIC TENSORACLE_STATIC_LIBRARY)
    set_target_properties(tensoracle PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(tensoracle PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
if(BUILD_SHARED_LIBS)
    target_link_libraries(tensoracle PRIVATE "${TENSORACLE_TENSORRT_LIBRARY}" tensoracle_json tensoracle_miniz ${CMAKE_DL_LIBS})
else()
    # Keep third-party objects out of the archive and resolve their targets in
    # the consumer's toolchain, without exporting this build's Conan paths.
    target_link_libraries(tensoracle PRIVATE
        "$<BUILD_INTERFACE:${TENSORACLE_TENSORRT_LIBRARY}>"
        "$<BUILD_INTERFACE:tensoracle_json>" miniz::miniz
        "$<BUILD_INTERFACE:tensoracle_internal_cuda_driver>"
        "$<INSTALL_INTERFACE:tensoracle_cuda_driver>" ${CMAKE_DL_LIBS})
endif()
set(TENSORACLE_INSTALL_RPATH "$ORIGIN" CACHE STRING "Relative runtime lookup paths for the installed library")
set_target_properties(tensoracle PROPERTIES
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH_USE_ORIGIN YES
    INSTALL_RPATH "${TENSORACLE_INSTALL_RPATH}")
if(BUILD_SHARED_LIBS AND UNIX AND NOT APPLE)
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
include("${CMAKE_BINARY_DIR}/conan/tensoracle_licenses.cmake")
install(FILES "${TENSORACLE_NLOHMANN_JSON_LICENSE}"
    DESTINATION "${_tensoracle_root}licenses/nlohmann-json")
install(FILES "${TENSORACLE_MINIZ_LICENSE}"
    DESTINATION "${_tensoracle_root}licenses/miniz")
install(FILES src/cubin_plugin.cpp src/sha256.hpp DESTINATION "${_tensoracle_root}src")
install(FILES cmake/TensoracleCuda.cmake cmake/TensoracleCubinPlugin.cmake
    DESTINATION "${TENSORACLE_HELPER_DIR}")
set(TENSORACLE_STATIC_DEPENDENCIES "")
if(NOT BUILD_SHARED_LIBS)
    set(TENSORACLE_STATIC_DEPENDENCIES [=[    include(CMakeFindDependencyMacro)
    find_dependency(miniz 3.1.1 EXACT CONFIG)
    if(NOT WIN32)
        find_dependency(Threads)
    endif()
    _tensoracle_find_cuda()
]=])
endif()
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
