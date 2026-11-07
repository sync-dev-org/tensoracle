include(FetchContent)
# Catch2 v3.16.1, pinned to the release commit.
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG 08092139210881f01a34e60c5f2fd6afa0ea4024
    EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(Catch2)

add_executable(tensoracle_abi_test abi.cpp comparison.cpp)
target_link_libraries(tensoracle_abi_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain)
include("${catch2_SOURCE_DIR}/extras/Catch.cmake")
catch_discover_tests(tensoracle_abi_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_executable(tensoracle_c_consumer c_consumer.c)
target_compile_features(tensoracle_c_consumer PRIVATE c_std_11)
target_compile_options(tensoracle_c_consumer PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-pedantic-errors;-Wall;-Wextra;-Werror>)
target_link_libraries(tensoracle_c_consumer PRIVATE tensoracle::tensoracle)
add_test(NAME "C callers use the ABI without C++ headers" COMMAND tensoracle_c_consumer)
set_tests_properties("C callers use the ABI without C++ headers" PROPERTIES LABELS "SP-TENSORACLE-014;SP-TENSORACLE-056;SP-TENSORACLE-061")

add_test(NAME "Installed SDK builds a separate C++ application"
    COMMAND "${CMAKE_COMMAND}"
        "-DBUILD_DIR=${PROJECT_BINARY_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer"
        "-DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
        "-DCONFIG=$<CONFIG>"
        "-DCXX=${CMAKE_CXX_COMPILER}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/installed.cmake")
set_tests_properties("Installed SDK builds a separate C++ application" PROPERTIES
    LABELS "SP-TENSORACLE-014;SP-TENSORACLE-021;SP-TENSORACLE-055;SP-TENSORACLE-057" TIMEOUT 60)
get_filename_component(_tensoracle_runtime "${TENSORACLE_TENSORRT_LIBRARY}" DIRECTORY)
if(UNIX)
    set_property(TEST "Installed SDK builds a separate C++ application" PROPERTY
        ENVIRONMENT "LD_LIBRARY_PATH=${_tensoracle_runtime}:$ENV{LD_LIBRARY_PATH}")
endif()

add_executable(tensoracle_execution_test execution.cpp)
target_link_options(tensoracle_execution_test PRIVATE "LINKER:--export-dynamic-symbol=utimensat")
target_sources(tensoracle_execution_test PRIVATE ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_execution_test PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_execution_test PRIVATE tensoracle::tensoracle
    Catch2::Catch2WithMain "${TENSORACLE_TENSORRT_LIBRARY}" tensoracle_cuda_driver tensoracle_json)
target_compile_definitions(tensoracle_execution_test PRIVATE
    TENSORACLE_CACHE_PROBE="$<TARGET_FILE:tensoracle_cache_probe>")
catch_discover_tests(tensoracle_execution_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 180)
add_executable(tensoracle_utilities_test utilities.cpp)
target_include_directories(tensoracle_utilities_test PRIVATE ../src)
target_link_libraries(tensoracle_utilities_test PRIVATE tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_utilities_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_executable(tensoracle_cache_test cache.cpp ../src/cache.cpp)
target_link_options(tensoracle_cache_test PRIVATE "LINKER:--wrap=stat" "LINKER:--wrap=fsync")
target_include_directories(tensoracle_cache_test PRIVATE ../src)
target_link_libraries(tensoracle_cache_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_cache_test PRIVATE
    TENSORACLE_CACHE_PROBE="$<TARGET_FILE:tensoracle_cache_probe>")
catch_discover_tests(tensoracle_cache_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)
add_executable(tensoracle_driver_failures_test driver_failures.cpp ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_driver_failures_test PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_driver_failures_test PRIVATE tensoracle::tensoracle
    tensoracle_json Catch2::Catch2WithMain "${TENSORACLE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
target_link_options(tensoracle_driver_failures_test PRIVATE "LINKER:--wrap=dlsym")
catch_discover_tests(tensoracle_driver_failures_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)
add_executable(tensoracle_cache_probe cache_probe.cpp ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_cache_probe PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_cache_probe PRIVATE tensoracle::tensoracle tensoracle_json "${TENSORACLE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
add_executable(tensoracle_cubin_test cubin.cpp)
target_link_libraries(tensoracle_cubin_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_cubin_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 60)
target_compile_definitions(tensoracle_cubin_test PRIVATE TENSORACLE_TEST_CUDA_ROOT="${CUDAToolkit_ROOT}")

add_library(tensoracle_exports_fixture SHARED native_exports_fixture.cpp)
set_target_properties(tensoracle_exports_fixture PROPERTIES CXX_VISIBILITY_PRESET hidden)
add_executable(tensoracle_native_exports_test native_exports.cpp)
target_include_directories(tensoracle_native_exports_test PRIVATE ../src)
target_link_libraries(tensoracle_native_exports_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_native_exports_test PRIVATE TENSORACLE_EXPORT_FIXTURE="$<TARGET_FILE:tensoracle_exports_fixture>")
add_dependencies(tensoracle_native_exports_test tensoracle_exports_fixture)
catch_discover_tests(tensoracle_native_exports_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_library(tensoracle_valid_plugin SHARED plugin_fixture.cpp)
target_compile_definitions(tensoracle_valid_plugin PRIVATE TENSORACLE_PLUGIN_VALID)
target_include_directories(tensoracle_valid_plugin PRIVATE
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
add_library(tensoracle_invalid_plugin SHARED plugin_fixture.cpp)
add_library(tensoracle_rejected_plugin SHARED plugin_fixture.cpp)
target_compile_definitions(tensoracle_rejected_plugin PRIVATE TENSORACLE_PLUGIN_VALID TENSORACLE_PLUGIN_NO_LOGGER)
target_include_directories(tensoracle_rejected_plugin PRIVATE
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
add_executable(tensoracle_plugins_test plugins.cpp)
target_include_directories(tensoracle_plugins_test PRIVATE ../src)
target_link_libraries(tensoracle_plugins_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain ${CMAKE_DL_LIBS})
target_compile_definitions(tensoracle_plugins_test PRIVATE
    TENSORACLE_VALID_PLUGIN="$<TARGET_FILE:tensoracle_valid_plugin>"
    TENSORACLE_INVALID_PLUGIN="$<TARGET_FILE:tensoracle_invalid_plugin>"
    TENSORACLE_REJECTED_PLUGIN="$<TARGET_FILE:tensoracle_rejected_plugin>")
add_dependencies(tensoracle_plugins_test tensoracle_valid_plugin tensoracle_invalid_plugin tensoracle_rejected_plugin)
catch_discover_tests(tensoracle_plugins_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_dependencies(tensoracle_execution_test tensoracle_cache_probe)
add_dependencies(tensoracle_cache_test tensoracle_cache_probe)

add_executable(tensoracle_cache_management_test cache_management.cpp ../src/cache.cpp ../src/cache_management_internal.cpp)
target_include_directories(tensoracle_cache_management_test PRIVATE ../src)
target_link_libraries(tensoracle_cache_management_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_cache_management_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_executable(tensoracle_weights_test weights.cpp ../src/safetensors.cpp)
target_include_directories(tensoracle_weights_test PRIVATE ../src)
target_link_libraries(tensoracle_weights_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_weights_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_executable(tensoracle_builder_probe builder_probe.cpp)
target_include_directories(tensoracle_builder_probe PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_builder_probe PRIVATE tensoracle::tensoracle tensoracle_json tensoracle_cuda_driver "${TENSORACLE_TENSORRT_LIBRARY}")
add_executable(tensoracle_builder_test builder.cpp)
target_include_directories(tensoracle_builder_test PRIVATE ../src "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_builder_test PRIVATE tensoracle::tensoracle tensoracle_json tensoracle_cuda_driver "${TENSORACLE_TENSORRT_LIBRARY}" Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_builder_test PRIVATE TENSORACLE_BUILDER_PROBE="$<TARGET_FILE:tensoracle_builder_probe>")
add_dependencies(tensoracle_builder_test tensoracle_builder_probe)
catch_discover_tests(tensoracle_builder_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 180)

if(TENSORACLE_BUILD_CUBIN_HELPER)
    add_executable(tensoracle_precision_test precision.cpp
        "${PROJECT_SOURCE_DIR}/../examples/cubin_plugin/bundle/bundle.cpp")
    target_include_directories(tensoracle_precision_test PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}")
    target_link_libraries(tensoracle_precision_test PRIVATE tensoracle::tensoracle tensoracle_cubin
        tensoracle_json Catch2::Catch2WithMain)
    catch_discover_tests(tensoracle_precision_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 60)
endif()
