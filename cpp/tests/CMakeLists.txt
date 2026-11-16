include("${CMAKE_CURRENT_LIST_DIR}/runtime.cmake")
find_package(Catch2 3.16.1 EXACT CONFIG REQUIRED)

add_executable(tensoracle_abi_test abi.cpp comparison.cpp)
target_link_libraries(tensoracle_abi_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain)
include(Catch)
catch_discover_tests(tensoracle_abi_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)

add_executable(tensoracle_c_consumer c_consumer.c)
target_compile_features(tensoracle_c_consumer PRIVATE c_std_11)
target_compile_options(tensoracle_c_consumer PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-pedantic-errors;-Wall;-Wextra;-Werror>)
target_link_libraries(tensoracle_c_consumer PRIVATE tensoracle::tensoracle)
add_test(NAME "C callers use the ABI without C++ headers" COMMAND tensoracle_c_consumer)
set_tests_properties("C callers use the ABI without C++ headers" PROPERTIES
    ${_tensoracle_test_properties} LABELS "SP-TENSORACLE-014;SP-TENSORACLE-056;SP-TENSORACLE-061")

add_test(NAME "Installed SDK builds a separate C++ application"
    COMMAND "${CMAKE_COMMAND}"
        "-DBUILD_DIR=${PROJECT_BINARY_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer"
        "-DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
        "-DCONFIG=$<CONFIG>"
        "-DCXX=${CMAKE_CXX_COMPILER}"
        "-DSHARED=${BUILD_SHARED_LIBS}"
        "-DTOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DTENSORRT_LIBRARY=${TENSORACLE_TENSORRT_LIBRARY}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/installed.cmake")
set_tests_properties("Installed SDK builds a separate C++ application" PROPERTIES
    ${_tensoracle_test_properties}
    LABELS "SP-TENSORACLE-014;SP-TENSORACLE-021;SP-TENSORACLE-055;SP-TENSORACLE-057;SP-TENSORACLE-065" TIMEOUT 60)
if(NOT BUILD_SHARED_LIBS)
    set_tests_properties("Installed SDK builds a separate C++ application" PROPERTIES
        LABELS "SP-TENSORACLE-014;SP-TENSORACLE-021;SP-TENSORACLE-055;SP-TENSORACLE-073;SP-TENSORACLE-074")
endif()

add_executable(tensoracle_execution_test execution.cpp)
if(NOT WIN32)
    target_link_options(tensoracle_execution_test PRIVATE "LINKER:--export-dynamic-symbol=utimensat")
endif()
target_sources(tensoracle_execution_test PRIVATE ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_execution_test PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_execution_test PRIVATE tensoracle::tensoracle
    Catch2::Catch2WithMain "${TENSORACLE_TENSORRT_LIBRARY}" tensoracle_cuda_driver tensoracle_json)
target_compile_definitions(tensoracle_execution_test PRIVATE
    TENSORACLE_CACHE_PROBE="$<TARGET_FILE:tensoracle_cache_probe>")
catch_discover_tests(tensoracle_execution_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 180)
add_executable(tensoracle_utilities_test utilities.cpp)
target_include_directories(tensoracle_utilities_test PRIVATE ../src)
target_link_libraries(tensoracle_utilities_test PRIVATE tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_utilities_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

add_executable(tensoracle_cache_test cache.cpp ../src/cache.cpp)
if(NOT WIN32)
    target_link_options(tensoracle_cache_test PRIVATE "LINKER:--wrap=stat" "LINKER:--wrap=fsync")
endif()
target_include_directories(tensoracle_cache_test PRIVATE ../src)
target_link_libraries(tensoracle_cache_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_cache_test PRIVATE
    TENSORACLE_CACHE_PROBE="$<TARGET_FILE:tensoracle_cache_probe>")
catch_discover_tests(tensoracle_cache_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)
add_executable(tensoracle_driver_failures_test driver_failures.cpp ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_driver_failures_test PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_driver_failures_test PRIVATE tensoracle::tensoracle
    tensoracle_json Catch2::Catch2WithMain "${TENSORACLE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
if(NOT WIN32)
    target_link_options(tensoracle_driver_failures_test PRIVATE "LINKER:--wrap=dlsym")
endif()
catch_discover_tests(tensoracle_driver_failures_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)
add_executable(tensoracle_cache_probe cache_probe.cpp ../src/cache.cpp ../src/cache_open.cpp)
target_include_directories(tensoracle_cache_probe PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_cache_probe PRIVATE tensoracle::tensoracle tensoracle_json "${TENSORACLE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
add_executable(tensoracle_cubin_test cubin.cpp)
target_link_libraries(tensoracle_cubin_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_cubin_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 60)
target_compile_definitions(tensoracle_cubin_test PRIVATE
    TENSORACLE_TEST_CUDA_ROOT="${_tensoracle_test_producer_cuda_root}"
    TENSORACLE_TEST_BUNDLE_ROOT="${PROJECT_SOURCE_DIR}/../examples/cubin_plugin/bundle")

add_library(tensoracle_exports_fixture SHARED native_exports_fixture.cpp)
set_target_properties(tensoracle_exports_fixture PROPERTIES CXX_VISIBILITY_PRESET hidden)
add_executable(tensoracle_native_exports_test native_exports.cpp)
target_include_directories(tensoracle_native_exports_test PRIVATE ../src)
target_link_libraries(tensoracle_native_exports_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_native_exports_test PRIVATE TENSORACLE_EXPORT_FIXTURE="$<TARGET_FILE:tensoracle_exports_fixture>")
add_dependencies(tensoracle_native_exports_test tensoracle_exports_fixture)
catch_discover_tests(tensoracle_native_exports_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)

add_library(tensoracle_valid_plugin SHARED plugin_fixture.cpp)
add_library(tensoracle_legacy_plugin SHARED plugin_fixture.cpp)
target_compile_definitions(tensoracle_legacy_plugin PRIVATE TENSORACLE_PLUGIN_VALID TENSORACLE_PLUGIN_LEGACY)
target_include_directories(tensoracle_legacy_plugin PRIVATE
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_compile_definitions(tensoracle_valid_plugin PRIVATE TENSORACLE_PLUGIN_VALID)
target_include_directories(tensoracle_valid_plugin PRIVATE
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
add_library(tensoracle_invalid_plugin SHARED plugin_fixture.cpp)
add_library(tensoracle_rejected_plugin SHARED plugin_fixture.cpp)
target_compile_definitions(tensoracle_rejected_plugin PRIVATE TENSORACLE_PLUGIN_VALID TENSORACLE_PLUGIN_NO_LOGGER)
target_include_directories(tensoracle_rejected_plugin PRIVATE
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
add_executable(tensoracle_plugins_test plugins.cpp)
foreach(_fixture IN ITEMS registration no_exports other_registration)
    add_library(tensoracle_${_fixture}_plugin SHARED registration_fixture.cpp)
    target_include_directories(tensoracle_${_fixture}_plugin PRIVATE
        "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
    target_compile_definitions(tensoracle_${_fixture}_plugin PRIVATE
        TENSORACLE_FIXTURE_NAMESPACE="tensoracle.tests.${_fixture}")
    set_target_properties(tensoracle_${_fixture}_plugin PROPERTIES CXX_VISIBILITY_PRESET hidden)
endforeach()
target_compile_definitions(tensoracle_no_exports_plugin PRIVATE TENSORACLE_NO_GET_CREATORS)
target_include_directories(tensoracle_plugins_test PRIVATE ../src
    "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_plugins_test PRIVATE tensoracle::tensoracle Catch2::Catch2WithMain
    "${TENSORACLE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
target_compile_definitions(tensoracle_plugins_test PRIVATE
    TENSORACLE_VALID_PLUGIN="$<TARGET_FILE:tensoracle_valid_plugin>"
    TENSORACLE_LEGACY_PLUGIN="$<TARGET_FILE:tensoracle_legacy_plugin>"
    TENSORACLE_INVALID_PLUGIN="$<TARGET_FILE:tensoracle_invalid_plugin>"
    TENSORACLE_REGISTRATION_PLUGIN="$<TARGET_FILE:tensoracle_registration_plugin>"
    TENSORACLE_NO_EXPORTS_PLUGIN="$<TARGET_FILE:tensoracle_no_exports_plugin>"
    TENSORACLE_OTHER_REGISTRATION_PLUGIN="$<TARGET_FILE:tensoracle_other_registration_plugin>"
    TENSORACLE_REJECTED_PLUGIN="$<TARGET_FILE:tensoracle_rejected_plugin>")
add_dependencies(tensoracle_plugins_test tensoracle_valid_plugin tensoracle_invalid_plugin tensoracle_rejected_plugin
    tensoracle_registration_plugin tensoracle_no_exports_plugin tensoracle_other_registration_plugin tensoracle_legacy_plugin)
catch_discover_tests(tensoracle_plugins_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)

add_dependencies(tensoracle_execution_test tensoracle_cache_probe)
add_dependencies(tensoracle_cache_test tensoracle_cache_probe)

add_executable(tensoracle_cache_management_test cache_management.cpp ../src/cache.cpp ../src/cache_management_internal.cpp)
target_include_directories(tensoracle_cache_management_test PRIVATE ../src)
target_link_libraries(tensoracle_cache_management_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_cache_management_test PRIVATE
    TENSORACLE_CACHE_PROBE="$<TARGET_FILE:tensoracle_cache_probe>")
add_dependencies(tensoracle_cache_management_test tensoracle_cache_probe)
catch_discover_tests(tensoracle_cache_management_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)

add_executable(tensoracle_weights_test weights.cpp ../src/safetensors.cpp)
target_include_directories(tensoracle_weights_test PRIVATE ../src)
target_link_libraries(tensoracle_weights_test PRIVATE tensoracle::tensoracle tensoracle_json Catch2::Catch2WithMain)
catch_discover_tests(tensoracle_weights_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 30)

add_executable(tensoracle_builder_probe builder_probe.cpp)
target_include_directories(tensoracle_builder_probe PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_builder_probe PRIVATE tensoracle::tensoracle tensoracle_json tensoracle_cuda_driver "${TENSORACLE_TENSORRT_LIBRARY}")
add_executable(tensoracle_builder_test builder.cpp)
target_include_directories(tensoracle_builder_test PRIVATE ../src "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
target_link_libraries(tensoracle_builder_test PRIVATE tensoracle::tensoracle tensoracle_json tensoracle_cuda_driver "${TENSORACLE_TENSORRT_LIBRARY}" Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_builder_test PRIVATE TENSORACLE_BUILDER_PROBE="$<TARGET_FILE:tensoracle_builder_probe>")
add_dependencies(tensoracle_builder_test tensoracle_builder_probe)
catch_discover_tests(tensoracle_builder_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
    ${_tensoracle_test_properties} TIMEOUT 180)

if(TENSORACLE_BUILD_CUBIN_HELPER)
    add_executable(tensoracle_precision_test precision.cpp
        "${PROJECT_SOURCE_DIR}/../examples/cubin_plugin/bundle/bundle.cpp")
    target_include_directories(tensoracle_precision_test PRIVATE "${TENSORACLE_TENSORRT_INCLUDE}")
    target_link_libraries(tensoracle_precision_test PRIVATE tensoracle::tensoracle tensoracle_cubin
        tensoracle_json Catch2::Catch2WithMain)
    catch_discover_tests(tensoracle_precision_test DL_PATHS ${_tensoracle_test_dl_paths} ADD_TAGS_AS_LABELS PROPERTIES
        ${_tensoracle_test_properties} TIMEOUT 60)
endif()

if(WIN32)
    foreach(_target IN ITEMS tensoracle_execution_test tensoracle_cache_test tensoracle_driver_failures_test)
        # Replace only Win32 external boundaries in these test executables.
        target_compile_options(${_target} PRIVATE "/FI${CMAKE_CURRENT_SOURCE_DIR}/windows_boundaries.hpp")
    endforeach()
endif()

if(WIN32)
    add_executable(tensoracle_old_ptxas old_ptxas.cpp)
    target_compile_definitions(tensoracle_cubin_test PRIVATE TENSORACLE_OLD_PTXAS="$<TARGET_FILE:tensoracle_old_ptxas>")
    add_dependencies(tensoracle_cubin_test tensoracle_old_ptxas)

    add_executable(tensoracle_hardlink_probe embedded_plugin/hardlink_probe.cpp)
    target_include_directories(tensoracle_hardlink_probe PRIVATE
        "${TENSORACLE_TENSORRT_INCLUDE}" "${TENSORACLE_CUDA_INCLUDE}")
    target_link_libraries(tensoracle_hardlink_probe PRIVATE tensoracle::tensoracle
        tensoracle_json "${TENSORACLE_TENSORRT_LIBRARY}")
    add_executable(tensoracle_windows_module_identity_test windows_module_identity.cpp)
    target_include_directories(tensoracle_windows_module_identity_test PRIVATE ../src)
    target_link_libraries(tensoracle_windows_module_identity_test PRIVATE tensoracle::tensoracle
        tensoracle_json Catch2::Catch2WithMain)
    target_compile_definitions(tensoracle_windows_module_identity_test PRIVATE
        TENSORACLE_HARDLINK_PROBE="$<TARGET_FILE:tensoracle_hardlink_probe>"
        TENSORACLE_REGISTRATION_PLUGIN="$<TARGET_FILE:tensoracle_registration_plugin>")
    add_dependencies(tensoracle_windows_module_identity_test tensoracle_hardlink_probe tensoracle_registration_plugin)
    catch_discover_tests(tensoracle_windows_module_identity_test DL_PATHS ${_tensoracle_test_dl_paths}
        ADD_TAGS_AS_LABELS PROPERTIES ${_tensoracle_test_properties} TIMEOUT 60)
endif()

# These exercise the actual file locks without creating a CUDA context.
add_executable(tensoracle_file_lock_probe file_lock_probe.cpp ../src/cache.cpp)
target_include_directories(tensoracle_file_lock_probe PRIVATE ../src ../include)
target_link_libraries(tensoracle_file_lock_probe PRIVATE tensoracle_json)
add_executable(tensoracle_cache_lock_test cache_lock.cpp ../src/cache.cpp)
target_include_directories(tensoracle_cache_lock_test PRIVATE ../src ../include)
target_link_libraries(tensoracle_cache_lock_test PRIVATE tensoracle_json Catch2::Catch2WithMain)
target_compile_definitions(tensoracle_cache_lock_test PRIVATE TENSORACLE_FILE_LOCK_PROBE="$<TARGET_FILE:tensoracle_file_lock_probe>")
add_dependencies(tensoracle_cache_lock_test tensoracle_file_lock_probe)
catch_discover_tests(tensoracle_cache_lock_test ADD_TAGS_AS_LABELS PROPERTIES TIMEOUT 30)

if(WIN32)
    # Catch's DL_PATHS handles discovery, test execution and their children.
    # These two plain CTests need the same directories without splitting PATH
    # through Catch's PROPERTIES list parser.
    set_tests_properties("C callers use the ABI without C++ headers"
        "Installed SDK builds a separate C++ application" PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${_tensoracle_test_runtime};PATH=path_list_prepend:$<TARGET_FILE_DIR:tensoracle>")
endif()
