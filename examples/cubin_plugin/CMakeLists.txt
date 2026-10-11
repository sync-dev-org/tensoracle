cmake_minimum_required(VERSION 3.20)
project(tensoracle_cubin_example LANGUAGES CXX)

if(NOT DEFINED TENSORACLE_CPP_ROOT)
    set(TENSORACLE_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../cpp")
endif()
include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleCubinPlugin.cmake")
if(NOT DEFINED TENSORRT_ROOT)
    message(FATAL_ERROR "Set TENSORRT_ROOT to the checked TensorRT 11 SDK")
endif()
if(NOT DEFINED TENSORACLE_BUNDLE_ROOT)
    set(TENSORACLE_BUNDLE_ROOT "${CMAKE_CURRENT_LIST_DIR}/bundle")
endif()

tensoracle_add_cubin_plugin(
    TARGET tensoracle_example_plugin
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${TENSORACLE_BUNDLE_ROOT}"
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/plugin.cpp")

install(TARGETS tensoracle_example_plugin LIBRARY DESTINATION example_cubin_plugin
    RUNTIME DESTINATION example_cubin_plugin)
install(FILES "${TENSORACLE_BUNDLE_ROOT}/bundle.json"
    DESTINATION example_cubin_plugin/bundle)
install(DIRECTORY "${TENSORACLE_BUNDLE_ROOT}/cubins/"
    DESTINATION example_cubin_plugin/bundle/cubins)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/example_cubin_plugin/weights.safetensors"
    DESTINATION example_cubin_plugin)
