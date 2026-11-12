cmake_minimum_required(VERSION 3.26)
if(NOT TENSORRT_ROOT)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cpp/conan/bootstrap.cmake")
endif()
project(tensoracle_cubin_example LANGUAGES CXX)

find_package(Python 3.12 REQUIRED COMPONENTS
    Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "from importlib.metadata import distribution; print(distribution('tensoracle').locate_file(''))"
    OUTPUT_VARIABLE _site OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${_site}/tensoracle/cpp")
find_package(tensoracle 2 CONFIG REQUIRED)
if(NOT TENSORRT_ROOT)
    find_package(TensorRT 11.2.1.2 EXACT CONFIG REQUIRED)
    list(GET TensorRT_INCLUDE_DIRS 0 _tensorrt_include)
    get_filename_component(TENSORRT_ROOT "${_tensorrt_include}" DIRECTORY)
endif()
if(NOT DEFINED TENSORACLE_BUNDLE_ROOT)
    set(TENSORACLE_BUNDLE_ROOT "${CMAKE_CURRENT_LIST_DIR}/bundle")
endif()
if(NOT CUDAToolkit_ROOT)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "from importlib.metadata import distribution; print(distribution('nvidia-cuda-runtime').locate_file('nvidia/cu13'))"
        OUTPUT_VARIABLE CUDAToolkit_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()

tensoracle_add_cubin_plugin(
    TARGET tensoracle_example_plugin
    TENSORRT_ROOT "${TENSORRT_ROOT}"
    BUNDLE_ROOT "${TENSORACLE_BUNDLE_ROOT}"
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/plugin.cpp")

nanobind_add_module(_native STABLE_ABI bindings.cpp)
target_include_directories(_native PRIVATE "${TENSORRT_ROOT}/include")
target_link_libraries(_native PRIVATE tensoracle::tensoracle tensoracle_cuda_driver)
file(RELATIVE_PATH _library_from_module
    "${_site}/example_cubin_plugin" "${_site}/tensoracle/cpp/lib")
set_target_properties(_native PROPERTIES
    BUILD_RPATH_USE_ORIGIN YES
    INSTALL_RPATH "$ORIGIN/${_library_from_module}"
    INSTALL_RPATH_USE_LINK_PATH NO)
install(TARGETS _native LIBRARY DESTINATION example_cubin_plugin)

install(TARGETS tensoracle_example_plugin LIBRARY DESTINATION example_cubin_plugin
    RUNTIME DESTINATION example_cubin_plugin)
install(FILES "${TENSORACLE_BUNDLE_ROOT}/bundle.json"
    DESTINATION example_cubin_plugin/bundle)
install(DIRECTORY "${TENSORACLE_BUNDLE_ROOT}/cubins/"
    DESTINATION example_cubin_plugin/bundle/cubins)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/example_cubin_plugin/weights.safetensors"
    DESTINATION example_cubin_plugin)
