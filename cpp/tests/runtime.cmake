# Compilation and linking keep using the Conan SDK. Tests need builder resources.
set(TENSORACLE_TEST_TENSORRT_RUNTIME "" CACHE PATH "Full TensorRT runtime directory for CTest")
if(TENSORACLE_TEST_TENSORRT_RUNTIME)
    set(_tensoracle_runtime_candidates "${TENSORACLE_TEST_TENSORRT_RUNTIME}")
else()
    file(GLOB _tensoracle_runtime_candidates LIST_DIRECTORIES TRUE
        "${CMAKE_CURRENT_LIST_DIR}/../../python/.venv/lib/python*/site-packages/tensorrt_libs")
endif()

set(_tensoracle_test_runtime "")
foreach(_candidate IN LISTS _tensoracle_runtime_candidates)
    file(GLOB _builder_resources "${_candidate}/libnvinfer_builder_resource*.so*")
    if(EXISTS "${_candidate}/libnvinfer.so.11" AND _builder_resources)
        get_filename_component(_tensoracle_test_runtime "${_candidate}" ABSOLUTE)
        break()
    endif()
endforeach()
if(NOT _tensoracle_test_runtime)
    message(FATAL_ERROR
        "C++ tests need a full TensorRT runtime with libnvinfer.so.11 and builder resource libraries. "
        "Set -DTENSORACLE_TEST_TENSORRT_RUNTIME=/path/to/tensorrt/lib, or run uv sync in python/ "
        "to prepare python/.venv. Searched: ${_tensoracle_runtime_candidates}")
endif()
message(STATUS "TensorRT runtime for CTest: ${_tensoracle_test_runtime}")
set(_tensoracle_test_environment "LD_LIBRARY_PATH=${_tensoracle_test_runtime}")
if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    string(APPEND _tensoracle_test_environment ":$ENV{LD_LIBRARY_PATH}")
endif()
