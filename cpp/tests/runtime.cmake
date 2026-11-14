# Compilation and linking keep using the Conan SDK. Tests need builder resources.
set(TENSORACLE_TEST_TENSORRT_RUNTIME "" CACHE PATH "Full TensorRT runtime directory for CTest")
if(TENSORACLE_TEST_TENSORRT_RUNTIME)
    set(_tensoracle_runtime_candidates "${TENSORACLE_TEST_TENSORRT_RUNTIME}")
else()
    file(GLOB _tensoracle_runtime_candidates LIST_DIRECTORIES TRUE
        "${CMAKE_CURRENT_LIST_DIR}/../../python/.venv/lib/python*/site-packages/tensorrt_libs"
        "${CMAKE_CURRENT_LIST_DIR}/../../python/.venv/Lib/site-packages/tensorrt_libs")
endif()

if(WIN32)
    set(_runtime_name nvinfer_11.dll)
else()
    set(_runtime_name libnvinfer.so.11)
endif()
set(_tensoracle_test_runtime "")
foreach(_candidate IN LISTS _tensoracle_runtime_candidates)
    if(WIN32)
        file(GLOB _builder_resources "${_candidate}/nvinfer_builder_resource*.dll")
    else()
        file(GLOB _builder_resources "${_candidate}/libnvinfer_builder_resource*.so*")
    endif()
    if(EXISTS "${_candidate}/${_runtime_name}" AND _builder_resources)
        get_filename_component(_tensoracle_test_runtime "${_candidate}" ABSOLUTE)
        break()
    endif()
endforeach()
if(NOT _tensoracle_test_runtime)
    message(FATAL_ERROR
        "C++ tests need a full TensorRT runtime with ${_runtime_name} and builder resource libraries. "
        "Set -DTENSORACLE_TEST_TENSORRT_RUNTIME=/path/to/tensorrt/lib, or run uv sync in python/ "
        "to prepare python/.venv. Searched: ${_tensoracle_runtime_candidates}")
endif()
message(STATUS "TensorRT runtime for CTest: ${_tensoracle_test_runtime}")
if(WIN32)
    # Catch's discovery subprocess needs the same DLL directories as CTest.
    set(_tensoracle_test_dl_paths "${_tensoracle_test_runtime}" "$<TARGET_FILE_DIR:tensoracle>")
    set(_tensoracle_test_environment "")
    set(_tensoracle_test_properties "")
else()
    set(_tensoracle_test_environment "LD_LIBRARY_PATH=${_tensoracle_test_runtime}")
    if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        string(APPEND _tensoracle_test_environment ":$ENV{LD_LIBRARY_PATH}")
    endif()
    set(_tensoracle_test_properties ENVIRONMENT "${_tensoracle_test_environment}")
endif()

# Cubin production executes the Python producer extra's tools. The Conan CUDA
# SDK remains the compilation input and need not contain a compiler or NVRTC.
set(_tensoracle_test_producer_cuda_root "${CUDAToolkit_ROOT}")
file(GLOB _tensoracle_producer_candidates LIST_DIRECTORIES TRUE
    "${CMAKE_CURRENT_LIST_DIR}/../../python/.venv/lib/python*/site-packages/nvidia/cu13"
    "${CMAKE_CURRENT_LIST_DIR}/../../python/.venv/Lib/site-packages/nvidia/cu13")
foreach(_candidate IN LISTS _tensoracle_producer_candidates)
    if((EXISTS "${_candidate}/bin/ptxas" AND EXISTS "${_candidate}/lib/libnvrtc.so.13") OR
       (EXISTS "${_candidate}/bin/ptxas.exe" AND EXISTS "${_candidate}/bin/x86_64/nvrtc64_130_0.dll"))
        get_filename_component(_tensoracle_test_producer_cuda_root "${_candidate}" ABSOLUTE)
        break()
    endif()
endforeach()
