include_guard(GLOBAL)

# Only the Driver API is needed by the C++ helpers. A PyPI CUDA root has neither
# the unversioned cudart linker name nor all the headers of a complete toolkit.
function(_tensoracle_find_cuda)
    if(CUDAToolkit_ROOT)
        set(_roots "${CUDAToolkit_ROOT}")
    elseif(DEFINED ENV{CUDAToolkit_ROOT} AND NOT "$ENV{CUDAToolkit_ROOT}" STREQUAL "")
        set(_roots "$ENV{CUDAToolkit_ROOT}")
    else()
        set(_roots "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
        foreach(_compiler IN ITEMS "${CMAKE_CUDA_COMPILER}" "$ENV{CUDACXX}")
            if(EXISTS "${_compiler}")
                get_filename_component(_bin "${_compiler}" DIRECTORY)
                get_filename_component(_root "${_bin}/.." ABSOLUTE)
                list(APPEND _roots "${_root}")
            endif()
        endforeach()
        find_program(_tensoracle_cuda_nvcc NAMES nvcc nvcc.exe)
        if(_tensoracle_cuda_nvcc)
            get_filename_component(_nvcc "${_tensoracle_cuda_nvcc}" REALPATH)
            get_filename_component(_bin "${_nvcc}" DIRECTORY)
            get_filename_component(_root "${_bin}/.." ABSOLUTE)
            list(APPEND _roots "${_root}")
        endif()
        # Remove private find results before CMake writes its cache (CMake 3.20
        # predates find_program/find_library's NO_CACHE option).
        unset(_tensoracle_cuda_nvcc CACHE)
        if(DEFINED ENV{CUDA_PATH})
            list(APPEND _roots "$ENV{CUDA_PATH}")
        endif()
        if(WIN32)
            file(GLOB _defaults "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        else()
            file(GLOB _defaults "/usr/local/cuda-*")
            list(APPEND _roots "/usr/local/cuda")
        endif()
        list(SORT _defaults COMPARE NATURAL ORDER DESCENDING)
        list(APPEND _roots ${_defaults})
    endif()

    set(_searched_includes)
    set(_include)
    foreach(_root IN LISTS _roots)
        foreach(_candidate IN ITEMS "${_root}/include" "${_root}/targets/x86_64-linux/include")
            list(APPEND _searched_includes "${_candidate}")
            if(EXISTS "${_candidate}/cuda.h")
                set(_include "${_candidate}")
                set(_selected_root "${_root}")
                break()
            endif()
        endforeach()
        if(_include)
            break()
        endif()
    endforeach()
    set(_root "${_selected_root}")
    if(NOT _include)
        message(FATAL_ERROR "CUDA header cuda.h was not found. Searched: ${_searched_includes}. "
            "Install PyPI nvidia-cuda-nvcc and nvidia-cuda-runtime 13.2 and set "
            "CUDAToolkit_ROOT to their site-packages/nvidia/cu13 directory.")
    endif()

    if(WIN32)
        set(_driver_paths "${_root}/lib/x64")
        find_library(_tensoracle_cuda_driver_library NAMES cuda
            PATHS ${_driver_paths} NO_DEFAULT_PATH)
    else()
        set(_driver_paths "/usr/lib/wsl/lib" "${_root}/lib64" "${_root}/lib"
            "${_root}/targets/x86_64-linux/lib")
        find_library(_tensoracle_cuda_driver_library NAMES cuda HINTS ${_driver_paths})
        if(NOT _tensoracle_cuda_driver_library)
            list(APPEND _driver_paths "${_root}/lib64/stubs" "${_root}/lib/stubs"
                "${_root}/targets/x86_64-linux/lib/stubs")
            find_library(_tensoracle_cuda_driver_library NAMES cuda
                PATHS ${_driver_paths} NO_DEFAULT_PATH)
        endif()
    endif()
    set(_driver "${_tensoracle_cuda_driver_library}")
    unset(_tensoracle_cuda_driver_library CACHE)
    if(NOT _driver)
        message(FATAL_ERROR "CUDA driver link library was not found. Searched: ${_driver_paths} "
            "and system library directories. Install the NVIDIA driver on Linux; on Windows "
            "install PyPI nvidia-cuda-runtime 13.2 and set CUDAToolkit_ROOT to its "
            "site-packages/nvidia/cu13 directory (lib/x64/cuda.lib).")
    endif()
    message(STATUS "tensoracle CUDA headers: ${_include}")
    message(STATUS "tensoracle CUDA driver: ${_driver}")
    if(NOT TARGET tensoracle_internal_cuda_driver)
        add_library(tensoracle_internal_cuda_driver UNKNOWN IMPORTED GLOBAL)
        set_target_properties(tensoracle_internal_cuda_driver PROPERTIES
            IMPORTED_LOCATION "${_driver}" INTERFACE_INCLUDE_DIRECTORIES "${_include}")
    endif()
    if(NOT TARGET tensoracle_cuda_driver)
        add_library(tensoracle_cuda_driver ALIAS tensoracle_internal_cuda_driver)
    endif()
    set(_tensoracle_cuda_root "${_root}" PARENT_SCOPE)
endfunction()

# Return the same CUDA inputs used by tensoracle_add_cubin_plugin and expose
# tensoracle_cuda_driver for consumers that only need to link the Driver API.
function(tensoracle_find_cuda)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "OUT_INCLUDE_DIR;OUT_DRIVER_LIBRARY" "")
    if(DEFINED ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unrecognized tensoracle_find_cuda arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(ARG_KEYWORDS_MISSING_VALUES OR
       NOT DEFINED ARG_OUT_INCLUDE_DIR OR "${ARG_OUT_INCLUDE_DIR}" STREQUAL "" OR
       NOT DEFINED ARG_OUT_DRIVER_LIBRARY OR "${ARG_OUT_DRIVER_LIBRARY}" STREQUAL "")
        message(FATAL_ERROR
            "tensoracle_find_cuda requires OUT_INCLUDE_DIR and OUT_DRIVER_LIBRARY variable names")
    endif()

    _tensoracle_find_cuda()
    get_target_property(_include tensoracle_cuda_driver INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_driver tensoracle_cuda_driver IMPORTED_LOCATION)
    set("${ARG_OUT_INCLUDE_DIR}" "${_include}" PARENT_SCOPE)
    set("${ARG_OUT_DRIVER_LIBRARY}" "${_driver}" PARENT_SCOPE)
endfunction()
