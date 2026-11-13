cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/install.cmake")
tensoracle_conan_profile()
set(TENSORACLE_CONAN_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.nf/tmp/conan-tests" CACHE PATH "Test dependency generators")
# Preparation is the single writer of custom recipes before parallel tests.
set(TENSORACLE_CONAN_OFFLINE OFF)
tensoracle_install_dependencies("${TENSORACLE_CONAN_PROFILE_PATH}" "${TENSORACLE_CONAN_OUTPUT_DIR}")

# The compatibility test builds readers with both JSON versions, offline.
if(TENSORACLE_CONAN_EXECUTABLE)
    set(_conan "${TENSORACLE_CONAN_EXECUTABLE}")
else()
    find_program(_conan conan REQUIRED)
endif()
set(_cache_args)
if(TENSORACLE_CONAN_CACHE)
    list(APPEND _cache_args --core-conf "core.cache:storage_path=${TENSORACLE_CONAN_CACHE}")
endif()
set(_compatibility_lock "${CMAKE_CURRENT_LIST_DIR}/cache-compatibility.lock")
file(READ "${_compatibility_lock}" _compatibility_contents)
string(JSON _compatibility_reference GET "${_compatibility_contents}" requires 0)
execute_process(COMMAND "${_conan}" install "--requires=${_compatibility_reference}"
    --lockfile "${_compatibility_lock}" --profile:host "${TENSORACLE_CONAN_PROFILE_PATH}"
    --profile:build "${TENSORACLE_CONAN_PROFILE_PATH}"
    --output-folder "${TENSORACLE_CONAN_OUTPUT_DIR}/cache-compatibility"
    --build=missing --remote conancenter ${TENSORACLE_CONAN_ARGS} ${_cache_args}
    COMMAND_ERROR_IS_FATAL ANY)
