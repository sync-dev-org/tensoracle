# Every build entry point includes this before its first project().
include_guard(GLOBAL)
if(CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "${CMAKE_BINARY_DIR}/conan/conan_toolchain.cmake")
    # The caller already resolved the dependency graph, for example in conan create.
    return()
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/install.cmake")
tensoracle_conan_profile()
file(GLOB_RECURSE dependency_inputs LIST_DIRECTORIES FALSE "${CMAKE_CURRENT_LIST_DIR}/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${dependency_inputs}
    "${CMAKE_CURRENT_LIST_DIR}/../conanfile.py" "${CMAKE_CURRENT_LIST_DIR}/../conan.lock"
    "${TENSORACLE_CONAN_PROFILE_PATH}")
tensoracle_install_dependencies("${TENSORACLE_CONAN_PROFILE_PATH}" "${CMAKE_BINARY_DIR}/conan")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_BINARY_DIR}/conan/conan_toolchain.cmake" CACHE FILEPATH "Conan generated toolchain" FORCE)
