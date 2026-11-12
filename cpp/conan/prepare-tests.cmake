cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/install.cmake")
tensoracle_conan_profile()
set(TENSORACLE_CONAN_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.nf/tmp/conan-tests" CACHE PATH "Test dependency generators")
# Preparation is the single writer of custom recipes before parallel tests.
set(TENSORACLE_CONAN_OFFLINE OFF)
tensoracle_install_dependencies("${TENSORACLE_CONAN_PROFILE_PATH}" "${TENSORACLE_CONAN_OUTPUT_DIR}")
