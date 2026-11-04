cmake_minimum_required(VERSION 3.26)
project(tensoracle_cache_fixture LANGUAGES CXX)

include("${TENSORACLE_CPP_ROOT}/cmake/TensoracleJson.cmake")
add_executable(cache_writer writer.cpp "${TENSORACLE_CPP_ROOT}/src/cache.cpp")
target_compile_features(cache_writer PRIVATE cxx_std_17)
target_include_directories(cache_writer PRIVATE
    "${TENSORACLE_CPP_ROOT}/include" "${TENSORACLE_CPP_ROOT}/src")
target_link_libraries(cache_writer PRIVATE tensoracle_json)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/artifact-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:cache_writer>")
