include_guard(GLOBAL)
include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/TensoracleJson.cmake")
# nlohmann/json v3.11.3 (MIT), https://github.com/nlohmann/json/tree/v3.11.3
# miniz 3.0.2 (MIT), https://github.com/richgel999/miniz/tree/3.0.2
# Release archives are verified before configuration. FETCHCONTENT_SOURCE_DIR_*
# can supply these same sources for an offline build.
FetchContent_Declare(tensoracle_miniz
    URL https://codeload.github.com/richgel999/miniz/tar.gz/refs/tags/3.0.2
    URL_HASH SHA256=c4b4c25a4eb81883448ff8924e6dba95c800094a198dc9ce66a292ac2ef8e018
    SOURCE_SUBDIR tensoracle-unused)
FetchContent_MakeAvailable(tensoracle_miniz)
add_library(tensoracle_miniz STATIC
    "${tensoracle_miniz_SOURCE_DIR}/miniz.c"
    "${tensoracle_miniz_SOURCE_DIR}/miniz_zip.c"
    "${tensoracle_miniz_SOURCE_DIR}/miniz_tinfl.c"
    "${tensoracle_miniz_SOURCE_DIR}/miniz_tdef.c")
file(WRITE "${tensoracle_miniz_BINARY_DIR}/miniz_export.h" "#pragma once\n#define MINIZ_EXPORT\n")
target_include_directories(tensoracle_miniz SYSTEM PUBLIC
    "${tensoracle_miniz_SOURCE_DIR}" "${tensoracle_miniz_BINARY_DIR}")
set_target_properties(tensoracle_miniz PROPERTIES POSITION_INDEPENDENT_CODE YES C_VISIBILITY_PRESET hidden)
