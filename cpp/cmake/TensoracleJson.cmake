include_guard(GLOBAL)
include(FetchContent)
# nlohmann/json v3.11.3 (MIT), https://github.com/nlohmann/json/tree/v3.11.3
# The release archive is verified. FETCHCONTENT_SOURCE_DIR_TENSORACLE_JSON
# can supply the same sources for an offline build.
FetchContent_Declare(tensoracle_json
    URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
    SOURCE_SUBDIR tensoracle-unused)
FetchContent_MakeAvailable(tensoracle_json)
add_library(tensoracle_json INTERFACE)
target_include_directories(tensoracle_json SYSTEM INTERFACE "${tensoracle_json_SOURCE_DIR}/single_include")
