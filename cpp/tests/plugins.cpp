#include <tensoracle/plugins.hpp>
#include "sha256.hpp"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
struct fixture {
    fs::path directory;
    fixture() {
        char pattern[] = "/tmp/tensoracle-plugin-XXXXXX";
        auto* path = mkdtemp(pattern);
        REQUIRE(path);
        directory = path;
    }
    ~fixture() { unsetenv("TENSORACLE_TEST_PLUGIN_MARKER"); fs::remove_all(directory); }
};
std::string bytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
}
}
TEST_CASE("Plugin export checks reject unsafe libraries before initializers run", "[SP-TENSORACLE-054][SP-TENSORACLE-011]") {
    fixture files;
    auto marker = files.directory / "loaded";
    setenv("TENSORACLE_TEST_PLUGIN_MARKER", marker.c_str(), 1);
    try {
        tensoracle::register_libraries({TENSORACLE_INVALID_PLUGIN});
        FAIL("missing getCreators was accepted");
    } catch (const tensoracle::error& failure) {
        REQUIRE(failure.status() == TENSORACLE_PLUGIN_LIBRARY_ERROR);
        REQUIRE(std::string(failure.what()).find("getCreators") != std::string::npos);
    }
    REQUIRE_FALSE(fs::exists(marker));
    auto* library = dlopen(TENSORACLE_INVALID_PLUGIN, RTLD_NOW | RTLD_LOCAL);
    REQUIRE(library);
    REQUIRE(fs::exists(marker));
    dlclose(library);
    REQUIRE_THROWS_AS(tensoracle::register_libraries({(files.directory / "missing.so").string()}), tensoracle::error);
}
TEST_CASE("Plugin registration retains and deduplicates loaded bytes and rejects changes", "[SP-TENSORACLE-054][SP-TENSORACLE-009]") {
    fixture files;
    auto marker = files.directory / "loaded";
    setenv("TENSORACLE_TEST_PLUGIN_MARKER", marker.c_str(), 1);
    auto path = files.directory / "first.so";
    auto copy = files.directory / "same.so";
    fs::copy_file(TENSORACLE_VALID_PLUGIN, path);
    fs::copy_file(path, copy);
    auto data = bytes(path);
    auto expected = tensoracle::detail::sha256(data.data(), data.size());
    REQUIRE(tensoracle::register_libraries({path.string(), path.string(), copy.string()}) ==
            std::vector<std::string>{expected, expected, expected});
    REQUIRE(tensoracle::register_libraries({path.string()}) == std::vector<std::string>{expected});
    REQUIRE(bytes(marker) == "x");
    { std::ofstream output(path, std::ios::binary | std::ios::app); output << "changed"; }
    try {
        tensoracle::register_libraries({path.string()});
        FAIL("changed plugin was accepted");
    } catch (const tensoracle::error& failure) {
        REQUIRE(failure.status() == TENSORACLE_PLUGIN_LIBRARY_ERROR);
        REQUIRE(std::string(failure.what()).find("changed") != std::string::npos);
    }
}
TEST_CASE("Plugin ABI returns owned diagnostics and validates digest indexes", "[SP-TENSORACLE-054][SP-TENSORACLE-056]") {
    tensoracle_plugin_libraries* result = reinterpret_cast<tensoracle_plugin_libraries*>(1);
    tensoracle_error* error = nullptr;
    REQUIRE(tensoracle_register_libraries(nullptr, 1, &result, &error) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(result == nullptr);
    REQUIRE(error);
    tensoracle_error_destroy(error);
    REQUIRE(tensoracle_register_libraries(nullptr, 0, &result, nullptr) == TENSORACLE_SUCCESS);
    size_t count = 1;
    REQUIRE(tensoracle_plugin_libraries_count(result, &count, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(count == 0);
    const char* digest = reinterpret_cast<const char*>(1);
    REQUIRE(tensoracle_plugin_libraries_digest(result, 0, &digest, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(digest == nullptr);
    REQUIRE(tensoracle_plugin_libraries_destroy(result) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_plugin_libraries_destroy(nullptr) == TENSORACLE_SUCCESS);
}

TEST_CASE("A library with exports but a rejected TensorRT contract reports a plugin error", "[SP-TENSORACLE-011][SP-TENSORACLE-054]") {
    try {
        tensoracle::register_libraries({TENSORACLE_REJECTED_PLUGIN});
        FAIL("TensorRT accepted a plugin missing setLoggerFinder");
    } catch (const tensoracle::error& failure) {
        REQUIRE(failure.status() == TENSORACLE_PLUGIN_LIBRARY_ERROR);
        REQUIRE(std::string(failure.what()).find("register") != std::string::npos);
    }
}
