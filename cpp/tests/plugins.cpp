#include <tensoracle/plugins.hpp>
#include "sha256.hpp"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <dlfcn.h>
#include <unistd.h>
#include <NvInferRuntime.h>

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
class embedded_creator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection*, nvinfer1::TensorRTPhase) noexcept override { return nullptr; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    const char* getPluginName() const noexcept override { return "TensoracleEmbeddedRegistrationFixture"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return "tensoracle.tests.embedded"; }
private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
struct linked_fixture {
    void* module;
    void* anchor;
    std::string plugin_namespace;
    std::vector<void*> creators;
    explicit linked_fixture(const fs::path& path, const std::string& suffix) {
        module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        REQUIRE(module);
        anchor = dlsym(module, "fixture_creators"); REQUIRE(anchor);
        plugin_namespace = "tensoracle.tests." + suffix;
        auto set_namespace = reinterpret_cast<void(*)(const char*)>(dlsym(module, "fixture_namespace"));
        REQUIRE(set_namespace); set_namespace(plugin_namespace.c_str());
        auto get = reinterpret_cast<nvinfer1::IPluginCreatorInterface* const*(*)(int32_t*)>(anchor);
        int32_t count = 0; auto* pointers = get(&count);
        creators.assign(pointers, pointers + count);
    }
    // Registration borrows creators and their module for the process lifetime.
    void* symbol(const char* name) const { auto* value = dlsym(module, name); REQUIRE(value); return value; }
};
void library_error(const fs::path& path, const char* element) {
    auto* result = reinterpret_cast<tensoracle_plugin_libraries*>(1);
    tensoracle_error* diagnostic = nullptr;
    const char* paths[]{path.c_str()};
    auto status = tensoracle_register_libraries(paths, 1, &result, &diagnostic);
    tensoracle::plugin_libraries result_owner(result);
    tensoracle::error_handle error(diagnostic);
    REQUIRE(status == TENSORACLE_PLUGIN_LIBRARY_ERROR);
    REQUIRE(result == nullptr);
    const char* message = nullptr;
    REQUIRE(tensoracle_error_message(error.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(message).find(element) != std::string::npos);
}
}
TEST_CASE("A directly registered creator does not mark its whole library as loaded",
          "[SP-TENSORACLE-077]") {
    fixture files; auto path = files.directory / "partial.so";
    fs::copy_file(TENSORACLE_REGISTRATION_PLUGIN, path);
    linked_fixture plugin(path, "partial-library");
    tensoracle::register_embedded_plugins(plugin.anchor, {plugin.creators[0]}, plugin.plugin_namespace.c_str());
    REQUIRE(getPluginRegistry()->getCreator("RegistrationSecond", "1", plugin.plugin_namespace.c_str()) == nullptr);
    library_error(path, "register");
}
TEST_CASE("Failed partial direct registration cannot bypass the library loader",
          "[SP-TENSORACLE-077]") {
    fixture files; auto path = files.directory / "failed.so";
    fs::copy_file(TENSORACLE_REGISTRATION_PLUGIN, path);
    linked_fixture plugin(path, "failed-partial-library");
    auto conflict = reinterpret_cast<nvinfer1::IPluginCreatorInterface*(*)()>(plugin.symbol("fixture_conflict"))();
    try {
        tensoracle::register_embedded_plugins(plugin.anchor, {plugin.creators[0], conflict}, plugin.plugin_namespace.c_str());
        FAIL("conflicting creator was accepted");
    } catch (const tensoracle::error& failure) { REQUIRE(failure.status() == TENSORACLE_PLUGIN_LIBRARY_ERROR); }
    REQUIRE(getPluginRegistry()->getCreator("RegistrationFirst", "1", plugin.plugin_namespace.c_str()) == plugin.creators[0]);
    library_error(path, "register");
}
TEST_CASE("Direct registration does not bypass the legacy getCreators export check",
          "[SP-TENSORACLE-077]") {
    fixture files; auto path = files.directory / "no-exports.so";
    fs::copy_file(TENSORACLE_NO_EXPORTS_PLUGIN, path);
    linked_fixture plugin(path, "missing-export");
    tensoracle::register_embedded_plugins(plugin.anchor, {plugin.creators[0]}, plugin.plugin_namespace.c_str());
    library_error(path, "getCreators");
}
TEST_CASE("Different modules can use direct and whole library registration together",
          "[SP-TENSORACLE-077]") {
    fixture files; auto path = files.directory / "direct.so";
    fs::copy_file(TENSORACLE_NO_EXPORTS_PLUGIN, path);
    linked_fixture plugin(path, "independent-direct");
    REQUIRE(tensoracle::register_embedded_plugins(plugin.anchor, plugin.creators, plugin.plugin_namespace.c_str()).digests().size() == 1);
    REQUIRE(tensoracle::register_libraries({TENSORACLE_OTHER_REGISTRATION_PLUGIN}).size() == 1);
    REQUIRE(getPluginRegistry()->getCreator("RegistrationSecond", "1", "tensoracle.tests.other_registration"));
}
TEST_CASE("Embedded registration rejects a replaced module file before registering creators",
          "[SP-TENSORACLE-078][SP-TENSORACLE-002][SP-TENSORACLE-009]") {
    fixture files; auto path = files.directory / "replaced.so";
    auto replacement = files.directory / "replacement.so";
    fs::copy_file(TENSORACLE_REGISTRATION_PLUGIN, path);
    linked_fixture plugin(path, "replaced-module");
    fs::copy_file(TENSORACLE_OTHER_REGISTRATION_PLUGIN, replacement);
    REQUIRE(bytes(path) != bytes(replacement));
    fs::rename(replacement, path);
    auto* result = reinterpret_cast<tensoracle_plugin_libraries*>(1);
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_register_embedded_plugins(plugin.anchor, plugin.creators.data(), plugin.creators.size(),
                                                       plugin.plugin_namespace.c_str(), &result, &diagnostic);
    tensoracle::plugin_libraries result_owner(result);
    tensoracle::error_handle error(diagnostic);
    REQUIRE(status == TENSORACLE_PLUGIN_LIBRARY_ERROR);
    REQUIRE(result == nullptr);
    REQUIRE(getPluginRegistry()->getCreator("RegistrationFirst", "1", plugin.plugin_namespace.c_str()) == nullptr);
    const char* message = nullptr;
    REQUIRE(tensoracle_error_message(error.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    for (const auto* element : {"module", "file", "restart"})
        REQUIRE(std::string(message).find(element) != std::string::npos);
}
TEST_CASE("Static objects in anonymous BSS still identify their containing module",
          "[SP-TENSORACLE-077][SP-TENSORACLE-078]") {
    fixture files; auto path = files.directory / "bss.so";
    fs::copy_file(TENSORACLE_REGISTRATION_PLUGIN, path);
    linked_fixture plugin(path, "anonymous-bss");
    auto address = reinterpret_cast<void*(*)()>(plugin.symbol("fixture_bss"))();
    bool anonymous = false;
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
        uintptr_t start, end; unsigned long long offset, inode; char permissions[5]; unsigned major, minor;
        if (std::sscanf(line.c_str(), "%lx-%lx %4s %llx %x:%x %llu", &start, &end, permissions, &offset, &major, &minor, &inode) == 7
            && reinterpret_cast<uintptr_t>(address) >= start && reinterpret_cast<uintptr_t>(address) < end)
            anonymous = inode == 0;
    }
    REQUIRE(anonymous);
    auto partial = tensoracle::register_embedded_plugins(address, {plugin.creators[0]}, plugin.plugin_namespace.c_str());
    auto result = tensoracle::register_embedded_plugins(address, plugin.creators, plugin.plugin_namespace.c_str());
    REQUIRE(result.digests() == partial.digests());
    auto data = bytes(path);
    REQUIRE(result.digests() == std::vector<std::string>{tensoracle::detail::sha256(data.data(), data.size())});
    REQUIRE(tensoracle::register_embedded_plugins(address, plugin.creators, plugin.plugin_namespace.c_str()).digests() == result.digests());
}
TEST_CASE("Embedded creators register in process and identify the complete executable",
          "[SP-TENSORACLE-077][SP-TENSORACLE-078]") {
    static embedded_creator creator, conflict;
    nvinfer1::IPluginCreatorInterface* interface = &creator;
    std::vector<void*> creators{interface};
    auto data = bytes("/proc/self/exe");
    auto expected = tensoracle::detail::sha256(data.data(), data.size());
    auto result = tensoracle::register_embedded_plugins(&creator, creators, creator.getPluginNamespace());
    REQUIRE(result.digests() == std::vector<std::string>{expected});
    REQUIRE(getPluginRegistry()->getCreator(creator.getPluginName(), "1", creator.getPluginNamespace()) == interface);
    REQUIRE(tensoracle::register_embedded_plugins(&creator, creators, creator.getPluginNamespace()).digests() == result.digests());
    creators[0] = static_cast<nvinfer1::IPluginCreatorInterface*>(&conflict);
    try {
        tensoracle::register_embedded_plugins(&conflict, creators, conflict.getPluginNamespace());
        FAIL("conflicting embedded creator was accepted");
    } catch (const tensoracle::error& failure) {
        REQUIRE(failure.status() == TENSORACLE_PLUGIN_LIBRARY_ERROR);
    }
}
TEST_CASE("Embedded registration clears its result when inputs cannot identify a module",
          "[SP-TENSORACLE-077][SP-TENSORACLE-056]") {
    auto* result = reinterpret_cast<tensoracle_plugin_libraries*>(1);
    REQUIRE(tensoracle_register_embedded_plugins(nullptr, nullptr, 1, "", &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(result == nullptr);
    void* missing[] = {nullptr};
    REQUIRE(tensoracle_register_embedded_plugins(&result, missing, 1, "", &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(result == nullptr);
    static embedded_creator creator;
    void* creators[] = {static_cast<nvinfer1::IPluginCreatorInterface*>(&creator)};
    REQUIRE(tensoracle_register_embedded_plugins(reinterpret_cast<void*>(1), creators, 1, "", &result, nullptr) == TENSORACLE_PLUGIN_LIBRARY_ERROR);
    REQUIRE(result == nullptr);
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
