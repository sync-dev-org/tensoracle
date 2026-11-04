#include <tensoracle/plugins.h>
#include "error.hpp"
#include "native_exports.hpp"
#include "sha256.hpp"
#include <NvInferRuntime.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <vector>

struct tensoracle_plugin_libraries { std::vector<std::string> digests; };
namespace {
struct registered_libraries {
    std::mutex mutex;
    std::map<std::filesystem::path, std::string> paths;
    std::map<std::string, nvinfer1::IPluginRegistry::PluginLibraryHandle> digests;
};
registered_libraries& registered() { static registered_libraries state; return state; }
[[noreturn]] void plugin_error(const std::string& message) {
    throw tensoracle::error(TENSORACLE_PLUGIN_LIBRARY_ERROR, message.c_str());
}
std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) plugin_error("plugin library is missing or unreadable: " + path.string());
    std::string data{std::istreambuf_iterator<char>(stream), {}};
    if (stream.bad()) plugin_error("could not read plugin library: " + path.string());
    return data;
}
}

extern "C" tensoracle_status tensoracle_register_libraries(const char* const* paths, size_t count,
    tensoracle_plugin_libraries** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        if (!output || (count && !paths)) throw std::invalid_argument("plugin result and path array are required");
        for (size_t i = 0; i < count; ++i)
            if (!paths[i] || !*paths[i]) throw std::invalid_argument("plugin library path must not be empty");
        auto result = std::make_unique<tensoracle_plugin_libraries>();
        auto& state = registered();
        std::lock_guard<std::mutex> lock(state.mutex);
        for (size_t i = 0; i < count; ++i) {
            std::error_code reason;
            auto path = std::filesystem::canonical(paths[i], reason);
            if (reason) plugin_error("plugin library is missing or unreadable: " + std::string(paths[i]));
            auto data = read(path);
            auto digest = tensoracle::detail::sha256(data.data(), data.size());
            auto found = state.paths.find(path);
            if (found != state.paths.end()) {
                if (found->second != digest) plugin_error("loaded plugin library changed: " + path.string());
                result->digests.push_back(digest);
                continue;
            }
            try {
                if (!tensoracle::detail::native_exports(data).count("getCreators"))
                    plugin_error("plugin library does not export getCreators: " + path.string());
            } catch (const tensoracle::error& failure) {
                plugin_error(std::string(failure.what()) + ": " + path.string());
            }
            if (!state.digests.count(digest)) {
                auto* registry = getPluginRegistry();
                if (!registry) plugin_error("TensorRT plugin registry is not available");
                auto handle = registry->loadLibrary(path.c_str());
                if (!handle) plugin_error("TensorRT could not register plugin library: " + path.string());
                auto after = read(path);
                if (tensoracle::detail::sha256(after.data(), after.size()) != digest)
                    plugin_error("plugin library changed while loading: " + path.string());
                state.digests.emplace(digest, handle);
            }
            state.paths.emplace(path, digest);
            result->digests.push_back(digest);
        }
        *output = result.release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_count(
    const tensoracle_plugin_libraries* libraries, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        if (!libraries || !count) throw std::invalid_argument("plugin libraries and count output are required");
        *count = libraries->digests.size();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_digest(
    const tensoracle_plugin_libraries* libraries, size_t index, const char** digest, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (digest) *digest = nullptr;
        if (!libraries || !digest || index >= libraries->digests.size())
            throw std::invalid_argument("plugin digest index is out of range or output is missing");
        *digest = libraries->digests[index].c_str();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_destroy(tensoracle_plugin_libraries* libraries) noexcept {
    delete libraries;
    return TENSORACLE_SUCCESS;
}
