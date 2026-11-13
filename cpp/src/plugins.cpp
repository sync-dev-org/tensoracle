#include <tensoracle/plugins.h>
#include "error.hpp"
#include "native_exports.hpp"
#include "sha256.hpp"
#include "plugins_internal.hpp"
#include <NvInferRuntime.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <vector>
#if defined(__linux__)
#include <array>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace {
struct registered_libraries {
    std::mutex mutex;
    // Byte identities seen by either registration API, including partial success.
    std::map<std::filesystem::path, std::string> paths;
    // Only completed loadLibrary calls permit skipping whole-library loading.
    std::map<std::string, nvinfer1::IPluginRegistry::PluginLibraryHandle> digests;
    // Creators successfully registered directly, independently of library loads.
    std::set<std::pair<nvinfer1::IPluginCreatorInterface*, std::string>> creators;
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
#if defined(__linux__)
struct module_file {
    std::filesystem::path path;
    std::string digest;
};
[[noreturn]] void module_file_error(const std::filesystem::path& path = {}) {
    plugin_error("cannot match the loaded plugin module to its current file; restart the process: " + path.string());
}
// Resolve the owning module even when address itself is in anonymous BSS, then
// verify a file-backed segment and hash its file through the same open fd.
module_file identify_module_file(const void* address) {
    struct search {
        uintptr_t address;
        uintptr_t file_address = 0;
        uint64_t file_offset = 0;
        bool executable = false;
    } found{reinterpret_cast<uintptr_t>(address)};
    dl_iterate_phdr([](dl_phdr_info* module, size_t, void* pointer) {
        auto& value = *static_cast<search*>(pointer);
        bool contains = false;
        for (size_t i = 0; i < module->dlpi_phnum; ++i) {
            const auto& segment = module->dlpi_phdr[i];
            auto start = module->dlpi_addr + segment.p_vaddr;
            if (segment.p_type == PT_LOAD && value.address >= start && value.address - start < segment.p_memsz)
                contains = true;
        }
        if (!contains) return 0;
        value.executable = !module->dlpi_name || !*module->dlpi_name;
        for (size_t i = 0; i < module->dlpi_phnum; ++i) {
            const auto& segment = module->dlpi_phdr[i];
            if (segment.p_type == PT_LOAD && segment.p_filesz) {
                value.file_address = module->dlpi_addr + segment.p_vaddr;
                value.file_offset = segment.p_offset;
                break;
            }
        }
        return 1;
    }, &found);
    if (!found.file_address) module_file_error();
    std::ifstream maps("/proc/self/maps");
    for (std::string line; std::getline(maps, line);) {
        unsigned long long start, end, offset, inode;
        unsigned device_major, device_minor;
        char permissions[5]; int pathname = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &start, &end,
                        permissions, &offset, &device_major, &device_minor, &inode, &pathname) != 7
            || found.file_address < start || found.file_address >= end)
            continue;
        if (!inode || offset + (found.file_address - start) != found.file_offset)
            module_file_error();
        auto path = found.executable ? std::filesystem::path("/proc/self/exe")
                                    : std::filesystem::path(line.substr(static_cast<size_t>(pathname)));
        if (!path.is_absolute()) module_file_error(path);
        struct descriptor {
            int value;
            ~descriptor() { if (value >= 0) ::close(value); }
        } file{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        struct stat status{};
        if (file.value < 0 || ::fstat(file.value, &status)
            || static_cast<uint64_t>(status.st_ino) != inode
            || major(status.st_dev) != device_major || minor(status.st_dev) != device_minor)
            module_file_error(path);
        tensoracle::detail::sha256_state hash;
        std::array<char, 64 * 1024> buffer{};
        for (;;) {
            auto count = ::read(file.value, buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) module_file_error(path);
            if (!count) break;
            hash.update(buffer.data(), static_cast<size_t>(count));
        }
        return {std::move(path), hash.finish()};
    }
    module_file_error();
}
#endif
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
            result->identities.emplace_back(path, digest);
            auto found = state.paths.find(path);
            if (found != state.paths.end() && found->second != digest)
                plugin_error("loaded plugin library changed: " + path.string());
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
        }
        *output = result.release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_register_embedded_plugins(const void* module_address,
    void* const* creators, size_t count, const char* plugin_namespace,
    tensoracle_plugin_libraries** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&]() -> tensoracle_status {
        if (output) *output = nullptr;
        if (!output || !module_address || !count || !creators || !plugin_namespace)
            throw std::invalid_argument("embedded registration requires module address, creators, namespace and result");
        for (size_t i = 0; i < count; ++i)
            if (!creators[i]) throw std::invalid_argument("embedded plugin creator must not be null");
#if defined(__linux__)
        auto& state = registered();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto identity = identify_module_file(module_address);
        const auto& path = identity.path;
        const auto& digest = identity.digest;
        auto found = state.paths.find(path);
        if (found != state.paths.end() && found->second != digest)
            plugin_error("loaded plugin containing file changed: " + path.string());
        auto result = std::make_unique<tensoracle_plugin_libraries>();
        result->identities.emplace_back(path, digest);
        auto* registry = getPluginRegistry();
        if (!registry) plugin_error("TensorRT plugin registry is not available");
        for (size_t i = 0; i < count; ++i) {
            auto* creator = static_cast<nvinfer1::IPluginCreatorInterface*>(creators[i]);
            auto key = std::make_pair(creator, std::string(plugin_namespace));
            if (state.creators.count(key)) continue;
            if (!registry->registerCreator(*creator, plugin_namespace))
                plugin_error("TensorRT could not register embedded plugin creator in namespace: " + std::string(plugin_namespace));
            state.creators.insert(std::move(key));
            state.paths.emplace(path, digest);
        }
        if (identify_module_file(module_address).digest != digest)
            plugin_error("plugin containing file changed while registering: " + path.string());
        *output = result.release();
        return TENSORACLE_SUCCESS;
#else
        plugin_error("embedded plugin registration is unsupported on Windows and other non-Linux platforms");
#endif
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_count(
    const tensoracle_plugin_libraries* libraries, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        if (!libraries || !count) throw std::invalid_argument("plugin libraries and count output are required");
        *count = libraries->identities.size();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_digest(
    const tensoracle_plugin_libraries* libraries, size_t index, const char** digest, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (digest) *digest = nullptr;
        if (!libraries || !digest || index >= libraries->identities.size())
            throw std::invalid_argument("plugin digest index is out of range or output is missing");
        *digest = libraries->identities[index].second.c_str();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_plugin_libraries_destroy(tensoracle_plugin_libraries* libraries) noexcept {
    delete libraries;
    return TENSORACLE_SUCCESS;
}
