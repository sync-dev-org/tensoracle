#include <tensoracle/plugins.h>
#include "error.hpp"
#include "native_exports.hpp"
#include "sha256.hpp"
#include "file.hpp"
#include "plugins_internal.hpp"
#include <NvInferRuntime.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <vector>
#include <array>
#include <cstring>
#if defined(_WIN32)
#include "engine_internal.hpp"
#include <psapi.h>
#endif
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
using tensoracle::detail::path_utf8;
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
void register_creator(nvinfer1::IPluginRegistry& registry, nvinfer1::IPluginCreatorInterface& creator,
                      const char* plugin_namespace, const std::string& failure) {
    if (!registry.registerCreator(creator, plugin_namespace)) plugin_error(failure);
}
#if defined(_WIN32)
const char* creator_namespace(nvinfer1::IPluginCreatorInterface& creator) {
    const auto info = creator.getInterfaceInfo();
    if (!std::strcmp(info.kind, "PLUGIN CREATOR_V3ONE"))
        return static_cast<nvinfer1::IPluginCreatorV3One&>(creator).getPluginNamespace();
    if (!std::strcmp(info.kind, "PLUGIN CREATOR_V1"))
        return static_cast<nvinfer1::IPluginCreator&>(creator).getPluginNamespace();
    plugin_error("TensorRT could not register plugin creator interface: " + std::string(info.kind));
}
HMODULE load_windows_library(const std::filesystem::path& path, nvinfer1::IPluginRegistry& registry) {
    auto failure = "TensorRT could not register plugin library: " + path_utf8(path);
    struct registration {
        nvinfer1::IPluginRegistry& registry;
        HMODULE module;
        std::vector<nvinfer1::IPluginCreatorInterface*> added;
        ~registration() {
            if (!module) return;
            for (auto* creator : added) registry.deregisterCreator(*creator);
            FreeLibrary(module);
        }
    } loading{registry, LoadLibraryExW(path.c_str(), nullptr, 0), {}};
    if (!loading.module) plugin_error(failure);
    auto set_logger = reinterpret_cast<void(*)(nvinfer1::ILoggerFinder*)>(GetProcAddress(loading.module, "setLoggerFinder"));
    auto get_creators = reinterpret_cast<nvinfer1::IPluginCreatorInterface* const*(*)(int32_t&)>(GetProcAddress(loading.module, "getCreators"));
    if (!set_logger || !get_creators) plugin_error(failure);
    struct logger_finder : nvinfer1::ILoggerFinder {
        nvinfer1::ILogger* findLogger() override { return &tensoracle::detail::process_engine_logger(); }
    };
    static logger_finder finder;
    set_logger(&finder);
    int32_t count = 0;
    auto* creators = get_creators(count);
    if (count < 0 || (count && !creators)) plugin_error(failure);
    loading.added.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        if (!creators[i]) plugin_error(failure);
        auto* plugin_namespace = creator_namespace(*creators[i]);
        if (!plugin_namespace) plugin_error(failure);
        register_creator(registry, *creators[i], plugin_namespace, failure);
        loading.added.push_back(creators[i]);
    }
    auto result = loading.module;
    // Creators and the logger finder remain valid until process exit.
    loading.module = nullptr;
    return result;
}
#endif
std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) plugin_error("plugin library is missing or unreadable: " + path_utf8(path));
    std::string data{std::istreambuf_iterator<char>(stream), {}};
    if (stream.bad()) plugin_error("could not read plugin library: " + path_utf8(path));
    return data;
}
struct module_file {
    std::filesystem::path path;
    std::string digest;
};
[[noreturn]] void module_file_error(const std::filesystem::path& path = {}) {
    plugin_error("cannot match the loaded plugin module to its current file; restart the process: " + path_utf8(path));
}
#if defined(__linux__)
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
#elif defined(_WIN32)
module_file identify_module_file(const void* address) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(address), &module)) module_file_error();
    std::vector<wchar_t> name(32768), mapped(32768);
    auto length = GetModuleFileNameW(module, name.data(), DWORD(name.size()));
    if (!length || length >= name.size()) module_file_error();
    std::filesystem::path path(std::wstring(name.data(), length));
    // Keep writes and replacements excluded while hashing the loaded file.
    auto handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) module_file_error(path);
    struct owner { HANDLE handle; ~owner() { CloseHandle(handle); } } file{handle};
    auto mapped_size = K32GetMappedFileNameW(GetCurrentProcess(), module, mapped.data(), DWORD(mapped.size()));
    if (!mapped_size || mapped_size >= mapped.size()) module_file_error(path);
    // A shared image section can retain another process's hardlink name.
    // Compare the files behind the two names, keeping both handles open.
    auto mapped_path = L"\\\\?\\GLOBALROOT" + std::wstring(mapped.data(), mapped_size);
    auto mapped_handle = CreateFileW(mapped_path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (mapped_handle == INVALID_HANDLE_VALUE) module_file_error(path);
    owner mapped_file{mapped_handle};
    FILE_ID_INFO opened_id{}, mapped_id{};
    if (GetFileInformationByHandleEx(handle, FileIdInfo, &opened_id, sizeof(opened_id))
        && GetFileInformationByHandleEx(mapped_handle, FileIdInfo, &mapped_id, sizeof(mapped_id))) {
        if (opened_id.VolumeSerialNumber != mapped_id.VolumeSerialNumber
            || std::memcmp(opened_id.FileId.Identifier, mapped_id.FileId.Identifier, sizeof(opened_id.FileId.Identifier)))
            module_file_error(path);
    } else {
        // File systems without 128-bit IDs can expose the legacy file index.
        BY_HANDLE_FILE_INFORMATION opened_info{}, mapped_info{};
        if (!GetFileInformationByHandle(handle, &opened_info)
            || !GetFileInformationByHandle(mapped_handle, &mapped_info)
            || opened_info.dwVolumeSerialNumber != mapped_info.dwVolumeSerialNumber
            || opened_info.nFileIndexHigh != mapped_info.nFileIndexHigh
            || opened_info.nFileIndexLow != mapped_info.nFileIndexLow)
            module_file_error(path);
    }
    tensoracle::detail::sha256_state hash;
    std::array<char, 64 * 1024> buffer{};
    for (;;) {
        DWORD count = 0;
        if (!ReadFile(handle, buffer.data(), DWORD(buffer.size()), &count, nullptr)) module_file_error(path);
        if (!count) break;
        hash.update(buffer.data(), count);
    }
    return {std::move(path), hash.finish()};
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
            auto path = std::filesystem::canonical(std::filesystem::u8path(paths[i]), reason);
            if (reason) plugin_error("plugin library is missing or unreadable: " + std::string(paths[i]));
            auto data = read(path);
            auto digest = tensoracle::detail::sha256(data.data(), data.size());
            result->identities.emplace_back(path, digest);
            auto found = state.paths.find(path);
            if (found != state.paths.end() && found->second != digest)
                plugin_error("loaded plugin library changed: " + path_utf8(path));
            try {
                if (!tensoracle::detail::native_exports(data).count("getCreators"))
                    plugin_error("plugin library does not export getCreators: " + path_utf8(path));
            } catch (const tensoracle::error& failure) {
                plugin_error(std::string(failure.what()) + ": " + path_utf8(path));
            }
            if (!state.digests.count(digest)) {
                auto* registry = getPluginRegistry();
                if (!registry) plugin_error("TensorRT plugin registry is not available");
#if defined(_WIN32)
                auto handle = load_windows_library(path, *registry);
#else
                auto handle = registry->loadLibrary(path_utf8(path).c_str());
#endif
                if (!handle) plugin_error("TensorRT could not register plugin library: " + path_utf8(path));
                auto after = read(path);
                if (tensoracle::detail::sha256(after.data(), after.size()) != digest)
                    plugin_error("plugin library changed while loading: " + path_utf8(path));
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
#if defined(__linux__) || defined(_WIN32)
        auto& state = registered();
        std::lock_guard<std::mutex> lock(state.mutex);
        auto identity = identify_module_file(module_address);
        const auto& path = identity.path;
        const auto& digest = identity.digest;
        auto found = state.paths.find(path);
        if (found != state.paths.end() && found->second != digest)
            plugin_error("loaded plugin containing file changed: " + path_utf8(path));
        auto result = std::make_unique<tensoracle_plugin_libraries>();
        result->identities.emplace_back(path, digest);
        auto* registry = getPluginRegistry();
        if (!registry) plugin_error("TensorRT plugin registry is not available");
        for (size_t i = 0; i < count; ++i) {
            auto* creator = static_cast<nvinfer1::IPluginCreatorInterface*>(creators[i]);
            auto key = std::make_pair(creator, std::string(plugin_namespace));
            if (state.creators.count(key)) continue;
            register_creator(*registry, *creator, plugin_namespace,
                "TensorRT could not register embedded plugin creator in namespace: " + std::string(plugin_namespace));
            state.creators.insert(std::move(key));
            state.paths.emplace(path, digest);
        }
        if (identify_module_file(module_address).digest != digest)
            plugin_error("plugin containing file changed while registering: " + path_utf8(path));
        *output = result.release();
        return TENSORACLE_SUCCESS;
#else
        plugin_error("embedded plugin registration is unsupported on this platform");
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
