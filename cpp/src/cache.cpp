#include "cache.hpp"
#include "sha256.hpp"
#include "file.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>
#include <fcntl.h>
#if !defined(_WIN32)
#include <sys/file.h>
#endif

namespace tensoracle::detail {
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
constexpr char magic[] = "TENSORACLE\2";
#if defined(_WIN32)
fs::path environment_path(const wchar_t* name) {
    auto size = GetEnvironmentVariableW(name, nullptr, 0);
    while (size) {
        std::vector<wchar_t> value(size);
        auto count = GetEnvironmentVariableW(name, value.data(), DWORD(value.size()));
        if (!count) return {};
        if (count < value.size()) return std::wstring(value.data(), count);
        size = count;
    }
    return {};
}
#endif
std::string canonical(const json& value) { return value.dump(-1, ' ', true); }
std::string digest(const std::string& bytes) { return sha256(bytes.data(), bytes.size()); }
[[noreturn]] void system_failure(const fs::path& path) {
    throw std::system_error(errno, std::generic_category(), path_utf8(path));
}
using descriptor = file_descriptor;
struct temporary_file {
    fs::path path;
    ~temporary_file() { std::error_code error; fs::remove(path, error); }
};
void write_all(int file, const void* input, size_t size) {
    const auto* bytes = static_cast<const char*>(input);
    while (size) {
        auto written = write_file_bytes(file, bytes, size);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) system_failure("writing cache file");
        bytes += written;
        size -= static_cast<size_t>(written);
    }
}
void atomic_write(const fs::path& path, const std::function<void(int)>& write) {
    fs::create_directories(path.parent_path());
#if defined(_WIN32)
    static std::atomic<uint64_t> sequence{0};
    fs::path name;
    int fd = -1;
    for (;;) {
        name = path.parent_path() / (L"." + path.filename().native() + L"." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(sequence.fetch_add(1)));
        try { fd = open_file(name, false, true); break; }
        catch (const std::system_error& error) { if (error.code().value() != ERROR_FILE_EXISTS && error.code().value() != ERROR_ALREADY_EXISTS) throw; }
    }
    descriptor file{fd};
    temporary_file temporary{name};
#else
    auto pattern = (path.parent_path() / ("." + path.filename().string() + ".XXXXXX")).string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    descriptor file{mkstemp(name.data())};
    if (file.value < 0) system_failure(path);
    temporary_file temporary{fs::path(name.data())};
#endif
    write(file.value);
#if defined(_WIN32)
    if (!FlushFileBuffers(file_handle(file.value))) windows_file_error(temporary.path);
#else
    if (fsync(file.value)) system_failure(temporary.path);
#endif
    if (close_file(file.value)) { file.value = -1; system_failure(temporary.path); }
    file.value = -1;
#if defined(_WIN32)
    auto source = CreateFileW(temporary.path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (source == INVALID_HANDLE_VALUE) windows_file_error(temporary.path);
    struct rename_owner { HANDLE value; ~rename_owner() { CloseHandle(value); } } rename_handle{source};
    auto destination = fs::absolute(path).wstring();
    std::vector<unsigned char> storage(sizeof(FILE_RENAME_INFO) + destination.size() * sizeof(wchar_t));
    auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    rename->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    rename->RootDirectory = nullptr;
    rename->FileNameLength = DWORD(destination.size() * sizeof(wchar_t));
    std::memcpy(rename->FileName, destination.data(), rename->FileNameLength);
    if (!SetFileInformationByHandle(source, FileRenameInfoEx, rename, DWORD(storage.size()))) windows_file_error(path);
#else
    fs::rename(temporary.path, path);
#endif
}
bool unchanged(const file_status& a, const file_status& b) {
    return a.size == b.size && a.mtime == b.mtime && same_file(a, b);
}
int64_t record_mtime_ns(file_time time) {
    // Version 1 digest records retain the low 64 bits of Unix nanoseconds.
    // Keep that encoding at the record boundary, without signed overflow.
    auto bits = uint64_t(time.first) * 1000000000 + uint64_t(time.second);
    return bits <= uint64_t(INT64_MAX) ? int64_t(bits) : -1 - int64_t(~bits);
}
bool hex_digest(const json& value) {
    if (!value.is_string()) return false;
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::string file_sha256(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) system_failure(path);
    std::array<char, 64 * 1024> buffer{};
    sha256_state hash;
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        hash.update(buffer.data(), static_cast<size_t>(stream.gcount()));
    }
    if (!stream.eof()) throw std::runtime_error("could not read weight file: " + path_utf8(path));
    return hash.finish();
}
}

engine_file_lock::engine_file_lock(const fs::path& path) : path_(path) {
    for (;;) {
        // Reusing an existing engine needs only read access. Publication replaces
        // the pathname while this handle keeps ownership of the old file.
        descriptor file{-1};
        try { file.value = open_file(path); }
        catch (const std::system_error& error) {
            if (error.code() != std::errc::no_such_file_or_directory) throw;
            try { file.value = open_file(path, false, true); }
            catch (const std::system_error& created) {
                if (created.code() == std::errc::file_exists) continue;
                throw;
            }
        }
#if defined(_WIN32)
        // Byte-range locks also prohibit reads through other handles. Reserve a
        // byte beyond any supported engine, leaving its header and plan readable.
        OVERLAPPED range{}; range.OffsetHigh = 0x7fffffff;
        if (!LockFileEx(file_handle(file.value), LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &range)) windows_file_error(path);
#else
        while (flock(file.value, LOCK_EX)) {
            if (errno != EINTR) system_failure(path);
        }
#endif
        auto locked = status_of(file.value);
        file_status current{};
        try { current = status_of(path); }
        catch (const std::system_error& error) {
            if (error.code() == std::errc::no_such_file_or_directory) continue;
            throw;
        }
        // Publication replaces the file. A waiter on an obsolete file must
        // acquire the current pathname before deciding whether to build.
        if (!same_file(locked, current)) continue;
        empty_ = locked.size == 0;
        descriptor_ = file.value;
        file.value = -1;
        return;
    }
}
engine_file_lock::~engine_file_lock() {
    if (descriptor_ < 0) return;
    if (empty_) {
        try {
            auto locked = status_of(descriptor_);
            if (!locked.size && same_file(locked, status_of(path_))) {
                std::error_code ignored;
#if defined(_WIN32)
                auto deletion = CreateFileW(path_.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (deletion != INVALID_HANDLE_VALUE) {
                    FILE_DISPOSITION_INFO_EX disposition{FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS};
                    SetFileInformationByHandle(deletion, FileDispositionInfoEx, &disposition, sizeof(disposition));
                    CloseHandle(deletion);
                }
#else
                fs::remove(path_, ignored);
#endif
            }
        } catch (const std::system_error&) {}
    }
    close_file(descriptor_);
}

fs::path cache_directory() {
#if defined(_WIN32)
    auto xdg = environment_path(L"XDG_CACHE_HOME");
    if (!xdg.empty()) return xdg / "tensoracle";
    auto home = environment_path(L"USERPROFILE");
    if (home.empty()) throw std::runtime_error("home directory is not available for tensoracle cache");
#else
    if (auto* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) return fs::path(xdg) / "tensoracle";
    const char* home = std::getenv("HOME");
    if (!home || !*home) throw std::runtime_error("home directory is not available for tensoracle cache");
#endif
    return fs::path(home) / ".cache" / "tensoracle";
}
std::optional<json> read_weight_record(const fs::path& path) {
    try {
        std::ifstream stream(path, std::ios::binary);
        auto record = json::parse(stream);
        auto payload = record;
        auto checksum = payload.at("checksum");
        payload.erase("checksum");
        if (payload.size() == 6 && payload.at("version").is_number_integer() && payload.at("version").get<uint64_t>() == 1
            && payload.at("path").is_string() && payload.at("size").is_number_integer()
            && payload.at("mtime_ns").is_number_integer() && payload.at("inode").is_number_integer()
            && hex_digest(payload.at("sha256")) && checksum.is_string()
            && checksum.get<std::string>() == digest(canonical(payload))) return record;
    } catch (const json::exception&) {
    } catch (const std::ios_base::failure&) {
    }
    return std::nullopt;
}
std::string weight_sha256(const fs::path& input, bool force_rebuild) {
    auto path = fs::absolute(input).lexically_normal();
    auto before = status_of(path);
    json attributes{{"path", path_utf8(path)}, {"size", before.size},
                    {"mtime_ns", record_mtime_ns(before.mtime)}, {"inode", before.inode}};
    auto record_path = cache_directory() / ("weight-" + digest(path_utf8(path)) + ".json");
    if (!force_rebuild) {
        auto record = read_weight_record(record_path);
        if (record) {
            bool matches = true;
            for (auto i = attributes.begin(); matches && i != attributes.end(); ++i)
                matches = record->at(i.key()) == i.value();
            if (matches) return record->at("sha256").get<std::string>();
        }
    }
    auto result = file_sha256(path);
    try {
        if (unchanged(before, status_of(path))) {
            auto record = attributes;
            record["version"] = 1;
            record["sha256"] = result;
            record["checksum"] = digest(canonical(record));
            auto bytes = canonical(record);
            atomic_write(record_path, [&](int fd) { write_all(fd, bytes.data(), bytes.size()); });
        }
    } catch (const std::system_error&) {
        // Digest records are optional; an unwritable directory must not block use.
    }
    return result;
}
json validate_cache_request(const cache_request& request) {
    if (request.name.empty() || request.graph_version.empty() || request.weights.empty())
        throw std::invalid_argument("engine name, graph version and at least one weight file are required");
    if (request.optimization_level < 0 || request.optimization_level > 5)
        throw std::invalid_argument("optimization level must be in 0..5");
    if (request.workspace_limit && !*request.workspace_limit)
        throw std::invalid_argument("workspace limit must be positive when supplied");
    json inputs = json::object();
    for (const auto& input : request.inputs) {
        auto rank = input.minimum.size();
        if (input.name.empty() || inputs.contains(input.name) || !rank || rank > 8
            || input.optimum.size() != rank || input.maximum.size() != rank)
            throw std::invalid_argument("input profiles require unique names and equal nonzero ranks up to eight");
        for (size_t i = 0; i < rank; ++i)
            if (!(0 < input.minimum[i] && input.minimum[i] <= input.optimum[i] && input.optimum[i] <= input.maximum[i]))
                throw std::invalid_argument("input dimensions must be positive and ordered minimum <= optimum <= maximum");
        inputs[input.name] = {{"minimum", input.minimum}, {"optimum", input.optimum}, {"maximum", input.maximum}};
    }
    return inputs;
}
json make_conditions(const cache_request& request, const std::vector<std::string>& plugins,
    const json& gpu, const std::string& tensorrt, const std::string& os, const std::string& architecture) {
    auto inputs = validate_cache_request(request);
    if (request.plugins.size() != plugins.size())
        throw std::invalid_argument("registered plugin digests must match the library paths");
    for (size_t i = 0; i < plugins.size(); ++i)
        if (file_sha256(request.plugins[i]) != plugins[i])
            throw std::invalid_argument("plugin library changed while determining engine conditions: " + path_utf8(request.plugins[i]));
    auto all_plugins = plugins;
    for (const auto& [path, registered] : request.embedded_plugins) {
        if (file_sha256(path) != registered)
            throw std::invalid_argument("embedded plugin containing file changed while determining engine conditions: " + path_utf8(path));
        all_plugins.push_back(registered);
    }
    json weights = json::array();
    for (const auto& path : request.weights) weights.push_back(weight_sha256(path, request.force_rebuild));
    return {{"name", request.name}, {"graph_version", request.graph_version}, {"weights", weights},
        {"plugins", all_plugins}, {"inputs", inputs}, {"builder_settings", {{"optimization_level", request.optimization_level}, {"tf32", false}}},
        {"gpu", gpu}, {"os", os}, {"architecture", architecture}, {"tensorrt_version", tensorrt}};
}
fs::path cache_path(const json& conditions) { return cache_directory() / (digest(canonical(conditions)) + ".engine"); }

namespace {
void read_exact(int fd, void* output, size_t size, const fs::path& path) {
    auto* bytes = static_cast<char*>(output);
    while (size) {
        auto count = read_file_bytes(fd, bytes, size);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) system_failure(path);
        if (!count) throw corrupt_engine_error("truncated TensorRT engine file: " + path_utf8(path));
        bytes += count; size -= static_cast<size_t>(count);
    }
}
}
engine_file_header read_engine_header(int fd, const fs::path& path, uint64_t size) {
    if (size < 18) throw corrupt_engine_error("incomplete TensorRT engine file: " + path_utf8(path));
    char prefix[18];
    read_exact(fd, prefix, sizeof(prefix), path);
    if (std::memcmp(prefix, magic, 10)) throw corrupt_engine_error("invalid TensorRT engine magic: " + path_utf8(path));
    uint64_t header_size = 0;
    for (size_t i = 10; i < 18; ++i) header_size = (header_size << 8U) | static_cast<uint8_t>(prefix[i]);
    if (!header_size || header_size >= size - 18) throw corrupt_engine_error("invalid TensorRT engine header length: " + path_utf8(path));
    std::string bytes(static_cast<size_t>(header_size), 0);
    read_exact(fd, bytes.data(), bytes.size(), path);
    engine_file_header result{{}, 18 + header_size, size - 18 - header_size};
    try {
        result.header = json::parse(bytes);
        if (!result.header.is_object() || !result.header.at("conditions").is_object()
            || !hex_digest(result.header.at("plan_sha256")))
            throw corrupt_engine_error("invalid TensorRT engine header: " + path_utf8(path));
        if (result.header.contains("shape_tensor_outputs")) {
            const auto& outputs = result.header["shape_tensor_outputs"];
            if (!outputs.is_array() || std::any_of(outputs.begin(), outputs.end(), [](const auto& name) { return !name.is_string(); }))
                throw corrupt_engine_error("invalid shape tensor outputs: " + path_utf8(path));
        }
    } catch (const json::exception&) { throw corrupt_engine_error("invalid TensorRT engine JSON: " + path_utf8(path)); }
    return result;
}
stored_plan read_engine_file(const fs::path& path, const json& expected) {
    descriptor file{open_file(path)};
    auto info = status_of(file.value);
    if (info.size < 0) throw corrupt_engine_error("invalid TensorRT engine size: " + path_utf8(path));
    auto metadata = read_engine_header(file.value, path, static_cast<uint64_t>(info.size));
    if (metadata.header["conditions"] != expected) throw engine_condition_error("TensorRT engine conditions differ: " + path_utf8(path));
    stored_plan result{std::move(metadata.header), std::vector<uint8_t>(static_cast<size_t>(metadata.plan_size))};
    read_exact(file.value, result.plan.data(), result.plan.size(), path);
    if (result.header["plan_sha256"].get<std::string>() != sha256(result.plan.data(), result.plan.size()))
        throw corrupt_engine_error("TensorRT engine checksum differs: " + path_utf8(path));
    return result;
}
void write_engine_file(const fs::path& path, const json& conditions, const std::vector<uint8_t>& plan,
    std::optional<uint64_t> workspace, const std::vector<std::string>& outputs) {
    if (plan.empty()) throw std::invalid_argument("cannot store an empty TensorRT plan");
    json header{{"conditions", conditions}, {"plan_sha256", sha256(plan.data(), plan.size())},
                {"workspace_limit", workspace ? json(*workspace) : json(nullptr)}};
    if (!outputs.empty()) header["shape_tensor_outputs"] = outputs;
    auto bytes = canonical(header);
    char length[8];
    for (size_t i = 0; i < 8; ++i) length[i] = static_cast<char>(uint64_t(bytes.size()) >> (56 - 8 * i));
    atomic_write(path, [&](int fd) {
        write_all(fd, magic, 10);
        write_all(fd, length, sizeof(length));
        write_all(fd, bytes.data(), bytes.size());
        write_all(fd, plan.data(), plan.size());
    });
}
void touch_engine(const fs::path& path) noexcept {
#if defined(_WIN32)
    auto handle = CreateFileW(path.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return;
    FILETIME now{}; GetSystemTimePreciseAsFileTime(&now);
    SetFileTime(handle, nullptr, nullptr, &now);
    CloseHandle(handle);
#else
    std::error_code error;
    fs::last_write_time(path, fs::file_time_type::clock::now(), error);
#endif
}
}
