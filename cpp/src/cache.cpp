#include "cache.hpp"
#include "sha256.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tensoracle::detail {
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
constexpr char magic[] = "TENSORACLE\2";
std::string canonical(const json& value) { return value.dump(-1, ' ', true); }
std::string digest(const std::string& bytes) { return sha256(bytes.data(), bytes.size()); }
[[noreturn]] void system_failure(const fs::path& path) {
    throw std::system_error(errno, std::generic_category(), path.string());
}
struct descriptor {
    int value;
    ~descriptor() { if (value >= 0) ::close(value); }
};
struct temporary_file {
    fs::path path;
    ~temporary_file() { std::error_code error; fs::remove(path, error); }
};
void write_all(int file, const void* input, size_t size) {
    const auto* bytes = static_cast<const char*>(input);
    while (size) {
        auto written = ::write(file, bytes, std::min(size, size_t(1024 * 1024)));
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) system_failure("writing cache file");
        bytes += written;
        size -= static_cast<size_t>(written);
    }
}
void atomic_write(const fs::path& path, const std::function<void(int)>& write) {
    fs::create_directories(path.parent_path());
    auto pattern = (path.parent_path() / ("." + path.filename().string() + ".XXXXXX")).string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    descriptor file{mkstemp(name.data())};
    if (file.value < 0) system_failure(path);
    temporary_file temporary{fs::path(name.data())};
    write(file.value);
    if (fsync(file.value)) system_failure(temporary.path);
    if (::close(file.value)) { file.value = -1; system_failure(temporary.path); }
    file.value = -1;
    fs::rename(temporary.path, path);
}
struct stat identity(const fs::path& path) {
    struct stat value{};
    if (::stat(path.c_str(), &value)) system_failure(path);
    return value;
}
int64_t mtime_ns(const struct stat& value) {
    return int64_t(value.st_mtim.tv_sec) * 1000000000 + value.st_mtim.tv_nsec;
}
bool unchanged(const struct stat& a, const struct stat& b) {
    return a.st_size == b.st_size && mtime_ns(a) == mtime_ns(b) && a.st_ino == b.st_ino;
}
bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
bool hex_digest(const json& value) {
    if (!value.is_string()) return false;
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::string file_sha256(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) system_failure(path);
    std::array<char, 1024 * 1024> buffer{};
    sha256_state hash;
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        hash.update(buffer.data(), static_cast<size_t>(stream.gcount()));
    }
    if (!stream.eof()) throw std::runtime_error("could not read weight file: " + path.string());
    return hash.finish();
}
}

engine_file_lock::engine_file_lock(const fs::path& path) : path_(path) {
    for (;;) {
        // Existing engines can be reused from a read-only cache. flock does not
        // require a writable descriptor; rebuilding publishes by rename too.
        descriptor file{::open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (file.value < 0) {
            if (errno != ENOENT) system_failure(path);
            file.value = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (file.value < 0) {
                if (errno == EEXIST) continue;
                system_failure(path);
            }
        }
        while (flock(file.value, LOCK_EX)) {
            if (errno != EINTR) system_failure(path);
        }
        struct stat locked{}, current{};
        if (fstat(file.value, &locked)) system_failure(path);
        if (::stat(path.c_str(), &current)) {
            if (errno == ENOENT) continue;
            system_failure(path);
        }
        // A previous owner may have published a new inode or removed its failed
        // placeholder. Waiting on that obsolete descriptor grants no ownership
        // of the current pathname. Close it and lock the current file instead.
        if (!same_file(locked, current)) continue;
        empty_ = locked.st_size == 0;
        descriptor_ = file.value;
        file.value = -1;
        return;
    }
}
engine_file_lock::~engine_file_lock() {
    if (descriptor_ < 0) return;
    if (empty_) {
        struct stat locked{}, current{};
        // Keep the lock through removal. A waiting descriptor will retry after
        // unlink, and a successful atomic publication has a different inode.
        if (!fstat(descriptor_, &locked) && !locked.st_size
            && !::stat(path_.c_str(), &current) && same_file(locked, current))
            ::unlink(path_.c_str());
    }
    ::close(descriptor_);
}

fs::path cache_directory() {
    if (auto* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) return fs::path(xdg) / "tensoracle";
#if defined(_WIN32)
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    if (!home || !*home) throw std::runtime_error("home directory is not available for tensoracle cache");
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
    auto before = identity(path);
    json attributes{{"path", path.string()}, {"size", before.st_size},
                    {"mtime_ns", mtime_ns(before)}, {"inode", before.st_ino}};
    auto record_path = cache_directory() / ("weight-" + digest(path.string()) + ".json");
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
        if (unchanged(before, identity(path))) {
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
            throw std::invalid_argument("plugin library changed while determining engine conditions: " + request.plugins[i].string());
    auto all_plugins = plugins;
    for (const auto& [path, registered] : request.embedded_plugins) {
        if (file_sha256(path) != registered)
            throw std::invalid_argument("embedded plugin containing file changed while determining engine conditions: " + path.string());
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
        auto count = ::read(fd, bytes, std::min(size, size_t(1024 * 1024)));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) system_failure(path);
        if (!count) throw corrupt_engine_error("truncated TensorRT engine file: " + path.string());
        bytes += count; size -= static_cast<size_t>(count);
    }
}
}
engine_file_header read_engine_header(int fd, const fs::path& path, uint64_t size) {
    if (size < 18) throw corrupt_engine_error("incomplete TensorRT engine file: " + path.string());
    char prefix[18];
    read_exact(fd, prefix, sizeof(prefix), path);
    if (std::memcmp(prefix, magic, 10)) throw corrupt_engine_error("invalid TensorRT engine magic: " + path.string());
    uint64_t header_size = 0;
    for (size_t i = 10; i < 18; ++i) header_size = (header_size << 8U) | static_cast<uint8_t>(prefix[i]);
    if (!header_size || header_size >= size - 18) throw corrupt_engine_error("invalid TensorRT engine header length: " + path.string());
    std::string bytes(static_cast<size_t>(header_size), 0);
    read_exact(fd, bytes.data(), bytes.size(), path);
    engine_file_header result{{}, 18 + header_size, size - 18 - header_size};
    try {
        result.header = json::parse(bytes);
        if (!result.header.is_object() || !result.header.at("conditions").is_object()
            || !hex_digest(result.header.at("plan_sha256")))
            throw corrupt_engine_error("invalid TensorRT engine header: " + path.string());
        if (result.header.contains("shape_tensor_outputs")) {
            const auto& outputs = result.header["shape_tensor_outputs"];
            if (!outputs.is_array() || std::any_of(outputs.begin(), outputs.end(), [](const auto& name) { return !name.is_string(); }))
                throw corrupt_engine_error("invalid shape tensor outputs: " + path.string());
        }
    } catch (const json::exception&) { throw corrupt_engine_error("invalid TensorRT engine JSON: " + path.string()); }
    return result;
}
stored_plan read_engine_file(const fs::path& path, const json& expected) {
    descriptor file{::open(path.c_str(), O_RDONLY)};
    if (file.value < 0) system_failure(path);
    struct stat info{};
    if (fstat(file.value, &info)) system_failure(path);
    if (info.st_size < 0) throw corrupt_engine_error("invalid TensorRT engine size: " + path.string());
    auto metadata = read_engine_header(file.value, path, static_cast<uint64_t>(info.st_size));
    if (metadata.header["conditions"] != expected) throw engine_condition_error("TensorRT engine conditions differ: " + path.string());
    stored_plan result{std::move(metadata.header), std::vector<uint8_t>(static_cast<size_t>(metadata.plan_size))};
    read_exact(file.value, result.plan.data(), result.plan.size(), path);
    if (result.header["plan_sha256"].get<std::string>() != sha256(result.plan.data(), result.plan.size()))
        throw corrupt_engine_error("TensorRT engine checksum differs: " + path.string());
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
    std::error_code error;
    fs::last_write_time(path, fs::file_time_type::clock::now(), error);
}
}
