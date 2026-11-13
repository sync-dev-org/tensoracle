#pragma once
#include <nlohmann/json.hpp>
#include <tensoracle/engine.hpp>
#include <filesystem>
#include <functional>
#include <optional>

namespace tensoracle::detail {
struct corrupt_engine_error : tensoracle::error {
    explicit corrupt_engine_error(const std::string& message) : error(TENSORACLE_CORRUPT_ENGINE, message.c_str()) {}
};
struct engine_condition_error : tensoracle::error {
    explicit engine_condition_error(const std::string& message) : error(TENSORACLE_ENGINE_CONDITION_MISMATCH, message.c_str()) {}
};
struct engine_io_error : tensoracle::error {
    explicit engine_io_error(const std::string& message) : error(TENSORACLE_ENGINE_IO_MISMATCH, message.c_str()) {}
};
struct input_range {
    std::string name;
    std::vector<int64_t> minimum, optimum, maximum;
};
struct tensor_spec {
    std::string name;
    tensoracle_dtype dtype;
    std::vector<int64_t> shape;
    bool input;
    std::optional<int32_t> location, format;
};
struct cache_request {
    std::string name, graph_version;
    std::vector<std::filesystem::path> weights, plugins;
    std::vector<input_range> inputs;
    std::vector<tensor_spec> io;
    int optimization_level = 3;
    std::optional<uint64_t> workspace_limit;
    bool force_rebuild = false;
    std::vector<std::pair<std::filesystem::path, std::string>> embedded_plugins;
};
struct stored_plan {
    nlohmann::json header;
    std::vector<uint8_t> plan;
};
struct engine_file_header {
    nlohmann::json header;
    uint64_t plan_offset;
    uint64_t plan_size;
};
struct built_plan {
    std::vector<uint8_t> plan;
    std::vector<std::string> shape_tensor_outputs;
};
struct cached_engine {
    engine_handle engine;
    std::filesystem::path path;
};

// Locks the current inode, retrying when a publisher replaces it while waiting.
// An empty file with no remaining lock owner can be taken over after a failure.
class engine_file_lock {
public:
    explicit engine_file_lock(const std::filesystem::path& path);
    ~engine_file_lock();
    engine_file_lock(const engine_file_lock&) = delete;
    engine_file_lock& operator=(const engine_file_lock&) = delete;
    bool empty() const noexcept { return empty_; }
private:
    std::filesystem::path path_;
    int descriptor_ = -1;
    bool empty_ = false;
};

std::filesystem::path cache_directory();
std::string weight_sha256(const std::filesystem::path& path, bool force_rebuild);
std::optional<nlohmann::json> read_weight_record(const std::filesystem::path& path);
nlohmann::json validate_cache_request(const cache_request& request);
nlohmann::json make_conditions(const cache_request& request, const std::vector<std::string>& plugin_digests,
    const nlohmann::json& gpu, const std::string& tensorrt_version,
    const std::string& os, const std::string& architecture);
std::filesystem::path cache_path(const nlohmann::json& conditions);
stored_plan read_engine_file(const std::filesystem::path& path, const nlohmann::json& expected);
// Reads exactly the prefix and header from a descriptor positioned at zero.
// Does not read/check the plan; the same schema validation is used by reuse.
engine_file_header read_engine_header(int descriptor, const std::filesystem::path& path,
                                     uint64_t file_size);
void write_engine_file(const std::filesystem::path& path, const nlohmann::json& conditions,
    const std::vector<uint8_t>& plan, std::optional<uint64_t> workspace_limit,
    const std::vector<std::string>& shape_tensor_outputs);
// mtime records opening, not ongoing inference. Files predating recording and
// files whose mtime could not be written may not represent the last real load.
// Only call after deserialization and IO checks succeed; never recreate files.
void touch_engine(const std::filesystem::path& path) noexcept;
nlohmann::json gpu_identity();
cached_engine open_cached_engine(const cache_request& request, const std::function<built_plan()>& build);
}
