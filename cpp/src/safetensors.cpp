#include "safetensors.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace tensoracle::detail {
using json = nlohmann::json;
mapped_weights_file::~mapped_weights_file() { if (mapping) munmap(mapping, size); }
namespace {
[[noreturn]] void invalid(const std::filesystem::path& path, const std::string& reason) {
    throw std::invalid_argument("invalid safetensors file " + path.string() + ": " + reason);
}
uint64_t integer(const json& value, const std::filesystem::path& path) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (value.is_number_integer() && value.get<int64_t>() >= 0) return uint64_t(value.get<int64_t>());
    invalid(path, "shapes and offsets require nonnegative integers");
}
std::pair<tensoracle_dtype, size_t> dtype(const std::string& name, const std::filesystem::path& path) {
    static const std::map<std::string, std::pair<tensoracle_dtype, size_t>> types{
        {"BOOL", {TENSORACLE_BOOL, 1}}, {"I8", {TENSORACLE_INT8, 1}}, {"U8", {TENSORACLE_UINT8, 1}},
        {"I16", {TENSORACLE_INT16, 2}}, {"U16", {TENSORACLE_UINT16, 2}},
        {"I32", {TENSORACLE_INT32, 4}}, {"U32", {TENSORACLE_UINT32, 4}},
        {"I64", {TENSORACLE_INT64, 8}}, {"U64", {TENSORACLE_UINT64, 8}},
        {"F16", {TENSORACLE_FLOAT16, 2}}, {"BF16", {TENSORACLE_BFLOAT16, 2}},
        {"F32", {TENSORACLE_FLOAT32, 4}}, {"F64", {TENSORACLE_FLOAT64, 8}}, {"F8_E4M3", {TENSORACLE_FLOAT8, 1}}};
    auto found = types.find(name);
    if (found == types.end()) invalid(path, "unsupported tensor dtype " + name);
    return found->second;
}
}
std::unique_ptr<tensoracle_weights> load_safetensors(const std::vector<std::filesystem::path>& paths) {
    auto weights = std::make_unique<tensoracle_weights>();
    for (const auto& path : paths) {
        struct descriptor { int fd; ~descriptor() { if (fd >= 0) close(fd); } } file{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
        if (file.fd < 0) throw std::system_error(errno, std::generic_category(), path.string());
        struct stat info{};
        if (fstat(file.fd, &info)) throw std::system_error(errno, std::generic_category(), path.string());
        if (!S_ISREG(info.st_mode) || info.st_size < 8) invalid(path, "missing length or regular file");
        auto storage = std::make_unique<mapped_weights_file>();
        storage->size = static_cast<size_t>(info.st_size);
        void* mapping = mmap(nullptr, storage->size, PROT_READ, MAP_PRIVATE, file.fd, 0);
        if (mapping == MAP_FAILED) throw std::system_error(errno, std::generic_category(), path.string());
        storage->mapping = mapping;
        auto* bytes = static_cast<const unsigned char*>(mapping);
        uint64_t header_size = 0;
        for (int i = 0; i < 8; ++i) header_size |= uint64_t(bytes[i]) << (8 * i);
        // Keep the reference safetensors reader's metadata limit.
        if (header_size > 100000000) invalid(path, "header exceeds 100000000 bytes");
        if (!header_size || header_size > storage->size - 8 || bytes[8] != '{') invalid(path, "invalid header length or object");
        if (std::memchr(bytes + 8, 0, static_cast<size_t>(header_size))) invalid(path, "NUL is not valid JSON header text");
        json header;
        std::vector<std::set<std::string>> keys;
        try {
            header = json::parse(bytes + 8, bytes + 8 + header_size, [&](int, json::parse_event_t event, json& value) {
                if (event == json::parse_event_t::object_start) keys.emplace_back();
                if (event == json::parse_event_t::key && !keys.back().insert(value.get<std::string>()).second)
                    invalid(path, "duplicate JSON key " + value.get<std::string>());
                if (event == json::parse_event_t::object_end) keys.pop_back();
                return true;
            });
        } catch (const json::exception& error) { invalid(path, error.what()); }
        if (!header.is_object()) invalid(path, "header must be an object");
        const uint64_t data_size = storage->size - 8 - header_size;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (auto item = header.begin(); item != header.end(); ++item) {
            const auto& name = item.key(); const auto& value = item.value();
            if (name == "__metadata__") {
                if (!value.is_object()) invalid(path, "metadata must be a string map");
                for (const auto& entry : value) if (!entry.is_string()) invalid(path, "metadata values must be strings");
                continue;
            }
            if (name.find('\0') != std::string::npos) invalid(path, "tensor name contains NUL");
            auto duplicate = weights->names.find(name);
            if (duplicate != weights->names.end()) invalid(path, "duplicate tensor " + name + " also declared in " + weights->tensors[duplicate->second].source.string());
            if (!value.is_object() || value.size() != 3 || !value.contains("dtype") || !value["dtype"].is_string()
                || !value.contains("shape") || !value["shape"].is_array() || !value.contains("data_offsets")
                || !value["data_offsets"].is_array() || value["data_offsets"].size() != 2) invalid(path, "invalid tensor descriptor " + name);
            auto type = dtype(value["dtype"].get<std::string>(), path);
            weight_record tensor{name, path, type.first, {}, nullptr, 1, 0, {}};
            bool empty = false;
            for (const auto& dimension : value["shape"]) {
                auto d = integer(dimension, path);
                if (d > uint64_t(INT64_MAX)) invalid(path, "tensor dimension exceeds int64 range");
                tensor.shape.push_back(static_cast<int64_t>(d)); empty = empty || d == 0;
            }
            if (empty) tensor.count = 0;
            else for (auto d : tensor.shape) {
                if (uint64_t(d) > uint64_t(INT64_MAX) / tensor.count) invalid(path, "tensor element count overflow");
                tensor.count *= uint64_t(d);
            }
            if (tensor.count > std::numeric_limits<uint64_t>::max() / type.second) invalid(path, "tensor byte size overflow");
            tensor.bytes = tensor.count * type.second;
            auto first = integer(value["data_offsets"][0], path), last = integer(value["data_offsets"][1], path);
            if (last < first || last > data_size || last - first != tensor.bytes) invalid(path, "invalid data offsets for " + name);
            tensor.mapped_data = bytes + 8 + header_size + first;
            if (tensor.bytes && reinterpret_cast<uintptr_t>(tensor.mapped_data) % type.second) {
                tensor.aligned.resize(static_cast<size_t>((tensor.bytes + 7) / 8));
                std::memcpy(tensor.aligned.data(), tensor.mapped_data, static_cast<size_t>(tensor.bytes));
            }
            weights->names.emplace(name, weights->tensors.size());
            weights->tensors.push_back(std::move(tensor)); ranges.emplace_back(first, last);
        }
        std::sort(ranges.begin(), ranges.end()); uint64_t cursor = 0;
        for (const auto& range : ranges) {
            if (range.first != cursor) invalid(path, "tensor data has overlaps or holes");
            cursor = range.second;
        }
        if (cursor != data_size) invalid(path, "unindexed trailing tensor data");
        weights->files.push_back(std::move(storage));
    }
    return weights;
}
}
