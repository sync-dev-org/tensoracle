#pragma once
#include "structure.hpp"
#include <cstring>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <tensoracle/cubin.hpp>

struct tensoracle_bundle_result {
    std::string manifest;
    std::string digest;
};
namespace tensoracle::detail {
using json = nlohmann::json;
namespace fs = std::filesystem;
[[noreturn]] inline void reject(tensoracle_status code, const std::string &message) {
    throw tensoracle::error(code, message.c_str());
}
inline void require(bool condition, const std::string &message,
                    tensoracle_status code = TENSORACLE_BUNDLE_ERROR) {
    if (!condition)
        reject(code, message);
}
inline bool present(const char *value) { return value && *value; }
std::string read_file(const fs::path &path, tensoracle_status code = TENSORACLE_BUNDLE_ERROR);
void write_file(const fs::path &path, const std::string &bytes);
uint32_t architecture(const std::string &data);
std::string digest(json value);
json parse_manifest(const std::string &text);
tensoracle_bundle_result result_for(const json &value);
tensoracle_bundle_result inspect(const std::string &text,
                                 const std::function<std::string(const std::string &)> &read,
                                 const std::string *plugin);
void validate_entry(const json &entry);
} // namespace tensoracle::detail
