#include "bundle_internal.hpp"
#include "error.hpp"
#include "native_exports.hpp"
#include "sha256.hpp"
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <miniz.h>
#include <regex>
#include <set>

namespace tensoracle::detail {
std::string read_file(const fs::path &path, tensoracle_status code) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "missing or unreadable file: " + path.string(), code);
    std::string bytes((std::istreambuf_iterator<char>(file)), {});
    require(!file.bad(), "could not read file: " + path.string(), code);
    return bytes;
}
void write_file(const fs::path &path, const std::string &bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    require(bool(file), "could not write file: " + path.string());
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    file.close();
    require(bool(file), "could not finish writing file: " + path.string());
}
static uint64_t number(const std::string &data, uint64_t offset, size_t count,
                       tensoracle_status code = TENSORACLE_PLUGIN_BUNDLE_MISMATCH) {
    require(offset <= data.size() && count <= data.size() - offset,
            "truncated native library or cubin", code);
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i)
        value |= uint64_t(static_cast<unsigned char>(data[static_cast<size_t>(offset) + i]))
                 << (8 * i);
    return value;
}
uint32_t architecture(const std::string &data) {
    constexpr auto code = TENSORACLE_CUBIN_ARCHITECTURE_ERROR;
    require(data.size() >= 64 && data.compare(0, 6, "\177ELF\2\1", 6) == 0,
            "cubin is not a 64-bit little-endian ELF", code);
    require(number(data, 18, 2, code) == 190, "cubin has no NVIDIA CUDA ELF machine", code);
    auto flags = number(data, 48, 4, code);
    auto abi = static_cast<unsigned char>(data[8]);
    require(abi == 7 || abi == 8, "unsupported cubin ELF ABI version: " + std::to_string(abi),
            code);
    auto sm = (flags >> (abi == 7 ? 16 : 8)) & 255;
    require(sm == 86 || sm == 89 || sm == 120, "unsupported cubin architecture in ELF flags", code);
    return static_cast<uint32_t>(sm);
}
static bool relative_path(const std::string &name) {
    if (name.empty() || name[0] == '/' || name[0] == '\\' || name.find('\0') != std::string::npos ||
        name.find(':') != std::string::npos)
        return false;
    std::string normalized = name;
    for (auto &ch : normalized)
        if (ch == '\\')
            ch = '/';
    for (auto const &part : fs::path(normalized))
        if (part == "..")
            return false;
    return true;
}
json parse_manifest(const std::string &text) {
    try {
        return json::parse(text);
    } catch (json::exception const &error) {
        reject(TENSORACLE_BUNDLE_ERROR, std::string("invalid bundle JSON: ") + error.what());
    }
}
std::string digest(json value) {
    require(value.is_object(), "bundle manifest must be an object");
    value.erase("bundle_sha256");
    auto text = value.dump(-1, ' ', true);
    return sha256(text.data(), text.size());
}
tensoracle_bundle_result result_for(const json &value) {
    return {value.dump(-1, ' ', true), digest(value)};
}
static bool identifier(const std::string &name) {
    return std::regex_match(name, std::regex("[a-zA-Z_][a-zA-Z0-9_]*"));
}
static bool integer(const json &value, uint64_t minimum, uint64_t maximum) {
    return value.is_number_integer() && !value.is_boolean() &&
           (value.is_number_unsigned() || value.get<int64_t>() >= 0) &&
           value.get<uint64_t>() >= minimum && value.get<uint64_t>() <= maximum;
}
void validate_entry(const json &entry) {
    require(entry.is_object(), "invalid cubin entry");
    require(entry.contains("target_sm") &&
                (entry["target_sm"] == "sm86" || entry["target_sm"] == "sm89" ||
                 entry["target_sm"] == "sm120"),
            "unsupported target SM");
    require(!entry.contains("hidden_trailing_parameters"),
            "CUDA C++ kernels have no hidden trailing parameters");
    auto resources = entry.value("resources", json());
    require(resources.is_object(), "missing cubin resources");
    require(!resources.contains("global_scratch_bytes") &&
                !resources.contains("profile_scratch_bytes"),
            "CUDA C++ resources cannot declare Triton scratch");
    for (auto const *key : {"compiled_shared_memory_bytes", "launch_dynamic_shared_memory_bytes",
                            "registers_per_thread"}) {
        require(resources.contains(key) &&
                    integer(resources[key], std::string(key) == "registers_per_thread" ? 1 : 0,
                            INT32_MAX),
                std::string("invalid cubin resource ") + key);
    }
    auto signature = entry.value("signature", json());
    require(entry.contains("entry_symbol") && entry["entry_symbol"].is_string() &&
                identifier(entry["entry_symbol"].get<std::string>()) && signature.is_array(),
            "missing cubin symbol or signature");
    require(!signature.empty() && signature.size() <= 14, "unsupported public argument count");
    std::set<std::string> names;
    for (auto const &argument : signature) {
        require(argument.is_object() && argument.contains("name") && argument["name"].is_string() &&
                    identifier(argument["name"].get<std::string>()) && argument.contains("type") &&
                    argument["type"].is_string(),
                "invalid public argument");
        auto name = argument["name"].get<std::string>();
        auto type = argument["type"].get<std::string>();
        require(names.insert(name).second && (type == "*fp16" || type == "*fp32" || type == "*i8" ||
                                              type == "*i32" || type == "i32" || type == "fp32"),
                "unsupported public argument");
    }
    auto specialization = entry.value("specialization", json());
    require(specialization.is_object() && specialization.size() == 1 &&
                specialization.contains("divisible_by") &&
                specialization["divisible_by"].is_object(),
            "unsupported specialization rule");
    auto divisors = specialization["divisible_by"];
    for (auto rule = divisors.begin(); rule != divisors.end(); ++rule) {
        require(names.count(rule.key()) && integer(rule.value(), 1, 1U << 30),
                "unsupported specialization rule");
        auto divisor = rule.value().get<uint32_t>();
        require((divisor & (divisor - 1)) == 0, "unsupported specialization rule");
        for (auto const &argument : signature)
            require(!(argument["name"] == rule.key() && argument["type"] == "fp32" && divisor != 1),
                    "unsupported floating point specialization rule");
    }
    for (auto const *key : {"sha256", "file"})
        require(entry.contains(key) && entry[key].is_string(), std::string("missing cubin ") + key);
    require(entry.contains("bytes") && integer(entry["bytes"], 1, SIZE_MAX), "invalid cubin size");
    if (entry.contains("embedded_symbol"))
        require(entry["embedded_symbol"].is_string() &&
                    identifier(entry["embedded_symbol"].get<std::string>()),
                "invalid embedded cubin symbol");
}

tensoracle_bundle_result inspect(const std::string &text,
                                 const std::function<std::string(const std::string &)> &read,
                                 const std::string *plugin) {
    auto manifest = parse_manifest(text);
    require(manifest.is_object() && manifest.value("schema_version", json()) == 2,
            "unsupported bundle manifest");
    require(manifest.contains("entries") && manifest["entries"].is_array() &&
                !manifest["entries"].empty(),
            "bundle has no cubin entries");
    auto hash = digest(manifest);
    if (manifest.contains("bundle_sha256"))
        require(manifest["bundle_sha256"] == hash, "bundle manifest digest differs");
    std::set<std::string> paths, ids, symbols;
    std::vector<std::pair<std::string, std::string>> cubins;
    for (auto const &entry : manifest["entries"]) {
        validate_entry(entry);
        auto name = entry["file"].get<std::string>();
        require(relative_path(name), "invalid cubin path");
        require(paths.insert(name).second, "duplicate cubin path");
        if (entry.contains("id"))
            require(entry["id"].is_string() && ids.insert(entry["id"].get<std::string>()).second,
                    "duplicate or invalid cubin id");
        auto data = read(name);
        require(data.size() == entry["bytes"].get<uint64_t>() &&
                    sha256(data.data(), data.size()) == entry["sha256"],
                "cubin digest or size differs: " + name, TENSORACLE_CORRUPT_CUBIN);
        auto sm = "sm" + std::to_string(architecture(data));
        require(entry["target_sm"] == sm,
                "cubin " + name + " declares " + entry["target_sm"].get<std::string>() +
                    " but ELF contains " + sm,
                TENSORACLE_CUBIN_ARCHITECTURE_ERROR);
        if (entry.contains("embedded_symbol")) {
            auto symbol = entry["embedded_symbol"].get<std::string>();
            require(symbols.insert(symbol).second, "duplicate embedded cubin symbol");
            cubins.emplace_back(symbol, std::move(data));
        } else
            require(!plugin, "plugin inspection requires an embedded cubin symbol",
                    TENSORACLE_PLUGIN_BUNDLE_MISMATCH);
    }
    if (plugin) {
        std::string embedded;
        try {
            embedded = exported_data(*plugin, "tensoracle_bundle_sha256", 65);
        } catch (tensoracle::error const &error) {
            reject(TENSORACLE_PLUGIN_BUNDLE_MISMATCH, error.what());
        }
        require(embedded == hash + std::string(1, '\0'), "plugin and cubin bundle differ",
                TENSORACLE_PLUGIN_BUNDLE_MISMATCH);
        for (auto const &[symbol, data] : cubins) {
            try {
                require(exported_data(*plugin, symbol, data.size()) == data,
                        "plugin embedded cubin differs: " + symbol,
                        TENSORACLE_PLUGIN_BUNDLE_MISMATCH);
            } catch (tensoracle::error const &) {
                reject(TENSORACLE_PLUGIN_BUNDLE_MISMATCH,
                       "plugin embedded cubin differs or is absent: " + symbol);
            }
        }
    }
    return {manifest.dump(-1, ' ', true), hash};
}
} // namespace tensoracle::detail
using namespace tensoracle::detail;
extern "C" tensoracle_status
tensoracle_bundle_result_destroy(tensoracle_bundle_result *value) noexcept {
    delete value;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_bundle_result_get(const tensoracle_bundle_result *result,
                                                          tensoracle_bundle_view *view,
                                                          tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        require(result && view && structure_size(view) >= sizeof(size_t),
                "bundle view requires a result and struct_size", TENSORACLE_INVALID_ARGUMENT);
        auto bytes = structure_size(view);
        write_field(view, bytes, offsetof(tensoracle_bundle_view, manifest_json),
                    result->manifest.c_str());
        write_field(view, bytes, offsetof(tensoracle_bundle_view, sha256), result->digest.c_str());
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_bundle_digest(const char *text,
                                                      tensoracle_bundle_result **result,
                                                      tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (result)
            *result = nullptr;
        require(text && result, "bundle digest requires JSON and result",
                TENSORACLE_INVALID_ARGUMENT);
        *result = new tensoracle_bundle_result(result_for(parse_manifest(text)));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cubin_sm(const void *data, size_t size, uint32_t *sm,
                                                 tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (sm)
            *sm = 0;
        require(data && sm, "cubin architecture requires bytes and output",
                TENSORACLE_INVALID_ARGUMENT);
        *sm = architecture(std::string(static_cast<const char *>(data), size));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_inspect_bundle(const char *root, const char *plugin,
                                                       tensoracle_bundle_result **result,
                                                       tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (result)
            *result = nullptr;
        require(present(root) && result, "bundle inspection requires root and result",
                TENSORACLE_INVALID_ARGUMENT);
        auto library =
            plugin ? read_file(plugin, TENSORACLE_PLUGIN_BUNDLE_MISMATCH) : std::string();
        auto value = inspect(
            read_file(fs::path(root) / "bundle.json"),
            [&](auto const &name) {
                return read_file(fs::path(root) / name, TENSORACLE_MISSING_CUBIN);
            },
            plugin ? &library : nullptr);
        *result = new tensoracle_bundle_result(std::move(value));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_inspect_embedded_bundle(
    const void *manifest, size_t manifest_size, const tensoracle_bundle_file *const *files,
    size_t count, tensoracle_bundle_result **result, tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (result)
            *result = nullptr;
        require(manifest && result && (!count || files),
                "embedded bundle inspection requires manifest, files and result",
                TENSORACLE_INVALID_ARGUMENT);
        std::map<std::string, const tensoracle_bundle_file*> members;
        for (size_t i = 0; i < count; ++i) {
            auto *file = files[i];
            require(file && structure_size(file) >= offsetof(tensoracle_bundle_file, size) + sizeof(file->size),
                    "embedded cubin requires all file fields", TENSORACLE_INVALID_ARGUMENT);
            require(present(file->name) && file->data,
                    "embedded cubin requires name and bytes", TENSORACLE_INVALID_ARGUMENT);
            require(members.emplace(file->name, file).second,
                    "duplicate embedded cubin name", TENSORACLE_INVALID_ARGUMENT);
        }
        auto value = inspect(
            std::string(static_cast<const char*>(manifest), manifest_size),
            [&](const std::string &name) {
                auto member = members.find(name);
                require(member != members.end(), "missing embedded cubin: " + name, TENSORACLE_MISSING_CUBIN);
                auto *file = member->second;
                return std::string(static_cast<const char*>(file->data), file->size);
            }, nullptr);
        *result = new tensoracle_bundle_result(std::move(value));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_inspect_wheel(const char *path, const char *plugin,
                                                      tensoracle_bundle_result **result,
                                                      tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (result)
            *result = nullptr;
        require(present(path) && result, "wheel inspection requires path and result",
                TENSORACLE_INVALID_ARGUMENT);
        require(!plugin || relative_path(plugin), "invalid plugin path in wheel");
        struct Archive {
            mz_zip_archive zip{};
            ~Archive() {
                if (zip.m_pState)
                    mz_zip_reader_end(&zip);
            }
        } archive;
        require(mz_zip_reader_init_file(&archive.zip, path, 0) != 0,
                "invalid or unreadable wheel ZIP");
        std::map<std::string, std::string> members;
        std::string manifest;
        for (mz_uint i = 0; i < mz_zip_reader_get_num_files(&archive.zip); ++i) {
            auto length = mz_zip_reader_get_filename(&archive.zip, i, nullptr, 0);
            require(length > 0, "invalid wheel member name");
            std::string name(length, '\0');
            mz_zip_reader_get_filename(&archive.zip, i, name.data(), length);
            name.pop_back();
            require(relative_path(name), "wheel contains an invalid path");
            if (mz_zip_reader_is_file_a_directory(&archive.zip, i))
                continue;
            require(!members.count(name), "duplicate wheel member");
            size_t size = 0;
            void *raw = mz_zip_reader_extract_to_heap(&archive.zip, i, &size, 0);
            std::unique_ptr<void, decltype(&mz_free)> bytes(raw, mz_free);
            require(raw != nullptr, "corrupt wheel member: " + name);
            members.emplace(name, std::string(static_cast<const char *>(raw), size));
            if (fs::path(name).filename() == "bundle.json") {
                require(manifest.empty(), "wheel must contain exactly one cubin bundle");
                manifest = name;
            }
        }
        require(!manifest.empty(), "wheel must contain exactly one cubin bundle");
        require(!plugin || members.count(plugin), "plugin is missing in wheel",
                TENSORACLE_PLUGIN_BUNDLE_MISMATCH);
        auto parent = fs::path(manifest).parent_path();
        auto value = inspect(
            members.at(manifest),
            [&](auto const &name) {
                auto member = (parent / name).generic_string();
                require(members.count(member), "missing cubin: " + member,
                        TENSORACLE_MISSING_CUBIN);
                return members.at(member);
            },
            plugin ? &members.at(plugin) : nullptr);
        *result = new tensoracle_bundle_result(std::move(value));
        return TENSORACLE_SUCCESS;
    });
}
