#include <tensoracle/builder.hpp>
#include "safetensors.hpp"
#include <nlohmann/json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
#include <map>
#include <type_traits>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
struct directory {
    fs::path path;
    directory() { char name[] = "/tmp/tensoracle-weights-XXXXXX"; path = mkdtemp(name); }
    ~directory() { fs::remove_all(path); }
};
void save(const fs::path& path, const std::string& header, const std::string& data) {
    std::ofstream file(path, std::ios::binary);
    uint64_t size = header.size(); char length[8];
    for (int i = 0; i < 8; ++i) length[i] = char(size >> (8 * i));
    file.write(length, 8); file << header << data;
}
}
TEST_CASE("Safetensors views retain named typed shaped values and scalar and empty tensors", "[SP-TENSORACLE-005][SP-TENSORACLE-059]") {
    directory files;
    auto path = files.path / "weights.safetensors";
    float values[] = {1.25f, -2.5f};
    json header{{"matrix", {{"dtype", "F32"}, {"shape", {1, 2}}, {"data_offsets", {0, 8}}}},
                {"empty", {{"dtype", "BF16"}, {"shape", {0}}, {"data_offsets", {8, 8}}}},
                {"scalar", {{"dtype", "I8"}, {"shape", json::array()}, {"data_offsets", {8, 9}}}}};
    std::string data(reinterpret_cast<const char*>(values), sizeof(values)); data += char(-7);
    save(path, header.dump(), data);
    auto storage = tensoracle::detail::load_safetensors({path});
    tensoracle::weights_view weights(storage.get());
    REQUIRE(weights.size() == 3);
    auto matrix = weights.at("matrix");
    REQUIRE(matrix.dtype == TENSORACLE_FLOAT32);
    REQUIRE(matrix.rank == 2);
    REQUIRE(matrix.dimensions[0] == 1);
    REQUIRE(matrix.dimensions[1] == 2);
    REQUIRE(matrix.element_count == 2);
    REQUIRE(matrix.byte_size == 8);
    REQUIRE(std::memcmp(matrix.data, values, sizeof(values)) == 0);
    REQUIRE(reinterpret_cast<uintptr_t>(matrix.data) % alignof(float) == 0);
    REQUIRE(weights.at("empty").element_count == 0);
    REQUIRE(weights.at("scalar").rank == 0);
    REQUIRE(*static_cast<const int8_t*>(weights.at("scalar").data) == -7);
    REQUIRE_THROWS_AS(weights.at("absent"), tensoracle::error);
    REQUIRE_THROWS_AS(weights.at(size_t(3)), tensoracle::error);
}
TEST_CASE("Safetensors rejects malformed JSON shapes offsets holes and duplicate names", "[SP-TENSORACLE-005][SP-TENSORACLE-059]") {
    directory files;
    auto path = files.path / "bad.safetensors";
    auto base = json{{"x", {{"dtype", "F32"}, {"shape", {2}}, {"data_offsets", {0, 8}}}}};
    for (int failure = 0; failure < 9; ++failure) {
        auto header = base;
        if (failure == 0) header["x"]["shape"] = {-1};
        if (failure == 1) header["x"]["shape"] = {true};
        if (failure == 2) header["x"]["data_offsets"] = {1, 9};
        if (failure == 3) header["x"]["data_offsets"] = {0, 4};
        if (failure == 4) header["x"]["dtype"] = "unsupported";
        if (failure == 5) header["__metadata__"] = {{"not-a-string", 1}};
        if (failure == 6) header["x"]["shape"] = {INT64_MAX, 8};
        if (failure == 7) header["y"] = header["x"];
        if (failure == 8) header["x"]["data_offsets"] = {4, 0};
        save(path, header.dump(), std::string(8, '\0'));
        REQUIRE_THROWS_AS(tensoracle::detail::load_safetensors({path}), std::invalid_argument);
    }
    save(path, "{\"x\":{},\"x\":{}}", "");
    REQUIRE_THROWS_AS(tensoracle::detail::load_safetensors({path}), std::invalid_argument);
    auto other = files.path / "other.safetensors";
    save(path, base.dump(), std::string(8, '\0')); save(other, base.dump(), std::string(8, '\0'));
    try { auto duplicate = tensoracle::detail::load_safetensors({path, other}); FAIL("duplicate tensor names accepted"); }
    catch (const std::invalid_argument& error) {
        REQUIRE(std::string(error.what()).find("x") != std::string::npos);
        REQUIRE(std::string(error.what()).find(path.string()) != std::string::npos);
        REQUIRE(std::string(error.what()).find(other.string()) != std::string::npos);
    }
}
TEST_CASE("Invalid builder declarations fail before CUDA or the network callback", "[SP-TENSORACLE-007][SP-TENSORACLE-052][SP-TENSORACLE-056]") {
    tensoracle::builder_options options;
    options.name = "test"; options.graph_version = "1"; options.weights = {"never-read.safetensors"};
    options.inputs = {{"x", {1}, {2}, {3}}};
    options.io = {{"x", TENSORACLE_FLOAT32, {-1}, true}, {"y", TENSORACLE_FLOAT32, {-1}, false}};
    bool called = false;
    auto callback = [&](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { called = true; };
    for (int failure = 0; failure < 4; ++failure) {
        auto invalid = options;
        if (failure == 0) invalid.workspace_limit = 0;
        if (failure == 1) invalid.optimization_level = 6;
        if (failure == 2) invalid.inputs[0].maximum = {0};
        if (failure == 3) invalid.io[0].name = "";
        try { auto engine = tensoracle::open_engine(invalid, callback); FAIL("invalid declaration accepted"); }
        catch (const tensoracle::error& error) { REQUIRE(error.status() == TENSORACLE_INVALID_ARGUMENT); }
    }
    REQUIRE_FALSE(called);
    tensoracle_engine* engine = reinterpret_cast<tensoracle_engine*>(1);
    REQUIRE(tensoracle_open_engine(nullptr, nullptr, &engine, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(engine == nullptr);
}

TEST_CASE("Safetensors rejects embedded NUL instead of treating a partial header as complete", "[SP-TENSORACLE-005][SP-TENSORACLE-059]") {
    directory files; auto path = files.path / "nul.safetensors";
    save(path, std::string("{}\0ignored", 10), "");
    REQUIRE_THROWS_AS(tensoracle::detail::load_safetensors({path}), std::invalid_argument);
}
TEST_CASE("Safetensors rejects a header beyond the upstream reader limit", "[SP-TENSORACLE-005][SP-TENSORACLE-059]") {
    directory files; auto path = files.path / "oversized.safetensors";
    std::string header(100000001, ' '); header[0] = '{'; header[1] = '}';
    save(path, header, "");
    REQUIRE_THROWS_AS(tensoracle::detail::load_safetensors({path}), std::invalid_argument);
}

TEST_CASE("Safetensors exposes nonempty half bfloat16 int8 and fp8 values without changing their bits", "[SP-TENSORACLE-005][SP-TENSORACLE-013][SP-TENSORACLE-059]") {
    directory files; auto path = files.path / "types.safetensors";
    json header{{"half", {{"dtype", "F16"}, {"shape", {1}}, {"data_offsets", {0, 2}}}},
                {"bfloat", {{"dtype", "BF16"}, {"shape", {1}}, {"data_offsets", {2, 4}}}},
                {"int8", {{"dtype", "I8"}, {"shape", {1}}, {"data_offsets", {4, 5}}}},
                {"fp8", {{"dtype", "F8_E4M3"}, {"shape", {1}}, {"data_offsets", {5, 6}}}}};
    save(path, header.dump(), std::string("\0\x3c\x80\x3f\x7f\x38", 6));
    auto storage = tensoracle::detail::load_safetensors({path}); tensoracle::weights_view weights(storage.get());
    REQUIRE(weights.at("half").dtype == TENSORACLE_FLOAT16);
    REQUIRE(weights.at("bfloat").dtype == TENSORACLE_BFLOAT16);
    REQUIRE(weights.at("int8").dtype == TENSORACLE_INT8);
    REQUIRE(weights.at("fp8").dtype == TENSORACLE_FLOAT8);
    REQUIRE(*static_cast<const uint16_t*>(weights.at("half").data) == 0x3c00);
    REQUIRE(*static_cast<const uint16_t*>(weights.at("bfloat").data) == 0x3f80);
    REQUIRE(*static_cast<const uint8_t*>(weights.at("fp8").data) == 0x38);
    struct prefix { size_t struct_size; const char* name; unsigned char trailing[8]; } output;
    std::memset(&output, 0xa5, sizeof(output)); output.struct_size = offsetof(prefix, trailing);
    REQUIRE(tensoracle_weights_find(storage.get(), "half", reinterpret_cast<tensoracle_weight*>(&output), nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(output.name) == "half"); for (auto byte : output.trailing) REQUIRE(byte == 0xa5);
}

TEST_CASE("Callers own weights opened from a path and retain tensor values until close", "[SP-TENSORACLE-061]") {
    static_assert(!std::is_copy_constructible_v<tensoracle::weights_handle>);
    static_assert(std::is_nothrow_move_constructible_v<tensoracle::weights_handle>);
    static_assert(std::is_nothrow_move_assignable_v<tensoracle::weights_handle>);
    directory files; auto path = files.path / "owned.safetensors";
    float values[] = {1.25f, -2.5f};
    json header{{"matrix", {{"dtype", "F32"}, {"shape", {1, 2}}, {"data_offsets", {0, 8}}}}};
    save(path, header.dump(), std::string(reinterpret_cast<const char*>(values), sizeof(values)));
    auto owner = tensoracle::open_weights(path);
    auto view = owner.view();
    REQUIRE(view.size() == 1);
    auto tensor = view.at(size_t(0));
    REQUIRE(std::string(tensor.name) == "matrix");
    REQUIRE(tensor.dtype == TENSORACLE_FLOAT32);
    REQUIRE(tensor.rank == 2);
    REQUIRE(tensor.dimensions[0] == 1);
    REQUIRE(tensor.dimensions[1] == 2);
    REQUIRE(tensor.element_count == 2);
    REQUIRE(tensor.byte_size == sizeof(values));
    REQUIRE(view.at("matrix").data == tensor.data);
    REQUIRE(reinterpret_cast<uintptr_t>(tensor.data) % alignof(float) == 0);
    auto moved = std::move(owner);
    REQUIRE(owner.get() == nullptr);
    tensoracle::weights_handle assigned;
    assigned = std::move(moved);
    REQUIRE(moved.get() == nullptr);
    fs::remove(path);
    REQUIRE(std::memcmp(tensor.data, values, sizeof(values)) == 0);
    auto* raw = assigned.release();
    REQUIRE(assigned.get() == nullptr);
    REQUIRE(tensoracle_weights_count(raw, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(tensor.data, values, sizeof(values)) == 0);
    REQUIRE(tensoracle_weights_destroy(raw) == TENSORACLE_SUCCESS);
    assigned.close();
    REQUIRE(tensoracle_weights_destroy(nullptr) == TENSORACLE_SUCCESS);
}

TEST_CASE("Callers read metadata keys and values and see an empty map when metadata is absent", "[SP-TENSORACLE-061]") {
    directory files; auto path = files.path / "metadata.safetensors";
    const std::map<std::string, std::string> expected{{"", ""}, {"format", "model-v1"}, {"説明", "重み"}};
    save(path, json{{"__metadata__", expected}}.dump(), "");
    auto owner = tensoracle::open_weights(path);
    auto view = owner.view();
    REQUIRE(view.size() == 0);
    REQUIRE(view.metadata_size() == expected.size());
    std::map<std::string, std::string> actual;
    for (size_t i = 0; i < view.metadata_size(); ++i) {
        auto entry = view.metadata_at(i);
        actual.emplace(entry.first, entry.second);
    }
    REQUIRE(actual == expected);
    auto retained = view.metadata_at("format");
    REQUIRE(std::string(view.metadata_at("説明")) == "重み");
    REQUIRE(std::string(view.metadata_at("")) == "");
    fs::remove(path);
    REQUIRE(std::string(retained) == "model-v1");
    REQUIRE(owner.metadata_size() == expected.size());
    REQUIRE(owner.metadata_at("format") == "model-v1");
    auto first = owner.metadata_at(size_t(0));
    REQUIRE(expected.at(std::string(first.first)) == first.second);
    for (const auto& header : {"{}", "{\"__metadata__\":{}}"}) {
        save(path, header, "");
        auto empty = tensoracle::open_weights(path);
        REQUIRE(empty.view().metadata_size() == 0);
    }
}

TEST_CASE("Opening weights rejects the same malformed files as engine weight loading", "[SP-TENSORACLE-061]") {
    directory files; auto path = files.path / "bad.safetensors";
    auto base = json{{"x", {{"dtype", "F32"}, {"shape", {2}}, {"data_offsets", {0, 8}}}}};
    std::vector<std::pair<std::string, std::string>> cases;
    for (int failure = 0; failure < 10; ++failure) {
        auto header = base;
        if (failure == 0) header["x"]["shape"] = {-1};
        if (failure == 1) header["x"]["shape"] = {true};
        if (failure == 2) header["x"]["data_offsets"] = {1, 9};
        if (failure == 3) header["x"]["data_offsets"] = {0, 4};
        if (failure == 4) header["x"]["dtype"] = "unsupported";
        if (failure == 5) header["__metadata__"] = {{"not-a-string", 1}};
        if (failure == 6) header["x"]["shape"] = {INT64_MAX, 8};
        if (failure == 7) header["y"] = header["x"];
        if (failure == 8) header["x"]["data_offsets"] = {4, 0};
        if (failure == 9) { header["x"]["shape"] = {1}; header["x"]["data_offsets"] = {4, 8}; }
        cases.emplace_back(header.dump(), std::string(8, '\0'));
    }
    cases.emplace_back("{\"x\":{},\"x\":{}}", "");
    cases.emplace_back("{\"__metadata__\":{\"x\":\"a\",\"x\":\"b\"}}", "");
    cases.emplace_back(std::string("{}\0ignored", 10), "");
    cases.emplace_back("{", "");
    for (const auto& item : cases) {
        save(path, item.first, item.second);
        std::string expected;
        try { auto weights = tensoracle::detail::load_safetensors({path}); FAIL("malformed file accepted"); }
        catch (const std::invalid_argument& error) { expected = error.what(); }
        tensoracle_weights* output = reinterpret_cast<tensoracle_weights*>(1);
        tensoracle_error* diagnostic = nullptr;
        REQUIRE(tensoracle_weights_open(path.c_str(), &output, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(output == nullptr);
        tensoracle::error_handle error(diagnostic);
        const char* message = nullptr;
        REQUIRE(tensoracle_error_message(error.get(), &message, nullptr) == TENSORACLE_SUCCESS);
        REQUIRE(std::string(message) == expected);
    }
    save(path, "{}", "");
    { std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out); file.put(char(127)); }
    REQUIRE_THROWS_AS(tensoracle::open_weights(path), tensoracle::error);
    fs::remove(path);
    tensoracle_weights* output = reinterpret_cast<tensoracle_weights*>(1);
    tensoracle_error* diagnostic = nullptr;
    REQUIRE(tensoracle_weights_open(path.c_str(), &output, &diagnostic) == TENSORACLE_INTERNAL_ERROR);
    REQUIRE(output == nullptr);
    tensoracle::error_handle error(diagnostic);
    const char* message = nullptr;
    REQUIRE(tensoracle_error_message(error.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(message).find(path.string()) != std::string::npos);
}

TEST_CASE("Invalid weight and metadata requests return diagnostics without transferring ownership", "[SP-TENSORACLE-061]") {
    directory files; auto path = files.path / "metadata.safetensors";
    save(path, "{\"__metadata__\":{\"format\":\"model-v1\"}}", "");
    auto owner = tensoracle::open_weights(path);
    const char* invalid_paths[] = {nullptr, ""};
    for (auto request : invalid_paths) {
        tensoracle_weights* output = reinterpret_cast<tensoracle_weights*>(1);
        tensoracle_error* diagnostic = nullptr;
        REQUIRE(tensoracle_weights_open(request, &output, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(output == nullptr);
        REQUIRE(diagnostic != nullptr);
        REQUIRE(tensoracle_error_destroy(diagnostic) == TENSORACLE_SUCCESS);
    }
    REQUIRE(tensoracle_weights_open(path.c_str(), nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    size_t count = 99;
    REQUIRE(tensoracle_weights_metadata_count(nullptr, &count, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(count == 0);
    REQUIRE(tensoracle_weights_metadata_count(owner.get(), nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle_weights_metadata_entry entry{}; entry.struct_size = sizeof(entry);
    REQUIRE(tensoracle_weights_metadata_at(owner.get(), 1, &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_at(nullptr, 0, &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_at(owner.get(), 0, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), "missing", &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_find(nullptr, "format", &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), nullptr, &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), "format", nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    entry.struct_size = sizeof(size_t) - 1;
    REQUIRE(tensoracle_weights_metadata_at(owner.get(), 0, &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), "format", &entry, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE_THROWS_AS(owner.view().metadata_at("missing"), tensoracle::error);
    REQUIRE_THROWS_AS(owner.view().metadata_at(size_t(1)), tensoracle::error);
}

TEST_CASE("Metadata preserves embedded NUL bytes through C entries and C++ views", "[SP-TENSORACLE-061]") {
    directory files; auto path = files.path / "nul-metadata.safetensors";
    const std::string key("model\0kind", 10), value("type\0v1", 7);
    save(path, json{{"__metadata__", {{key, value}, {"format", value}}}}.dump(), "");
    auto owner = tensoracle::open_weights(path);
    REQUIRE(owner.metadata_size() == 2);
    REQUIRE(owner.metadata_at("format") == value);
    tensoracle_weights_metadata_entry entry{}; entry.struct_size = sizeof(entry);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), "format", &entry, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(entry.key, entry.key_size) == "format");
    REQUIRE(std::string(entry.value, entry.value_size) == value);
    REQUIRE(entry.key[entry.key_size] == '\0');
    REQUIRE(entry.value[entry.value_size] == '\0');
    bool found = false;
    for (size_t i = 0; i < owner.metadata_size(); ++i) {
        REQUIRE(tensoracle_weights_metadata_at(owner.get(), i, &entry, nullptr) == TENSORACLE_SUCCESS);
        auto item = owner.view().metadata_at(i);
        REQUIRE(item.first == std::string_view(entry.key, entry.key_size));
        REQUIRE(item.second == std::string_view(entry.value, entry.value_size));
        REQUIRE(entry.key[entry.key_size] == '\0');
        REQUIRE(entry.value[entry.value_size] == '\0');
        if (item.first == key) { REQUIRE(item.second == value); found = true; }
    }
    REQUIRE(found);
    // This is the same loader used to supply the network-building callback.
    auto storage = tensoracle::detail::load_safetensors({path});
    auto borrowed = tensoracle::weights_view(storage.get());
    REQUIRE(borrowed.metadata_size() == 2);
    REQUIRE(borrowed.metadata_at("format") == value);
}

TEST_CASE("Metadata entries preserve fields outside the caller's structure size", "[SP-TENSORACLE-055][SP-TENSORACLE-061]") {
    directory files; auto path = files.path / "metadata.safetensors";
    save(path, "{\"__metadata__\":{\"format\":\"model-v1\"}}", "");
    auto owner = tensoracle::open_weights(path);
    struct prefix { size_t struct_size; const char* key; unsigned char trailing[8]; } entry;
    std::memset(&entry, 0xa5, sizeof(entry)); entry.struct_size = offsetof(prefix, trailing);
    auto* output = reinterpret_cast<tensoracle_weights_metadata_entry*>(&entry);
    REQUIRE(tensoracle_weights_metadata_at(owner.get(), 0, output, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(entry.key) == "format");
    for (auto byte : entry.trailing) REQUIRE(byte == 0xa5);
    REQUIRE(tensoracle_weights_metadata_find(owner.get(), "format", output, nullptr) == TENSORACLE_SUCCESS);
    for (auto byte : entry.trailing) REQUIRE(byte == 0xa5);
}

TEST_CASE("Borrowed build weights expose metadata from all declared files with the first value for repeated keys", "[SP-TENSORACLE-061]") {
    directory files; auto first = files.path / "first.safetensors", second = files.path / "second.safetensors";
    save(first, "{\"__metadata__\":{\"format\":\"first\",\"first-only\":\"one\"}}", "");
    save(second, "{\"__metadata__\":{\"format\":\"second\",\"second-only\":\"two\"}}", "");
    auto storage = tensoracle::detail::load_safetensors({first, second});
    tensoracle::weights_view borrowed(storage.get());
    REQUIRE(borrowed.metadata_size() == 3);
    REQUIRE(std::string(borrowed.metadata_at("format")) == "first");
    REQUIRE(std::string(borrowed.metadata_at("first-only")) == "one");
    REQUIRE(std::string(borrowed.metadata_at("second-only")) == "two");
    auto reversed = tensoracle::detail::load_safetensors({second, first});
    REQUIRE(std::string(tensoracle::weights_view(reversed.get()).metadata_at("format")) == "second");
}
