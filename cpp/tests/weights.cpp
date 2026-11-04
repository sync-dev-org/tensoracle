#include <tensoracle/builder.hpp>
#include "safetensors.hpp"
#include <nlohmann/json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fstream>
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
