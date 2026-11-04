#pragma once
#include <tensoracle/builder.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace builder_fixture {
namespace fs = std::filesystem;
inline void cuda_check(CUresult value) { if (value != CUDA_SUCCESS) throw std::runtime_error("CUDA driver failure " + std::to_string(value)); }
struct context {
    CUcontext value{};
    context() {
        cuda_check(cuInit(0)); CUdevice device; cuda_check(cuDeviceGet(&device, 0));
        cuda_check(cuDevicePrimaryCtxRetain(&value, device)); cuda_check(cuCtxPushCurrent(value));
    }
    ~context() { CUcontext old; cuCtxPopCurrent(&old); cuDevicePrimaryCtxRelease(0); }
};
struct device_buffer {
    CUdeviceptr value{};
    explicit device_buffer(size_t size) { cuda_check(cuMemAlloc(&value, size)); }
    ~device_buffer() { cuMemFree(value); }
};
struct stream {
    CUstream value{};
    stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
struct temporary_directory {
    fs::path path;
    std::string old;
    bool existed;
    temporary_directory() {
        char name[] = "/tmp/tensoracle-builder-XXXXXX"; auto* created = mkdtemp(name);
        if (!created) throw std::runtime_error("temporary directory failed"); path = created;
        const char* previous = std::getenv("XDG_CACHE_HOME"); existed = previous != nullptr; if (previous) old = previous;
        setenv("XDG_CACHE_HOME", path.c_str(), 1);
    }
    ~temporary_directory() { fs::remove_all(path); if (existed) setenv("XDG_CACHE_HOME", old.c_str(), 1); else unsetenv("XDG_CACHE_HOME"); }
};
inline std::vector<float> input(int rows = 16) {
    std::mt19937 random(11); std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<float> values(rows * 32); for (auto& value : values) value = distribution(random); return values;
}
inline void write_weights(const fs::path& path, float change = 0) {
    std::mt19937 random(29); std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<float> matrix(32 * 16), bias(16);
    for (auto& value : matrix) value = distribution(random) + change;
    for (auto& value : bias) value = distribution(random) / 2;
    float maximum = 0; for (float value : matrix) maximum = std::max(maximum, std::abs(value));
    std::map<std::string, std::pair<std::vector<int64_t>, std::vector<float>>> tensors{
        {"matrix", {{32, 16}, matrix}}, {"bias", {{1, 16}, bias}},
        {"int8.input_scale", {{1}, {1.0f / 127}}}, {"int8.weight_scale", {{1}, {maximum / 127}}},
        {"fp8.input_scale", {{1}, {1.0f / 448}}}, {"fp8.weight_scale", {{1}, {maximum / 448}}}};
    nlohmann::json header = nlohmann::json::object(); std::string data;
    for (const auto& item : tensors) {
        auto first = data.size(); data.append(reinterpret_cast<const char*>(item.second.second.data()), item.second.second.size() * sizeof(float));
        header[item.first] = {{"dtype", "F32"}, {"shape", item.second.first}, {"data_offsets", {first, data.size()}}};
    }
    auto encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
    std::ofstream output(path, std::ios::binary); uint64_t size = encoded.size(); char length[8];
    for (int i = 0; i < 8; ++i) length[i] = char(size >> (8 * i));
    output.write(length, 8); output << encoded << data;
}
inline tensoracle::builder_options options(const fs::path& file, const std::string& precision = "fp32", bool ranged = false) {
    tensoracle::builder_options request; request.name = "linear-" + precision; request.graph_version = "1"; request.weights = {file.string()};
    request.workspace_limit = 64 * 1024 * 1024; request.optimization_level = 0;
    request.inputs = {{"features", {ranged ? 1 : 16, 32}, {ranged ? 8 : 16, 32}, {16, 32}}};
    request.io = {{"features", TENSORACLE_FLOAT32, {ranged ? -1 : 16, 32}, true}, {"result", TENSORACLE_FLOAT32, {ranged ? -1 : 16, 16}, false}};
    return request;
}
inline void graph(nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights,
                  const std::string& precision, bool ranged = false) {
    auto constant = [&](const char* name) {
        auto value = weights.at(name);
        if (value.dtype != TENSORACLE_FLOAT32) throw std::runtime_error("fixture requires float32 weights");
        nvinfer1::Dims shape{}; shape.nbDims = int32_t(value.rank);
        std::copy_n(value.dimensions, value.rank, shape.d);
        return network.addConstant(shape, {nvinfer1::DataType::kFLOAT, value.data, int64_t(value.element_count)})->getOutput(0);
    };
    auto* features = network.addInput("features", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(ranged ? -1 : 16, 32));
    auto* matrix = constant("matrix"); auto* bias = constant("bias");
    if (precision == "fp16" || precision == "bf16") {
        auto type = precision == "fp16" ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kBF16;
        features = network.addCast(*features, type)->getOutput(0); matrix = network.addCast(*matrix, type)->getOutput(0); bias = network.addCast(*bias, type)->getOutput(0);
    } else if (precision == "int8" || precision == "fp8") {
        auto type = precision == "int8" ? nvinfer1::DataType::kINT8 : nvinfer1::DataType::kFP8;
        auto qdq = [&](nvinfer1::ITensor* value, const std::string& field) {
            auto* scale = constant((precision + "." + field + "_scale").c_str());
            auto* quantized = network.addQuantize(*value, *scale, type)->getOutput(0);
            return network.addDequantize(*quantized, *scale, nvinfer1::DataType::kFLOAT)->getOutput(0);
        };
        features = qdq(features, "input"); matrix = qdq(matrix, "weight");
    }
    auto* result = network.addMatrixMultiply(*features, nvinfer1::MatrixOperation::kNONE, *matrix, nvinfer1::MatrixOperation::kNONE)->getOutput(0);
    result = network.addElementWise(*result, *bias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    if (precision == "fp16" || precision == "bf16") result = network.addCast(*result, nvinfer1::DataType::kFLOAT)->getOutput(0);
    result->setName("result"); network.markOutput(*result);
}
inline std::vector<float> run(const tensoracle::engine_handle& engine, int rows = 16, bool ranged = false) {
    auto execution = engine.create_execution(); if (ranged) execution.set_input_shapes({{"features", {rows, 32}}});
    auto values = input(rows); std::vector<float> output(rows * 16);
    device_buffer source(values.size() * sizeof(float)), target(output.size() * sizeof(float)); stream queue;
    cuda_check(cuMemcpyHtoD(source.value, values.data(), values.size() * sizeof(float)));
    execution.enqueue({{"features", reinterpret_cast<void*>(source.value)}, {"result", reinterpret_cast<void*>(target.value)}}, queue.value);
    cuda_check(cuStreamSynchronize(queue.value)); cuda_check(cuMemcpyDtoH(output.data(), target.value, output.size() * sizeof(float))); return output;
}
}
