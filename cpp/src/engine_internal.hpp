#pragma once
#include <tensoracle/engine.hpp>
#include <NvInferRuntime.h>
#include <memory>
#include <string>
#include <vector>

namespace tensoracle::detail {
struct engine_logger : nvinfer1::ILogger {
    void log(Severity, const char*) noexcept override {}
};
// TensorRT may reuse the first logger across runtimes; retain it independently
// of individual engines, including when an older handle is released first.
inline engine_logger& process_engine_logger() {
    static engine_logger logger;
    return logger;
}
struct engine_state {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
};
}

struct tensoracle_engine {
    std::shared_ptr<tensoracle::detail::engine_state> state;
    std::string path;
};

struct tensoracle_execution {
    struct input {
        std::string name;
        nvinfer1::Dims low, optimum, high;
        bool ranged;
    };
    std::shared_ptr<tensoracle::detail::engine_state> state;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<std::string> names;
    std::vector<input> inputs;
    bool ready = false;
};

namespace tensoracle::detail {
inline tensoracle_dtype public_dtype(nvinfer1::DataType value) {
    switch (value) {
    case nvinfer1::DataType::kFLOAT: return TENSORACLE_FLOAT32;
    case nvinfer1::DataType::kHALF: return TENSORACLE_FLOAT16;
    case nvinfer1::DataType::kINT8: return TENSORACLE_INT8;
    case nvinfer1::DataType::kINT32: return TENSORACLE_INT32;
    case nvinfer1::DataType::kBOOL: return TENSORACLE_BOOL;
    case nvinfer1::DataType::kUINT8: return TENSORACLE_UINT8;
    case nvinfer1::DataType::kFP8: return TENSORACLE_FLOAT8;
    case nvinfer1::DataType::kBF16: return TENSORACLE_BFLOAT16;
    case nvinfer1::DataType::kINT64: return TENSORACLE_INT64;
    case nvinfer1::DataType::kINT4: return TENSORACLE_INT4;
    case nvinfer1::DataType::kFP4: return TENSORACLE_FLOAT4;
    case nvinfer1::DataType::kE8M0: return TENSORACLE_E8M0;
    }
    throw std::runtime_error("unrecognized TensorRT tensor dtype");
}
// Kept internal: public engine handles are returned only after cache/IO checks.
inline engine_handle deserialize_engine(const std::vector<uint8_t>& plan) {
    if (plan.empty()) throw std::invalid_argument("serialized plan must not be empty");
    auto state = std::make_shared<engine_state>();
    state->runtime.reset(nvinfer1::createInferRuntime(process_engine_logger()));
    if (!state->runtime) throw std::runtime_error("TensorRT could not create a runtime");
    state->engine.reset(state->runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!state->engine) throw tensoracle::error(TENSORACLE_CORRUPT_ENGINE,
        "TensorRT could not deserialize engine; check plugin libraries and creators");
    return engine_handle(new tensoracle_engine{std::move(state), {}});
}
}
