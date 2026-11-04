#pragma once
#include "cache.hpp"
#include <NvInfer.h>

namespace tensoracle::detail {
inline std::unique_ptr<nvinfer1::IBuilderConfig> configure_builder(nvinfer1::IBuilder& builder, const cache_request& request) {
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
    if (!config) throw std::runtime_error("TensorRT failed to create a builder config");
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    if (request.workspace_limit) config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, *request.workspace_limit);
    config->setBuilderOptimizationLevel(request.optimization_level);
    return config;
}
}
