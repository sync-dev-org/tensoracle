#include <tensoracle/builder.hpp>
#include <tensoracle/cubin.hpp>
#include <tensoracle/plugins.hpp>
#include <NvInfer.h>
#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

extern "C" nvinfer1::IPluginCreatorInterface* const* embeddedExampleCreators(int32_t&) noexcept;
const char* embedded_manifest();
const tensoracle_bundle_file* const* embedded_files(size_t&);
namespace {
void cuda_check(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("CUDA error " + std::to_string(status));
}
struct context {
    CUcontext value{};
    context() {
        cuda_check(cuInit(0)); CUdevice device; cuda_check(cuDeviceGet(&device, 0));
        cuda_check(cuDevicePrimaryCtxRetain(&value, device)); cuda_check(cuCtxPushCurrent(value));
    }
    ~context() { CUcontext old; cuCtxPopCurrent(&old); cuDevicePrimaryCtxRelease(0); }
};
struct buffer {
    CUdeviceptr value{};
    buffer() { cuda_check(cuMemAlloc(&value, 32 * sizeof(float))); }
    ~buffer() { cuMemFree(value); }
};
struct stream {
    CUstream value{};
    stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
}
extern "C" __attribute__((visibility("default"))) int embedded_model(const char* weights) {
    size_t file_count = 0;
    auto files = embedded_files(file_count);
    auto bundle = tensoracle::inspect_embedded_bundle(embedded_manifest(), files, file_count);
    int32_t count = 0;
    auto source = embeddedExampleCreators(count);
    std::vector<void*> creators(source, source + count);
    auto anchor = reinterpret_cast<const void*>(&embeddedExampleCreators);
    auto registration = tensoracle::register_embedded_plugins(anchor, creators, "tensoracle.example");
    if (tensoracle::register_embedded_plugins(anchor, creators, "tensoracle.example").digests() != registration.digests())
        throw std::runtime_error("repeated registration changed identity");
    if (getPluginRegistry()->getCreator("TensoracleExampleBias", "1", "tensoracle.example") != source[1])
        throw std::runtime_error("TensorRT did not retain the linked creator");
    auto identity = registration.digests().at(0);
    if (!weights) {
        std::printf("{\"identity\":\"%s\",\"bundle\":\"%s\"}\n", identity.c_str(), bundle.sha256().c_str());
        return 0;
    }
    context gpu;
    tensoracle::builder_options options;
    options.name = "embedded-example"; options.graph_version = "1"; options.weights = {weights};
    options.inputs = {{"features", {1, 16}, {2, 16}, {4, 16}}};
    options.io = {{"features", TENSORACLE_FLOAT32, {-1, 16}, true}, {"result", TENSORACLE_FLOAT32, {-1, 16}, false}};
    options.workspace_limit = 64 * 1024 * 1024; options.optimization_level = 0;
    options.embedded_plugins = {registration.get()};
    int builds = 0;
    auto digest = bundle.sha256();
    auto engine = tensoracle::open_engine(options, [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& values) {
        ++builds;
        auto bias = values.at("bias");
        nvinfer1::PluginField fields[] = {{"bundle_sha256", digest.data(), nvinfer1::PluginFieldType::kCHAR, 64},
            {"bias", bias.data, nvinfer1::PluginFieldType::kFLOAT32, 16}};
        nvinfer1::PluginFieldCollection collection{2, fields};
        auto* creator = static_cast<nvinfer1::IPluginCreatorV3One*>(source[1]);
        std::unique_ptr<nvinfer1::IPluginV3> plugin(creator->createPlugin("bias", &collection, nvinfer1::TensorRTPhase::kBUILD));
        if (!plugin) throw std::runtime_error("linked creator rejected bias fields");
        auto* input = network.addInput("features", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(-1, 16));
        nvinfer1::ITensor* inputs[]{input};
        auto* layer = network.addPluginV3(inputs, 1, nullptr, 0, *plugin);
        if (!layer) throw std::runtime_error("linked plugin layer was rejected");
        auto* output = layer->getOutput(0); output->setName("result"); network.markOutput(*output);
    });
    auto execution = engine.create_execution(); execution.set_input_shapes({{"features", {2, 16}}});
    std::vector<float> input(32), actual(32);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i) / 16 - 1;
    buffer source_buffer, target_buffer; stream queue;
    cuda_check(cuMemcpyHtoD(source_buffer.value, input.data(), input.size() * sizeof(float)));
    execution.enqueue({{"features", reinterpret_cast<void*>(source_buffer.value)},
                       {"result", reinterpret_cast<void*>(target_buffer.value)}}, queue.value);
    cuda_check(cuStreamSynchronize(queue.value));
    cuda_check(cuMemcpyDtoH(actual.data(), target_buffer.value, actual.size() * sizeof(float)));
    auto values = tensoracle::open_weights(weights);
    auto bias = static_cast<const float*>(values.view().at("bias").data);
    for (size_t i = 0; i < actual.size(); ++i)
        if (std::abs(actual[i] - (input[i] + bias[i % 16])) > 1e-6f)
            throw std::runtime_error("embedded plugin output differs from CPU bias addition");
    std::printf("{\"identity\":\"%s\",\"path\":\"%s\",\"build_calls\":%d}\n",
                identity.c_str(), engine.path().c_str(), builds);
    return 0;
}
