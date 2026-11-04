#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <tensoracle/builder.hpp>
#include <tensoracle/cubin.hpp>
#include <NvInfer.h>
#include <cuda.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;
namespace {
void cuda_check(CUresult result) {
    if (result != CUDA_SUCCESS) throw std::runtime_error("CUDA driver error " + std::to_string(result));
}
struct primary_context {
    CUdevice device{};
    CUcontext value{};
    primary_context() {
        cuda_check(cuInit(0)); cuda_check(cuDeviceGet(&device, 0));
        cuda_check(cuDevicePrimaryCtxRetain(&value, device));
    }
    ~primary_context() { cuDevicePrimaryCtxRelease(device); }
};
struct current_context {
    explicit current_context(CUcontext value) { cuda_check(cuCtxPushCurrent(value)); }
    ~current_context() { CUcontext old{}; cuCtxPopCurrent(&old); }
};
struct buffer {
    CUdeviceptr value{};
    explicit buffer(size_t size) { cuda_check(cuMemAlloc(&value, size)); }
    ~buffer() { cuMemFree(value); }
};
struct stream {
    CUstream value{};
    stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream() { cuStreamSynchronize(value); cuStreamDestroy(value); }
};
struct captured_graph {
    CUgraph graph{};
    CUgraphExec executable{};
    ~captured_graph() { if (executable) cuGraphExecDestroy(executable); if (graph) cuGraphDestroy(graph); }
};

// The caller-owned network callback sees safetensors through tensoracle's C++
// weights view. Plugin field values remain alive through plugin construction.
void define_network(nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights,
                    const std::string& digest) {
    using namespace nvinfer1;
    auto matrix = weights.at("matrix"), scale = weights.at("scale"), bias = weights.at("bias");
    if (matrix.dtype != TENSORACLE_FLOAT32 || matrix.rank != 2 || matrix.dimensions[0] != 32 || matrix.dimensions[1] != 16
        || scale.dtype != TENSORACLE_FLOAT32 || scale.element_count != 1
        || bias.dtype != TENSORACLE_FLOAT32 || bias.element_count != 16)
        throw std::invalid_argument("example weights require float32 matrix (32,16), scalar scale and 16 bias values");
    auto* registry = getPluginRegistry();
    auto* dot_creator = static_cast<IPluginCreatorV3One*>(registry->getCreator("TensoracleExampleDot", "1", "tensoracle.example"));
    auto* bias_creator = static_cast<IPluginCreatorV3One*>(registry->getCreator("TensoracleExampleBias", "1", "tensoracle.example"));
    if (!dot_creator || !bias_creator) throw std::runtime_error("example creators must be registered before network construction");
    PluginField dot_fields[] = {
        {"bundle_sha256", digest.data(), PluginFieldType::kCHAR, 64},
        {"weights", matrix.data, PluginFieldType::kFLOAT32, 512},
        {"scale", scale.data, PluginFieldType::kFLOAT32, 1}};
    PluginField bias_fields[] = {
        {"bundle_sha256", digest.data(), PluginFieldType::kCHAR, 64},
        {"bias", bias.data, PluginFieldType::kFLOAT32, 16}};
    PluginFieldCollection dot_collection{3, dot_fields}, bias_collection{2, bias_fields};
    std::unique_ptr<IPluginV3> dot(dot_creator->createPlugin("dot", &dot_collection, TensorRTPhase::kBUILD));
    std::unique_ptr<IPluginV3> bias_plugin(bias_creator->createPlugin("bias", &bias_collection, TensorRTPhase::kBUILD));
    if (!dot || !bias_plugin) throw std::runtime_error("example plugin fields were rejected");
    auto* input = network.addInput("features", DataType::kFLOAT, Dims2(-1, 32));
    if (!input) throw std::runtime_error("cannot create example input");
    ITensor* dot_inputs[]{input};
    auto* dotted = network.addPluginV3(dot_inputs, 1, nullptr, 0, *dot);
    if (!dotted) throw std::runtime_error("cannot create dot layer");
    ITensor* bias_inputs[]{dotted->getOutput(0)};
    auto* biased = network.addPluginV3(bias_inputs, 1, nullptr, 0, *bias_plugin);
    if (!biased) throw std::runtime_error("cannot create bias layer");
    auto* output = biased->getOutput(0); output->setName("result"); network.markOutput(*output);
}

using input_array = nb::ndarray<const float, nb::numpy, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
class model {
public:
    model(const std::string& weights, const std::string& plugin, const std::string& bundle, bool force_rebuild) {
        current_context current(gpu_.value);
        auto digest = tensoracle::inspect_bundle(bundle.c_str(), plugin.c_str()).sha256();
        tensoracle::builder_options options;
        options.name = "cubin-example"; options.graph_version = "1";
        options.weights = {weights}; options.plugins = {plugin};
        options.inputs = {{"features", {1, 32}, {16, 32}, {32, 32}}};
        options.io = {{"features", TENSORACLE_FLOAT32, {-1, 32}, true}, {"result", TENSORACLE_FLOAT32, {-1, 16}, false}};
        options.workspace_limit = 64 * 1024 * 1024; options.optimization_level = 0;
        options.force_rebuild = force_rebuild;
        auto engine = tensoracle::open_engine(options,
            [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& values) {
                ++build_calls_; define_network(network, values, digest);
            });
        path_ = engine.path();
        for (auto& execution : executions_) execution = engine.create_execution();
        // Each execution retains its engine and its own plugin resource state.
    }
    ~model() {
        current_context current(gpu_.value);
        for (auto& execution : executions_) execution.close();
    }
    nb::ndarray<nb::numpy, float, nb::ndim<2>> infer(input_array input, size_t index, bool capture) {
        if (input.shape(1) != 32 || input.shape(0) < 1 || input.shape(0) > 32)
            throw std::invalid_argument("features must have shape (rows,32), with 1 <= rows <= 32");
        if (index >= executions_.size()) throw std::invalid_argument("execution must be 0 or 1");
        current_context current(gpu_.value);
        auto& execution = executions_[index];
        auto rows = input.shape(0);
        execution.set_input_shapes({{"features", {int64_t(rows), 32}}});
        auto output = std::make_unique<float[]>(rows * 16);
        buffer source(rows * 32 * sizeof(float)), target(rows * 16 * sizeof(float));
        stream queue;
        std::vector<tensoracle::tensor_address> addresses{
            {"features", reinterpret_cast<void*>(source.value)}, {"result", reinterpret_cast<void*>(target.value)}};
        cuda_check(cuMemsetD8(source.value, 0, rows * 32 * sizeof(float)));
        // Warm up after every shape change before capturing enqueue on a non-default stream.
        execution.enqueue(addresses, queue.value);
        cuda_check(cuStreamSynchronize(queue.value));
        captured_graph graph;
        if (capture) {
            cuda_check(cuStreamBeginCapture(queue.value, CU_STREAM_CAPTURE_MODE_GLOBAL));
            try { execution.enqueue(addresses, queue.value); }
            catch (...) { cuStreamEndCapture(queue.value, &graph.graph); throw; }
            cuda_check(cuStreamEndCapture(queue.value, &graph.graph));
            cuda_check(cuGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
        }
        // New input bytes arrive after capture. The replay must compute these values.
        cuda_check(cuMemcpyHtoD(source.value, input.data(), rows * 32 * sizeof(float)));
        if (capture) cuda_check(cuGraphLaunch(graph.executable, queue.value));
        else execution.enqueue(addresses, queue.value);
        cuda_check(cuStreamSynchronize(queue.value));
        cuda_check(cuMemcpyDtoH(output.get(), target.value, rows * 16 * sizeof(float)));
        nb::capsule owner(output.get(), [](void* pointer) noexcept { delete[] static_cast<float*>(pointer); });
        auto* values = output.release();
        return {values, {rows, 16}, owner};
    }
    std::vector<int64_t> output_shape(size_t index) {
        if (index >= executions_.size()) throw std::invalid_argument("execution must be 0 or 1");
        current_context current(gpu_.value);
        return executions_[index].get_tensor_shape("result");
    }
    const std::string& path() const { return path_; }
    size_t build_calls() const { return build_calls_; }
private:
    primary_context gpu_;
    std::array<tensoracle::engine_execution, 2> executions_;
    std::string path_;
    size_t build_calls_ = 0;
};
}
NB_MODULE(_native, module) {
    nb::class_<model>(module, "Model")
        .def(nb::init<const std::string&, const std::string&, const std::string&, bool>())
        .def("infer", &model::infer, nb::arg("features").noconvert(), nb::arg("execution") = 0, nb::arg("capture") = false)
        .def("output_shape", &model::output_shape, nb::arg("execution") = 0)
        .def_prop_ro("engine_path", &model::path)
        .def_prop_ro("build_calls", &model::build_calls);
}
