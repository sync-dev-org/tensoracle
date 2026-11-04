#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <tensoracle/builder.hpp>
#include <NvInfer.h>
#include <cuda.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
namespace {
void cuda_check(CUresult status) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("CUDA driver error " + std::to_string(status));
}

struct context {
    CUdevice device{};
    CUcontext value{};
    context() {
        cuda_check(cuInit(0));
        cuda_check(cuDeviceGet(&device, 0));
        cuda_check(cuDevicePrimaryCtxRetain(&value, device));
        auto status = cuCtxPushCurrent(value);
        if (status != CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(device);
            cuda_check(status);
        }
    }
    ~context() {
        CUcontext previous{};
        cuCtxPopCurrent(&previous);
        cuDevicePrimaryCtxRelease(device);
    }
    context(const context&) = delete;
    context& operator=(const context&) = delete;
};

struct device_buffer {
    CUdeviceptr value{};
    device_buffer() { cuda_check(cuMemAlloc(&value, 4 * sizeof(float))); }
    ~device_buffer() { cuMemFree(value); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
};

struct stream {
    CUstream value{};
    stream() { cuda_check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING)); }
    ~stream() {
        cuStreamSynchronize(value);
        cuStreamDestroy(value);
    }
    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;
};

// TensorRT's network, safetensors access, and engine operations stay in C++.
// No TensorRT or tensoracle Python API participates in construction or execution.
void build_network(nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) {
    auto bias = weights.at("bias");
    if (bias.dtype != TENSORACLE_FLOAT32 || bias.rank != 1 || bias.dimensions[0] != 4)
        throw std::invalid_argument("bias must be a float32 safetensors tensor of shape (4,)");
    nvinfer1::Dims shape{};
    shape.nbDims = 1;
    shape.d[0] = 4;
    auto* input = network.addInput("input", nvinfer1::DataType::kFLOAT, shape);
    auto* constant = network.addConstant(shape, {nvinfer1::DataType::kFLOAT, bias.data, 4});
    if (!input || !constant) throw std::runtime_error("cannot create network inputs");
    auto* sum = network.addElementWise(*input, *constant->getOutput(0),
                                     nvinfer1::ElementWiseOperation::kSUM);
    if (!sum) throw std::runtime_error("cannot create addition layer");
    auto* output = sum->getOutput(0);
    output->setName("output");
    network.markOutput(*output);
}

nb::dict infer(const std::string& weights,
               nb::ndarray<const float, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu> input) {
    if (input.shape(0) != 4) throw std::invalid_argument("input must have shape (4,)");
    context gpu;
    tensoracle::builder_options options;
    options.name = "wheel-consumer-add";
    options.graph_version = "1";
    options.weights = {weights};
    options.inputs = {{"input", {4}, {4}, {4}}};
    options.io = {{"input", TENSORACLE_FLOAT32, {4}, true},
                  {"output", TENSORACLE_FLOAT32, {4}, false}};
    options.workspace_limit = 64 * 1024 * 1024;
    options.optimization_level = 0;
    size_t calls = 0;
    auto engine = tensoracle::open_engine(options,
        [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& values) {
            ++calls;
            build_network(network, values);
        });
    auto execution = engine.create_execution();
    device_buffer source, target;
    stream queue;
    auto output = std::make_unique<float[]>(4);
    cuda_check(cuMemcpyHtoD(source.value, input.data(), 4 * sizeof(float)));
    execution.enqueue({{"input", reinterpret_cast<void*>(source.value)},
                       {"output", reinterpret_cast<void*>(target.value)}}, queue.value);
    cuda_check(cuStreamSynchronize(queue.value));
    cuda_check(cuMemcpyDtoH(output.get(), target.value, 4 * sizeof(float)));
    nb::capsule owner(output.get(), [](void* pointer) noexcept { delete[] static_cast<float*>(pointer); });
    auto* data = output.release();
    nb::dict result;
    result["output"] = nb::ndarray<nb::numpy, float, nb::ndim<1>>(data, {4}, owner);
    result["engine_path"] = nb::str(engine.path().c_str());
    result["build_calls"] = calls;
    return result;
}
}

NB_MODULE(_native, module) {
    module.def("version", [] {
        auto value = tensoracle::version();
        return nb::make_tuple(value.major, value.minor, value.patch);
    });
    module.def("infer", &infer, nb::arg("weights"), nb::arg("input").noconvert());
}
