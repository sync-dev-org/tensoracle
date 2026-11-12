#include "fixture.h"
#include <NvInfer.h>
#include <cuda.h>
#include <cmath>
#include <cstdio>

extern "C" int fixture_start(gpu_fixture* fixture) {
    CUdevice device{};
    CUcontext context{};
    CUdeviceptr input{}, output{};
    CUstream stream{};
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) return 1;
    fixture->context = context;
    if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return 1;
    if (cuMemAlloc(&input, 4 * sizeof(float)) != CUDA_SUCCESS) return 1;
    fixture->input = reinterpret_cast<void*>(input);
    if (cuMemAlloc(&output, 4 * sizeof(float)) != CUDA_SUCCESS) return 1;
    fixture->output = reinterpret_cast<void*>(output);
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return 1;
    fixture->stream = stream;
    const float values[]{1, 2, 3, 4};
    return cuMemcpyHtoD(input, values, sizeof(values)) != CUDA_SUCCESS;
}

extern "C" int fixture_check(gpu_fixture* fixture) {
    const float expected[]{1.5f, 0, 6.25f, 3.25f};
    float actual[4]{};
    if (cuStreamSynchronize(fixture->stream) != CUDA_SUCCESS ||
        cuMemcpyDtoH(actual, reinterpret_cast<CUdeviceptr>(fixture->output), sizeof(actual)) != CUDA_SUCCESS)
        return 1;
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-5f) return 1;
    }
    std::puts("OUTPUT 1.5 0 6.25 3.25");
    return 0;
}

extern "C" void fixture_destroy(gpu_fixture* fixture) {
    if (fixture->stream) { cuStreamSynchronize(fixture->stream); cuStreamDestroy(fixture->stream); }
    if (fixture->input) cuMemFree(reinterpret_cast<CUdeviceptr>(fixture->input));
    if (fixture->output) cuMemFree(reinterpret_cast<CUdeviceptr>(fixture->output));
    if (fixture->context) { CUcontext previous{}; cuCtxPopCurrent(&previous); cuDevicePrimaryCtxRelease(0); }
}

extern "C" tensoracle_status fixture_graph(void* context, void* opaque,
    const tensoracle_weights* weights, const char** message) noexcept {
    ++*static_cast<int*>(context);
    tensoracle_weight bias{}; bias.struct_size = sizeof(bias);
    if (tensoracle_weights_find(weights, "bias", &bias, nullptr) != TENSORACLE_SUCCESS ||
        bias.dtype != TENSORACLE_FLOAT32 || bias.element_count != 4) {
        *message = "Expected four float32 bias values"; return TENSORACLE_INVALID_ARGUMENT;
    }
    auto status = fixture_add(opaque, &bias);
    if (status != TENSORACLE_SUCCESS) *message = "Network addition failed";
    return status;
}

extern "C" tensoracle_status fixture_add(void* opaque, const tensoracle_weight* bias) noexcept {
    auto& network = *static_cast<nvinfer1::INetworkDefinition*>(opaque);
    nvinfer1::Dims shape{}; shape.nbDims = 1; shape.d[0] = 4;
    auto* input = network.addInput("input", nvinfer1::DataType::kFLOAT, shape);
    auto* constant = network.addConstant(shape, {nvinfer1::DataType::kFLOAT, bias->data, 4});
    if (!input || !constant) return TENSORACLE_INTERNAL_ERROR;
    auto* sum = network.addElementWise(*input, *constant->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (!sum) return TENSORACLE_INTERNAL_ERROR;
    sum->getOutput(0)->setName("output"); network.markOutput(*sum->getOutput(0));
    return TENSORACLE_SUCCESS;
}
