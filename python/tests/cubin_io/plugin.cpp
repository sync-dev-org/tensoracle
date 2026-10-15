#include "tensoracle/cubin_plugin.hpp"
#include "kernel_bytes.hpp"

#include <climits>
#include <new>
#include <vector>

using namespace nvinfer1;
using namespace tensoracle::cubin;

namespace {
char const* const kNames[] = {"TensoracleHalfUnary", "TensoracleHalfResidual", "TensoracleHalfSelect"};
constexpr char kNamespace[] = "tensoracle.tests.io";

PluginIO declaration(int mode)
{
    TensorSpec half{DataType::kHALF, TensorFormat::kHWC8};
    return {std::vector<TensorSpec>(mode == 0 ? 1 : 2, half), half, mode == 2 ? 1 : 0};
}

class HalfPlugin final : public PluginBase {
public:
    explicit HalfPlugin(int mode) : PluginBase(kNames[mode], kNamespace, {}, declaration(mode)), mode_(mode) {}
    IPluginV3* clone() noexcept override
    {
        try { return new HalfPlugin(mode_); }
        catch (...) { return nullptr; }
    }
    int32_t enqueue(PluginTensorDesc const* input_desc, PluginTensorDesc const* output_desc,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        int input_count = mode_ == 0 ? 1 : 2;
        Dims dims[2] = {input_desc[0].dims, input_desc[input_count - 1].dims};
        if (validate_shape(dims, input_count, output_desc[0].dims) != 0) return -1;
        auto const& shape = output_desc[0].dims;
        int64_t count = shape.d[0] * ((shape.d[1] + 7) / 8 * 8) * shape.d[2] * shape.d[3];
        if (count <= 0 || count > INT_MAX) return -1;
        int size = static_cast<int>(count);
        void const* x = inputs[0];
        void const* residual = inputs[input_count - 1];
        void* output = outputs[0];
        void* arguments[] = {&x, &residual, &output, &size, &mode_};
        CUmodule module{};
        CUfunction function{};
        auto status = cuModuleLoadData(&module, kKernelBytes);
        if (status == CUDA_SUCCESS) status = cuModuleGetFunction(&function, module, "half_io");
        auto native_stream = reinterpret_cast<CUstream>(stream);
        if (status == CUDA_SUCCESS)
            status = cuLaunchKernel(function, (size + 127) / 128, 1, 1, 128, 1, 1, 0,
                native_stream, arguments, nullptr);
        if (status == CUDA_SUCCESS) status = cuStreamSynchronize(native_stream);
        if (module) static_cast<void>(cuModuleUnload(module));
        return status == CUDA_SUCCESS ? 0 : -1;
    }

protected:
    int32_t validate_shape(Dims const* inputs, int32_t input_count,
        Dims const& output) const noexcept override
    {
        if (input_count != (mode_ == 0 ? 1 : 2) || output.nbDims != 4) return -1;
        for (int i = mode_ == 2 ? 1 : 0; i < input_count; ++i) {
            if (inputs[i].nbDims != 4) return -1;
            for (int j = 0; j < 4; ++j)
                if (inputs[i].d[j] != -1 && output.d[j] != -1 &&
                    inputs[i].d[j] != output.d[j]) return -1;
        }
        return 0;
    }

private:
    int mode_;
};

class Creator final : public IPluginCreatorV3One {
public:
    explicit Creator(int mode) : mode_(mode) {}
    char const* getPluginName() const noexcept override { return kNames[mode_]; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields,
        TensorRTPhase) noexcept override
    {
        if (fields == nullptr || fields->nbFields != 0) return nullptr;
        try { return new HalfPlugin(mode_); }
        catch (...) { return nullptr; }
    }
private:
    int mode_;
    PluginFieldCollection fields_{};
};
Creator gUnary(0), gResidual(1), gSelect(2);
}

#if defined(_WIN32)
#define IO_EXPORT __declspec(dllexport)
#else
#define IO_EXPORT __attribute__((visibility("default")))
#endif
extern "C" IO_EXPORT void setLoggerFinder(ILoggerFinder*) noexcept {}
extern "C" IO_EXPORT IPluginCreatorInterface* const* getCreators(int32_t& count) noexcept
{
    static IPluginCreatorInterface* creators[] = {&gUnary, &gResidual, &gSelect};
    count = 3;
    return creators;
}
