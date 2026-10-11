#include "tensoracle/cubin_plugin.hpp"

#include <array>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <new>
#include <utility>

#if defined(_WIN32)
#define EXAMPLE_EXPORT __declspec(dllexport)
#else
#define EXAMPLE_EXPORT __attribute__((visibility("default")))
#endif

extern "C" char const tensoracle_bundle_sha256[];

namespace {
using namespace nvinfer1;
using tensoracle::cubin::DeviceBytes;
using tensoracle::cubin::Fields;
using tensoracle::cubin::Kernel;
using tensoracle::cubin::PluginBase;

constexpr char kNamespace[] = "tensoracle.example";
constexpr char kDotName[] = "TensoracleExampleDot";
constexpr char kBiasName[] = "TensoracleExampleBias";
constexpr std::array<int, 3> kSms{{86, 89, 120}};
std::atomic<std::uint64_t> gSpecialized{};
std::atomic<std::uint64_t> gUnspecialized{};

int sm_index() noexcept
{
    CUdevice device{};
    int major{}, minor{};
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != CUDA_SUCCESS)
        return -1;
    int const sm = major * 10 + minor;
    for (std::size_t index = 0; index < kSms.size(); ++index)
        if (kSms[index] == sm) return static_cast<int>(index);
    return -1;
}

bool field_rule(Fields const& fields, bool dot) noexcept
{
    auto const& values = fields.values();
    if (values.size() != (dot ? 3U : 2U)) return false;
    if (values[0].name != "bundle_sha256" || values[0].type != PluginFieldType::kCHAR ||
        values[0].length != 64 ||
        std::memcmp(values[0].bytes.data(), tensoracle_bundle_sha256, 64) != 0) return false;
    if (dot) {
        return values[1].name == "weights" && values[1].type == PluginFieldType::kFLOAT32 &&
            values[1].length == 32 * 16 && values[2].name == "scale" &&
            values[2].type == PluginFieldType::kFLOAT32 && values[2].length == 1;
    }
    return values[1].name == "bias" && values[1].type == PluginFieldType::kFLOAT32 &&
        values[1].length == 16;
}

bool shape_matches(Dims const& input, Dims const& output, int input_columns) noexcept
{
    return input.nbDims == 2 && output.nbDims == 2 && input.d[1] == input_columns &&
        output.d[1] == 16 && (input.d[0] == -1 || input.d[0] > 0) &&
        (output.d[0] == -1 || output.d[0] > 0) &&
        (input.d[0] == -1 || output.d[0] == -1 || input.d[0] == output.d[0]);
}

std::unique_ptr<Kernel> kernel_for(int sm, char const* suffix)
{
    char id[80];
    std::snprintf(id, sizeof(id), "sm%d.%s", sm, suffix);
    auto const* image = tensoracle::cubin::find_bundle_image(id);
    return image == nullptr ? nullptr : std::make_unique<Kernel>(*image);
}

class DotPlugin final : public PluginBase {
public:
    explicit DotPlugin(Fields fields)
        : PluginBase(kDotName, kNamespace, fields), weights_(fields.find("weights")->bytes)
    {
        std::memcpy(&scale_, fields.find("scale")->bytes.data(), sizeof(scale_));
        for (std::size_t index = 0; index < kSms.size(); ++index) {
            specialized_[index] = kernel_for(kSms[index], "dot.specialized");
            unspecialized_[index] = kernel_for(kSms[index], "dot.unspecialized");
            scale_kernels_[index] = kernel_for(kSms[index], "scale");
            if (!specialized_[index] || !unspecialized_[index] || !scale_kernels_[index])
                valid_ = false;
        }
    }

    IPluginV3* clone() noexcept override
    {
        try { return new (std::nothrow) DotPlugin(fields()); }
        catch (...) { return nullptr; }
    }

    std::int32_t getOutputShapes(DimsExprs const* inputs, std::int32_t input_count,
        DimsExprs const*, std::int32_t shape_input_count, DimsExprs* outputs,
        std::int32_t output_count, IExprBuilder& builder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || input_count != 1 ||
            shape_input_count != 0 || output_count != 1 || inputs[0].nbDims != 2) return -1;
        outputs[0] = inputs[0];
        outputs[0].d[1] = builder.constant(16);
        return 0;
    }

    std::int32_t enqueue(PluginTensorDesc const* input_descriptors,
        PluginTensorDesc const* output_descriptors, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (!valid_ || input_descriptors == nullptr || output_descriptors == nullptr ||
            inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
            validate_shape(&input_descriptors[0].dims, 1, output_descriptors[0].dims) != 0) return -1;
        int rows = input_descriptors[0].dims.d[0];
        if (rows < 1 || rows > INT_MAX / 16) return -1;
        int const index = sm_index();
        if (index < 0) return -1;
        CUdeviceptr weight{};
        if (weights_.for_current_context(weight) != CUDA_SUCCESS) return -1;
        CUdeviceptr input = reinterpret_cast<CUdeviceptr>(inputs[0]);
        CUdeviceptr output = reinterpret_cast<CUdeviceptr>(outputs[0]);
        void* dot_arguments[] = {&input, &weight, &output, &rows};
        bool const specialized = specialized_[index]->accepts(dot_arguments, std::size(dot_arguments));
        Kernel* dot = specialized ? specialized_[index].get() : unspecialized_[index].get();
        auto native_stream = reinterpret_cast<CUstream>(stream);
        if (dot->launch(static_cast<unsigned int>((rows + 15) / 16), 128,
                dot_arguments, std::size(dot_arguments), native_stream) != CUDA_SUCCESS) return -1;
        int count = rows * 16;
        void* scale_arguments[] = {&output, &output, &scale_, &count};
        if (scale_kernels_[index]->launch(static_cast<unsigned int>((count + 127) / 128), 128,
                scale_arguments, std::size(scale_arguments), native_stream) != CUDA_SUCCESS) return -1;
        (specialized ? gSpecialized : gUnspecialized).fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

protected:
    std::int32_t validate_shape(Dims const* inputs, std::int32_t input_count,
        Dims const& output) const noexcept override
    {
        return input_count == 1 && shape_matches(inputs[0], output, 32) ? 0 : -1;
    }

private:
    DeviceBytes weights_;
    float scale_{};
    bool valid_{true};
    std::array<std::unique_ptr<Kernel>, 3> specialized_{};
    std::array<std::unique_ptr<Kernel>, 3> unspecialized_{};
    std::array<std::unique_ptr<Kernel>, 3> scale_kernels_{};
};

class BiasPlugin final : public PluginBase {
public:
    explicit BiasPlugin(Fields fields)
        : PluginBase(kBiasName, kNamespace, fields), bias_(fields.find("bias")->bytes)
    {
        for (std::size_t index = 0; index < kSms.size(); ++index) {
            kernels_[index] = kernel_for(kSms[index], "bias");
            if (!kernels_[index]) valid_ = false;
        }
    }

    IPluginV3* clone() noexcept override
    {
        try { return new (std::nothrow) BiasPlugin(fields()); }
        catch (...) { return nullptr; }
    }

    std::int32_t enqueue(PluginTensorDesc const* input_descriptors,
        PluginTensorDesc const* output_descriptors, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (!valid_ || input_descriptors == nullptr || output_descriptors == nullptr ||
            inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr ||
            validate_shape(&input_descriptors[0].dims, 1, output_descriptors[0].dims) != 0) return -1;
        int const rows = input_descriptors[0].dims.d[0];
        if (rows < 1 || rows > INT_MAX / 16) return -1;
        int const index = sm_index();
        if (index < 0) return -1;
        CUdeviceptr bias{};
        if (bias_.for_current_context(bias) != CUDA_SUCCESS) return -1;
        CUdeviceptr input = reinterpret_cast<CUdeviceptr>(inputs[0]);
        CUdeviceptr output = reinterpret_cast<CUdeviceptr>(outputs[0]);
        int count = rows * 16;
        void* arguments[] = {&input, &bias, &output, &count};
        return kernels_[index]->launch(static_cast<unsigned int>((count + 127) / 128), 128,
            arguments, std::size(arguments), reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? 0 : -1;
    }

protected:
    std::int32_t validate_shape(Dims const* inputs, std::int32_t input_count,
        Dims const& output) const noexcept override
    {
        return input_count == 1 && shape_matches(inputs[0], output, 16) ? 0 : -1;
    }

private:
    DeviceBytes bias_;
    bool valid_{true};
    std::array<std::unique_ptr<Kernel>, 3> kernels_{};
};

class Creator final : public IPluginCreatorV3One {
public:
    explicit Creator(bool dot) : dot_(dot)
    {
        names_[0] = PluginField("bundle_sha256", nullptr, PluginFieldType::kCHAR, 64);
        names_[1] = dot ? PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 512)
                        : PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 16);
        if (dot) names_[2] = PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1);
        collection_.nbFields = dot ? 3 : 2;
        collection_.fields = names_.data();
    }

    char const* getPluginName() const noexcept override { return dot_ ? kDotName : kBiasName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return kNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &collection_; }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* source,
        TensorRTPhase phase) noexcept override
    {
        if (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME) return nullptr;
        try {
            Fields fields;
            if (!Fields::parse(source, fields) || !field_rule(fields, dot_)) return nullptr;
            return dot_ ? static_cast<IPluginV3*>(new (std::nothrow) DotPlugin(std::move(fields)))
                        : static_cast<IPluginV3*>(new (std::nothrow) BiasPlugin(std::move(fields)));
        } catch (...) { return nullptr; }
    }

private:
    bool dot_;
    std::array<PluginField, 3> names_{};
    PluginFieldCollection collection_{};
};

Creator gDotCreator(true);
Creator gBiasCreator(false);
}  // namespace

extern "C" EXAMPLE_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}

extern "C" EXAMPLE_EXPORT nvinfer1::IPluginCreatorInterface* const*
getCreators(std::int32_t& count) noexcept
{
    static nvinfer1::IPluginCreatorInterface* creators[] = {&gDotCreator, &gBiasCreator};
    count = 2;
    return creators;
}

extern "C" EXAMPLE_EXPORT std::uint64_t tensoracle_example_specialized_launches() noexcept
{
    return gSpecialized.load(std::memory_order_relaxed);
}

extern "C" EXAMPLE_EXPORT std::uint64_t tensoracle_example_unspecialized_launches() noexcept
{
    return gUnspecialized.load(std::memory_order_relaxed);
}
