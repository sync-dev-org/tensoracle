#pragma once

#include "NvInfer.h"
#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tensoracle::cubin {

struct FieldValue {
    std::string name;
    nvinfer1::PluginFieldType type;
    std::int32_t length;
    std::vector<std::uint8_t> bytes;
};

// Own every byte. TensorRT may discard its input PluginFieldCollection after createPlugin.
class Fields {
public:
    static bool parse(nvinfer1::PluginFieldCollection const* source, Fields& result) noexcept;
    Fields() = default;
    Fields(Fields const& other);
    Fields& operator=(Fields const& other);
    Fields(Fields&& other);
    Fields& operator=(Fields&& other);

    FieldValue const* find(char const* name) const noexcept;
    std::vector<FieldValue> const& values() const noexcept { return values_; }
    nvinfer1::PluginFieldCollection const* collection() noexcept;

private:
    void rebuild();
    std::vector<FieldValue> values_;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection collection_{};
};

// Implements TensorRT V3 lifecycle and the format/serialization boilerplate.
class PluginBase : public nvinfer1::IPluginV3,
                   public nvinfer1::IPluginV3OneCore,
                   public nvinfer1::IPluginV3OneBuild,
                   public nvinfer1::IPluginV3OneRuntime {
public:
    PluginBase(char const* name, char const* plugin_namespace, Fields fields);
    ~PluginBase() noexcept override = default;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    std::int32_t getNbOutputs() const noexcept override;
    bool supportsFormatCombination(std::int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* in_out,
        std::int32_t input_count, std::int32_t output_count) noexcept override;
    std::int32_t getOutputDataTypes(nvinfer1::DataType* output_types, std::int32_t output_count,
        nvinfer1::DataType const* input_types, std::int32_t input_count) const noexcept override;
    std::int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, std::int32_t input_count,
        nvinfer1::DimsExprs const*, std::int32_t shape_input_count,
        nvinfer1::DimsExprs* outputs, std::int32_t output_count,
        nvinfer1::IExprBuilder& builder) noexcept override;
    std::int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        std::int32_t input_count, nvinfer1::DynamicPluginTensorDesc const* outputs,
        std::int32_t output_count) noexcept override;
    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, std::int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, std::int32_t) const noexcept override;
    char const* getTimingCacheID() noexcept override;
    char const* getMetadataString() noexcept override;
    std::int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
        std::int32_t input_count, nvinfer1::PluginTensorDesc const* outputs,
        std::int32_t output_count) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

protected:
    Fields const& fields() const noexcept { return fields_; }
    virtual std::int32_t validate_shape(nvinfer1::Dims const& input,
        nvinfer1::Dims const& output) const noexcept = 0;

private:
    char const* name_;
    char const* namespace_;
    Fields fields_;
};

enum class ArgumentKind : std::uint8_t { kPointer, kInt32, kFloat32 };

struct ArgumentRule {
    ArgumentKind kind;
    std::uint32_t divisible_by;
};

struct KernelImage {
    unsigned char const* bytes;
    std::size_t size;
    char const* sha256;
    char const* entry_symbol;
    std::int32_t target_sm;
    std::int32_t registers_per_thread;
    std::int32_t dynamic_shared_bytes;
    std::int32_t global_scratch_bytes;
    std::int32_t profile_scratch_bytes;
    // The producer derives these rules from the manifest's ordered signature.
    std::size_t public_argument_count;
    ArgumentRule const* argument_rules;
};

// Implemented by the producer-generated bundle.cpp linked into a caller's plugin.
KernelImage const* find_bundle_image(char const* id) noexcept;

std::string sha256(void const* bytes, std::size_t size);

// Each CUDA context owns its own module; destruction unloads in the owning context.
class Kernel {
public:
    explicit Kernel(KernelImage image) noexcept : image_(image) {}
    Kernel(Kernel const&) = delete;
    Kernel& operator=(Kernel const&) = delete;
    ~Kernel() noexcept;

    // Check the generated public ABI and divisibility rules before choosing a variant.
    bool accepts(void* const* arguments, std::size_t argument_count) const noexcept;
    CUresult launch(unsigned int grid, unsigned int block, void** arguments,
        std::size_t argument_count, CUstream stream) noexcept;

private:
    struct Loaded {
        CUmodule module{};
        CUfunction function{};
    };
    CUresult function_for_current_context(CUfunction& function) noexcept;
    KernelImage image_;
    std::mutex mutex_;
    std::map<CUcontext, Loaded> loaded_;
};

class DeviceBytes {
public:
    explicit DeviceBytes(std::vector<std::uint8_t> bytes) : bytes_(std::move(bytes)) {}
    DeviceBytes(DeviceBytes const&) = delete;
    DeviceBytes& operator=(DeviceBytes const&) = delete;
    ~DeviceBytes() noexcept;
    CUresult for_current_context(CUdeviceptr& pointer) noexcept;

private:
    std::vector<std::uint8_t> bytes_;
    std::mutex mutex_;
    std::map<CUcontext, CUdeviceptr> pointers_;
};

}  // namespace tensoracle::cubin
