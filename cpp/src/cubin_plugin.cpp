#include "tensoracle/cubin_plugin.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensoracle::cubin {
namespace {

nvinfer1::PluginTensorDesc const& tensor_desc(nvinfer1::PluginTensorDesc const& value) noexcept
{
    return value;
}

nvinfer1::PluginTensorDesc const& tensor_desc(nvinfer1::DynamicPluginTensorDesc const& value) noexcept
{
    return value.desc;
}

bool matches_tensor(nvinfer1::PluginTensorDesc const& descriptor, TensorSpec const& spec) noexcept
{
    return descriptor.type == spec.type && descriptor.format == spec.format;
}

template <typename Descriptor, typename Validate>
std::int32_t validate_tensors(Descriptor const* inputs, std::int32_t input_count,
    Descriptor const* outputs, std::int32_t output_count, PluginIO const& io,
    Validate validate) noexcept
{
    if (inputs == nullptr || outputs == nullptr || input_count < 0 || output_count != 1 ||
        static_cast<std::size_t>(input_count) != io.inputs.size()) return -1;
    for (std::int32_t index = 0; index < input_count; ++index)
        if (!matches_tensor(tensor_desc(inputs[index]), io.inputs[index])) return -1;
    if (!matches_tensor(tensor_desc(outputs[0]), io.output)) return -1;
    try {
        std::vector<nvinfer1::Dims> dims;
        dims.reserve(io.inputs.size());
        for (std::int32_t index = 0; index < input_count; ++index)
            dims.push_back(tensor_desc(inputs[index]).dims);
        return validate(dims.data(), input_count, tensor_desc(outputs[0]).dims);
    } catch (...) {
        return -1;
    }
}

std::size_t field_size(nvinfer1::PluginFieldType type) noexcept
{
    using nvinfer1::PluginFieldType;
    switch (type) {
    case PluginFieldType::kFLOAT32:
    case PluginFieldType::kINT32: return 4;
    case PluginFieldType::kFLOAT16:
    case PluginFieldType::kINT16: return 2;
    case PluginFieldType::kINT8:
    case PluginFieldType::kCHAR: return 1;
    default: return 0;
    }
}

std::uint32_t rotate_right(std::uint32_t value, unsigned int count) noexcept
{
    return (value >> count) | (value << (32U - count));
}

template <typename Handle, typename Release>
void release_in_context(CUcontext context, Handle handle, Release release) noexcept
{
    if (handle == 0 || context == nullptr) return;
    if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
        release(handle);
        CUcontext previous{};
        static_cast<void>(cuCtxPopCurrent(&previous));
    }
    // A destroyed CUDA context has already released its allocations and modules.
}

}  // namespace

bool Fields::parse(nvinfer1::PluginFieldCollection const* source, Fields& result) noexcept
{
    if (source == nullptr || source->nbFields < 0 ||
        (source->nbFields != 0 && source->fields == nullptr)) return false;
    try {
        Fields parsed;
        parsed.values_.reserve(static_cast<std::size_t>(source->nbFields));
        for (std::int32_t index = 0; index < source->nbFields; ++index) {
            auto const& field = source->fields[index];
            std::size_t const item_size = field_size(field.type);
            if (field.name == nullptr || item_size == 0 || field.length < 0 ||
                (field.length != 0 && field.data == nullptr)) return false;
            if (parsed.find(field.name) != nullptr) return false;
            std::size_t const length = static_cast<std::size_t>(field.length) * item_size;
            if (length > 128U * 1024U * 1024U) return false;
            auto const* begin = static_cast<std::uint8_t const*>(field.data);
            FieldValue value{field.name, field.type, field.length, {}};
            if (length != 0) value.bytes.assign(begin, begin + length);
            parsed.values_.push_back(std::move(value));
        }
        parsed.rebuild();
        result = std::move(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

Fields::Fields(Fields const& other) : values_(other.values_) { rebuild(); }
Fields& Fields::operator=(Fields const& other)
{
    if (this != &other) { values_ = other.values_; rebuild(); }
    return *this;
}
Fields::Fields(Fields&& other) : values_(std::move(other.values_)) { rebuild(); }
Fields& Fields::operator=(Fields&& other)
{
    if (this != &other) { values_ = std::move(other.values_); rebuild(); }
    return *this;
}

FieldValue const* Fields::find(char const* name) const noexcept
{
    auto found = std::find_if(values_.begin(), values_.end(),
        [name](FieldValue const& value) { return value.name == name; });
    return found == values_.end() ? nullptr : &*found;
}

void Fields::rebuild()
{
    fields_.clear();
    fields_.reserve(values_.size());
    for (auto& value : values_)
        fields_.emplace_back(value.name.c_str(), value.bytes.data(), value.type, value.length);
    collection_.nbFields = static_cast<std::int32_t>(fields_.size());
    collection_.fields = fields_.data();
}

nvinfer1::PluginFieldCollection const* Fields::collection() noexcept { return &collection_; }

PluginBase::PluginBase(char const* name, char const* plugin_namespace, Fields fields, PluginIO io)
    : name_(name), namespace_(plugin_namespace), fields_(std::move(fields)), io_(std::move(io))
{
    if (io_.inputs.empty() || io_.inputs.size() > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()) || io_.shape_input_index < 0 ||
        static_cast<std::size_t>(io_.shape_input_index) >= io_.inputs.size())
        throw std::invalid_argument("PluginIO requires inputs and a valid shape_input_index");
}

nvinfer1::IPluginCapability* PluginBase::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    using nvinfer1::PluginCapabilityType;
    if (type == PluginCapabilityType::kCORE) return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    if (type == PluginCapabilityType::kBUILD) return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME) return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    return nullptr;
}

char const* PluginBase::getPluginName() const noexcept { return name_; }
char const* PluginBase::getPluginVersion() const noexcept { return "1"; }
char const* PluginBase::getPluginNamespace() const noexcept { return namespace_; }
std::int32_t PluginBase::getNbOutputs() const noexcept { return 1; }

bool PluginBase::supportsFormatCombination(std::int32_t position,
    nvinfer1::DynamicPluginTensorDesc const* in_out,
    std::int32_t input_count, std::int32_t output_count) noexcept
{
    if (in_out == nullptr || input_count < 0 ||
        static_cast<std::size_t>(input_count) != io_.inputs.size() ||
        output_count != 1 || position < 0 || position > input_count)
        return false;
    auto const& spec = position < input_count ? io_.inputs[position] : io_.output;
    return matches_tensor(in_out[position].desc, spec);
}

std::int32_t PluginBase::getOutputDataTypes(nvinfer1::DataType* output_types,
    std::int32_t output_count, nvinfer1::DataType const* input_types,
    std::int32_t input_count) const noexcept
{
    if (output_types == nullptr || input_types == nullptr || output_count != 1 || input_count < 0 ||
        static_cast<std::size_t>(input_count) != io_.inputs.size()) return -1;
    for (std::int32_t index = 0; index < input_count; ++index)
        if (input_types[index] != io_.inputs[index].type) return -1;
    output_types[0] = io_.output.type;
    return 0;
}

std::int32_t PluginBase::getOutputShapes(nvinfer1::DimsExprs const* inputs,
    std::int32_t input_count, nvinfer1::DimsExprs const*, std::int32_t shape_input_count,
    nvinfer1::DimsExprs* outputs, std::int32_t output_count,
    nvinfer1::IExprBuilder&) noexcept
{
    if (inputs == nullptr || outputs == nullptr || input_count < 0 ||
        static_cast<std::size_t>(input_count) != io_.inputs.size() ||
        shape_input_count != 0 || output_count != 1) return -1;
    outputs[0] = inputs[io_.shape_input_index];
    return 0;
}

std::int32_t PluginBase::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
    std::int32_t input_count, nvinfer1::DynamicPluginTensorDesc const* outputs,
    std::int32_t output_count) noexcept
{
    return validate_tensors(inputs, input_count, outputs, output_count, io_,
        [this](nvinfer1::Dims const* dims, std::int32_t count, nvinfer1::Dims const& output) {
            return validate_shape(dims, count, output);
        });
}

std::size_t PluginBase::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, std::int32_t,
    nvinfer1::DynamicPluginTensorDesc const*, std::int32_t) const noexcept { return 0; }
char const* PluginBase::getTimingCacheID() noexcept { return name_; }
char const* PluginBase::getMetadataString() noexcept { return name_; }

std::int32_t PluginBase::onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
    std::int32_t input_count, nvinfer1::PluginTensorDesc const* outputs,
    std::int32_t output_count) noexcept
{
    return validate_tensors(inputs, input_count, outputs, output_count, io_,
        [this](nvinfer1::Dims const* dims, std::int32_t count, nvinfer1::Dims const& output) {
            return validate_shape(dims, count, output);
        });
}

nvinfer1::IPluginV3* PluginBase::attachToContext(nvinfer1::IPluginResourceContext*) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* PluginBase::getFieldsToSerialize() noexcept
{
    return fields_.collection();
}

std::string sha256(void const* bytes, std::size_t size)
{
    // SHA-256 is kept local so a caller's native plugin has no crypto dependency.
    constexpr std::array<std::uint32_t, 64> constants{{
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
    }};
    std::array<std::uint32_t, 8> state{{
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U,
    }};
    auto const* begin = static_cast<std::uint8_t const*>(bytes);
    std::vector<std::uint8_t> message(begin, begin + size);
    message.push_back(0x80U);
    while (message.size() % 64U != 56U) message.push_back(0U);
    auto const bit_count = static_cast<std::uint64_t>(size) * 8U;
    for (int shift = 56; shift >= 0; shift -= 8)
        message.push_back(static_cast<std::uint8_t>((bit_count >> shift) & 0xffU));
    for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            auto const position = offset + index * 4U;
            words[index] = (static_cast<std::uint32_t>(message[position]) << 24U) |
                (static_cast<std::uint32_t>(message[position + 1U]) << 16U) |
                (static_cast<std::uint32_t>(message[position + 2U]) << 8U) |
                message[position + 3U];
        }
        for (std::size_t index = 16; index < 64; ++index) {
            auto const a = words[index - 15U];
            auto const b = words[index - 2U];
            words[index] = words[index - 16U] +
                (rotate_right(a, 7U) ^ rotate_right(a, 18U) ^ (a >> 3U)) +
                words[index - 7U] +
                (rotate_right(b, 17U) ^ rotate_right(b, 19U) ^ (b >> 10U));
        }
        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];
        for (std::size_t index = 0; index < 64; ++index) {
            auto const s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            auto const choice = (e & f) ^ (~e & g);
            auto const t1 = h + s1 + choice + constants[index] + words[index];
            auto const s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            auto const majority = (a & b) ^ (a & c) ^ (b & c);
            auto const t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result(64U, '0');
    for (std::size_t word = 0; word < 8; ++word)
        for (std::size_t nibble = 0; nibble < 8; ++nibble)
            result[word * 8U + nibble] = hex[(state[word] >> (4U * (7U - nibble))) & 15U];
    return result;
}

Kernel::~Kernel() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto const& [context, loaded] : loaded_)
        release_in_context(context, loaded.module, [](CUmodule module) { static_cast<void>(cuModuleUnload(module)); });
}

bool Kernel::accepts(void* const* arguments, std::size_t argument_count) const noexcept
{
    if (arguments == nullptr || image_.argument_rules == nullptr || argument_count == 0 ||
        argument_count > 14 || argument_count != image_.public_argument_count) return false;
    for (std::size_t index = 0; index < argument_count; ++index) {
        if (arguments[index] == nullptr) return false;
        ArgumentRule const& rule = image_.argument_rules[index];
        std::uint32_t const divisor = rule.divisible_by;
        if (divisor == 0 || (divisor & (divisor - 1U)) != 0) return false;
        switch (rule.kind) {
        case ArgumentKind::kPointer: {
            CUdeviceptr value{};
            std::memcpy(&value, arguments[index], sizeof(value));
            if (divisor > 1 && (value == 0 || value % divisor != 0)) return false;
            break;
        }
        case ArgumentKind::kInt32: {
            std::int32_t value{};
            std::memcpy(&value, arguments[index], sizeof(value));
            if (divisor > 1 && value % static_cast<std::int32_t>(divisor) != 0) return false;
            break;
        }
        case ArgumentKind::kFloat32:
            if (divisor != 1) return false;
            break;
        default: return false;
        }
    }
    return true;
}

CUresult Kernel::function_for_current_context(CUfunction& function) noexcept
{
    CUcontext context{};
    auto status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS || context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = loaded_.find(context);
    if (found != loaded_.end()) { function = found->second.function; return CUDA_SUCCESS; }
    if (image_.bytes == nullptr || image_.size == 0 || image_.sha256 == nullptr ||
        image_.entry_symbol == nullptr || image_.global_scratch_bytes != 0 ||
        image_.profile_scratch_bytes != 0) return CUDA_ERROR_INVALID_VALUE;
    try {
        if (sha256(image_.bytes, image_.size) != image_.sha256) return CUDA_ERROR_INVALID_IMAGE;
    } catch (...) { return CUDA_ERROR_OUT_OF_MEMORY; }
    CUdevice device{};
    int major{}, minor{};
    status = cuCtxGetDevice(&device);
    if (status == CUDA_SUCCESS)
        status = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (status == CUDA_SUCCESS)
        status = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (status != CUDA_SUCCESS) return status;
    if (major * 10 + minor != image_.target_sm) return CUDA_ERROR_INVALID_IMAGE;
    Loaded loaded{};
    status = cuModuleLoadData(&loaded.module, image_.bytes);
    if (status == CUDA_SUCCESS)
        status = cuModuleGetFunction(&loaded.function, loaded.module, image_.entry_symbol);
    if (status == CUDA_SUCCESS && image_.dynamic_shared_bytes > 0)
        status = cuFuncSetAttribute(loaded.function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, image_.dynamic_shared_bytes);
    int registers{};
    if (status == CUDA_SUCCESS)
        status = cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, loaded.function);
    if (status == CUDA_SUCCESS && registers != image_.registers_per_thread)
        status = CUDA_ERROR_INVALID_IMAGE;
    if (status != CUDA_SUCCESS) {
        if (loaded.module != nullptr) static_cast<void>(cuModuleUnload(loaded.module));
        return status;
    }
    function = loaded.function;
    try { loaded_.emplace(context, loaded); }
    catch (...) { static_cast<void>(cuModuleUnload(loaded.module)); return CUDA_ERROR_OUT_OF_MEMORY; }
    return CUDA_SUCCESS;
}

CUresult Kernel::launch(unsigned int grid, unsigned int block,
    void** arguments, std::size_t argument_count, CUstream stream) noexcept
{
    if (grid == 0 || block == 0 || !accepts(arguments, argument_count))
        return CUDA_ERROR_INVALID_VALUE;
    CUfunction function{};
    auto const status = function_for_current_context(function);
    if (status != CUDA_SUCCESS) return status;
    CUdeviceptr global_scratch{}, profile_scratch{};
    std::array<void*, 16> parameters{};
    std::copy(arguments, arguments + argument_count, parameters.begin());
    parameters[argument_count] = &global_scratch;
    parameters[argument_count + 1] = &profile_scratch;
    return cuLaunchKernel(function, grid, 1U, 1U, block, 1U, 1U,
        static_cast<unsigned int>(image_.dynamic_shared_bytes), stream, parameters.data(), nullptr);
}

DeviceBytes::~DeviceBytes() noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto const& [context, pointer] : pointers_)
        release_in_context(context, pointer, [](CUdeviceptr value) { static_cast<void>(cuMemFree(value)); });
}

CUresult DeviceBytes::for_current_context(CUdeviceptr& pointer) noexcept
{
    CUcontext context{};
    auto status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS || context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
    std::lock_guard<std::mutex> guard(mutex_);
    auto found = pointers_.find(context);
    if (found != pointers_.end()) { pointer = found->second; return CUDA_SUCCESS; }
    if (bytes_.empty()) return CUDA_ERROR_INVALID_VALUE;
    CUdeviceptr allocated{};
    status = cuMemAlloc(&allocated, bytes_.size());
    if (status != CUDA_SUCCESS) return status;
    status = cuMemcpyHtoD(allocated, bytes_.data(), bytes_.size());
    if (status != CUDA_SUCCESS) { static_cast<void>(cuMemFree(allocated)); return status; }
    try { pointers_.emplace(context, allocated); }
    catch (...) { static_cast<void>(cuMemFree(allocated)); return CUDA_ERROR_OUT_OF_MEMORY; }
    pointer = allocated;
    return CUDA_SUCCESS;
}

}  // namespace tensoracle::cubin
