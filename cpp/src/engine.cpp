#include "engine_internal.hpp"
#include "error.hpp"
#include "structure.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}
void succeeded(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool same(const nvinfer1::Dims& a, const nvinfer1::Dims& b) {
    return a.nbDims == b.nbDims && std::equal(a.d, a.d + a.nbDims, b.d);
}
void check_tensor(const tensoracle_execution* execution, const char* name) {
    require(execution && name, "execution and tensor name are required");
    require(std::find(execution->names.begin(), execution->names.end(), name) != execution->names.end(),
            std::string("unknown engine tensor: ") + name);
}
void infer(tensoracle_execution& execution, bool creating) {
    succeeded(execution.context->inferShapes(0, nullptr) == 0, "TensorRT shape inference failed");
    for (const auto& name : execution.names) {
        auto shape = execution.context->getTensorShape(name.c_str());
        bool resolved = shape.nbDims >= 0 && std::none_of(shape.d, shape.d + shape.nbDims,
                                                       [](auto dimension) { return dimension < 0; });
        if (creating) require(resolved, "tensor has data-dependent shape: " + name);
        else succeeded(resolved, "tensor shape is unresolved: " + name);
    }
    execution.ready = true;
}

using tensoracle::detail::write_field;
using tensoracle::detail::structure_size;

std::unique_ptr<tensoracle_execution> create_execution(
    const tensoracle_engine* engine, const tensoracle_scratch_memory* scratch) {
    auto result = std::make_unique<tensoracle_execution>();
    result->state = engine->state;
    auto& native = *engine->state->engine;
    for (int32_t i = 0; i < native.getNbIOTensors(); ++i) {
        const char* name = native.getIOTensorName(i);
        result->names.emplace_back(name);
        require(!(native.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT
                  && native.isShapeInferenceIO(name)),
                std::string("shape inference output is not supported: ") + name);
        require(native.getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE,
                std::string("tensor has HOST location: ") + name);
        if (native.getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        require(!native.isShapeInferenceIO(name), std::string("external shape tensor input: ") + name);
        auto shape = native.getTensorShape(name);
        tensoracle_execution::input input{name, shape, shape, shape, false};
        if (std::find(shape.d, shape.d + shape.nbDims, -1) != shape.d + shape.nbDims) {
            input.low = native.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
            input.optimum = native.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
            input.high = native.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            input.ranged = !same(input.low, input.high);
        }
        result->inputs.push_back(input);
    }
    int64_t required = 0;
    if (scratch) {
        require(structure_size(scratch) >= offsetof(tensoracle_scratch_memory, size) + sizeof(scratch->size),
                "scratch memory requires all known fields");
        require(scratch->address, "scratch device address must not be null");
        required = native.getDeviceMemorySizeV2();
        succeeded(required >= 0, "TensorRT returned an invalid scratch size");
        require(scratch->size >= static_cast<size_t>(required), "scratch device memory is too small");
    }
    result->context.reset(scratch
        ? native.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED)
        : native.createExecutionContext());
    succeeded(bool(result->context), "TensorRT failed to create an execution context");
    if (scratch) result->context->setDeviceMemoryV2(scratch->address, required);
    for (const auto& input : result->inputs)
        succeeded(result->context->setInputShape(input.name.c_str(), input.optimum),
                  "TensorRT rejected initial shape: " + input.name);
    infer(*result, true);
    result->ready = std::none_of(result->inputs.begin(), result->inputs.end(),
                                 [](const auto& input) { return input.ranged; });
    return result;
}
}

extern "C" tensoracle_status tensoracle_engine_destroy(tensoracle_engine* engine) noexcept {
    delete engine;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_engine_tensor_count(
    const tensoracle_engine* engine, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        require(engine && count, "engine and tensor count output are required");
        *count = engine->state->engine->getNbIOTensors();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_engine_tensor_name(
    const tensoracle_engine* engine, size_t index, const char** name, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (name) *name = nullptr;
        require(engine && name, "engine and tensor name output are required");
        require(index < static_cast<size_t>(engine->state->engine->getNbIOTensors()), "tensor index out of range");
        *name = engine->state->engine->getIOTensorName(static_cast<int32_t>(index));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_create(
    const tensoracle_engine* engine, tensoracle_execution** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(engine && output, "engine and execution output are required");
        *output = create_execution(engine, nullptr).release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_engine_scratch_size(
    const tensoracle_engine* engine, size_t* size, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (size) *size = 0;
        require(engine && size, "engine and scratch size output are required");
        auto required = engine->state->engine->getDeviceMemorySizeV2();
        succeeded(required >= 0, "TensorRT returned an invalid scratch size");
        *size = static_cast<size_t>(required);
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_create_with_scratch(
    const tensoracle_engine* engine, const tensoracle_scratch_memory* scratch,
    tensoracle_execution** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(engine && scratch && output, "engine, scratch memory and execution output are required");
        *output = create_execution(engine, scratch).release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_destroy(tensoracle_execution* execution) noexcept {
    delete execution;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_execution_set_input_shapes(
    tensoracle_execution* execution, const tensoracle_input_shape* const* shapes,
    size_t count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(execution && (!count || shapes), "execution and input shape array are required");
        auto expected = std::count_if(execution->inputs.begin(), execution->inputs.end(),
                                      [](const auto& input) { return input.ranged; });
        require(count == static_cast<size_t>(expected), "input shape names must match every ranged input");
        std::set<std::string> seen;
        for (size_t i = 0; i < count; ++i) {
            const auto* shape = shapes[i];
            require(shape && structure_size(shape) >= offsetof(tensoracle_input_shape, dimensions) + sizeof(shape->dimensions),
                    "input shape requires all known fields");
            require(shape->name && seen.emplace(shape->name).second, "missing or duplicate input shape name");
            auto input = std::find_if(execution->inputs.begin(), execution->inputs.end(),
                                      [&](const auto& item) { return item.ranged && item.name == shape->name; });
            require(input != execution->inputs.end(), std::string("unexpected input shape: ") + shape->name);
            require(shape->rank == input->low.nbDims && shape->dimensions, "input shape rank differs: " + input->name);
            for (int32_t j = 0; j < shape->rank; ++j)
                require(shape->dimensions[j] >= input->low.d[j] && shape->dimensions[j] <= input->high.d[j],
                        "input shape is outside profile range: " + input->name);
        }
        execution->ready = false;
        for (size_t i = 0; i < count; ++i) {
            nvinfer1::Dims dimensions{};
            dimensions.nbDims = shapes[i]->rank;
            std::copy_n(shapes[i]->dimensions, dimensions.nbDims, dimensions.d);
            succeeded(execution->context->setInputShape(shapes[i]->name, dimensions),
                      std::string("TensorRT rejected input shape: ") + shapes[i]->name);
        }
        infer(*execution, false);
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_tensor_info(
    tensoracle_execution* execution, const char* name, tensoracle_tensor_info* info,
    tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        check_tensor(execution, name);
        require(info && structure_size(info) >= sizeof(size_t), "tensor info requires struct_size");
        succeeded(execution->ready, "input shapes need a complete successful setting");
        auto shape = execution->context->getTensorShape(name);
        succeeded(shape.nbDims >= 0 && std::none_of(shape.d, shape.d + shape.nbDims,
                      [](auto value) { return value < 0; }), "tensor shape is unresolved");
        auto type = tensoracle::detail::public_dtype(execution->state->engine->getTensorDataType(name));
        auto size = structure_size(info);
        write_field(info, size, offsetof(tensoracle_tensor_info, dtype), type);
        write_field(info, size, offsetof(tensoracle_tensor_info, rank), shape.nbDims);
        for (int32_t i = 0; i < shape.nbDims; ++i)
            write_field(info, size, offsetof(tensoracle_tensor_info, dimensions) + i * sizeof(int64_t), shape.d[i]);
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_tensor_dtype(
    const tensoracle_execution* execution, const char* name, tensoracle_dtype* dtype,
    tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        check_tensor(execution, name);
        require(dtype, "tensor dtype output is required");
        *dtype = tensoracle::detail::public_dtype(execution->state->engine->getTensorDataType(name));
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_execution_enqueue(
    tensoracle_execution* execution, const tensoracle_tensor_address* const* addresses,
    size_t count, tensoracle_cuda_stream stream, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(execution && (!count || addresses), "execution and address array are required");
        require(count == execution->names.size(), "addresses must name every engine IO tensor");
        std::set<std::string> seen;
        for (size_t i = 0; i < count; ++i) {
            auto* address = addresses[i];
            require(address && structure_size(address) >= offsetof(tensoracle_tensor_address, address) + sizeof(address->address),
                    "tensor address requires all known fields");
            check_tensor(execution, address->name);
            require(address->address && seen.emplace(address->name).second,
                    std::string("null or duplicate tensor address: ") + address->name);
        }
        succeeded(execution->ready, "input shapes need a complete successful setting");
        for (size_t i = 0; i < count; ++i)
            succeeded(execution->context->setTensorAddress(addresses[i]->name, addresses[i]->address),
                      std::string("TensorRT rejected tensor address: ") + addresses[i]->name);
        succeeded(execution->context->enqueueV3(stream), "TensorRT enqueue failed");
        return TENSORACLE_SUCCESS;
    });
}
