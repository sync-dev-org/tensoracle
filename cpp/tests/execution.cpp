#include <tensoracle/engine.hpp>
#include "engine_internal.hpp"
#include "cache.hpp"
#include <NvInfer.h>
#include <cuda.h>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include <fstream>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <cerrno>
#include <unistd.h>

namespace {
std::string denied_timestamp;
int denied_timestamp_calls = 0;
}
// Interpose the OS boundary used by libstdc++'s last_write_time, including calls
// originating in shared libraries. All unrelated filesystem operations stay real.
extern "C" int utimensat(int directory, const char* path, const timespec times[2], int flags) noexcept {
    if (path && path == denied_timestamp) {
        ++denied_timestamp_calls; errno = EACCES; return -1;
    }
    return int(syscall(SYS_utimensat, directory, path, times, flags));
}

namespace {
struct logger : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "TensorRT: %s\n", message);
    }
};
logger log_sink;
template<class T> using trt_ptr = std::unique_ptr<T>;

void cuda_check(CUresult result) { REQUIRE(result == CUDA_SUCCESS); }
struct device_context {
    CUcontext value{};
    device_context() {
        auto version = tensoracle::tensorrt_version();
        std::fprintf(stderr, "Loaded TensorRT %u.%u.%u\n", version.major, version.minor, version.patch);
        REQUIRE(version.major == 11);
        REQUIRE(version.minor >= 2);
        cuda_check(cuInit(0));
        CUdevice device;
        cuda_check(cuDeviceGet(&device, 0));
        cuda_check(cuDevicePrimaryCtxRetain(&value, device));
        cuda_check(cuCtxPushCurrent(value));
    }
    ~device_context() {
        CUcontext popped;
        cuCtxPopCurrent(&popped);
        cuDevicePrimaryCtxRelease(0);
    }
};

std::vector<uint8_t> sum_plan(bool ranged = false, bool point = false, bool nonzero = false,
                              int shape_output = 0) {
    trt_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(log_sink));
    REQUIRE(builder);
    trt_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    trt_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setBuilderOptimizationLevel(0);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64 * 1024 * 1024);
    auto dims = nvinfer1::Dims2(ranged || point ? -1 : 3, 2);
    auto* left = network->addInput("left", nvinfer1::DataType::kFLOAT, dims);
    auto* right = network->addInput("right", nvinfer1::DataType::kFLOAT, dims);
    auto* sum = network->addElementWise(*left, *right, nvinfer1::ElementWiseOperation::kSUM);
    auto* output = nonzero ? network->addNonZero(*sum->getOutput(0))->getOutput(0) : sum->getOutput(0);
    output->setName("result");
    network->markOutput(*output);
    if (shape_output) {
        auto* shape = network->addShape(*left)->getOutput(0);
        shape->setName("dimensions");
        if (shape_output == 1) REQUIRE(network->markOutputForShapes(*shape));
        else network->markOutput(*shape);
    }
    if (ranged || point) {
        auto* profile = builder->createOptimizationProfile();
        for (auto* name : {"left", "right"}) {
            REQUIRE(profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN,
                                          nvinfer1::Dims2(point ? 3 : 1, 2)));
            REQUIRE(profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT,
                                          nvinfer1::Dims2(3, 2)));
            REQUIRE(profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX,
                                          nvinfer1::Dims2(point ? 3 : 5, 2)));
        }
        REQUIRE(config->addOptimizationProfile(profile) >= 0);
    }
    trt_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    REQUIRE(plan);
    const auto* bytes = static_cast<const uint8_t*>(plan->data());
    return {bytes, bytes + plan->size()};
}

struct buffers {
    CUstream stream{};
    std::array<CUdeviceptr, 3> pointers{};
    buffers() {
        cuda_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        for (auto& pointer : pointers) cuda_check(cuMemAlloc(&pointer, 10 * sizeof(float)));
    }
    ~buffers() {
        cuStreamSynchronize(stream);
        for (auto pointer : pointers) cuMemFree(pointer);
        cuStreamDestroy(stream);
    }
    std::vector<tensoracle::tensor_address> addresses() const {
        return {{"left", reinterpret_cast<void*>(pointers[0])},
                {"right", reinterpret_cast<void*>(pointers[1])},
                {"result", reinterpret_cast<void*>(pointers[2])}};
    }
    void inputs(int rows, float offset) {
        std::vector<float> left(rows * 2), right(rows * 2, -0.25f);
        for (int i = 0; i < rows * 2; ++i) left[i] = i / 4.0f + offset;
        cuda_check(cuMemcpyHtoD(pointers[0], left.data(), left.size() * sizeof(float)));
        cuda_check(cuMemcpyHtoD(pointers[1], right.data(), right.size() * sizeof(float)));
    }
    void check(int rows, float offset) {
        std::vector<float> result(rows * 2);
        cuda_check(cuStreamSynchronize(stream));
        cuda_check(cuMemcpyDtoH(result.data(), pointers[2], result.size() * sizeof(float)));
        for (int i = 0; i < rows * 2; ++i) REQUIRE(result[i] == i / 4.0f + offset - 0.25f);
    }
};

// Two matrix products separated by sin require intermediate device storage;
// an elementwise-only engine may need no scratch at all.
constexpr int matrix_width = 512;
std::vector<uint8_t> scratch_plan() {
    trt_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(log_sink));
    REQUIRE(builder);
    trt_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    trt_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setBuilderOptimizationLevel(0);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64 * 1024 * 1024);
    auto* left = network->addInput("left", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(-1, matrix_width));
    auto* right = network->addInput("right", nvinfer1::DataType::kFLOAT,
                                  nvinfer1::Dims2(matrix_width, matrix_width));
    auto* product = network->addMatrixMultiply(*left, nvinfer1::MatrixOperation::kNONE,
                                             *right, nvinfer1::MatrixOperation::kNONE);
    auto* sine = network->addUnary(*product->getOutput(0), nvinfer1::UnaryOperation::kSIN);
    auto* output = network->addMatrixMultiply(*sine->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                            *right, nvinfer1::MatrixOperation::kNONE)->getOutput(0);
    output->setName("result");
    network->markOutput(*output);
    auto* profile = builder->createOptimizationProfile();
    REQUIRE(profile->setDimensions("left", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, matrix_width)));
    REQUIRE(profile->setDimensions("left", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(16, matrix_width)));
    REQUIRE(profile->setDimensions("left", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(matrix_width, matrix_width)));
    REQUIRE(config->addOptimizationProfile(profile) >= 0);
    trt_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    REQUIRE(plan);
    const auto* bytes = static_cast<const uint8_t*>(plan->data());
    return {bytes, bytes + plan->size()};
}

struct device_memory {
    CUdeviceptr pointer{};
    explicit device_memory(size_t size) { cuda_check(cuMemAlloc(&pointer, size ? size : 1)); }
    ~device_memory() { cuMemFree(pointer); }
    void* address() const { return reinterpret_cast<void*>(pointer); }
};

struct matrix_buffers {
    CUstream stream{};
    device_memory left{matrix_width * matrix_width * sizeof(float)};
    device_memory right{matrix_width * matrix_width * sizeof(float)};
    device_memory result{matrix_width * matrix_width * sizeof(float)};
    matrix_buffers() {
        cuda_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        std::vector<float> identity(matrix_width * matrix_width, 0);
        for (int i = 0; i < matrix_width; ++i) identity[i * matrix_width + i] = 1;
        cuda_check(cuMemcpyHtoD(right.pointer, identity.data(), identity.size() * sizeof(float)));
        cuda_check(cuCtxSynchronize());
    }
    ~matrix_buffers() { cuStreamSynchronize(stream); cuStreamDestroy(stream); }
    std::vector<tensoracle::tensor_address> addresses() const {
        return {{"left", left.address()}, {"right", right.address()}, {"result", result.address()}};
    }
    void inputs(int rows, float offset) {
        std::vector<float> values(rows * matrix_width);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(i % 31) / 32 + offset - 0.5f;
        cuda_check(cuMemcpyHtoD(left.pointer, values.data(), values.size() * sizeof(float)));
        cuda_check(cuCtxSynchronize());
    }
    std::vector<float> read(int rows, float offset) {
        cuda_check(cuStreamSynchronize(stream));
        std::vector<float> values(rows * matrix_width);
        cuda_check(cuMemcpyDtoH(values.data(), result.pointer, values.size() * sizeof(float)));
        for (size_t i = 0; i < values.size(); ++i) {
            auto expected = std::sin(float(i % 31) / 32 + offset - 0.5f);
            if (!(std::abs(values[i] - expected) < 1e-5f)) {
                INFO("rows=" << rows << " offset=" << offset << " index=" << i
                     << " observed=" << values[i] << " expected=" << expected);
                FAIL("matrix output differs from sin(left) with identity right input");
            }
        }
        return values;
    }
};
}

TEST_CASE("Null execution arguments fail without initializing CUDA", "[SP-TENSORACLE-012][SP-TENSORACLE-056]") {
    tensoracle_error* error = nullptr;
    tensoracle_execution* execution = reinterpret_cast<tensoracle_execution*>(1);
    REQUIRE(tensoracle_execution_create(nullptr, &execution, &error) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(execution == nullptr);
    REQUIRE(error);
    tensoracle_error_destroy(error);
    REQUIRE(tensoracle_execution_destroy(nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_engine_destroy(nullptr) == TENSORACLE_SUCCESS);
    REQUIRE_THROWS_AS(tensoracle::detail::deserialize_engine({}), std::invalid_argument);
    REQUIRE(tensoracle_execution_enqueue(nullptr, nullptr, 0, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
}

TEST_CASE("Scratch operations reject missing handles and clear outputs without CUDA", "[SP-TENSORACLE-068][SP-TENSORACLE-056]") {
    size_t size = 123;
    tensoracle_error* diagnostic = nullptr;
    REQUIRE(tensoracle_engine_scratch_size(nullptr, &size, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle::error_handle error(diagnostic);
    REQUIRE(error);
    REQUIRE(size == 0);
    REQUIRE(tensoracle_engine_scratch_size(nullptr, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle_execution* execution = reinterpret_cast<tensoracle_execution*>(1);
    REQUIRE(tensoracle_execution_create_with_scratch(nullptr, nullptr, &execution, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(execution == nullptr);
    tensoracle::engine_handle empty;
    REQUIRE_THROWS_AS(empty.scratch_size(), tensoracle::error);
    REQUIRE_THROWS_AS(empty.create_execution(nullptr, 0), tensoracle::error);
}

TEST_CASE("Engine scratch size is positive and stable across queries and executions", "[SP-TENSORACLE-068][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(scratch_plan());
    auto size = engine.scratch_size();
    REQUIRE(size > 0);
    size_t queried = 0;
    REQUIRE(tensoracle_engine_scratch_size(engine.get(), &queried, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(queried == size);
    REQUIRE(tensoracle_engine_scratch_size(engine.get(), nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    auto execution = engine.create_execution();
    execution.set_input_shapes({{"left", {matrix_width, matrix_width}}});
    REQUIRE(engine.scratch_size() == size);
    REQUIRE(engine.scratch_size() == size);
}

TEST_CASE("Executions reject missing or undersized caller scratch without returning a handle", "[SP-TENSORACLE-068][SP-TENSORACLE-056][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(scratch_plan());
    auto size = engine.scratch_size();
    REQUIRE(size > 0);
    device_memory memory(size);
    tensoracle_scratch_memory scratch{sizeof(scratch), memory.address(), size};
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto value = scratch;
        if (invalid == 0) value.address = nullptr;
        if (invalid == 1) value.size = size - 1;
        if (invalid == 2) value.struct_size = offsetof(tensoracle_scratch_memory, size);
        tensoracle_execution* execution = reinterpret_cast<tensoracle_execution*>(1);
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_create_with_scratch(engine.get(), invalid == 3 ? nullptr : &value,
                                                               &execution, &diagnostic);
        tensoracle::error_handle error(diagnostic);
        INFO("invalid scratch case " << invalid);
        REQUIRE(status == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(execution == nullptr);
        REQUIRE(error);
    }
    REQUIRE(tensoracle_execution_create_with_scratch(engine.get(), &scratch, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE_THROWS_AS(engine.create_execution(memory.address(), size - 1), tensoracle::error);
    REQUIRE_THROWS_AS(engine.create_execution(nullptr, size), tensoracle::error);
    // Unknown trailing input fields are ignored, as on the other C ABI inputs.
    struct extended_scratch { tensoracle_scratch_memory value; uint64_t future; } extended{scratch, UINT64_MAX};
    extended.value.struct_size = sizeof(extended);
    tensoracle_execution* execution = nullptr;
    REQUIRE(tensoracle_execution_create_with_scratch(engine.get(), &extended.value, &execution, nullptr) == TENSORACLE_SUCCESS);
    tensoracle::engine_execution owned(execution);
    REQUIRE(owned.get());
}

TEST_CASE("Two executions sharing caller scratch match ordinary execution on one stream", "[SP-TENSORACLE-068][SP-TENSORACLE-012][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(scratch_plan());
    auto size = engine.scratch_size();
    REQUIRE(size > 0);
    device_memory memory(size);
    auto ordinary = engine.create_execution();
    auto first = engine.create_execution(memory.address(), size);
    auto second = engine.create_execution(memory.address(), size);
    matrix_buffers values;
    for (auto* execution : {&first, &second}) {
        for (int rows : {16, 128, matrix_width}) {
            INFO("rows=" << rows << " execution=" << (execution == &first ? "first" : "second"));
            ordinary.set_input_shapes({{"left", {rows, matrix_width}}});
            execution->set_input_shapes({{"left", {rows, matrix_width}}});
            values.inputs(rows, execution == &first ? 0.0f : 0.25f);
            ordinary.enqueue(values.addresses(), values.stream);
            auto reference = values.read(rows, execution == &first ? 0.0f : 0.25f);
            INFO("caller-owned scratch execution");
            cuda_check(cuMemsetD8Async(values.result.pointer, 0xff, reference.size() * sizeof(float), values.stream));
            execution->enqueue(values.addresses(), values.stream);
            REQUIRE(values.read(rows, execution == &first ? 0.0f : 0.25f) == reference);
        }
    }
    // Both contexts retain the engine, and destroying one must not free scratch.
    engine = {};
    first.close();
    values.inputs(matrix_width, 0.5f);
    second.enqueue(values.addresses(), values.stream);
    values.read(matrix_width, 0.5f);
}

TEST_CASE("Sharing one scratch allocation uses less device memory than ordinary executions", "[SP-TENSORACLE-068][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(scratch_plan());
    auto size = engine.scratch_size();
    REQUIRE(size > 0);
    { auto warm = engine.create_execution(); }
    size_t ordinary_before{}, ordinary_after{}, shared_before{}, shared_after{}, total{};
    std::unique_ptr<device_memory> memory;
    cuda_check(cuCtxSynchronize());
    cuda_check(cuMemGetInfo(&ordinary_before, &total));
    std::vector<tensoracle::engine_execution> executions;
    constexpr size_t count = 8;
    for (size_t i = 0; i < count; ++i) executions.push_back(engine.create_execution());
    cuda_check(cuCtxSynchronize());
    cuda_check(cuMemGetInfo(&ordinary_after, &total));
    executions.clear();
    cuda_check(cuCtxSynchronize());
    cuda_check(cuMemGetInfo(&shared_before, &total));
    memory = std::make_unique<device_memory>(size);
    for (size_t i = 0; i < count; ++i) executions.push_back(engine.create_execution(memory->address(), size));
    cuda_check(cuCtxSynchronize());
    cuda_check(cuMemGetInfo(&shared_after, &total));
    auto ordinary_used = int64_t(ordinary_before) - int64_t(ordinary_after);
    auto shared_used = int64_t(shared_before) - int64_t(shared_after);
    std::fprintf(stderr, "scratch memory: count=%zu scratch_bytes=%zu ordinary_before=%zu ordinary_after=%zu "
                         "shared_before=%zu shared_after=%zu ordinary_used=%lld shared_used=%lld saved=%lld\n",
                 count, size, ordinary_before, ordinary_after, shared_before, shared_after,
                 static_cast<long long>(ordinary_used), static_cast<long long>(shared_used),
                 static_cast<long long>(ordinary_used - shared_used));
    REQUIRE(ordinary_used > 0);
    REQUIRE(shared_used < ordinary_used);
}

TEST_CASE("Enqueue with caller scratch captures and replays on a caller CUDA Graph", "[SP-TENSORACLE-068][SP-TENSORACLE-028][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(scratch_plan());
    auto size = engine.scratch_size();
    REQUIRE(size > 0);
    device_memory memory(size);
    auto execution = engine.create_execution(memory.address(), size);
    constexpr int rows = 16;
    execution.set_input_shapes({{"left", {rows, matrix_width}}});
    matrix_buffers values;
    values.inputs(rows, 0);
    execution.enqueue(values.addresses(), values.stream);
    auto reference = values.read(rows, 0);
    CUgraph graph{};
    CUgraphExec instance{};
    cuda_check(cuStreamBeginCapture(values.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    execution.enqueue(values.addresses(), values.stream);
    cuda_check(cuStreamEndCapture(values.stream, &graph));
    cuda_check(cuGraphInstantiate(&instance, graph, 0));
    for (float offset : {0.0f, 0.25f, 0.0f}) {
        values.inputs(rows, offset);
        cuda_check(cuMemsetD8Async(values.result.pointer, 0xff, rows * matrix_width * sizeof(float), values.stream));
        cuda_check(cuGraphLaunch(instance, values.stream));
        auto result = values.read(rows, offset);
        if (offset == 0) REQUIRE(result == reference);
    }
    cuda_check(cuGraphExecDestroy(instance));
    cuda_check(cuGraphDestroy(graph));
}

TEST_CASE("TensorRT-only dtypes stay outside CPU output comparison", "[SP-TENSORACLE-054][SP-TENSORACLE-056]") {
    const size_t shape[] = {1};
    uint64_t bytes = 0;
    for (auto dtype : {TENSORACLE_BFLOAT16, TENSORACLE_FLOAT8, TENSORACLE_INT4, TENSORACLE_FLOAT4, TENSORACLE_E8M0}) {
        tensoracle_array value{sizeof(value), dtype, 1, shape, &bytes};
        tensoracle_output_comparison result{sizeof(result)};
        REQUIRE(tensoracle_compare_outputs(&value, &value, nullptr, &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    }
}

TEST_CASE("Fixed and point profiles run on caller streams and retain their engine", "[SP-TENSORACLE-012][gpu]") {
    device_context context;
    for (bool point : {false, true}) {
        auto engine = tensoracle::detail::deserialize_engine(sum_plan(false, point));
        REQUIRE(engine.tensor_count() == 3);
        REQUIRE(engine.tensor_name(0) == "left");
        auto execution = engine.create_execution();
        engine = {};
        REQUIRE(execution.get_tensor_shape("result") == std::vector<int64_t>{3, 2});
        REQUIRE(execution.get_tensor_dtype("result") == TENSORACLE_FLOAT32);
        struct prefix { size_t struct_size; tensoracle_dtype dtype; unsigned char sentinel[8]; } info;
        std::memset(&info, 0xa5, sizeof(info));
        info.struct_size = offsetof(prefix, sentinel);
        REQUIRE(tensoracle_execution_tensor_info(execution.get(), "result",
            reinterpret_cast<tensoracle_tensor_info*>(&info), nullptr) == TENSORACLE_SUCCESS);
        REQUIRE(info.dtype == TENSORACLE_FLOAT32);
        for (auto byte : info.sentinel) REQUIRE(byte == 0xa5);
        REQUIRE_THROWS_AS(execution.get_tensor_dtype("typo"), tensoracle::error);
        REQUIRE_THROWS_AS(execution.set_input_shapes({{"left", {3, 2}}}), tensoracle::error);
        buffers values;
        values.inputs(3, 0);
        execution.enqueue(values.addresses(), values.stream);
        values.check(3, 0);
        values.inputs(3, 2);
        execution.enqueue(values.addresses(), nullptr);
        cuda_check(cuCtxSynchronize());
        values.check(3, 2);
        execution.close();
        execution.close();
        REQUIRE_THROWS_AS(execution.get_tensor_shape("result"), tensoracle::error);
    }
}

TEST_CASE("Ranged inputs require complete valid shapes and independent executions", "[SP-TENSORACLE-007][SP-TENSORACLE-012][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(sum_plan(true));
    auto first = engine.create_execution();
    auto second = engine.create_execution();
    REQUIRE_THROWS_AS(first.get_tensor_shape("result"), tensoracle::error);
    REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {3, 2}}}), tensoracle::error);
    REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {3, 2}}, {"right", {6, 2}}}), tensoracle::error);
    REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {3, 2}}, {"typo", {3, 2}}}), tensoracle::error);
    REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {3, 2}}, {"right", {3}}}), tensoracle::error);
    second.set_input_shapes({{"left", {5, 2}}, {"right", {5, 2}}});
    REQUIRE(first.get_tensor_dtype("left") == TENSORACLE_FLOAT32);
    REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {2, 2}}, {"right", {4, 2}}}), std::runtime_error);
    REQUIRE_THROWS_AS(first.get_tensor_shape("result"), std::runtime_error);
    buffers values;
    for (int rows : {1, 3, 5, 4}) {
        first.set_input_shapes({{"left", {rows, 2}}, {"right", {rows, 2}}});
        REQUIRE(first.get_tensor_shape("result") == std::vector<int64_t>{rows, 2});
        REQUIRE(second.get_tensor_shape("result") == std::vector<int64_t>{5, 2});
        // Validation must not invalidate the preceding successful setting.
        REQUIRE_THROWS_AS(first.set_input_shapes({{"left", {rows, 2}}, {"right", {0, 2}}}), tensoracle::error);
        values.inputs(rows, 1);
        auto addresses = values.addresses();
        addresses.pop_back();
        REQUIRE_THROWS_AS(first.enqueue(addresses, values.stream), tensoracle::error);
        addresses = values.addresses();
        addresses[2].address = nullptr;
        REQUIRE_THROWS_AS(first.enqueue(addresses, values.stream), tensoracle::error);
        first.enqueue(values.addresses(), values.stream);
        values.check(rows, 1);
    }
}

TEST_CASE("Data dependent outputs are rejected before enqueue", "[SP-TENSORACLE-012][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(sum_plan(false, false, true));
    REQUIRE_THROWS_AS(engine.create_execution(), tensoracle::error);
}

TEST_CASE("Shape inference outputs are rejected before creating an execution context", "[SP-TENSORACLE-012][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(sum_plan(true, false, false, 1));
    REQUIRE(engine.get()->state->engine->isShapeInferenceIO("dimensions"));
    tensoracle_execution* execution = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_execution_create(engine.get(), &execution, &diagnostic);
    tensoracle::unique_handle<tensoracle_execution, tensoracle_execution_destroy> cleanup(execution);
    tensoracle::error_handle error(diagnostic);
    REQUIRE(status == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(execution == nullptr);
    REQUIRE(error);
    auto ordinary = tensoracle::detail::deserialize_engine(sum_plan(true, false, false, 2));
    REQUIRE_FALSE(ordinary.get()->state->engine->isShapeInferenceIO("dimensions"));
    auto supported = ordinary.create_execution();
    supported.set_input_shapes({{"left", {3, 2}}, {"right", {3, 2}}});
    REQUIRE(supported.get_tensor_shape("dimensions") == std::vector<int64_t>{2});
}

TEST_CASE("One enqueue captures and replays with new inputs on caller graphs", "[SP-TENSORACLE-028][gpu]") {
    device_context context;
    auto engine = tensoracle::detail::deserialize_engine(sum_plan(true));
    for (int rows : {2, 5}) {
        auto execution = engine.create_execution();
        execution.set_input_shapes({{"left", {rows, 2}}, {"right", {rows, 2}}});
        buffers values;
        auto addresses = values.addresses();
        values.inputs(rows, 0);
        execution.enqueue(addresses, values.stream);
        values.check(rows, 0);
        CUgraph graph{};
        CUgraphExec instance{};
        cuda_check(cuStreamBeginCapture(values.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        execution.enqueue(addresses, values.stream);
        cuda_check(cuStreamEndCapture(values.stream, &graph));
        cuda_check(cuGraphInstantiate(&instance, graph, 0));
        for (float offset : {0.0f, 2.0f, 0.0f}) {
            values.inputs(rows, offset);
            cuda_check(cuMemsetD8Async(values.pointers[2], 0xff, rows * 2 * sizeof(float), values.stream));
            cuda_check(cuGraphLaunch(instance, values.stream));
            values.check(rows, offset);
        }
        cuGraphExecDestroy(instance);
        cuGraphDestroy(graph);
    }
}

TEST_CASE("Cached engines reuse across processes validate IO and touch only successful loads", "[SP-TENSORACLE-002][SP-TENSORACLE-009][SP-TENSORACLE-011][SP-TENSORACLE-012][SP-TENSORACLE-042][SP-TENSORACLE-048][SP-TENSORACLE-052][gpu]") {
    using namespace tensoracle::detail;
    namespace fs = std::filesystem;
    device_context context;
    char pattern[] = "/tmp/tensoracle-engine-XXXXXX";
    auto* directory = mkdtemp(pattern);
    REQUIRE(directory);
    struct cleanup {
        fs::path path;
        std::string old;
        bool existed;
        ~cleanup() {
            if (existed) setenv("XDG_CACHE_HOME", old.c_str(), 1); else unsetenv("XDG_CACHE_HOME");
            fs::remove_all(path);
        }
    } files{directory, std::getenv("XDG_CACHE_HOME") ? std::getenv("XDG_CACHE_HOME") : "", std::getenv("XDG_CACHE_HOME") != nullptr};
    setenv("XDG_CACHE_HOME", directory, 1);
    auto weights = files.path / "weights";
    std::ofstream(weights) << "weights";
    cache_request request;
    request.name = "sum"; request.graph_version = "1"; request.weights = {weights};
    request.inputs = {{"left", {3, 2}, {3, 2}, {3, 2}}, {"right", {3, 2}, {3, 2}, {3, 2}}};
    request.io = {{"left", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                  {"right", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                  {"result", TENSORACLE_FLOAT32, {3, 2}, false, {}, {}}};
    request.workspace_limit = 64 * 1024 * 1024;
    request.optimization_level = 0;
    int builds = 0;
    auto build = [&] { ++builds; return built_plan{sum_plan(), {}}; };
    auto first = open_cached_engine(request, build);
    REQUIRE(builds == 1);
    REQUIRE(fs::is_regular_file(first.path));
    auto old = fs::file_time_type::clock::now() - std::chrono::hours(48);
    fs::last_write_time(first.path, old);
    request.workspace_limit = 1;
    auto reused = open_cached_engine(request, build);
    REQUIRE(reused.path == first.path);
    REQUIRE(builds == 1);
    REQUIRE(fs::last_write_time(first.path) > old);
    auto execution = reused.engine.create_execution();
    buffers values;
    values.inputs(3, 2);
    execution.enqueue(values.addresses(), values.stream);
    values.check(3, 2);

    auto pid = fork();
    REQUIRE(pid >= 0);
    if (!pid) {
        execl(TENSORACLE_CACHE_PROBE, TENSORACLE_CACHE_PROBE, "--engine", weights.c_str(), first.path.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    fs::last_write_time(first.path, old);
    {
        struct reset { ~reset() { denied_timestamp.clear(); } } restore;
        denied_timestamp = first.path.string(); denied_timestamp_calls = 0;
        auto readonly = open_cached_engine(request, build);
        REQUIRE(denied_timestamp_calls == 1);
        REQUIRE(readonly.path == first.path);
        REQUIRE(fs::last_write_time(first.path) == old);
        auto run = readonly.engine.create_execution();
        values.inputs(3, 4); run.enqueue(values.addresses(), values.stream); values.check(3, 4);
        REQUIRE(builds == 1);
    }

    fs::last_write_time(first.path, old);
    for (int mismatch = 0; mismatch < 6; ++mismatch) {
        INFO("IO mismatch case " << mismatch);
        auto changed = request;
        if (mismatch == 0) changed.io[0].dtype = TENSORACLE_FLOAT16;
        if (mismatch == 1) changed.io[0].name = "typo";
        if (mismatch == 2) changed.io[0].shape = {4, 2};
        if (mismatch == 3) changed.io[0].input = false;
        if (mismatch == 4) changed.io[0].location = int32_t(nvinfer1::TensorLocation::kHOST);
        if (mismatch == 5) changed.io[0].format = int32_t(nvinfer1::TensorFormat::kCHW2);
        try {
            auto ignored = open_cached_engine(changed, build);
            FAIL("mismatching engine IO was accepted");
        } catch (const engine_io_error& failure) {
            auto message = std::string(failure.what());
            const char* fields[] = {"dtype", "name", "shape", "mode", "location", "format"};
            REQUIRE(message.find(mismatch == 1 ? "typo" : "left") != std::string::npos);
            REQUIRE(message.find(fields[mismatch]) != std::string::npos);
            REQUIRE(message.find("observed=") != std::string::npos);
            REQUIRE(message.find("expected=") != std::string::npos);
            REQUIRE(message.find(first.path.string()) != std::string::npos);
            if (mismatch == 0) {
                REQUIRE(message.find("observed=" + std::to_string(TENSORACLE_FLOAT32)) != std::string::npos);
                REQUIRE(message.find("expected=" + std::to_string(TENSORACLE_FLOAT16)) != std::string::npos);
            }
            if (mismatch == 1) {
                REQUIRE(message.find("observed=[\"left\",\"result\",\"right\"]") != std::string::npos);
                REQUIRE(message.find("expected=[\"result\",\"right\",\"typo\"]") != std::string::npos);
            }
            if (mismatch == 2) {
                REQUIRE(message.find("observed=[3,2]") != std::string::npos);
                REQUIRE(message.find("expected=[4,2]") != std::string::npos);
            }
            if (mismatch >= 3) {
                auto observed = mismatch == 3 ? int(nvinfer1::TensorIOMode::kINPUT) : 0;
                auto expected = mismatch == 3 ? int(nvinfer1::TensorIOMode::kOUTPUT)
                    : mismatch == 4 ? int(nvinfer1::TensorLocation::kHOST) : int(nvinfer1::TensorFormat::kCHW2);
                REQUIRE(message.find("observed=" + std::to_string(observed)) != std::string::npos);
                REQUIRE(message.find("expected=" + std::to_string(expected)) != std::string::npos);
            }
        }
        REQUIRE(fs::last_write_time(first.path) == old);
        REQUIRE(builds == 1);
    }
    auto trt = tensoracle::tensorrt_version();
    auto trt_text = std::to_string(trt.major) + "." + std::to_string(trt.minor) + "." + std::to_string(trt.patch);
    auto conditions = make_conditions(request, {}, gpu_identity(), trt_text, "Linux", "x86_64");
    REQUIRE(conditions["gpu"].contains("ecc_enabled"));
    REQUIRE(conditions["gpu"].contains("total_memory"));
    REQUIRE(conditions["gpu"].size() == 13);
    REQUIRE(read_engine_file(first.path, conditions).header["workspace_limit"] == 64 * 1024 * 1024);
    { std::ofstream output(first.path, std::ios::binary | std::ios::app); output << "broken"; }
    fs::last_write_time(first.path, old);
    REQUIRE_THROWS_AS(open_cached_engine(request, build), corrupt_engine_error);
    REQUIRE(fs::last_write_time(first.path) == old);
    REQUIRE(builds == 1);
    request.force_rebuild = true;
    REQUIRE(open_cached_engine(request, build).path == first.path);
    REQUIRE(builds == 2);
    auto before_weight = fs::last_write_time(weights);
    std::ofstream(weights) << "changed";
    fs::last_write_time(weights, before_weight);
    auto changed = open_cached_engine(request, build);
    REQUIRE(changed.path != first.path);
    REQUIRE(fs::exists(first.path));
    REQUIRE(builds == 3);
    auto broken_request = request;
    broken_request.name = "broken-builder";
    REQUIRE_THROWS_AS(open_cached_engine(broken_request, []() -> built_plan { throw std::runtime_error("graph failed"); }), std::runtime_error);
    auto broken_conditions = make_conditions(broken_request, {}, gpu_identity(), trt_text, "Linux", "x86_64");
    REQUIRE_FALSE(fs::exists(cache_path(broken_conditions)));

    // An invalid plan is distinguishable from both a builder failure and IO mismatch.
    REQUIRE_THROWS_AS(open_cached_engine(broken_request, [] { return built_plan{{1, 2, 3}, {}}; }), corrupt_engine_error);
    REQUIRE_FALSE(fs::exists(cache_path(broken_conditions)));
}
