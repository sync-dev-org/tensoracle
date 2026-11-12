#include "cache.hpp"
#include "engine_internal.hpp"
#include <tensoracle/plugins.hpp>
#include <cuda.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <cerrno>
#include <map>
#include <set>
#include <system_error>

namespace tensoracle::detail {
using json = nlohmann::json;
namespace fs = std::filesystem;
namespace {
struct driver_library {
    void* value = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    ~driver_library() { if (value) dlclose(value); }
    template<class T> T get(const char* name) {
        if (!value) throw std::runtime_error("CUDA driver library was not found");
        auto pointer = dlsym(value, name);
        if (!pointer) throw std::runtime_error(std::string("CUDA driver symbol was not found: ") + name);
        return reinterpret_cast<T>(pointer);
    }
};
std::vector<int64_t> dimensions(const nvinfer1::Dims& value) {
    if (value.nbDims < 0) return {-1};
    return {value.d, value.d + value.nbDims};
}
[[noreturn]] void mismatch(const fs::path& path, const std::string& name, const std::string& field,
                           const json& observed, const json& expected) {
    throw engine_io_error("TensorRT engine tensor '" + name + "' " + field + " differs: observed="
        + observed.dump() + ", expected=" + expected.dump() + "; path=" + path.string());
}
void validate_io(const engine_handle& engine, const fs::path& path, const cache_request& request) {
    auto& native = *engine.get()->state->engine;
    std::map<std::string, const tensor_spec*> expected;
    std::set<std::string> actual;
    for (int32_t i = 0; i < native.getNbIOTensors(); ++i) actual.emplace(native.getIOTensorName(i));
    for (const auto& spec : request.io)
        if (!expected.emplace(spec.name, &spec).second)
            throw std::invalid_argument("duplicate engine IO tensor: " + spec.name);
    std::set<std::string> expected_names;
    for (const auto& item : expected) expected_names.emplace(item.first);
    if (expected_names != actual) {
        auto differing = expected_names;
        for (const auto& name : actual) if (!differing.erase(name)) differing.emplace(name);
        mismatch(path, *differing.begin(), "name", actual, expected_names);
    }
    std::set<std::string> input_names;
    for (const auto& item : expected) {
        const auto& name = item.first;
        const auto& spec = *item.second;
        auto mode = native.getTensorIOMode(name.c_str());
        auto wanted = spec.input ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
        if (mode != wanted) mismatch(path, name, "mode", int(mode), int(wanted));
        if (spec.input) input_names.emplace(name);
        auto type = public_dtype(native.getTensorDataType(name.c_str()));
        if (type != spec.dtype) mismatch(path, name, "dtype", type, spec.dtype);
        auto shape = dimensions(native.getTensorShape(name.c_str()));
        if (shape != spec.shape) mismatch(path, name, "shape", shape, spec.shape);
        if (spec.location) {
            auto location = native.getTensorLocation(name.c_str());
            if (int32_t(location) != *spec.location) mismatch(path, name, "location", int32_t(location), *spec.location);
        }
        if (spec.format) {
            auto format = native.getTensorFormat(name.c_str());
            if (int32_t(format) != *spec.format) mismatch(path, name, "format", int32_t(format), *spec.format);
        }
    }
    std::set<std::string> declared;
    for (const auto& input : request.inputs) declared.emplace(input.name);
    if (declared != input_names) mismatch(path, "inputs", "build input names", input_names, declared);
    for (const auto& input : request.inputs) {
        auto shape = dimensions(native.getTensorShape(input.name.c_str()));
        std::vector<std::vector<int64_t>> observed(3, shape);
        if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
            for (int i = 0; i < 3; ++i)
                observed[i] = dimensions(native.getProfileShape(input.name.c_str(), 0, static_cast<nvinfer1::OptProfileSelector>(i)));
        }
        std::vector<std::vector<int64_t>> expected_range{input.minimum, input.optimum, input.maximum};
        if (observed != expected_range) mismatch(path, input.name, "profile range", observed, expected_range);
    }
}
engine_handle open_plan(const std::vector<uint8_t>& plan, const fs::path& path, const cache_request& request) {
    engine_handle result;
    try { result = deserialize_engine(plan); }
    catch (const tensoracle::error& failure) {
        if (failure.status() == TENSORACLE_CORRUPT_ENGINE)
            throw corrupt_engine_error(std::string(failure.what()) + "; path=" + path.string());
        throw;
    }
    validate_io(result, path, request);
    return result;
}
}

json gpu_identity() {
    driver_library driver;
    if (driver.get<decltype(&cuInit)>("cuInit")(0) != CUDA_SUCCESS)
        throw std::runtime_error("CUDA driver initialization failed");
    CUdevice device;
    if (driver.get<decltype(&cuCtxGetDevice)>("cuCtxGetDevice")(&device) != CUDA_SUCCESS
        && driver.get<decltype(&cuDeviceGet)>("cuDeviceGet")(&device, 0) != CUDA_SUCCESS)
        throw std::runtime_error("CUDA device selection failed");
    char name[256]{};
    if (driver.get<decltype(&cuDeviceGetName)>("cuDeviceGetName")(name, sizeof(name), device) != CUDA_SUCCESS)
        throw std::runtime_error("CUDA device name lookup failed");
    json result{{"name", name}};
    const std::pair<const char*, CUdevice_attribute> attributes[] = {
        {"compute_capability_major", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
        {"compute_capability_minor", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
        {"global_memory_bus_width", CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
        {"l2_cache_size", CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
        {"max_shared_memory_per_block", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK},
        {"max_shared_memory_per_block_optin", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN},
        {"max_shared_memory_per_multiprocessor", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
        {"texture_alignment", CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT},
        {"multiprocessor_count", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
        {"integrated", CU_DEVICE_ATTRIBUTE_INTEGRATED},
        {"ecc_enabled", CU_DEVICE_ATTRIBUTE_ECC_ENABLED},
    };
    auto attribute = driver.get<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    for (const auto& item : attributes) {
        int value;
        if (attribute(&value, item.second, device) != CUDA_SUCCESS)
            throw std::runtime_error(std::string("CUDA device attribute lookup failed: ") + item.first);
        result[item.first] = value;
    }
    size_t total;
    if (driver.get<decltype(&cuDeviceTotalMem)>("cuDeviceTotalMem_v2")(&total, device) != CUDA_SUCCESS)
        throw std::runtime_error("CUDA device total memory lookup failed");
    result["total_memory"] = total;
    return result;
}

cached_engine open_cached_engine(const cache_request& request, const std::function<built_plan()>& build) {
    std::vector<std::string> paths;
    for (const auto& path : request.plugins) paths.push_back(path.string());
    auto plugins = tensoracle::register_libraries(paths);
    auto version = tensoracle::tensorrt_version();
    auto version_text = std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
    struct utsname system{};
    if (uname(&system)) throw std::system_error(errno, std::generic_category(), "uname");
    auto conditions = make_conditions(request, plugins, gpu_identity(), version_text, system.sysname, system.machine);
    auto path = cache_path(conditions);
    fs::create_directories(path.parent_path());
    engine_file_lock lock(path);
    if (!request.force_rebuild && !lock.empty()) {
        auto saved = read_engine_file(path, conditions);
        auto engine = open_plan(saved.plan, path, request);
        touch_engine(path);
        return {std::move(engine), path};
    }
    auto result = build();
    auto engine = open_plan(result.plan, path, request);
    write_engine_file(path, conditions, result.plan, request.workspace_limit, result.shape_tensor_outputs);
    return {std::move(engine), path};
}
}
