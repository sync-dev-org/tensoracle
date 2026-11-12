#include "builder_fixture.hpp"
#include "cache.hpp"
#include <tensoracle/cache.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <dlfcn.h>
#include <cstring>
#include <future>

namespace {
bool no_context = false, fail_memory = false;
int selected_device = -1, fallback_ordinal = -1;
CUdevice_attribute failed_attribute{};
CUresult init(unsigned) { return CUDA_SUCCESS; }
CUresult current(CUdevice* device) {
    if (no_context) return CUDA_ERROR_INVALID_CONTEXT;
    *device = 3; return CUDA_SUCCESS;
}
CUresult fallback(CUdevice* device, int ordinal) { fallback_ordinal = ordinal; *device = ordinal; return CUDA_SUCCESS; }
CUresult name(char* value, int, CUdevice device) { selected_device = device; std::strcpy(value, "test GPU"); return CUDA_SUCCESS; }
CUresult attribute(int* value, CUdevice_attribute field, CUdevice) {
    if (field == failed_attribute) return CUDA_ERROR_INVALID_VALUE;
    *value = 100 + int(field); return CUDA_SUCCESS;
}
CUresult memory(size_t* value, CUdevice) { *value = 4096; return fail_memory ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS; }
struct reset_driver {
    ~reset_driver() { no_context = false; fail_memory = false; failed_attribute = {}; }
};
}

// cache_open.cpp discovers these driver entry points through dlsym. Replace that
// external boundary in this executable while retaining the complete cache path.
extern "C" void* __real_dlsym(void*, const char*);
extern "C" void* __wrap_dlsym(void* library, const char* symbol) {
    if (!std::strcmp(symbol, "cuInit")) return reinterpret_cast<void*>(&init);
    if (!std::strcmp(symbol, "cuCtxGetDevice")) return reinterpret_cast<void*>(&current);
    if (!std::strcmp(symbol, "cuDeviceGet")) return reinterpret_cast<void*>(&fallback);
    if (!std::strcmp(symbol, "cuDeviceGetName")) return reinterpret_cast<void*>(&name);
    if (!std::strcmp(symbol, "cuDeviceGetAttribute")) return reinterpret_cast<void*>(&attribute);
    if (!std::strcmp(symbol, "cuDeviceTotalMem_v2")) return reinterpret_cast<void*>(&memory);
    return __real_dlsym(library, symbol);
}

TEST_CASE("Device identity falls back to device zero only without a current CUDA context", "[SP-TENSORACLE-009]") {
    reset_driver restore;
    fallback_ordinal = -1;
    auto identity = tensoracle::detail::gpu_identity();
    REQUIRE(selected_device == 3); REQUIRE(fallback_ordinal == -1);
    no_context = true;
    identity = tensoracle::detail::gpu_identity();
    REQUIRE(fallback_ordinal == 0); REQUIRE(selected_device == 0);
    REQUIRE(identity.size() == 13);
    REQUIRE(identity["name"] == "test GPU"); REQUIRE(identity["total_memory"] == 4096);
    REQUIRE(identity["ecc_enabled"] == 100 + int(CU_DEVICE_ATTRIBUTE_ECC_ENABLED));
}

TEST_CASE("Unavailable GPU attributes prevent cached reuse and never invoke the builder", "[SP-TENSORACLE-009][SP-TENSORACLE-011]") {
    reset_driver restore;
    builder_fixture::temporary_directory files;
    auto weight = files.path / "weights"; std::ofstream(weight) << "weights";
    tensoracle::detail::cache_request request;
    request.name = "gpu-identity"; request.graph_version = "1"; request.weights = {weight};
    auto version = tensoracle::tensorrt_version();
    auto version_text = std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
    auto conditions = tensoracle::detail::make_conditions(request, {}, tensoracle::detail::gpu_identity(), version_text, "Linux", "x86_64");
    auto path = tensoracle::detail::cache_path(conditions);
    tensoracle::detail::write_engine_file(path, conditions, {'p', 'l', 'a', 'n'}, {}, {});
    auto old = std::filesystem::last_write_time(path);
    int calls = 0;
    auto build = [&]() -> tensoracle::detail::built_plan { ++calls; throw std::runtime_error("builder must not run"); };
    const CUdevice_attribute fields[] = {
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_ECC_ENABLED};
    for (auto field : fields) {
        INFO("unavailable attribute " << int(field)); failed_attribute = field;
        REQUIRE_THROWS_WITH(tensoracle::detail::open_cached_engine(request, build), Catch::Matchers::StartsWith("CUDA device attribute lookup failed:"));
        REQUIRE(std::filesystem::last_write_time(path) == old);
    }
    failed_attribute = {}; fail_memory = true;
    REQUIRE_THROWS_WITH(tensoracle::detail::open_cached_engine(request, build), "CUDA device total memory lookup failed");
    REQUIRE(calls == 0); REQUIRE(std::filesystem::last_write_time(path) == old);
    REQUIRE(tensoracle::detail::read_engine_file(path, conditions).plan == std::vector<uint8_t>{'p', 'l', 'a', 'n'});
}

TEST_CASE("A failed callback passes construction to the waiting thread without a corrupt file error", "[SP-TENSORACLE-067]") {
    builder_fixture::temporary_directory files;
    auto weight = files.path / "weights"; std::ofstream(weight) << "weights";
    tensoracle::detail::cache_request request;
    request.name = "failed-callbacks"; request.graph_version = "1"; request.weights = {weight};
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    auto open = [&](bool first) {
        try {
            auto engine = tensoracle::detail::open_cached_engine(request, [&]() -> tensoracle::detail::built_plan {
                if (first) { entered.set_value(); released.wait(); }
                throw std::runtime_error(first ? "first callback failed" : "waiting callback failed");
            });
            return std::string("unexpected success");
        } catch (const std::exception& failure) { return std::string(failure.what()); }
    };
    auto first = std::async(std::launch::async, [&] { return open(true); });
    entered.get_future().wait();
    auto second = std::async(std::launch::async, [&] { return open(false); });
    auto waiting = second.wait_for(std::chrono::milliseconds(300));
    release.set_value();
    CHECK(waiting == std::future_status::timeout);
    REQUIRE(first.get() == "first callback failed");
    REQUIRE(second.get() == "waiting callback failed");
    REQUIRE(tensoracle::list_cache().entries().empty());
}
