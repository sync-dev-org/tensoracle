#include <tensoracle/tensoracle.h>
#include <tensoracle/tensoracle.hpp>
#include <tensoracle/builder.h>

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

TEST_CASE("Older builder requests ignore appended embedded registrations",
          "[SP-TENSORACLE-055][SP-TENSORACLE-077]") {
    const char* weights[]{"unused.safetensors"};
    tensoracle_builder_options request{};
    request.struct_size = offsetof(tensoracle_builder_options, embedded_plugins);
    request.name = "old-header"; request.graph_version = "1";
    request.weight_files = weights; request.weight_count = 1;
    request.optimization_level = -1; // Stop during validation, before any GPU access.
    request.embedded_plugins = reinterpret_cast<const tensoracle_plugin_libraries* const*>(1);
    request.embedded_plugin_count = 1;
    tensoracle_network_builder callback{sizeof(callback), nullptr,
        [](void*, void*, const tensoracle_weights*, const char**) noexcept -> tensoracle_status { return TENSORACLE_SUCCESS; }};
    tensoracle_engine* engine = nullptr;
    tensoracle_error* diagnostic = nullptr;
    REQUIRE(tensoracle_open_engine(&request, &callback, &engine, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle::error_handle owner(diagnostic);
    const char* message = nullptr;
    REQUIRE(tensoracle_error_message(owner.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::strstr(message, "optimization level"));
    REQUIRE(engine == nullptr);

    request.struct_size = sizeof(request);
    request.embedded_plugins = nullptr;
    diagnostic = nullptr;
    REQUIRE(tensoracle_open_engine(&request, &callback, &engine, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
    owner.reset(diagnostic);
    REQUIRE(tensoracle_error_message(owner.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::strstr(message, "registration handles"));
    REQUIRE(engine == nullptr);
}

TEST_CASE("C callers read the loaded library version", "[SP-TENSORACLE-055][SP-TENSORACLE-056]") {
    tensoracle_version version{sizeof(version), 99, 99, 99};
    tensoracle_error* error = nullptr;
    REQUIRE(tensoracle_get_version(&version, &error) == TENSORACLE_SUCCESS);
    REQUIRE(error == nullptr);
    REQUIRE(version.struct_size == sizeof(version));
    REQUIRE(version.major == TENSORACLE_VERSION_MAJOR);
    REQUIRE(version.minor == TENSORACLE_VERSION_MINOR);
    REQUIRE(version.patch == TENSORACLE_VERSION_PATCH);
}

TEST_CASE("Older output structures receive only complete known fields", "[SP-TENSORACLE-055][SP-TENSORACLE-056]") {
    alignas(tensoracle_version) std::array<unsigned char, sizeof(tensoracle_version) + 16> storage;
    for (auto size : {sizeof(size_t), offsetof(tensoracle_version, minor),
                      offsetof(tensoracle_version, patch), sizeof(tensoracle_version) + 16}) {
        storage.fill(0xa5);
        std::memcpy(storage.data(), &size, sizeof(size));
        auto* version = reinterpret_cast<tensoracle_version*>(storage.data());
        REQUIRE(tensoracle_get_version(version, nullptr) == TENSORACLE_SUCCESS);
        if (size >= offsetof(tensoracle_version, minor)) {
            uint32_t major = 0;
            std::memcpy(&major, storage.data() + offsetof(tensoracle_version, major), sizeof(major));
            REQUIRE(major == TENSORACLE_VERSION_MAJOR);
        }
        auto end = size > sizeof(tensoracle_version) ? sizeof(tensoracle_version) : size;
        for (size_t i = end; i < storage.size(); ++i) REQUIRE(storage[i] == 0xa5);
    }
}

TEST_CASE("Version checks accept older headers and reject incompatible headers", "[SP-TENSORACLE-055][SP-TENSORACLE-056]") {
    tensoracle_version header{sizeof(header), TENSORACLE_VERSION_MAJOR, 0, 999};
    REQUIRE(tensoracle_check_version(&header, nullptr) == TENSORACLE_SUCCESS);
    header.struct_size = offsetof(tensoracle_version, minor);
    header.minor = UINT32_MAX; // Must not be read when outside struct_size.
    REQUIRE(tensoracle_check_version(&header, nullptr) == TENSORACLE_SUCCESS);
    header.struct_size = sizeof(header);
    for (auto incompatible : {tensoracle_version{sizeof(header), TENSORACLE_VERSION_MAJOR + 1, 0, 0},
                              tensoracle_version{sizeof(header), TENSORACLE_VERSION_MAJOR, TENSORACLE_VERSION_MINOR + 1, 0}}) {
        tensoracle_error* raw = nullptr;
        REQUIRE(tensoracle_check_version(&incompatible, &raw) == TENSORACLE_INCOMPATIBLE_VERSION);
        tensoracle::error_handle error(raw);
        REQUIRE(error != nullptr);
        const char* message = nullptr;
        REQUIRE(tensoracle_error_message(error.get(), &message, nullptr) == TENSORACLE_SUCCESS);
        REQUIRE(message != nullptr);
        REQUIRE(std::strstr(message, "version") != nullptr);
    }
}

TEST_CASE("Invalid pointers and undersized structures return an owned diagnostic", "[SP-TENSORACLE-056]") {
    tensoracle_error* error = nullptr;
    REQUIRE(tensoracle_get_version(nullptr, &error) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(error != nullptr);
    const char* message = nullptr;
    REQUIRE(tensoracle_error_message(error, &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::strstr(message, "version") != nullptr);
    REQUIRE(tensoracle_error_destroy(error) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_error_destroy(nullptr) == TENSORACLE_SUCCESS);
    tensoracle_error* read_error = nullptr;
    REQUIRE(tensoracle_error_message(nullptr, &message, &read_error) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(message == nullptr);
    REQUIRE(read_error != nullptr);
    REQUIRE(tensoracle_error_message(read_error, &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::strstr(message, "error handle") != nullptr);
    REQUIRE(tensoracle_error_destroy(read_error) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_error_message(nullptr, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle_version small{sizeof(size_t) - 1, 71, 72, 73};
    REQUIRE(tensoracle_get_version(&small, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(small.major == 71);
    REQUIRE(tensoracle_get_tensorrt_version(&small, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    small.struct_size = sizeof(size_t);
    REQUIRE(tensoracle_check_version(&small, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_check_version(nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
}

TEST_CASE("C++ callers own diagnostic handles and receive typed exceptions", "[SP-TENSORACLE-056]") {
    static_assert(!std::is_copy_constructible_v<tensoracle::error_handle>);
    static_assert(std::is_nothrow_move_constructible_v<tensoracle::error_handle>);
    tensoracle::check_version();
    REQUIRE(tensoracle::version().major == TENSORACLE_VERSION_MAJOR);
    tensoracle_error* raw = nullptr;
    auto status = tensoracle_get_version(nullptr, &raw);
    tensoracle::error_handle original(raw);
    auto moved = std::move(original);
    REQUIRE(original == nullptr);
    REQUIRE(moved.get() == raw);
    try {
        tensoracle::check(status, moved.release());
        FAIL("the failed operation must throw");
    } catch (const tensoracle::error& e) {
        REQUIRE(e.status() == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(std::strstr(e.what(), "version") != nullptr);
    }
}

TEST_CASE("TensorRT library discovery requires no CUDA device", "[SP-TENSORACLE-054][SP-TENSORACLE-053]") {
    auto version = tensoracle::tensorrt_version();
    REQUIRE(version.major == 11);
    REQUIRE(version.minor >= 2);
    REQUIRE(version.patch < 100);
    static_assert(std::is_same_v<tensoracle_cuda_stream, CUstream_st*>);
}
