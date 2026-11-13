#pragma once
#include "builder.h"
#include "engine.hpp"
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace nvinfer1 { class INetworkDefinition; }
namespace tensoracle {
class weights_view {
public:
    explicit weights_view(const tensoracle_weights* value) : value_(value) {}
    size_t size() const {
        size_t output = 0; tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_count(value_, &output, &error); check(status, error); return output;
    }
    tensoracle_weight at(size_t index) const {
        tensoracle_weight output{}; output.struct_size = sizeof(output); tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_at(value_, index, &output, &error); check(status, error); return output;
    }
    tensoracle_weight at(const char* name) const {
        tensoracle_weight output{}; output.struct_size = sizeof(output); tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_find(value_, name, &output, &error); check(status, error); return output;
    }
    size_t metadata_size() const {
        size_t output = 0; tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_metadata_count(value_, &output, &error); check(status, error); return output;
    }
    std::pair<std::string_view, std::string_view> metadata_at(size_t index) const {
        tensoracle_weights_metadata_entry output{}; output.struct_size = sizeof(output); tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_metadata_at(value_, index, &output, &error); check(status, error);
        return {{output.key, output.key_size}, {output.value, output.value_size}};
    }
    std::string_view metadata_at(const char* key) const {
        tensoracle_weights_metadata_entry output{}; output.struct_size = sizeof(output); tensoracle_error* error = nullptr;
        auto status = tensoracle_weights_metadata_find(value_, key, &output, &error); check(status, error);
        return {output.value, output.value_size};
    }
private:
    const tensoracle_weights* value_;
};

class weights_handle {
public:
    weights_handle() = default;
    explicit weights_handle(tensoracle_weights* handle) : handle_(handle) {}
    tensoracle_weights* get() const noexcept { return handle_.get(); }
    tensoracle_weights* release() noexcept { return handle_.release(); }
    void close() noexcept { handle_.reset(); }
    weights_view view() const noexcept { return weights_view(get()); }
    size_t metadata_size() const { return view().metadata_size(); }
    std::pair<std::string_view, std::string_view> metadata_at(size_t index) const { return view().metadata_at(index); }
    std::string_view metadata_at(const char* key) const { return view().metadata_at(key); }
private:
    unique_handle<tensoracle_weights, tensoracle_weights_destroy> handle_;
};

inline weights_handle open_weights(const std::filesystem::path& path) {
    tensoracle_weights* result = nullptr; tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_weights_open(path.string().c_str(), &result, &diagnostic);
    check(status, diagnostic); return weights_handle(result);
}
struct input_profile { std::string name; std::vector<int64_t> minimum, optimum, maximum; };
struct tensor_spec {
    std::string name; tensoracle_dtype dtype; std::vector<int64_t> shape; bool input;
    std::optional<int32_t> location{}, format{};
};
struct builder_options {
    std::string name, graph_version;
    std::vector<std::string> weights, plugins;
    std::vector<input_profile> inputs;
    std::vector<tensor_spec> io;
    std::optional<int64_t> workspace_limit;
    int32_t optimization_level = 3;
    bool force_rebuild = false;
    std::vector<const tensoracle_plugin_libraries*> embedded_plugins;
};
// A callback returns void or throws; every exception is translated before the
// C callback returns. TensorRT headers are needed only where the graph is built.
template<class F> engine_handle open_engine(const builder_options& options, F&& build) {
    using callable = std::remove_reference_t<F>;
    static_assert(std::is_void_v<std::invoke_result_t<callable&, nvinfer1::INetworkDefinition&, const weights_view&>>,
                  "network callback must return void, using an exception for failure");
    struct context { callable* function; std::exception_ptr failure; } state{std::addressof(build), {}};
    tensoracle_network_builder callback{sizeof(callback), &state,
        [](void* pointer, void* network, const tensoracle_weights* weights, const char** message) noexcept -> tensoracle_status {
            auto& value = *static_cast<context*>(pointer);
            try { (*value.function)(*static_cast<nvinfer1::INetworkDefinition*>(network), weights_view(weights)); return TENSORACLE_SUCCESS; }
            catch (const error& failure) { value.failure = std::current_exception(); *message = failure.what(); return failure.status(); }
            catch (const std::bad_alloc& failure) { value.failure = std::current_exception(); *message = failure.what(); return TENSORACLE_OUT_OF_MEMORY; }
            catch (const std::invalid_argument& failure) { value.failure = std::current_exception(); *message = failure.what(); return TENSORACLE_INVALID_ARGUMENT; }
            catch (const std::exception& failure) { value.failure = std::current_exception(); *message = failure.what(); return TENSORACLE_INTERNAL_ERROR; }
            catch (...) { value.failure = std::current_exception(); *message = "unknown exception in network callback"; return TENSORACLE_INTERNAL_ERROR; }
        }};
    std::vector<const char*> files, plugins;
    for (const auto& path : options.weights) files.push_back(path.c_str());
    for (const auto& path : options.plugins) plugins.push_back(path.c_str());
    std::vector<tensoracle_input_profile> profiles; profiles.reserve(options.inputs.size());
    for (const auto& input : options.inputs) {
        if (input.minimum.size() != input.optimum.size() || input.minimum.size() != input.maximum.size())
            throw error(TENSORACLE_INVALID_ARGUMENT, "input profile ranks differ");
        profiles.push_back({sizeof(tensoracle_input_profile), input.name.c_str(), input.minimum.size(),
            input.minimum.data(), input.optimum.data(), input.maximum.data()});
    }
    std::vector<const tensoracle_input_profile*> profile_pointers;
    for (const auto& profile : profiles) profile_pointers.push_back(&profile);
    std::vector<tensoracle_tensor_spec> io; io.reserve(options.io.size());
    for (const auto& value : options.io)
        io.push_back({sizeof(tensoracle_tensor_spec), value.name.c_str(), value.dtype, value.shape.size(), value.shape.data(),
            uint32_t(value.input), uint32_t(value.location.has_value()), value.location.value_or(0),
            uint32_t(value.format.has_value()), value.format.value_or(0)});
    std::vector<const tensoracle_tensor_spec*> io_pointers;
    for (const auto& value : io) io_pointers.push_back(&value);
    tensoracle_builder_options request{sizeof(request), options.name.c_str(), options.graph_version.c_str(),
        files.data(), files.size(), profile_pointers.data(), profile_pointers.size(), io_pointers.data(), io_pointers.size(),
        plugins.data(), plugins.size(), uint32_t(options.workspace_limit.has_value()), options.workspace_limit.value_or(0),
        options.optimization_level, uint32_t(options.force_rebuild),
        options.embedded_plugins.data(), options.embedded_plugins.size()};
    tensoracle_engine* result = nullptr; tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_open_engine(&request, &callback, &result, &diagnostic);
    check(status, diagnostic); return engine_handle(result);
}
}
