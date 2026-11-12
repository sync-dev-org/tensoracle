#pragma once
#include "tensoracle.hpp"
#include "engine.h"

#include <string>
#include <utility>
#include <vector>

namespace tensoracle {
struct input_shape { std::string name; std::vector<int64_t> dimensions; };
struct tensor_address { std::string name; void* address; };

class engine_execution {
public:
    engine_execution() = default;
    explicit engine_execution(tensoracle_execution* handle) : handle_(handle) {}
    tensoracle_execution* get() const noexcept { return handle_.get(); }
    void close() noexcept { handle_.reset(); }
    void set_input_shapes(const std::vector<input_shape>& values) {
        std::vector<tensoracle_input_shape> shapes;
        shapes.reserve(values.size());
        for (const auto& value : values)
            shapes.push_back({sizeof(tensoracle_input_shape), value.name.c_str(),
                static_cast<int32_t>(value.dimensions.size()), value.dimensions.data()});
        std::vector<const tensoracle_input_shape*> pointers;
        for (const auto& shape : shapes) pointers.push_back(&shape);
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_set_input_shapes(get(), pointers.data(), pointers.size(), &diagnostic);
        check(status, diagnostic);
    }
    std::vector<int64_t> get_tensor_shape(const char* name) {
        tensoracle_tensor_info info{sizeof(info), 0, 0, {}};
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_tensor_info(get(), name, &info, &diagnostic);
        check(status, diagnostic);
        return {info.dimensions, info.dimensions + info.rank};
    }
    tensoracle_dtype get_tensor_dtype(const char* name) const {
        tensoracle_dtype result{};
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_tensor_dtype(get(), name, &result, &diagnostic);
        check(status, diagnostic);
        return result;
    }
    void enqueue(const std::vector<tensor_address>& values, tensoracle_cuda_stream stream) {
        std::vector<tensoracle_tensor_address> addresses;
        addresses.reserve(values.size());
        for (const auto& value : values)
            addresses.push_back({sizeof(tensoracle_tensor_address), value.name.c_str(), value.address});
        std::vector<const tensoracle_tensor_address*> pointers;
        for (const auto& address : addresses) pointers.push_back(&address);
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_enqueue(get(), pointers.data(), pointers.size(), stream, &diagnostic);
        check(status, diagnostic);
    }
private:
    unique_handle<tensoracle_execution, tensoracle_execution_destroy> handle_;
};

class engine_handle {
public:
    engine_handle() = default;
    explicit engine_handle(tensoracle_engine* handle) : handle_(handle) {}
    tensoracle_engine* get() const noexcept { return handle_.get(); }
    tensoracle_engine* release() noexcept { return handle_.release(); }
    std::string path() const {
        const char* result = nullptr; tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_engine_path(get(), &result, &diagnostic); check(status, diagnostic); return result;
    }
    engine_execution create_execution() const {
        tensoracle_execution* result = nullptr;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_create(get(), &result, &diagnostic);
        check(status, diagnostic);
        return engine_execution(result);
    }
    size_t scratch_size() const {
        size_t size = 0;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_engine_scratch_size(get(), &size, &diagnostic);
        check(status, diagnostic);
        return size;
    }
    engine_execution create_execution(void* scratch, size_t size) const {
        tensoracle_scratch_memory memory{sizeof(memory), scratch, size};
        tensoracle_execution* result = nullptr;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_execution_create_with_scratch(get(), &memory, &result, &diagnostic);
        check(status, diagnostic);
        return engine_execution(result);
    }
    size_t tensor_count() const {
        size_t count = 0;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_engine_tensor_count(get(), &count, &diagnostic);
        check(status, diagnostic);
        return count;
    }
    std::string tensor_name(size_t index) const {
        const char* name = nullptr;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_engine_tensor_name(get(), index, &name, &diagnostic);
        check(status, diagnostic);
        return name;
    }
private:
    unique_handle<tensoracle_engine, tensoracle_engine_destroy> handle_;
};
}
