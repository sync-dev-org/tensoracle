#pragma once
#include "tensoracle.hpp"
#include "plugins.h"
#include <string>
#include <vector>

namespace tensoracle {
class plugin_libraries {
public:
    explicit plugin_libraries(tensoracle_plugin_libraries* value) noexcept : handle_(value) {}
    tensoracle_plugin_libraries* get() const noexcept { return handle_.get(); }
    std::vector<std::string> digests() const {
        size_t count = 0;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_plugin_libraries_count(get(), &count, &diagnostic);
        check(status, diagnostic);
        std::vector<std::string> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const char* digest = nullptr;
            status = tensoracle_plugin_libraries_digest(get(), i, &digest, &diagnostic);
            check(status, diagnostic);
            result.emplace_back(digest);
        }
        return result;
    }
private:
    unique_handle<tensoracle_plugin_libraries, tensoracle_plugin_libraries_destroy> handle_;
};
inline plugin_libraries register_embedded_plugins(const void* module_address,
    const std::vector<void*>& creators, const char* plugin_namespace = "") {
    tensoracle_plugin_libraries* result = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_register_embedded_plugins(module_address, creators.data(), creators.size(),
                                                       plugin_namespace, &result, &diagnostic);
    check(status, diagnostic);
    return plugin_libraries(result);
}
inline std::vector<std::string> register_libraries(const std::vector<std::string>& paths) {
    std::vector<const char*> borrowed;
    for (const auto& path : paths) borrowed.push_back(path.c_str());
    tensoracle_plugin_libraries* result = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_register_libraries(borrowed.data(), borrowed.size(), &result, &diagnostic);
    check(status, diagnostic);
    return plugin_libraries(result).digests();
}
}
