#pragma once
#include "tensoracle.hpp"
#include "plugins.h"
#include <string>
#include <vector>

namespace tensoracle {
inline std::vector<std::string> register_libraries(const std::vector<std::string>& paths) {
    std::vector<const char*> borrowed;
    for (const auto& path : paths) borrowed.push_back(path.c_str());
    tensoracle_plugin_libraries* result = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_register_libraries(borrowed.data(), borrowed.size(), &result, &diagnostic);
    check(status, diagnostic);
    unique_handle<tensoracle_plugin_libraries, tensoracle_plugin_libraries_destroy> owner(result);
    size_t count = 0;
    status = tensoracle_plugin_libraries_count(result, &count, &diagnostic);
    check(status, diagnostic);
    std::vector<std::string> digests;
    digests.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const char* digest = nullptr;
        status = tensoracle_plugin_libraries_digest(result, i, &digest, &diagnostic);
        check(status, diagnostic);
        digests.emplace_back(digest);
    }
    return digests;
}
}
