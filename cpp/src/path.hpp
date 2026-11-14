#pragma once
#include <filesystem>
#include <string>
#include <system_error>

namespace tensoracle::detail {
// Filesystem paths remain native. JSON and text returned to the Python/CLI
// boundary use UTF-8 bytes, independently of the Windows narrow code page.
inline std::string path_utf8(const std::filesystem::path& path, bool generic = false) {
    // C++17 uses char and C++20 uses char8_t for the same UTF-8 bytes.
    auto bytes = generic ? path.generic_u8string() : path.u8string();
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
inline std::string system_error_utf8(const std::system_error& error) {
#if defined(_WIN32)
    if (auto* filesystem = dynamic_cast<const std::filesystem::filesystem_error*>(&error)) {
        // MSVC's filesystem_error::what() formats paths with the narrow code page.
        std::string message = filesystem->std::system_error::what();
        for (const auto* path : {&filesystem->path1(), &filesystem->path2()})
            if (!path->empty()) message += ": \"" + path_utf8(*path) + "\"";
        return message;
    }
#endif
    return error.what();
}
}
