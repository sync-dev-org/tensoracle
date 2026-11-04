#include <tensoracle/tensoracle.h>
#include "error.hpp"
#include "structure.hpp"

#include <NvInferRuntime.h>

static_assert(NV_TENSORRT_MAJOR == 11, "tensoracle v2 requires TensorRT 11 headers");

namespace {

using tensoracle::detail::structure_size;
using tensoracle::detail::read_field;
using tensoracle::detail::write_field;

tensoracle_status write_version(tensoracle_version* version, uint32_t major,
                               uint32_t minor, uint32_t patch, tensoracle_error** error) {
    if (!version || structure_size(version) < sizeof(size_t))
        return tensoracle::detail::fail(TENSORACLE_INVALID_ARGUMENT,
            "version output requires a struct_size field", error);
    auto size = structure_size(version);
    write_field(version, size, offsetof(tensoracle_version, major), major);
    write_field(version, size, offsetof(tensoracle_version, minor), minor);
    write_field(version, size, offsetof(tensoracle_version, patch), patch);
    return TENSORACLE_SUCCESS;
}

}

extern "C" tensoracle_status tensoracle_get_version(
    tensoracle_version* version, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        return write_version(version, TENSORACLE_VERSION_MAJOR, TENSORACLE_VERSION_MINOR,
                             TENSORACLE_VERSION_PATCH, error);
    });
}

extern "C" tensoracle_status tensoracle_check_version(
    const tensoracle_version* header, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&]() -> tensoracle_status {
        if (!header || structure_size(header) < offsetof(tensoracle_version, major) + sizeof(uint32_t))
            return tensoracle::detail::fail(TENSORACLE_INVALID_ARGUMENT,
                "header version requires struct_size and major fields", error);
        auto size = structure_size(header);
        if (read_field<uint32_t>(header, size, offsetof(tensoracle_version, major)) != TENSORACLE_VERSION_MAJOR)
            return tensoracle::detail::fail(TENSORACLE_INCOMPATIBLE_VERSION,
                "tensoracle major version is incompatible with the header version", error);
        if (read_field<uint32_t>(header, size, offsetof(tensoracle_version, minor)) > TENSORACLE_VERSION_MINOR)
            return tensoracle::detail::fail(TENSORACLE_INCOMPATIBLE_VERSION,
                "tensoracle minor version is older than the header version", error);
        return TENSORACLE_SUCCESS;
    });
}

extern "C" tensoracle_status tensoracle_get_tensorrt_version(
    tensoracle_version* version, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        auto value = static_cast<uint32_t>(getInferLibVersion());
        return write_version(version, value / 10000, (value / 100) % 100, value % 100, error);
    });
}
