#pragma once
#include "cubin.h"
#include "tensoracle.hpp"
#include <string>

namespace tensoracle {
class bundle_result {
  public:
    explicit bundle_result(tensoracle_bundle_result *value) noexcept : handle_(value) {}
    bundle_result(bundle_result &&) noexcept = default;
    bundle_result &operator=(bundle_result &&) noexcept = default;
    tensoracle_bundle_view view() const {
        tensoracle_bundle_view value{sizeof(value), nullptr, nullptr};
        tensoracle_error *diagnostic = nullptr;
        auto status = tensoracle_bundle_result_get(handle_.get(), &value, &diagnostic);
        check(status, diagnostic);
        return value;
    }
    std::string manifest_json() const { return view().manifest_json; }
    std::string sha256() const { return view().sha256; }

  private:
    unique_handle<tensoracle_bundle_result, tensoracle_bundle_result_destroy> handle_;
};
inline bundle_result compile_bundle(const tensoracle_kernel_spec *const *specs, size_t count,
                                    const tensoracle_producer_options &options) {
    tensoracle_bundle_result *result = nullptr;
    tensoracle_error *diagnostic = nullptr;
    auto status = tensoracle_compile_bundle(specs, count, &options, &result, &diagnostic);
    check(status, diagnostic);
    return bundle_result(result);
}
inline bundle_result inspect_bundle(const char *root, const char *plugin = nullptr) {
    tensoracle_bundle_result *result = nullptr;
    tensoracle_error *diagnostic = nullptr;
    auto status = tensoracle_inspect_bundle(root, plugin, &result, &diagnostic);
    check(status, diagnostic);
    return bundle_result(result);
}
inline bundle_result inspect_wheel(const char *path, const char *plugin = nullptr) {
    tensoracle_bundle_result *result = nullptr;
    tensoracle_error *diagnostic = nullptr;
    auto status = tensoracle_inspect_wheel(path, plugin, &result, &diagnostic);
    check(status, diagnostic);
    return bundle_result(result);
}
inline bundle_result bundle_digest(const char *json) {
    tensoracle_bundle_result *result = nullptr;
    tensoracle_error *diagnostic = nullptr;
    auto status = tensoracle_bundle_digest(json, &result, &diagnostic);
    check(status, diagnostic);
    return bundle_result(result);
}
inline uint32_t cubin_sm(const void *data, size_t size) {
    uint32_t sm = 0;
    tensoracle_error *diagnostic = nullptr;
    auto status = tensoracle_cubin_sm(data, size, &sm, &diagnostic);
    check(status, diagnostic);
    return sm;
}
} // namespace tensoracle
