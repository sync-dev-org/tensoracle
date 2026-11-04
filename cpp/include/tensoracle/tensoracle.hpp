#pragma once

#include "tensoracle.h"

#include <memory>
#include <stdexcept>

namespace tensoracle {

template <class T, tensoracle_status (*Release)(T*)>
struct handle_deleter {
    void operator()(T* value) const noexcept { (void)Release(value); }
};

template <class T, tensoracle_status (*Release)(T*)>
using unique_handle = std::unique_ptr<T, handle_deleter<T, Release>>;

using error_handle = unique_handle<tensoracle_error, tensoracle_error_destroy>;

class error : public std::runtime_error {
public:
    error(tensoracle_status status, const char* message)
        : std::runtime_error(message), status_(status) {}
    tensoracle_status status() const noexcept { return status_; }
private:
    tensoracle_status status_;
};

/* Always consumes the diagnostic, including if constructing the exception
 * throws. Keep status and error in separate statements: C++17 does not order
 * evaluation of function arguments. */
inline void check(tensoracle_status status, tensoracle_error* diagnostic) {
    error_handle owner(diagnostic);
    if (status == TENSORACLE_SUCCESS) return;
    const char* message = "tensoracle operation failed without a diagnostic";
    if (diagnostic) (void)tensoracle_error_message(diagnostic, &message, nullptr);
    throw error(status, message);
}

inline tensoracle_version version() {
    tensoracle_version result{sizeof(result), 0, 0, 0};
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_get_version(&result, &diagnostic);
    check(status, diagnostic);
    return result;
}

inline void check_version() {
    const tensoracle_version header{sizeof(header), TENSORACLE_VERSION_MAJOR,
                                    TENSORACLE_VERSION_MINOR, TENSORACLE_VERSION_PATCH};
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_check_version(&header, &diagnostic);
    check(status, diagnostic);
}

inline tensoracle_version tensorrt_version() {
    tensoracle_version result{sizeof(result), 0, 0, 0};
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_get_tensorrt_version(&result, &diagnostic);
    check(status, diagnostic);
    return result;
}

inline tensoracle_output_comparison compare_outputs(
    const tensoracle_array& candidate, const tensoracle_array& reference,
    const tensoracle_comparison_options* options = nullptr) {
    tensoracle_output_comparison result{};
    result.struct_size = sizeof(result);
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_compare_outputs(&candidate, &reference, options, &result, &diagnostic);
    check(status, diagnostic);
    return result;
}

} // namespace tensoracle
