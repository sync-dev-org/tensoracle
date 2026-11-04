#include "error.hpp"

#include <string>

struct tensoracle_error {
    std::string message;
};

namespace {
// This diagnostic survives allocation failure; destruction is a no-op for it.
const char allocation_message[] = "tensoracle could not allocate an error diagnostic";
// A default-constructed string needs no allocation.
tensoracle_error emergency_error{};
tensoracle_error* emergency() noexcept {
    return &emergency_error;
}
}

tensoracle_status tensoracle::detail::fail(tensoracle_status status, const char* message,
                                         tensoracle_error** error) noexcept {
    if (error) {
        try {
            *error = new tensoracle_error{message};
        } catch (...) {
            *error = emergency();
            return TENSORACLE_OUT_OF_MEMORY;
        }
    }
    return status;
}

extern "C" tensoracle_status tensoracle_error_message(
    const tensoracle_error* error, const char** message, tensoracle_error** diagnostic) noexcept {
    return tensoracle::detail::guard(diagnostic, [&]() -> tensoracle_status {
        if (message) *message = nullptr;
        if (!error || !message)
            return tensoracle::detail::fail(TENSORACLE_INVALID_ARGUMENT,
                "reading an error handle requires a non-null handle and message output", diagnostic);
        *message = error == emergency() ? allocation_message : error->message.c_str();
        return TENSORACLE_SUCCESS;
    });
}

extern "C" tensoracle_status tensoracle_error_destroy(tensoracle_error* error) noexcept {
    if (error != emergency()) delete error;
    return TENSORACLE_SUCCESS;
}
