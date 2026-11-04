#pragma once


#include <tensoracle/tensoracle.hpp>

#include <exception>
#include <new>
#include <stdexcept>

namespace tensoracle::detail {

tensoracle_status fail(tensoracle_status status, const char* message,
                      tensoracle_error** error) noexcept;

/* The common exception boundary for operations added to the C ABI. */
template <class Operation>
tensoracle_status guard(tensoracle_error** error, Operation&& operation) noexcept {
    if (error) *error = nullptr;
    try {
        return operation();
    } catch (const tensoracle::error& e) {
        return fail(e.status(), e.what(), error);
    } catch (const std::bad_alloc&) {
        return fail(TENSORACLE_OUT_OF_MEMORY, "tensoracle allocation failed", error);
    } catch (const std::invalid_argument& e) {
        return fail(TENSORACLE_INVALID_ARGUMENT, e.what(), error);
    } catch (const std::exception& e) {
        return fail(TENSORACLE_INTERNAL_ERROR, e.what(), error);
    } catch (...) {
        return fail(TENSORACLE_INTERNAL_ERROR, "unknown tensoracle exception", error);
    }
}

} // namespace tensoracle::detail
