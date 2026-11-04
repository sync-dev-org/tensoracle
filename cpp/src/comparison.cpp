#include <tensoracle/tensoracle.h>
#include "error.hpp"
#include "structure.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace {
using tensoracle::detail::fail;
using tensoracle::detail::read_field;
using tensoracle::detail::structure_size;
using tensoracle::detail::write_field;

size_t element_size(tensoracle_dtype dtype) {
    switch (dtype) {
        case TENSORACLE_BOOL: case TENSORACLE_INT8: case TENSORACLE_UINT8: return 1;
        case TENSORACLE_INT16: case TENSORACLE_UINT16: case TENSORACLE_FLOAT16: return 2;
        case TENSORACLE_INT32: case TENSORACLE_UINT32: case TENSORACLE_FLOAT32: return 4;
        case TENSORACLE_INT64: case TENSORACLE_UINT64: case TENSORACLE_FLOAT64: return 8;
        case TENSORACLE_LONG_DOUBLE: return sizeof(long double);
        default: return 0;
    }
}

int numeric_kind(tensoracle_dtype dtype) {
    if (dtype == TENSORACLE_BOOL) return 1;
    if (dtype >= TENSORACLE_INT8 && dtype <= TENSORACLE_UINT64) return 2;
    if (dtype >= TENSORACLE_FLOAT16 && dtype <= TENSORACLE_LONG_DOUBLE) return 3;
    return 0;
}

bool valid_array(const tensoracle_array* input) {
    return input && structure_size(input) >= offsetof(tensoracle_array, data) + sizeof(input->data) &&
           input->data && (!input->ndim || input->shape);
}

template <class T>
T element(const tensoracle_array& input, size_t index) {
    T value;
    std::memcpy(&value, static_cast<const unsigned char*>(input.data) + index * sizeof(T), sizeof(T));
    return value;
}

long double floating(const tensoracle_array& input, size_t index) {
    switch (input.dtype) {
        case TENSORACLE_FLOAT32: return element<float>(input, index);
        case TENSORACLE_FLOAT64: return element<double>(input, index);
        case TENSORACLE_LONG_DOUBLE: return element<long double>(input, index);
        default: {
            const auto bits = element<uint16_t>(input, index);
            const auto exponent = (bits >> 10) & 31;
            const auto fraction = bits & 1023;
            double value;
            if (exponent == 31)
                value = fraction ? std::numeric_limits<double>::quiet_NaN() :
                                   std::numeric_limits<double>::infinity();
            else if (exponent == 0)
                value = std::ldexp(static_cast<double>(fraction), -24);
            else
                value = std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
            return (bits & 0x8000) ? -value : value;
        }
    }
}

struct integer_value {
    bool negative;
    uint64_t bits;
};

template <class T>
integer_value integer_element(const tensoracle_array& input, size_t index) {
    const auto value = element<T>(input, index);
    return {std::numeric_limits<T>::is_signed && value < 0, static_cast<uint64_t>(value)};
}

integer_value integer(const tensoracle_array& input, size_t index) {
    switch (input.dtype) {
        case TENSORACLE_BOOL: return {false, element<uint8_t>(input, index) != 0};
        case TENSORACLE_INT8: return integer_element<int8_t>(input, index);
        case TENSORACLE_UINT8: return integer_element<uint8_t>(input, index);
        case TENSORACLE_INT16: return integer_element<int16_t>(input, index);
        case TENSORACLE_UINT16: return integer_element<uint16_t>(input, index);
        case TENSORACLE_INT32: return integer_element<int32_t>(input, index);
        case TENSORACLE_UINT32: return integer_element<uint32_t>(input, index);
        case TENSORACLE_INT64: return integer_element<int64_t>(input, index);
        default: return integer_element<uint64_t>(input, index);
    }
}

tensoracle_status tolerances(const tensoracle_comparison_options* options,
                             bool& present, double& atol, double& rtol, tensoracle_error** error) {
    present = false;
    if (!options) return TENSORACLE_SUCCESS;
    const auto size = structure_size(options);
    if (size < sizeof(size_t))
        return fail(TENSORACLE_INVALID_ARGUMENT, "comparison options require struct_size", error);
    auto has_atol = read_field<uint32_t>(options, size, offsetof(tensoracle_comparison_options, has_atol));
    auto has_rtol = read_field<uint32_t>(options, size, offsetof(tensoracle_comparison_options, has_rtol));
    if (!!has_atol != !!has_rtol)
        return fail(TENSORACLE_INVALID_ARGUMENT, "atol and rtol must be supplied together", error);
    if (!has_atol) return TENSORACLE_SUCCESS;
    if (size < offsetof(tensoracle_comparison_options, rtol) + sizeof(double))
        return fail(TENSORACLE_INVALID_ARGUMENT, "present tolerances require complete value fields", error);
    atol = options->atol;
    rtol = options->rtol;
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
        return fail(TENSORACLE_INVALID_ARGUMENT, "atol and rtol must be finite nonnegative numbers", error);
    present = true;
    return TENSORACLE_SUCCESS;
}

tensoracle_output_comparison measure(const tensoracle_array& candidate,
                                     const tensoracle_array& reference, size_t count,
                                     bool with_tolerance, double atol, double rtol) {
    tensoracle_output_comparison result{};
    result.struct_size = sizeof(result);
    if (numeric_kind(candidate.dtype) != 3) {
        for (size_t i = 0; i < count; ++i) {
            const auto a = integer(candidate, i), b = integer(reference, i);
            result.mismatch_count += a.negative != b.negative || a.bits != b.bits;
        }
        result.available = TENSORACLE_HAS_MISMATCH;
        result.mismatch_fraction = static_cast<double>(result.mismatch_count) / count;
        return result;
    }

    // Inspect the original precision before converting to float64, just as the
    // Python API does for finite long doubles outside float64's range.
    for (size_t i = 0; i < count; ++i) {
        result.candidate_nonfinite_count += !std::isfinite(floating(candidate, i));
        result.reference_nonfinite_count += !std::isfinite(floating(reference, i));
    }
    if (result.candidate_nonfinite_count || result.reference_nonfinite_count) return result;

    double candidate_norm = 0, reference_norm = 0, difference_norm = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto a = static_cast<double>(floating(candidate, i));
        const auto b = static_cast<double>(floating(reference, i));
        const auto difference = a - b;
        const auto absolute = std::abs(difference);
        if (std::isnan(absolute) || absolute > result.max_absolute_error)
            result.max_absolute_error = absolute;
        candidate_norm = std::hypot(candidate_norm, a);
        reference_norm = std::hypot(reference_norm, b);
        difference_norm = std::hypot(difference_norm, difference);
        if (with_tolerance && !(absolute <= atol + rtol * std::abs(b) || a == b))
            ++result.outside_tolerance_count;
    }
    result.available = TENSORACLE_HAS_MAX_ABSOLUTE_ERROR;
    if (reference_norm) {
        result.available |= TENSORACLE_HAS_RELATIVE_L2_ERROR;
        result.relative_l2_error = difference_norm / reference_norm;
    }
    if (candidate_norm && reference_norm) {
        result.available |= TENSORACLE_HAS_COSINE_SIMILARITY;
        for (size_t i = 0; i < count; ++i)
            result.cosine_similarity += (static_cast<double>(floating(candidate, i)) / candidate_norm) *
                                        (static_cast<double>(floating(reference, i)) / reference_norm);
    }
    if (with_tolerance) {
        result.available |= TENSORACLE_HAS_TOLERANCE;
        result.outside_tolerance_fraction = static_cast<double>(result.outside_tolerance_count) / count;
    }
    return result;
}

void write_result(tensoracle_output_comparison* output, const tensoracle_output_comparison& result) {
    const auto size = structure_size(output);
#define WRITE_RESULT(Field) write_field(output, size, offsetof(tensoracle_output_comparison, Field), result.Field)
    WRITE_RESULT(available);
    WRITE_RESULT(candidate_nonfinite_count);
    WRITE_RESULT(reference_nonfinite_count);
    WRITE_RESULT(max_absolute_error);
    WRITE_RESULT(relative_l2_error);
    WRITE_RESULT(cosine_similarity);
    WRITE_RESULT(outside_tolerance_count);
    WRITE_RESULT(outside_tolerance_fraction);
    WRITE_RESULT(mismatch_count);
    WRITE_RESULT(mismatch_fraction);
#undef WRITE_RESULT
}
}

extern "C" tensoracle_status tensoracle_compare_outputs(
    const tensoracle_array* candidate, const tensoracle_array* reference,
    const tensoracle_comparison_options* options,
    tensoracle_output_comparison* result, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&]() -> tensoracle_status {
        if (!result || structure_size(result) < sizeof(size_t))
            return fail(TENSORACLE_INVALID_ARGUMENT, "comparison output requires struct_size", error);
        if (!valid_array(candidate) || !valid_array(reference))
            return fail(TENSORACLE_INVALID_ARGUMENT, "arrays require complete fields and non-null storage", error);
        if (candidate->ndim != reference->ndim)
            return fail(TENSORACLE_INVALID_ARGUMENT, "candidate and reference must have the same nonempty shape", error);
        size_t count = 1;
        for (size_t i = 0; i < candidate->ndim; ++i) {
            const auto dimension = candidate->shape[i];
            if (dimension != reference->shape[i] || !dimension ||
                count > std::numeric_limits<size_t>::max() / dimension)
                return fail(TENSORACLE_INVALID_ARGUMENT, "candidate and reference must have the same nonempty shape", error);
            count *= dimension;
        }
        const auto kind = numeric_kind(candidate->dtype);
        if (!kind || kind != numeric_kind(reference->dtype))
            return fail(TENSORACLE_INVALID_ARGUMENT, "candidate and reference must have the same numeric kind", error);
        if (count > std::numeric_limits<size_t>::max() / element_size(candidate->dtype) ||
            count > std::numeric_limits<size_t>::max() / element_size(reference->dtype))
            return fail(TENSORACLE_INVALID_ARGUMENT, "array byte size overflows size_t", error);
        bool with_tolerance = false;
        double atol = 0, rtol = 0;
        auto status = tolerances(options, with_tolerance, atol, rtol, error);
        if (status != TENSORACLE_SUCCESS) return status;
        write_result(result, measure(*candidate, *reference, count, with_tolerance, atol, rtol));
        return TENSORACLE_SUCCESS;
    });
}
