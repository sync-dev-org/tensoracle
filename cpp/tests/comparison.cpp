#include <tensoracle/tensoracle.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
template <class T, size_t N>
tensoracle_array array(const T (&data)[N], tensoracle_dtype dtype, const size_t* shape) {
    return {sizeof(tensoracle_array), dtype, 1, shape, data};
}

tensoracle_output_comparison compare(const tensoracle_array& candidate,
                                    const tensoracle_array& reference,
                                    const tensoracle_comparison_options* options = nullptr) {
    return tensoracle::compare_outputs(candidate, reference, options);
}
}

TEST_CASE("Float outputs report distances and reference-relative tolerance misses", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {2};
    const float candidate[] = {2, 1}, reference[] = {1, -1};
    auto left = array(candidate, TENSORACLE_FLOAT32, shape);
    auto right = array(reference, TENSORACLE_FLOAT32, shape);
    tensoracle_comparison_options options{sizeof(options), 1, 1, 0.5, 0.5};
    auto result = compare(left, right, &options);
    REQUIRE(result.candidate_nonfinite_count == 0);
    REQUIRE(result.reference_nonfinite_count == 0);
    REQUIRE(result.available == (TENSORACLE_HAS_MAX_ABSOLUTE_ERROR | TENSORACLE_HAS_RELATIVE_L2_ERROR |
                                TENSORACLE_HAS_COSINE_SIMILARITY | TENSORACLE_HAS_TOLERANCE));
    REQUIRE(result.max_absolute_error == Catch::Approx(2));
    REQUIRE(result.relative_l2_error == Catch::Approx(std::sqrt(2.5)));
    REQUIRE(result.cosine_similarity == Catch::Approx(1 / std::sqrt(10.0)));
    REQUIRE(result.outside_tolerance_count == 1);
    REQUIRE(result.outside_tolerance_fraction == Catch::Approx(0.5));

    const double a[] = {11.5, 1}, b[] = {10, 1};
    left = array(a, TENSORACLE_FLOAT64, shape);
    right = array(b, TENSORACLE_FLOAT64, shape);
    options.atol = 0;
    options.rtol = 0.14;
    REQUIRE(compare(left, right, &options).outside_tolerance_count == 1);
    REQUIRE(compare(right, left, &options).outside_tolerance_count == 0);
    REQUIRE((compare(left, right).available & TENSORACLE_HAS_TOLERANCE) == 0);
}

TEST_CASE("Half and extended precision inputs use float64 arithmetic", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {2};
    const uint16_t half[] = {0x3e00, 0xc000}; // 1.5, -2
    const float reference[] = {1.25f, -2.25f};
    auto result = compare(array(half, TENSORACLE_FLOAT16, shape),
                          array(reference, TENSORACLE_FLOAT32, shape));
    REQUIRE(result.max_absolute_error == Catch::Approx(0.25));
    REQUIRE(result.relative_l2_error == Catch::Approx(std::sqrt(0.125 / 6.625)));
    REQUIRE(result.cosine_similarity == Catch::Approx(6.375 / std::sqrt(6.25 * 6.625)));
    const long double extended[] = {1.5L, -2.0L};
    REQUIRE(compare(array(extended, TENSORACLE_LONG_DOUBLE, shape),
                    array(reference, TENSORACLE_FLOAT32, shape)).relative_l2_error ==
            Catch::Approx(result.relative_l2_error));
}

TEST_CASE("Zero norms suppress only measurements requiring that norm", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {2};
    const double zero[] = {0, 0}, nonzero[] = {1, 2};
    auto z = array(zero, TENSORACLE_FLOAT64, shape);
    auto n = array(nonzero, TENSORACLE_FLOAT64, shape);
    auto result = compare(n, z);
    REQUIRE(result.available == TENSORACLE_HAS_MAX_ABSOLUTE_ERROR);
    REQUIRE(result.max_absolute_error == 2);
    result = compare(z, n);
    REQUIRE(result.available == (TENSORACLE_HAS_MAX_ABSOLUTE_ERROR | TENSORACLE_HAS_RELATIVE_L2_ERROR));
    REQUIRE(result.relative_l2_error == 1);
}

TEST_CASE("Nonfinite values on each side suppress all float measurements", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {3};
    const double invalid[] = {NAN, -INFINITY, 3}, valid[] = {1, 2, 3};
    auto a = array(invalid, TENSORACLE_FLOAT64, shape);
    auto b = array(valid, TENSORACLE_FLOAT64, shape);
    tensoracle_comparison_options options{sizeof(options), 1, 1, 0, 0};
    auto forward = compare(a, b, &options), reverse = compare(b, a, &options);
    REQUIRE(forward.available == 0);
    REQUIRE(forward.candidate_nonfinite_count == 2);
    REQUIRE(forward.reference_nonfinite_count == 0);
    REQUIRE(reverse.available == 0);
    REQUIRE(reverse.candidate_nonfinite_count == 0);
    REQUIRE(reverse.reference_nonfinite_count == 2);
}

TEST_CASE("Large signed and unsigned integers compare exactly without float promotion", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {4};
    const int64_t signed_values[] = {INT64_MIN, -1, INT64_MAX, INT64_C(9007199254740993)};
    const uint64_t unsigned_values[] = {UINT64_C(9223372036854775808), UINT64_MAX,
                                      INT64_MAX, UINT64_C(9007199254740992)};
    auto a = array(signed_values, TENSORACLE_INT64, shape);
    auto b = array(unsigned_values, TENSORACLE_UINT64, shape);
    for (auto result : {compare(a, b), compare(b, a)}) {
        REQUIRE(result.available == TENSORACLE_HAS_MISMATCH);
        REQUIRE(result.mismatch_count == 3);
        REQUIRE(result.mismatch_fraction == 0.75);
        REQUIRE(result.candidate_nonfinite_count == 0);
        REQUIRE(result.reference_nonfinite_count == 0);
    }
    REQUIRE(compare(b, b).mismatch_count == 0);
}

TEST_CASE("Every integer width and bool report exact mismatches", "[SP-TENSORACLE-014]") {
    const size_t shape[] = {2};
    const uint64_t expected[] = {1, 2};
    auto reference = array(expected, TENSORACLE_UINT64, shape);
#define CHECK_INTEGER(Type, Dtype) do { \
    const Type values[] = {1, 3}; \
    auto input = array(values, Dtype, shape); \
    REQUIRE(compare(input, reference).mismatch_count == 1); \
    REQUIRE(compare(reference, input).mismatch_count == 1); \
} while (false)
    CHECK_INTEGER(int8_t, TENSORACLE_INT8);
    CHECK_INTEGER(uint8_t, TENSORACLE_UINT8);
    CHECK_INTEGER(int16_t, TENSORACLE_INT16);
    CHECK_INTEGER(uint16_t, TENSORACLE_UINT16);
    CHECK_INTEGER(int32_t, TENSORACLE_INT32);
    CHECK_INTEGER(uint32_t, TENSORACLE_UINT32);
    CHECK_INTEGER(int64_t, TENSORACLE_INT64);
    CHECK_INTEGER(uint64_t, TENSORACLE_UINT64);
#undef CHECK_INTEGER
    const uint8_t booleans[] = {1, 0}, other[] = {1, 1};
    auto result = compare(array(booleans, TENSORACLE_BOOL, shape), array(other, TENSORACLE_BOOL, shape));
    REQUIRE(result.mismatch_count == 1);
    REQUIRE(result.mismatch_fraction == 0.5);
}

TEST_CASE("Invalid comparison inputs return the shared status and diagnostic", "[SP-TENSORACLE-014][SP-TENSORACLE-056]") {
    const size_t shape[] = {1}, different[] = {2}, empty[] = {0};
    const double value[] = {1};
    auto good = array(value, TENSORACLE_FLOAT64, shape);
    auto bad = good;
    tensoracle_output_comparison result{sizeof(result)};
    auto reject = [&](const tensoracle_array* a, const tensoracle_array* b,
                      const tensoracle_comparison_options* options = nullptr) {
        tensoracle_error* raw = nullptr;
        REQUIRE(tensoracle_compare_outputs(a, b, options, &result, &raw) == TENSORACLE_INVALID_ARGUMENT);
        tensoracle::error_handle owner(raw);
        REQUIRE(owner != nullptr);
    };
    reject(nullptr, &good);
    bad.shape = different; reject(&bad, &good);
    bad.shape = empty; reject(&bad, &bad);
    bad = good; bad.dtype = TENSORACLE_INT64; reject(&bad, &good);
    bad.dtype = 999; reject(&bad, &bad);
    bad = good; bad.data = nullptr; reject(&bad, &good);
    bad = good; bad.shape = nullptr; reject(&bad, &good);
    bad = good; bad.struct_size = sizeof(size_t); reject(&bad, &good);
    tensoracle_comparison_options options{sizeof(options), 1, 0, 0, 0};
    reject(&good, &good, &options);
    options.has_rtol = 1;
    for (auto invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        options.atol = invalid; options.rtol = 0; reject(&good, &good, &options);
        options.rtol = invalid; options.atol = 0; reject(&good, &good, &options);
    }
    options.struct_size = sizeof(size_t); // Missing trailing options mean no tolerances.
    REQUIRE(tensoracle_compare_outputs(&good, &good, &options, &result, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_compare_outputs(&good, &good, nullptr, nullptr, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    result.struct_size = sizeof(size_t) - 1;
    REQUIRE(tensoracle_compare_outputs(&good, &good, nullptr, &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
}

TEST_CASE("Scalar shapes retain stable float measurements for large finite values", "[SP-TENSORACLE-014]") {
    const double value = 1e300;
    tensoracle_array scalar{sizeof(scalar), TENSORACLE_FLOAT64, 0, nullptr, &value};
    auto result = compare(scalar, scalar);
    REQUIRE(result.max_absolute_error == 0);
    REQUIRE(result.relative_l2_error == 0);
    REQUIRE(result.cosine_similarity == 1);
}

TEST_CASE("Comparison output sizes preserve unknown bytes and incomplete fields", "[SP-TENSORACLE-055][SP-TENSORACLE-056]") {
    const size_t shape[] = {1};
    const double value[] = {1};
    auto input = array(value, TENSORACLE_FLOAT64, shape);
    alignas(tensoracle_output_comparison)
        std::array<unsigned char, sizeof(tensoracle_output_comparison) + 16> storage;
    for (size_t size = sizeof(size_t); size <= storage.size(); ++size) {
        storage.fill(0xa5);
        std::memcpy(storage.data(), &size, sizeof(size));
        auto* output = reinterpret_cast<tensoracle_output_comparison*>(storage.data());
        REQUIRE(tensoracle_compare_outputs(&input, &input, nullptr, output, nullptr) == TENSORACLE_SUCCESS);
        size_t actual_size = 0;
        std::memcpy(&actual_size, storage.data(), sizeof(actual_size));
        REQUIRE(actual_size == size);
        for (size_t i = size; i < storage.size(); ++i) REQUIRE(storage[i] == 0xa5);
        const size_t offset = offsetof(tensoracle_output_comparison, max_absolute_error);
        if (size > offset && size < offset + sizeof(double))
            for (size_t i = offset; i < offset + sizeof(double); ++i) REQUIRE(storage[i] == 0xa5);
    }
}
