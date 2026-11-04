#ifndef TENSORACLE_TENSORACLE_H
#define TENSORACLE_TENSORACLE_H

#include <stddef.h>
#include <stdint.h>

#define TENSORACLE_VERSION_MAJOR 2
#define TENSORACLE_VERSION_MINOR 0
#define TENSORACLE_VERSION_PATCH 0

#if defined(_WIN32)
#  if defined(TENSORACLE_BUILDING_LIBRARY)
#    define TENSORACLE_API __declspec(dllexport)
#  else
#    define TENSORACLE_API __declspec(dllimport)
#  endif
#else
#  define TENSORACLE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#  define TENSORACLE_NOEXCEPT noexcept
extern "C" {
#else
#  define TENSORACLE_NOEXCEPT
#endif

/* Status values have a fixed-width representation across the C boundary. */
typedef int32_t tensoracle_status;
enum {
    TENSORACLE_SUCCESS = 0,
    TENSORACLE_INVALID_ARGUMENT = 1,
    TENSORACLE_INCOMPATIBLE_VERSION = 2,
    TENSORACLE_OUT_OF_MEMORY = 3,
    TENSORACLE_INTERNAL_ERROR = 4
};

typedef struct tensoracle_error tensoracle_error;
typedef struct CUstream_st* tensoracle_cuda_stream;

/* CPU array storage uses native byte order and contiguous C element order.
 * FLOAT16 is IEEE binary16; LONG_DOUBLE uses the host C long double format.
 * BOOL occupies one byte, with any nonzero value representing true. */
typedef int32_t tensoracle_dtype;
enum {
    TENSORACLE_BOOL = 1,
    TENSORACLE_INT8 = 2,
    TENSORACLE_UINT8 = 3,
    TENSORACLE_INT16 = 4,
    TENSORACLE_UINT16 = 5,
    TENSORACLE_INT32 = 6,
    TENSORACLE_UINT32 = 7,
    TENSORACLE_INT64 = 8,
    TENSORACLE_UINT64 = 9,
    TENSORACLE_FLOAT16 = 10,
    TENSORACLE_FLOAT32 = 11,
    TENSORACLE_FLOAT64 = 12,
    TENSORACLE_LONG_DOUBLE = 13,
    /* TensorRT tensor types, unsupported by the CPU comparison operation. */
    TENSORACLE_BFLOAT16 = 14,
    TENSORACLE_FLOAT8 = 15,
    TENSORACLE_INT4 = 16,
    TENSORACLE_FLOAT4 = 17,
    TENSORACLE_E8M0 = 18
};

/* All fields are required. data and the ndim shape entries are borrowed for
 * the call. ndim == 0 denotes one scalar and permits a NULL shape. */
typedef struct tensoracle_array {
    size_t struct_size;
    tensoracle_dtype dtype;
    size_t ndim;
    const size_t* shape;
    const void* data;
} tensoracle_array;

/* NULL options or absent trailing fields mean no tolerances. Both presence
 * flags must agree. A present tolerance requires its complete value field. */
typedef struct tensoracle_comparison_options {
    size_t struct_size;
    uint32_t has_atol;
    uint32_t has_rtol;
    double atol;
    double rtol;
} tensoracle_comparison_options;

enum {
    TENSORACLE_HAS_MAX_ABSOLUTE_ERROR = 1,
    TENSORACLE_HAS_RELATIVE_L2_ERROR = 2,
    TENSORACLE_HAS_COSINE_SIMILARITY = 4,
    TENSORACLE_HAS_TOLERANCE = 8,
    TENSORACLE_HAS_MISMATCH = 16
};

/* Nonfinite counts are always present. available identifies defined metrics;
 * missing values are zero-filled and must be ignored. TOLERANCE and MISMATCH
 * each cover both their count and fraction. Zero reference norm omits relative
 * L2 error; either zero norm omits cosine similarity. A nonfinite input omits
 * all float metrics. Integers and bool only provide mismatch metrics. */
typedef struct tensoracle_output_comparison {
    size_t struct_size;
    uint32_t available;
    uint64_t candidate_nonfinite_count;
    uint64_t reference_nonfinite_count;
    double max_absolute_error;
    double relative_l2_error;
    double cosine_similarity;
    uint64_t outside_tolerance_count;
    double outside_tolerance_fraction;
    uint64_t mismatch_count;
    double mismatch_fraction;
} tensoracle_output_comparison;

/* Initialize struct_size to the number of bytes actually supplied. Output
 * functions preserve struct_size, write only complete fields inside that size,
 * and leave unknown trailing bytes untouched. A size_t-only output is valid.
 * Fields may be appended within a major release; existing fields never move. */
typedef struct tensoracle_version {
    size_t struct_size;
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} tensoracle_version;

/* error is optional. When supplied, *error is set to NULL on success and to an
 * owned diagnostic on failure. Release it with tensoracle_error_destroy before
 * reusing the slot. Error text is borrowed until that handle is destroyed.
 * No C++ exception crosses any of these functions. None initialize CUDA. */
TENSORACLE_API tensoracle_status tensoracle_get_version(
    tensoracle_version* version, tensoracle_error** error) TENSORACLE_NOEXCEPT;

/* Requires at least the major field. A missing minor field means zero. The
 * loaded major must match and its minor must be >= the header's minor. Patch
 * differences do not break ABI compatibility. Unknown input fields are ignored. */
TENSORACLE_API tensoracle_status tensoracle_check_version(
    const tensoracle_version* header_version, tensoracle_error** error) TENSORACLE_NOEXCEPT;

TENSORACLE_API tensoracle_status tensoracle_get_tensorrt_version(
    tensoracle_version* version, tensoracle_error** error) TENSORACLE_NOEXCEPT;

/* Compare equal, nonempty shapes with matching numeric kinds (float, integer,
 * or bool). Signed and unsigned integers may be mixed and compare exactly.
 * Float arithmetic uses float64 and tolerance is relative to reference.
 * Output follows the struct_size contract above and is unchanged on failure.
 * No CUDA device or stream is needed for this CPU operation. */
TENSORACLE_API tensoracle_status tensoracle_compare_outputs(
    const tensoracle_array* candidate, const tensoracle_array* reference,
    const tensoracle_comparison_options* options,
    tensoracle_output_comparison* result, tensoracle_error** error) TENSORACLE_NOEXCEPT;

/* Reading a NULL handle or using a NULL message output pointer is invalid;
 * diagnostic follows the same optional ownership contract as other operations.
 * Destroying NULL succeeds and destruction never fails. */
TENSORACLE_API tensoracle_status tensoracle_error_message(
    const tensoracle_error* error, const char** message,
    tensoracle_error** diagnostic) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_error_destroy(
    tensoracle_error* error) TENSORACLE_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#endif
