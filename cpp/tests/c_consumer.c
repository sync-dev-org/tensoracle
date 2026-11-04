/* SP-TENSORACLE-014, SP-TENSORACLE-055, SP-TENSORACLE-056:
 * C11 callers compare CPU arrays through the public header. */
#include <tensoracle/tensoracle.h>
#include <tensoracle/engine.h>
#include <tensoracle/plugins.h>
#include <tensoracle/cache.h>
#include <tensoracle/builder.h>

int main(void) {
    tensoracle_version version = {sizeof(tensoracle_version), 0, 0, 0};
    tensoracle_error* error = NULL;
    tensoracle_cuda_stream stream = NULL;
    (void)stream;
    if (tensoracle_cache_result_destroy(NULL) != TENSORACLE_SUCCESS) return 5;
    tensoracle_engine* unused_engine = NULL;
    if (tensoracle_open_engine(NULL, NULL, &unused_engine, NULL) != TENSORACLE_INVALID_ARGUMENT || unused_engine) return 6;
    if (tensoracle_get_version(&version, &error) != TENSORACLE_SUCCESS || error) return 1;
    if (version.major != TENSORACLE_VERSION_MAJOR) return 2;
    if (tensoracle_check_version(&version, &error) != TENSORACLE_SUCCESS) return 3;
    const size_t shape[] = {2};
    const uint64_t candidate[] = {UINT64_MAX, 1}, reference[] = {UINT64_MAX - 1, 1};
    const tensoracle_array a = {sizeof(a), TENSORACLE_UINT64, 1, shape, candidate};
    const tensoracle_array b = {sizeof(b), TENSORACLE_UINT64, 1, shape, reference};
    tensoracle_output_comparison result = {0};
    result.struct_size = sizeof(result);
    if (tensoracle_compare_outputs(&a, &b, NULL, &result, &error) != TENSORACLE_SUCCESS || error)
        return 5;
    if (result.available != TENSORACLE_HAS_MISMATCH || result.mismatch_count != 1 ||
        result.mismatch_fraction != 0.5) return 6;
    return tensoracle_error_destroy(error) == TENSORACLE_SUCCESS ? 0 : 4;
}
