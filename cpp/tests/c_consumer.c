#define _POSIX_C_SOURCE 200809L
/* SP-TENSORACLE-014, SP-TENSORACLE-055, SP-TENSORACLE-056, SP-TENSORACLE-061:
 * C11 callers compare CPU arrays through the public header. */
#include <tensoracle/tensoracle.h>
#include <tensoracle/engine.h>
#include <tensoracle/plugins.h>
#include <tensoracle/cache.h>
#include <tensoracle/builder.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int read_weights(void) {
    char path[] = "/tmp/tensoracle-c-weights-XXXXXX";
    const char header[] = "{\"__metadata__\":{\"format\":\"test\"},"
                          "\"bias\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    int fd = mkstemp(path);
    if (fd < 0) return 10;
    FILE* file = fdopen(fd, "wb");
    if (!file) { close(fd); unlink(path); return 11; }
    uint64_t length = sizeof(header) - 1;
    for (int i = 0; i < 8; ++i) fputc((int)((length >> (8 * i)) & 255), file);
    fwrite(header, 1, (size_t)length, file);
    fputc(7, file);
    if (fclose(file)) { unlink(path); return 12; }
    tensoracle_weights* weights = NULL;
    tensoracle_error* error = NULL;
    int result = 13;
    if (tensoracle_weights_open(path, &weights, &error) != TENSORACLE_SUCCESS || error) goto cleanup;
    size_t count = 0;
    if (tensoracle_weights_count(weights, &count, &error) != TENSORACLE_SUCCESS || count != 1) goto cleanup;
    tensoracle_weight tensor = {0}; tensor.struct_size = sizeof(tensor);
    if (tensoracle_weights_at(weights, 0, &tensor, &error) != TENSORACLE_SUCCESS) goto cleanup;
    if (strcmp(tensor.name, "bias") || tensor.dtype != TENSORACLE_INT8 || tensor.rank != 1 ||
        tensor.dimensions[0] != 1 || tensor.element_count != 1 || tensor.byte_size != 1 ||
        *(const int8_t*)tensor.data != 7) goto cleanup;
    if (tensoracle_weights_metadata_count(weights, &count, &error) != TENSORACLE_SUCCESS || count != 1) goto cleanup;
    tensoracle_weights_metadata_entry entry = {0}; entry.struct_size = sizeof(entry);
    if (tensoracle_weights_metadata_at(weights, 0, &entry, &error) != TENSORACLE_SUCCESS ||
        entry.key_size != 6 || strcmp(entry.key, "format") || entry.value_size != 4 || strcmp(entry.value, "test")) goto cleanup;
    if (tensoracle_weights_metadata_find(weights, "format", &entry, &error) != TENSORACLE_SUCCESS ||
        entry.value_size != 4 || strcmp(entry.value, "test")) goto cleanup;
    result = 0;
cleanup:
    tensoracle_error_destroy(error);
    tensoracle_weights_destroy(weights);
    unlink(path);
    return result;
}

int main(void) {
    int weights_result = read_weights();
    if (weights_result) return weights_result;
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
