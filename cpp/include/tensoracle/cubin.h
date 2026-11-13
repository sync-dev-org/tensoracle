#ifndef TENSORACLE_CUBIN_H
#define TENSORACLE_CUBIN_H
#include "tensoracle.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
    TENSORACLE_BUNDLE_ERROR = 20,
    TENSORACLE_MISSING_CUBIN = 21,
    TENSORACLE_CORRUPT_CUBIN = 22,
    TENSORACLE_CUBIN_ARCHITECTURE_ERROR = 23,
    TENSORACLE_PLUGIN_BUNDLE_MISMATCH = 24,
    TENSORACLE_COMPILATION_ERROR = 25,
    TENSORACLE_TOOL_NOT_FOUND = 26
};

/* All strings are UTF-8, NUL terminated and borrowed for the call. Argument and
 * constant arrays are pointer arrays so each element can grow independently. */
typedef struct tensoracle_kernel_argument {
    size_t struct_size;
    const char *name;
    const char *type;      /* *fp16, *fp32, *i8, *i32, i32, fp32 */
    uint32_t divisible_by; /* power of two; 1 means no specialization */
} tensoracle_kernel_argument;

typedef struct tensoracle_compile_constant {
    size_t struct_size;
    const char *name;
    int64_t value; /* supplied as a CUDA C++ preprocessor definition */
} tensoracle_compile_constant;

typedef struct tensoracle_kernel_spec {
    size_t struct_size;
    const char *name;
    const char *source;       /* CUDA C++, never a Python/Triton function */
    const char *entry_symbol; /* extern "C" __global__ entry */
    const tensoracle_kernel_argument *const *arguments;
    size_t argument_count;
    const tensoracle_compile_constant *const *constants;
    size_t constant_count;
    uint32_t dynamic_shared_memory_bytes;
} tensoracle_kernel_spec;

typedef struct tensoracle_producer_options {
    size_t struct_size;
    const char *output_directory;
    const char *cuda_root; /* PyPI nvidia/cu13 directory; NULL uses wheel layout */
    const char *ptxas;     /* optional independent override */
    const char *cuobjdump; /* optional independent override */
    const char *nvrtc;     /* optional independent shared library override */
} tensoracle_producer_options;

typedef struct tensoracle_bundle_result tensoracle_bundle_result;
/* Borrowed fields remain valid until the result handle is destroyed. */
typedef struct tensoracle_bundle_view {
    size_t struct_size;
    const char *manifest_json;
    const char *sha256;
} tensoracle_bundle_view;

/* Failed operations set *result to NULL. No operation initializes CUDA or loads
 * a plugin. Inspection reads exported constants directly from the library file. */
TENSORACLE_API tensoracle_status tensoracle_compile_bundle(
    const tensoracle_kernel_spec *const *specs, size_t count,
    const tensoracle_producer_options *options, tensoracle_bundle_result **result,
    tensoracle_error **error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status
tensoracle_inspect_bundle(const char *root, const char *plugin, tensoracle_bundle_result **result,
                          tensoracle_error **error) TENSORACLE_NOEXCEPT;
typedef struct tensoracle_bundle_file {
    size_t struct_size;
    const char *name; /* Relative file name from the manifest. */
    const void *data;
    size_t size;
} tensoracle_bundle_file;
/* Inspect manifest and cubin bytes already in memory. All input is borrowed
 * only during this call. Names must be unique. No files are opened, no plugin
 * is loaded, and CUDA is not initialized. Uses the same manifest, size, digest
 * and architecture checks and error codes as directory inspection. The JSON
 * length excludes a terminating NUL. Initialize every file's struct_size. */
TENSORACLE_API tensoracle_status tensoracle_inspect_embedded_bundle(
    const void *manifest_json, size_t manifest_size,
    const tensoracle_bundle_file *const *files, size_t count,
    tensoracle_bundle_result **result, tensoracle_error **error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_inspect_wheel(
    const char *wheel, const char *plugin_member, tensoracle_bundle_result **result,
    tensoracle_error **error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status
tensoracle_bundle_result_get(const tensoracle_bundle_result *result, tensoracle_bundle_view *view,
                             tensoracle_error **error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_bundle_result_destroy(tensoracle_bundle_result *result)
    TENSORACLE_NOEXCEPT;
/* Digest excludes bundle_sha256 itself. result contains canonical JSON and digest. */
TENSORACLE_API tensoracle_status
tensoracle_bundle_digest(const char *manifest_json, tensoracle_bundle_result **result,
                         tensoracle_error **error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_cubin_sm(const void *data, size_t size, uint32_t *sm,
                                                     tensoracle_error **error) TENSORACLE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
