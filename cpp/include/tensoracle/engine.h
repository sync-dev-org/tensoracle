#ifndef TENSORACLE_ENGINE_H
#define TENSORACLE_ENGINE_H
#include "tensoracle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensoracle_engine tensoracle_engine;
typedef struct tensoracle_execution tensoracle_execution;

enum {
    TENSORACLE_CORRUPT_ENGINE = 100,
    TENSORACLE_ENGINE_CONDITION_MISMATCH = 101,
    TENSORACLE_ENGINE_IO_MISMATCH = 102
};

typedef struct tensoracle_input_shape {
    size_t struct_size;
    const char* name;
    int32_t rank;
    const int64_t* dimensions;
} tensoracle_input_shape;

typedef struct tensoracle_tensor_address {
    size_t struct_size;
    const char* name;
    void* address;
} tensoracle_tensor_address;

/* Output uses the same prefix/trailing-byte rules as tensoracle_version.
 * dimensions has TensorRT's maximum rank of eight. */
typedef struct tensoracle_tensor_info {
    size_t struct_size;
    tensoracle_dtype dtype;
    int32_t rank;
    int64_t dimensions[8];
} tensoracle_tensor_info;

/* Execution handles retain the engine independently of this handle. */
TENSORACLE_API tensoracle_status tensoracle_engine_destroy(tensoracle_engine* engine) TENSORACLE_NOEXCEPT;
/* Borrowed path to the validated saved engine, until this handle is destroyed. */
TENSORACLE_API tensoracle_status tensoracle_engine_path(
    const tensoracle_engine* engine, const char** path, tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_engine_tensor_count(
    const tensoracle_engine* engine, size_t* count, tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* The returned name is borrowed until the engine handle is destroyed. */
TENSORACLE_API tensoracle_status tensoracle_engine_tensor_name(
    const tensoracle_engine* engine, size_t index, const char** name,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;

/* DEVICE tensors with dimensions inferable from shapes are supported. External
 * shape inputs, shape inference outputs, HOST tensors and data-dependent
 * outputs are rejected. Ordinary shape-layer outputs remain supported. */
TENSORACLE_API tensoracle_status tensoracle_execution_create(
    const tensoracle_engine* engine, tensoracle_execution** execution,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_execution_destroy(tensoracle_execution* execution) TENSORACLE_NOEXCEPT;
/* Supply every ranged input exactly once; omit fixed and single-point inputs.
 * Arrays contain pointers so appending fields never changes their ABI stride.
 * Invalid arguments leave prior shapes intact. TensorRT failures require a
 * complete successful setting before querying shapes or enqueuing again. */
TENSORACLE_API tensoracle_status tensoracle_execution_set_input_shapes(
    tensoracle_execution* execution, const tensoracle_input_shape* const* shapes,
    size_t count, tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_execution_tensor_info(
    tensoracle_execution* execution, const char* name, tensoracle_tensor_info* info,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Dtype may be queried before ranged input shapes are set. */
TENSORACLE_API tensoracle_status tensoracle_execution_tensor_dtype(
    const tensoracle_execution* execution, const char* name, tensoracle_dtype* dtype,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Supply every IO tensor exactly once, with a non-null device address. There
 * are no transfers or synchronization. NULL stream selects the default stream.
 * Caller owns buffer correctness, lifetime, device and ordering. Finish work
 * before mutating shapes, closing, or using this execution from another thread.
 *
 * For CUDA graph capture: use a non-default stream and enqueue once after each
 * shape change before capture. Shapes, addresses and this execution are fixed
 * in that graph. Loops, conditionals and layers/plugins synchronizing with the
 * host may not be captured. The caller creates, recreates and destroys graphs. */
TENSORACLE_API tensoracle_status tensoracle_execution_enqueue(
    tensoracle_execution* execution, const tensoracle_tensor_address* const* addresses,
    size_t count, tensoracle_cuda_stream stream, tensoracle_error** error) TENSORACLE_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#endif
