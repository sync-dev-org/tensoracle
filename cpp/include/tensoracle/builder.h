#ifndef TENSORACLE_BUILDER_H
#define TENSORACLE_BUILDER_H
#include "engine.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct tensoracle_weights tensoracle_weights;
/* Borrowed views into declared safetensors. Values remain valid until the
 * outer open_engine call returns, including after the network callback returns.
 * Values are contiguous little-endian scalars; data is naturally aligned.
 * Tensor names must be unique across ALL declared files. A duplicate rejects
 * the build before its callback, identifying the name and both source paths.
 * Supported storage types: BOOL, I/U8/16/32/64, F16, BF16, F32, F64 and
 * F8_E4M3. Unsupported encodings fail instead of being reinterpreted. */
typedef struct tensoracle_weight {
    size_t struct_size;
    const char* name;
    tensoracle_dtype dtype;
    size_t rank;
    const int64_t* dimensions;
    const void* data;
    uint64_t element_count;
    uint64_t byte_size;
} tensoracle_weight;
TENSORACLE_API tensoracle_status tensoracle_weights_count(const tensoracle_weights*, size_t*, tensoracle_error**) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_weights_at(const tensoracle_weights*, size_t, tensoracle_weight*, tensoracle_error**) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_weights_find(const tensoracle_weights*, const char*, tensoracle_weight*, tensoracle_error**) TENSORACLE_NOEXCEPT;

/* network borrows nvinfer1::INetworkDefinition*. Callback/context are borrowed
 * only during this call and never retained. On failure return a nonzero status
 * and optionally a message valid until open_engine returns (literal/context
 * storage, never a local stack buffer). The library copies it to its existing
 * owned error handle. No C++ exception may escape this callback. */
typedef tensoracle_status (*tensoracle_build_network_fn)(void* context, void* network,
    const tensoracle_weights* weights, const char** message) TENSORACLE_NOEXCEPT;
typedef struct tensoracle_network_builder {
    size_t struct_size;
    void* context;
    tensoracle_build_network_fn build;
} tensoracle_network_builder;
typedef struct tensoracle_input_profile {
    size_t struct_size;
    const char* name;
    size_t rank;
    const int64_t* minimum;
    const int64_t* optimum;
    const int64_t* maximum;
} tensoracle_input_profile;
typedef struct tensoracle_tensor_spec {
    size_t struct_size;
    const char* name;
    tensoracle_dtype dtype;
    size_t rank;
    const int64_t* dimensions;
    uint32_t input;
    uint32_t has_location;
    int32_t location; /* TensorRT TensorLocation */
    uint32_t has_format;
    int32_t format; /* TensorRT TensorFormat */
} tensoracle_tensor_spec;
typedef struct tensoracle_builder_options {
    size_t struct_size;
    const char* name;
    const char* graph_version;
    const char* const* weight_files;
    size_t weight_count;
    const tensoracle_input_profile* const* inputs;
    size_t input_count;
    const tensoracle_tensor_spec* const* io;
    size_t io_count;
    const char* const* plugins;
    size_t plugin_count;
    uint32_t has_workspace_limit;
    int64_t workspace_limit;
    int32_t optimization_level;
    uint32_t force_rebuild;
} tensoracle_builder_options;
/* name/version/files/profiles/IO fields are required. Missing trailing fields
 * mean no plugins, default TensorRT workspace, optimization level 3, no force.
 * Present workspace must be positive; optimization level is 0..5. No precision
 * or TF32 option: network tensor types determine precision and TF32 is disabled.
 * All inputs are borrowed only for this call. Profile/IO arrays use pointers so
 * extending a structure never changes the stride. The callback is skipped for
 * a valid cache hit. Shape inference outputs (markOutputForShapes) fail BEFORE
 * serialization; no warning policy or partial engine is saved. */
TENSORACLE_API tensoracle_status tensoracle_open_engine(const tensoracle_builder_options*,
    const tensoracle_network_builder*, tensoracle_engine**, tensoracle_error**) TENSORACLE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
