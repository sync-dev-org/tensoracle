#ifndef TENSORACLE_PLUGINS_H
#define TENSORACLE_PLUGINS_H
#include "tensoracle.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { TENSORACLE_PLUGIN_LIBRARY_ERROR = 110 };
typedef struct tensoracle_plugin_libraries tensoracle_plugin_libraries;
/* Inspect getCreators before loading. Library handles remain registered for the
 * lifetime of the process. Byte-identical files load once; changing bytes at an
 * already loaded canonical path is an error. Returned digests own their bytes
 * independently of the supplied paths, which are borrowed only for this call.
 * Empty input is valid. Registration may partially succeed before an error. */
TENSORACLE_API tensoracle_status tensoracle_register_libraries(
    const char* const* paths, size_t count, tensoracle_plugin_libraries** result,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Register creators already linked into this process, without loading a plugin
 * file. Each creator is an nvinfer1::IPluginCreatorInterface*. module_address
 * must point into the library/executable containing their implementation.
 * plugin_namespace is the TensorRT registration namespace (empty is valid).
 * The result contains one SHA-256: the COMPLETE containing shared library, or
 * the executable for statically linked code. Reading that file is required for
 * identity, but registration itself uses the supplied creators in memory.
 * Repeating the same creators and namespace is valid. Conflicting creators or
 * changed bytes at a registered path fail. Registration may partially succeed.
 * The caller must keep the creators and their containing module alive for the
 * lifetime of the process; destroying the result never deregisters them.
 * The creator array and namespace string are borrowed only for this call;
 * *result is NULL on failure.
 * Currently supported on Linux; Windows reports PLUGIN_LIBRARY_ERROR. */
TENSORACLE_API tensoracle_status tensoracle_register_embedded_plugins(
    const void* module_address, void* const* creators, size_t count,
    const char* plugin_namespace, tensoracle_plugin_libraries** result,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_plugin_libraries_count(
    const tensoracle_plugin_libraries* libraries, size_t* count,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* The returned digest is borrowed until the result handle is destroyed. */
TENSORACLE_API tensoracle_status tensoracle_plugin_libraries_digest(
    const tensoracle_plugin_libraries* libraries, size_t index, const char** digest,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Frees only the result, never deregisters or unloads a library. NULL is valid. */
TENSORACLE_API tensoracle_status tensoracle_plugin_libraries_destroy(
    tensoracle_plugin_libraries* libraries) TENSORACLE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
