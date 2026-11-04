#include <cstdlib>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
// Inspection must never load this file or execute its initializers.
__attribute__((constructor)) static void never_load() { std::abort(); }
#endif
extern "C" {
extern EXPORT unsigned char const tensoracle_test_constant[] = {1, 2, 0, 4};
EXPORT void* getCreators(int* count) { *count = 0; return nullptr; }
}
