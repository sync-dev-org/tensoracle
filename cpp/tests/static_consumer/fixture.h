#ifndef STATIC_CONSUMER_FIXTURE_H
#define STATIC_CONSUMER_FIXTURE_H
#include <tensoracle/builder.h>
#ifndef TENSORACLE_STATIC_LIBRARY
#error "The installed static target must supply its public export definition"
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gpu_fixture {
    void* context;
    void* input;
    void* output;
    tensoracle_cuda_stream stream;
} gpu_fixture;
int fixture_start(gpu_fixture*);
int fixture_check(gpu_fixture*);
void fixture_destroy(gpu_fixture*);
tensoracle_status fixture_add(void*, const tensoracle_weight*) TENSORACLE_NOEXCEPT;
tensoracle_status fixture_graph(void*, void*, const tensoracle_weights*, const char**) TENSORACLE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
