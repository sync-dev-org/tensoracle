/* SP-TENSORACLE-054, SP-TENSORACLE-073, SP-TENSORACLE-074:
 * C callers build and run an engine with the installed static target. */
#include "fixture.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    tensoracle_version version = {sizeof(version), 0, 0, 0};
    if (tensoracle_get_version(&version, NULL) != TENSORACLE_SUCCESS ||
        version.major != TENSORACLE_VERSION_MAJOR) return 1;
    if (argc == 1) return 0;
    if (argc != 3) return 2;
    gpu_fixture gpu = {0};
    if (fixture_start(&gpu)) { fixture_destroy(&gpu); return 3; }
    const int64_t shape[] = {4};
    tensoracle_input_profile profile = {sizeof(profile), "input", 1, shape, shape, shape};
    const tensoracle_input_profile* profiles[] = {&profile};
    tensoracle_tensor_spec input = {0}, output = {0};
    input.struct_size = sizeof(input); input.name = "input";
    input.dtype = TENSORACLE_FLOAT32; input.rank = 1; input.dimensions = shape; input.input = 1;
    output = input; output.name = "output"; output.input = 0;
    const tensoracle_tensor_spec* io[] = {&input, &output};
    const char* files[] = {argv[1]};
    tensoracle_builder_options options = {0};
    options.struct_size = sizeof(options); options.name = "static-c"; options.graph_version = "1";
    options.weight_files = files; options.weight_count = 1;
    options.inputs = profiles; options.input_count = 1; options.io = io; options.io_count = 2;
    options.has_workspace_limit = 1; options.workspace_limit = 64 * 1024 * 1024;
    int calls = 0, result = 4;
    tensoracle_network_builder builder = {sizeof(builder), &calls, fixture_graph};
    tensoracle_engine* engine = NULL;
    tensoracle_execution* execution = NULL;
    tensoracle_error* error = NULL;
    if (tensoracle_open_engine(&options, &builder, &engine, &error) != TENSORACLE_SUCCESS) goto cleanup;
    if (calls != atoi(argv[2])) goto cleanup;
    if (tensoracle_execution_create(engine, &execution, &error) != TENSORACLE_SUCCESS) goto cleanup;
    tensoracle_tensor_address source = {sizeof(source), "input", gpu.input};
    tensoracle_tensor_address target = {sizeof(target), "output", gpu.output};
    const tensoracle_tensor_address* addresses[] = {&source, &target};
    if (tensoracle_execution_enqueue(execution, addresses, 2, gpu.stream, &error) != TENSORACLE_SUCCESS) goto cleanup;
    result = fixture_check(&gpu);
cleanup:
    if (error) {
        const char* message = NULL;
        if (tensoracle_error_message(error, &message, NULL) == TENSORACLE_SUCCESS)
            fprintf(stderr, "%s\n", message);
    }
    tensoracle_error_destroy(error);
    tensoracle_execution_destroy(execution); tensoracle_engine_destroy(engine);
    fixture_destroy(&gpu);
    return result;
}
