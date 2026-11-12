// SP-TENSORACLE-054, SP-TENSORACLE-073, SP-TENSORACLE-074:
// The same installed static target supports the public C++ wrappers.
#include "fixture.h"
#include <tensoracle/builder.hpp>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    gpu_fixture gpu{};
    try {
        tensoracle::check_version();
        if (argc == 1) return 0;
        if (argc != 3 || fixture_start(&gpu)) { fixture_destroy(&gpu); return 2; }
        int result = 1;
        {
            tensoracle::builder_options options;
            options.name = "static-cpp"; options.graph_version = "1"; options.weights = {argv[1]};
            options.inputs = {{"input", {4}, {4}, {4}}};
            options.io = {{"input", TENSORACLE_FLOAT32, {4}, true}, {"output", TENSORACLE_FLOAT32, {4}, false}};
            options.workspace_limit = 64 * 1024 * 1024; options.optimization_level = 0;
            int calls = 0;
            auto engine = tensoracle::open_engine(options,
                [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) {
                    ++calls;
                    auto bias = weights.at("bias");
                    if (fixture_add(&network, &bias) != TENSORACLE_SUCCESS)
                        throw std::runtime_error("Network addition failed");
                });
            if (calls != std::atoi(argv[2])) throw std::runtime_error("Unexpected builder callback count");
            auto execution = engine.create_execution();
            execution.enqueue({{"input", gpu.input}, {"output", gpu.output}}, gpu.stream);
            result = fixture_check(&gpu);
        }
        fixture_destroy(&gpu);
        return result;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what()); fixture_destroy(&gpu); return 1;
    }
}
