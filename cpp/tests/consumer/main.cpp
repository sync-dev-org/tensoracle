#include <tensoracle/tensoracle.hpp>
#include <tensoracle/engine.hpp>
#include <tensoracle/plugins.hpp>
#include <tensoracle/cache.hpp>
#include <tensoracle/builder.hpp>
#include <functional>
#include <iostream>

// Compile the complete opening call using only installed tensoracle headers.
// Application graph code can include TensorRT headers at its own build boundary;
// this non-GPU consumer validates the callback signature and C ABI linkage.
[[maybe_unused]] static tensoracle::engine_handle application_open(const std::string& weights) {
    tensoracle::builder_options options;
    options.name = "application"; options.graph_version = "1"; options.weights = {weights};
    options.inputs = {{"x", {1}, {1}, {1}}};
    options.io = {{"x", TENSORACLE_FLOAT32, {1}, true}, {"y", TENSORACLE_FLOAT32, {1}, false}};
    std::function<void(nvinfer1::INetworkDefinition&, const tensoracle::weights_view&)> callback =
        [](nvinfer1::INetworkDefinition&, const tensoracle::weights_view& values) {
            (void)values.size(); // The application's TensorRT graph implementation goes here.
        };
    return tensoracle::open_engine(options, callback);
}

int main() {
    tensoracle::check_version();
    auto own = tensoracle::version();
    auto trt = tensoracle::tensorrt_version();
    if (own.major != TENSORACLE_VERSION_MAJOR || trt.major != 11) return 1;
    tensoracle::engine_handle empty;
    tensoracle::engine_execution execution;
    tensoracle::cache_result cache;
    execution.close();
    if (!tensoracle::register_libraries({}).empty()) return 2;
    const size_t shape[] = {2};
    const double candidate[] = {2, 1}, reference[] = {1, 1};
    const tensoracle_array a{sizeof(a), TENSORACLE_FLOAT64, 1, shape, candidate};
    const tensoracle_array b{sizeof(b), TENSORACLE_FLOAT64, 1, shape, reference};
    auto comparison = tensoracle::compare_outputs(a, b);
    if (!(comparison.available & TENSORACLE_HAS_MAX_ABSOLUTE_ERROR) ||
        comparison.max_absolute_error != 1) return 2;
    std::cout << own.major << '.' << own.minor << '.' << own.patch << '\n';
    return 0;
}
