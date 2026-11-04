#include "builder_fixture.hpp"
#include "builder_internal.hpp"
#include <tensoracle/cache.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace builder_fixture;
TEST_CASE("Public builder retains weights through serialization and reuses ranged named engines", "[SP-TENSORACLE-007][SP-TENSORACLE-008][SP-TENSORACLE-009][SP-TENSORACLE-052][SP-TENSORACLE-059][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "linear.safetensors"; write_weights(path);
    auto request = options(path, "fp32", true); int calls = 0;
    auto define = [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) { ++calls; graph(network, weights, "fp32", true); };
    auto engine = tensoracle::open_engine(request, define);
    REQUIRE(calls == 1); REQUIRE(fs::exists(engine.path()));
    for (int rows : {1, 8, 16}) REQUIRE(run(engine, rows, true).size() == size_t(rows * 16));
    request.workspace_limit.reset();
    auto reopened = tensoracle::open_engine(request, define);
    REQUIRE(calls == 1); REQUIRE(reopened.path() == engine.path());
    request.name = "another-role";
    auto other = tensoracle::open_engine(request, define); REQUIRE(other.path() != engine.path()); REQUIRE(calls == 2);
    request.name = "linear-fp32"; request.force_rebuild = true;
    auto forced = tensoracle::open_engine(request, define); REQUIRE(calls == 3); REQUIRE(forced.path() == engine.path());
    auto original_time = fs::last_write_time(path); write_weights(path, 0.01f); fs::last_write_time(path, original_time);
    auto changed = tensoracle::open_engine(request, define);
    REQUIRE(calls == 4); REQUIRE(changed.path() != engine.path()); REQUIRE(fs::exists(engine.path()));
    REQUIRE(run(changed, 8, true) != run(engine, 8, true));
    auto entries = tensoracle::list_cache();
    for (const auto& entry : entries.entries()) if (std::string(entry.file) == fs::path(forced.path()).filename())
        REQUIRE(nlohmann::json::parse(entry.header_json)["workspace_limit"].is_null());
}
TEST_CASE("FP32 FP16 BF16 and explicit INT8 networks compute caller limits and reopen after exec", "[SP-TENSORACLE-009][SP-TENSORACLE-013][SP-TENSORACLE-020][SP-TENSORACLE-059][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "linear.safetensors"; write_weights(path);
    auto build = [&](const std::string& precision) { return tensoracle::open_engine(options(path, precision),
        [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) { graph(network, weights, precision); }); };
    auto reference_engine = build("fp32"); auto reference = run(reference_engine);
    // Independent CPU matrix product from the deterministic fixture's source values.
    std::mt19937 random(29); std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<float> matrix(512), bias(16); for (auto& v : matrix) v = distribution(random); for (auto& v : bias) v = distribution(random) / 2;
    auto x = input();
    for (int row = 0; row < 16; ++row) for (int col = 0; col < 16; ++col) {
        double expected = bias[col]; for (int k = 0; k < 32; ++k) expected += double(x[row * 32 + k]) * matrix[k * 16 + col];
        REQUIRE(std::abs(reference[row * 16 + col] - expected) < 2e-6);
    }
    for (const auto& precision : {std::string("fp16"), std::string("bf16"), std::string("int8")}) {
        INFO(precision);
        auto engine = build(precision); auto actual = run(engine); float maximum = 0;
        for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
        std::printf("%s max absolute error %.9g\n", precision.c_str(), double(maximum));
        REQUIRE(maximum > 1e-5f);
        REQUIRE(maximum < (precision == "fp16" ? 0.003f : precision == "bf16" ? 0.015f : 0.03f));
        auto output = files.path / (precision + ".bin");
        auto pid = fork(); REQUIRE(pid >= 0);
        if (!pid) { execl(TENSORACLE_BUILDER_PROBE, TENSORACLE_BUILDER_PROBE, path.c_str(), precision.c_str(), output.c_str(), nullptr); _exit(127); }
        int status; REQUIRE(waitpid(pid, &status, 0) == pid); REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 0);
        std::vector<float> restarted(actual.size()); std::ifstream saved(output, std::ios::binary); saved.read(reinterpret_cast<char*>(restarted.data()), restarted.size() * sizeof(float));
        REQUIRE(saved.good()); REQUIRE(restarted == actual);
    }
}
TEST_CASE("FP8 explicitly quantized networks reuse identical outputs after exec on sm89 or sm120", "[SP-TENSORACLE-009][SP-TENSORACLE-013][SP-TENSORACLE-014][gpu]") {
    context gpu; int major, minor; cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0)); cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0));
    if (major * 10 + minor != 89 && major * 10 + minor != 120)
        SKIP("FP8 reuse requires sm89 or sm120; the detected GPU is sm" + std::to_string(major * 10 + minor));
    temporary_directory files; auto path = files.path / "weights"; write_weights(path);
    auto engine = tensoracle::open_engine(options(path, "fp8"), [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) { graph(network, weights, "fp8"); });
    auto reference_engine = tensoracle::open_engine(options(path), [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) { graph(network, weights, "fp32"); });
    auto actual = run(engine), reference = run(reference_engine); float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
    REQUIRE(maximum > 1e-5f); REQUIRE(maximum < 0.13f);
    auto output = files.path / "fp8.bin";
    auto pid = fork(); REQUIRE(pid >= 0);
    if (!pid) { execl(TENSORACLE_BUILDER_PROBE, TENSORACLE_BUILDER_PROBE, path.c_str(), "fp8", output.c_str(), nullptr); _exit(127); }
    int status{}; REQUIRE(waitpid(pid, &status, 0) == pid); REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 0);
    std::vector<float> restarted(actual.size());
    std::ifstream saved(output, std::ios::binary); saved.read(reinterpret_cast<char*>(restarted.data()), restarted.size() * sizeof(float));
    REQUIRE(saved.good()); REQUIRE(restarted == actual);
}

TEST_CASE("An empty engine being built in another process is rejected and can be force rebuilt", "[SP-TENSORACLE-009][SP-TENSORACLE-011][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "weights"; write_weights(path);
    int ready[2], release[2]; REQUIRE(pipe(ready) == 0); REQUIRE(pipe(release) == 0);
    auto pid = fork(); REQUIRE(pid >= 0);
    if (!pid) {
        close(ready[0]); close(release[1]);
        auto ready_fd = std::to_string(ready[1]), release_fd = std::to_string(release[0]);
        execl(TENSORACLE_BUILDER_PROBE, TENSORACLE_BUILDER_PROBE, "--placeholder", path.c_str(), ready_fd.c_str(), release_fd.c_str(), nullptr);
        _exit(127);
    }
    close(ready[1]); close(release[0]); char token;
    REQUIRE(read(ready[0], &token, 1) == 1);
    auto request = options(path); int calls = 0;
    auto define = [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) { ++calls; graph(network, weights, "fp32"); };
    try { auto engine = tensoracle::open_engine(request, define); FAIL("empty in-progress file accepted"); }
    catch (const tensoracle::error& failure) { REQUIRE(failure.status() == TENSORACLE_CORRUPT_ENGINE); }
    REQUIRE(calls == 0);
    REQUIRE(write(release[1], "x", 1) == 1); close(ready[0]); close(release[1]);
    int status{}; REQUIRE(waitpid(pid, &status, 0) == pid); REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 0);
    fs::path placeholder;
    for (const auto& entry : fs::directory_iterator(files.path / "tensoracle"))
        if (entry.path().extension() == ".engine") placeholder = entry.path();
    REQUIRE_FALSE(placeholder.empty()); REQUIRE(fs::file_size(placeholder) == 0);
    request.force_rebuild = true;
    auto recovered = tensoracle::open_engine(request, define);
    REQUIRE(calls == 1); REQUIRE(recovered.path() == placeholder); REQUIRE(fs::file_size(placeholder) > 0);
    REQUIRE(run(recovered).size() == 16 * 16);
}
TEST_CASE("Builder settings disable TF32 and preserve the default or explicit workspace", "[SP-TENSORACLE-013][SP-TENSORACLE-052][gpu]") {
    context gpu;
    struct logger : nvinfer1::ILogger { void log(Severity, const char*) noexcept override {} } log;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(log)); REQUIRE(builder);
    std::unique_ptr<nvinfer1::IBuilderConfig> baseline(builder->createBuilderConfig());
    tensoracle::detail::cache_request request; request.optimization_level = 2;
    auto automatic = tensoracle::detail::configure_builder(*builder, request);
    REQUIRE_FALSE(automatic->getFlag(nvinfer1::BuilderFlag::kTF32));
    REQUIRE(automatic->getBuilderOptimizationLevel() == 2);
    REQUIRE(automatic->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE) == baseline->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE));
    request.workspace_limit = 64 * 1024 * 1024;
    auto explicit_limit = tensoracle::detail::configure_builder(*builder, request);
    REQUIRE(explicit_limit->getMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE) == *request.workspace_limit);
}
TEST_CASE("Callback failures duplicate weights and shape outputs leave no saved engine", "[SP-TENSORACLE-005][SP-TENSORACLE-011][SP-TENSORACLE-059][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "weights"; write_weights(path); auto request = options(path);
    try {
        auto engine = tensoracle::open_engine(request, [](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { throw std::runtime_error("caller failure retained"); }); FAIL("callback failure ignored");
    } catch (const tensoracle::error& failure) { REQUIRE(failure.status() == TENSORACLE_INTERNAL_ERROR); REQUIRE(std::string(failure.what()) == "caller failure retained"); }
    const char* weight_path = request.weights[0].c_str();
    tensoracle_builder_options raw{}; raw.struct_size = sizeof(raw); raw.name = "raw-error"; raw.graph_version = "1"; raw.weight_files = &weight_path; raw.weight_count = 1;
    int raw_calls = 0;
    tensoracle_network_builder callback{sizeof(callback), &raw_calls,
        [](void* state, void*, const tensoracle_weights*, const char** message) noexcept -> tensoracle_status {
            ++*static_cast<int*>(state); *message = "C callback status retained"; return TENSORACLE_INVALID_ARGUMENT;
        }};
    tensoracle_engine* raw_engine = nullptr; tensoracle_error* diagnostic = nullptr;
    REQUIRE(tensoracle_open_engine(&raw, &callback, &raw_engine, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
    tensoracle::error_handle owned(diagnostic); const char* message = nullptr;
    REQUIRE(raw_engine == nullptr); REQUIRE(raw_calls == 1);
    REQUIRE(tensoracle_error_message(owned.get(), &message, nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(message) == "C callback status retained");
    auto other = files.path / "other"; fs::copy_file(path, other); request.weights.push_back(other.string()); bool called = false;
    REQUIRE_THROWS_AS(tensoracle::open_engine(request, [&](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { called = true; }), tensoracle::error);
    REQUIRE_FALSE(called);
    request.weights.pop_back();
    request.io.push_back({"dimensions", TENSORACLE_INT64, {2}, false});
    try {
        auto engine = tensoracle::open_engine(request, [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) {
            graph(network, weights, "fp32"); auto* shape = network.addShape(*network.getInput(0))->getOutput(0); shape->setName("dimensions"); REQUIRE(network.markOutputForShapes(*shape));
        });
        FAIL("shape inference output was accepted");
    } catch (const tensoracle::error& failure) {
        REQUIRE(failure.status() == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(std::string(failure.what()).find("shape") != std::string::npos);
        REQUIRE(std::string(failure.what()).find("dimensions") != std::string::npos);
    }
    REQUIRE(tensoracle::list_cache().entries().empty());
}

TEST_CASE("Ordinary shape layer outputs remain supported by the public builder", "[SP-TENSORACLE-007][SP-TENSORACLE-054][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "weights"; write_weights(path); auto request = options(path);
    request.io.push_back({"dimensions", TENSORACLE_INT64, {2}, false});
    auto engine = tensoracle::open_engine(request, [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) {
        graph(network, weights, "fp32"); auto* shape = network.addShape(*network.getInput(0))->getOutput(0);
        shape->setName("dimensions"); network.markOutput(*shape);
    });
    REQUIRE(engine.create_execution().get_tensor_shape("dimensions") == std::vector<int64_t>{2});
}

TEST_CASE("Build success and failure return unused host allocations after releasing mapped weights", "[SP-TENSORACLE-054][SP-TENSORACLE-059][gpu]") {
#if !defined(__GLIBC__)
    SKIP("allocator RSS check requires glibc malloc_trim");
#else
    context gpu; temporary_directory files; auto path = files.path / "matrix.safetensors";
    nlohmann::json header{{"matrix", {{"dtype", "F32"}, {"shape", {512, 512}}, {"data_offsets", {0, 512 * 512 * 4}}}}};
    auto encoded = header.dump(); while (encoded.size() % 8) encoded += ' ';
    std::vector<float> values(512 * 512, 1.0f / 512);
    { std::ofstream file(path, std::ios::binary); char length[8]; for (int i = 0; i < 8; ++i) length[i] = char(uint64_t(encoded.size()) >> (8 * i));
      file.write(length, 8); file << encoded; file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float)); }
    auto rss = [] { std::ifstream stat("/proc/self/statm"); uint64_t virtual_pages, resident; stat >> virtual_pages >> resident; return resident * uint64_t(sysconf(_SC_PAGESIZE)); };
    for (bool failure : {false, true}) {
        auto request = options(path); request.name = failure ? "memory-failure" : "memory-success";
        request.inputs = {{"features", {1, 512}, {1, 512}, {1, 512}}};
        request.io = {{"features", TENSORACLE_FLOAT32, {1, 512}, true}, {"result", TENSORACLE_FLOAT32, {1, 512}, false}};
        tensoracle::engine_handle engine;
        try {
            engine = tensoracle::open_engine(request, [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& weights) {
                auto weight = weights.at("matrix"); auto* value = network.addInput("features", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(1, 512));
                for (int layer = 0; layer < 64; ++layer) {
                    auto* matrix = network.addConstant(nvinfer1::Dims2(512, 512), {nvinfer1::DataType::kFLOAT, weight.data, int64_t(weight.element_count)})->getOutput(0);
                    value = network.addMatrixMultiply(*value, nvinfer1::MatrixOperation::kNONE, *matrix, nvinfer1::MatrixOperation::kNONE)->getOutput(0);
                }
                if (!failure) { value->setName("result"); network.markOutput(*value); }
            });
            REQUIRE_FALSE(failure);
        } catch (const tensoracle::error& error) { REQUIRE(failure); REQUIRE(error.status() == TENSORACLE_INTERNAL_ERROR); }
        auto before = rss(); (void)malloc_trim(0); auto after = rss(); auto reduction = before > after ? before - after : 0;
        std::printf("memory failure=%d RSS=%llu second_trim=%llu\n", failure, (unsigned long long)before, (unsigned long long)reduction);
        REQUIRE(reduction <= 32ULL * 1024 * 1024);
    }
#endif
}
