#include "builder_fixture.hpp"
#include "concurrent_builder.hpp"
#include <future>
#include <csignal>
#include "builder_internal.hpp"
#include <tensoracle/cache.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace builder_fixture;
namespace {
struct builder_process {
    pid_t pid = -1;
    int channel = -1;
    builder_process(const fs::path& weights, const std::string& label, bool force = false,
                    const std::string& name = "linear-fp32") {
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair)) throw std::runtime_error("socketpair failed");
        auto number = std::to_string(pair[1]);
        pid = fork();
        if (pid == 0) {
            close(pair[0]); fcntl(pair[1], F_SETFD, 0);
            execl(TENSORACLE_BUILDER_PROBE, TENSORACLE_BUILDER_PROBE, "--concurrent",
                weights.c_str(), number.c_str(), force ? "force" : "reuse", name.c_str(), label.c_str(), nullptr);
            _exit(127);
        }
        close(pair[1]); channel = pair[0];
        if (pid < 0) { close(channel); throw std::runtime_error("fork failed"); }
    }
    ~builder_process() {
        if (pid > 0) { kill(pid, SIGKILL); waitpid(pid, nullptr, 0); }
        close(channel);
    }
    char next() { return concurrent_builder::receive(channel); }
    void command(char value) { concurrent_builder::send(channel, value); }
    bool quiet() { return concurrent_builder::quiet(channel); }
    int finish() {
        int status{};
        if (waitpid(pid, &status, 0) != pid) throw std::runtime_error("waitpid failed");
        pid = -1; return status;
    }
};
struct builder_thread {
    int channel = -1;
    std::future<int> task;
    builder_thread(const fs::path& weights, const std::string& label) {
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair)) throw std::runtime_error("socketpair failed");
        channel = pair[0];
        task = std::async(std::launch::async, [weights, label, endpoint = pair[1]] {
            struct cleanup { int fd; ~cleanup() { close(fd); } } release{endpoint};
            return concurrent_builder::open(weights, endpoint, false, "linear-fp32", label);
        });
    }
    ~builder_thread() { shutdown(channel, SHUT_RDWR); if (task.valid()) task.wait(); close(channel); }
    char next() { return concurrent_builder::receive(channel); }
    void command(char value) { concurrent_builder::send(channel, value); }
    bool quiet() { return concurrent_builder::quiet(channel); }
};
nlohmann::json outcome(const fs::path& weights, const std::string& label) {
    std::ifstream input(weights.parent_path() / (label + ".json"));
    return nlohmann::json::parse(input);
}
std::vector<std::string> calls(const fs::path& weights) {
    std::ifstream input(weights.parent_path() / "calls");
    std::vector<std::string> result;
    for (std::string line; std::getline(input, line);) result.push_back(line);
    return result;
}
void successful(builder_process& process) {
    REQUIRE(process.next() == 's');
    auto status = process.finish(); REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 0);
}
void failed(builder_process& process, const fs::path& weights, const std::string& label) {
    REQUIRE(process.next() == 'e');
    auto status = process.finish(); REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 1);
    REQUIRE(outcome(weights, label).at("status") == TENSORACLE_INTERNAL_ERROR);
    REQUIRE(outcome(weights, label).at("error") == "controlled callback failure");
}
void reusable(const fs::path& weights, const std::string& label, const std::string& name = "linear-fp32") {
    context gpu;
    auto request = options(weights); request.name = name;
    auto engine = tensoracle::open_engine(request,
        [](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { throw std::runtime_error("saved engine was lost"); });
    auto result = outcome(weights, label);
    REQUIRE(engine.path() == result.at("path").get<std::string>());
    REQUIRE(run(engine) == result.at("values").get<std::vector<float>>());
}
}

TEST_CASE("Concurrent first opens in separate processes build once and all reuse the published engine", "[SP-TENSORACLE-009][SP-TENSORACLE-067][SP-TENSORACLE-046][SP-TENSORACLE-049][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first");
    REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process second(weights, "second"), third(weights, "third");
    REQUIRE(second.next() == 'o'); REQUIRE(third.next() == 'o');
    CHECK(second.quiet()); CHECK(third.quiet());
    REQUIRE(calls(weights) == std::vector<std::string>{"first"});
    auto listing = tensoracle::list_cache(); auto entries = listing.entries();
    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].status == TENSORACLE_CACHE_BUILDING_OR_LEFTOVER);
    REQUIRE(entries[0].size_bytes == 0);
    for (const auto& entry : fs::directory_iterator(files.path / "tensoracle")) {
        auto name = entry.path().filename().string();
        REQUIRE((entry.path().extension() == ".engine" || (name.rfind("weight-", 0) == 0 && entry.path().extension() == ".json")));
    }
    first.command('g'); successful(first); successful(second); successful(third);
    REQUIRE(calls(weights) == std::vector<std::string>{"first"});
    REQUIRE(outcome(weights, "first") == outcome(weights, "second"));
    REQUIRE(outcome(weights, "first") == outcome(weights, "third"));
    reusable(weights, "first");
}

TEST_CASE("Concurrent first opens in separate threads build once and return independent usable engines", "[SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_thread first(weights, "first");
    REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_thread second(weights, "second"); REQUIRE(second.next() == 'o');
    CHECK(second.quiet());
    first.command('g'); REQUIRE(first.next() == 's'); REQUIRE(second.next() == 's');
    REQUIRE(first.task.get() == 0); REQUIRE(second.task.get() == 0);
    REQUIRE(calls(weights) == std::vector<std::string>{"first"});
    REQUIRE(outcome(weights, "first") == outcome(weights, "second"));
    reusable(weights, "second");
}

TEST_CASE("A waiting process rebuilds after the first network callback fails", "[SP-TENSORACLE-011][SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first"); REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process second(weights, "second"); REQUIRE(second.next() == 'o'); CHECK(second.quiet());
    first.command('f'); failed(first, weights, "first");
    REQUIRE(second.next() == 'b'); second.command('g'); successful(second);
    REQUIRE(calls(weights) == std::vector<std::string>{"first", "second"});
    reusable(weights, "second");
}

TEST_CASE("A waiting process rebuilds after SIGKILL releases the first builder lock", "[SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first"); REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process second(weights, "second"); REQUIRE(second.next() == 'o'); CHECK(second.quiet());
    REQUIRE(kill(first.pid, SIGKILL) == 0);
    auto status = first.finish(); REQUIRE(WIFSIGNALED(status)); REQUIRE(WTERMSIG(status) == SIGKILL);
    REQUIRE(second.next() == 'b'); second.command('g'); successful(second);
    REQUIRE(calls(weights) == std::vector<std::string>{"first", "second"});
    reusable(weights, "second");
}

TEST_CASE("Forced rebuilds wait for the same engine and survive an earlier callback failure", "[SP-TENSORACLE-009][SP-TENSORACLE-011][SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    for (bool first_forced : {false, true}) {
        INFO("first builder forced=" << first_forced);
        fs::remove_all(files.path / "tensoracle"); fs::remove(files.path / "calls");
        builder_process first(weights, "first", first_forced);
        REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
        builder_process second(weights, "second", !first_forced); REQUIRE(second.next() == 'o');
        CHECK(second.quiet());
        first.command('f'); failed(first, weights, "first");
        REQUIRE(second.next() == 'b'); second.command('g'); successful(second);
        REQUIRE(calls(weights) == std::vector<std::string>{"first", "second"});
        reusable(weights, "second");
    }
}

TEST_CASE("Different engine conditions can enter their network callbacks together", "[SP-TENSORACLE-009][SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first", false, "first-engine");
    REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process second(weights, "second", false, "second-engine");
    REQUIRE(second.next() == 'o'); REQUIRE(second.next() == 'b');
    REQUIRE(calls(weights) == std::vector<std::string>{"first", "second"});
    first.command('g'); second.command('g'); successful(first); successful(second);
    REQUIRE(outcome(weights, "first").at("path") != outcome(weights, "second").at("path"));
    REQUIRE(outcome(weights, "first").at("values") == outcome(weights, "second").at("values"));
    reusable(weights, "first", "first-engine"); reusable(weights, "second", "second-engine");
}

TEST_CASE("Overlapping forced rebuilds each build serially after the previous engine is published", "[SP-TENSORACLE-011][SP-TENSORACLE-067][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first", true); REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process second(weights, "second", true); REQUIRE(second.next() == 'o'); CHECK(second.quiet());
    first.command('g'); successful(first);
    REQUIRE(second.next() == 'b'); second.command('g'); successful(second);
    REQUIRE(calls(weights) == std::vector<std::string>{"first", "second"});
    REQUIRE(outcome(weights, "first").at("path") == outcome(weights, "second").at("path"));
    REQUIRE(outcome(weights, "first").at("values") == outcome(weights, "second").at("values"));
    reusable(weights, "second");
}

TEST_CASE("A failed forced rebuild retains the earlier complete engine for waiting readers", "[SP-TENSORACLE-009][SP-TENSORACLE-011][SP-TENSORACLE-067][SP-TENSORACLE-049][gpu]") {
    temporary_directory files; auto weights = files.path / "weights"; write_weights(weights);
    builder_process first(weights, "first"); REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    builder_process forced(weights, "forced", true); REQUIRE(forced.next() == 'o'); CHECK(forced.quiet());
    first.command('g'); successful(first);
    REQUIRE(forced.next() == 'b');
    auto listing = tensoracle::list_cache(); auto entries = listing.entries();
    REQUIRE(entries.size() == 1); REQUIRE(entries[0].status == TENSORACLE_CACHE_RECOGNIZED);
    std::ifstream original(outcome(weights, "first").at("path").get<std::string>(), std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(original), {}};
    builder_process reader(weights, "reader"); REQUIRE(reader.next() == 'o'); CHECK(reader.quiet());
    forced.command('f'); failed(forced, weights, "forced"); successful(reader);
    REQUIRE(calls(weights) == std::vector<std::string>{"first", "forced"});
    REQUIRE(outcome(weights, "reader") == outcome(weights, "first"));
    std::ifstream retained(outcome(weights, "first").at("path").get<std::string>(), std::ios::binary);
    const bool unchanged = std::string(std::istreambuf_iterator<char>(retained), {}) == bytes;
    REQUIRE(unchanged);
    reusable(weights, "reader");
}

TEST_CASE("Saved LeakyReLU engines stay small and reuse correct half precision outputs", "[SP-TENSORACLE-009][gpu]") {
    context gpu; temporary_directory files; auto path = files.path / "weights"; write_weights(path);
    auto request = options(path); request.name = "leaky-relu";
    request.workspace_limit = 4ULL * 1024 * 1024 * 1024; request.optimization_level = 3;
    const std::vector<int64_t> dimensions{1, 64, 135, 240};
    request.inputs = {{"features", dimensions, dimensions, dimensions}};
    request.io = {{"features", TENSORACLE_FLOAT16, dimensions, true}, {"result", TENSORACLE_FLOAT16, dimensions, false}};
    int calls = 0;
    auto define = [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view&) {
        ++calls;
        auto* input = network.addInput("features", nvinfer1::DataType::kHALF, nvinfer1::Dims4(1, 64, 135, 240));
        REQUIRE(input);
        auto* activation = network.addActivation(*input, nvinfer1::ActivationType::kLEAKY_RELU);
        REQUIRE(activation); activation->setAlpha(0.1f);
        auto* result = activation->getOutput(0); REQUIRE(result);
        result->setName("result"); network.markOutput(*result);
    };
    auto started = std::chrono::steady_clock::now();
    auto engine = tensoracle::open_engine(request, define);
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    auto bytes = fs::file_size(engine.path());
    std::printf("LeakyReLU saved bytes=%llu build seconds=%.6f\n", (unsigned long long)bytes, elapsed);
    // Without a warm timing cache, TensorRT 11.4 serializes about 29 MB for this one layer.
    CHECK(bytes < 1024 * 1024);
    REQUIRE(calls == 1);

    // IEEE binary16 values: [-10, -5, -2.5, -1.25, 0, 1, 2, 4].
    const uint16_t input_pattern[]{0xc900, 0xc500, 0xc100, 0xbd00, 0, 0x3c00, 0x4000, 0x4400};
    // LeakyReLU(x, 0.1): [-1, -0.5, -0.25, -0.125, 0, 1, 2, 4].
    const uint16_t output_pattern[]{0xbc00, 0xb800, 0xb400, 0xb000, 0, 0x3c00, 0x4000, 0x4400};
    const size_t count = 64 * 135 * 240;
    std::vector<uint16_t> values(count), expected(count), actual(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = input_pattern[i % 8]; expected[i] = output_pattern[i % 8];
    }
    device_buffer source(count * sizeof(uint16_t)), target(count * sizeof(uint16_t)); stream queue;
    cuda_check(cuMemcpyHtoD(source.value, values.data(), count * sizeof(uint16_t)));
    auto run_activation = [&](const tensoracle::engine_handle& saved) {
        auto execution = saved.create_execution();
        execution.enqueue({{"features", reinterpret_cast<void*>(source.value)}, {"result", reinterpret_cast<void*>(target.value)}}, queue.value);
        cuda_check(cuStreamSynchronize(queue.value));
        cuda_check(cuMemcpyDtoH(actual.data(), target.value, count * sizeof(uint16_t)));
        REQUIRE(std::equal(actual.begin(), actual.end(), expected.begin()));
    };
    run_activation(engine);
    auto reopened = tensoracle::open_engine(request, define);
    REQUIRE(calls == 1); REQUIRE(reopened.path() == engine.path());
    run_activation(reopened);
}

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
