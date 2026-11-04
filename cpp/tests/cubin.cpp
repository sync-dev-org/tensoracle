#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <tensoracle/cubin.hpp>

namespace fs = std::filesystem;
namespace {
struct Directory {
    fs::path path = fs::temp_directory_path() /
                    ("tensoracle-cubin-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Directory() { fs::create_directories(path); }
    ~Directory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
void write(fs::path const &path, std::string const &text) {
    std::ofstream(path, std::ios::binary) << text;
}
} // namespace

TEST_CASE("C++ callers inspect cubin architectures without initializing CUDA",
          "[SP-TENSORACLE-004][SP-TENSORACLE-056]") {
    std::string bytes(64, '\0');
    bytes.replace(0, 6, "\177ELF\2\1", 6);
    bytes[8] = 7;
    bytes[18] = static_cast<char>(190);
    bytes[50] = 86;
    REQUIRE(tensoracle::cubin_sm(bytes.data(), bytes.size()) == 86);
    bytes[50] = 90;
    try {
        tensoracle::cubin_sm(bytes.data(), bytes.size());
        FAIL("unsupported SM accepted");
    } catch (tensoracle::error const &error) {
        REQUIRE(error.status() == TENSORACLE_CUBIN_ARCHITECTURE_ERROR);
    }
}

TEST_CASE("C++ producer rejects ptxas outside CUDA 13.2 before writing a bundle",
          "[SP-TENSORACLE-037][SP-TENSORACLE-058]") {
    Directory directory;
    auto ptxas = directory.path / "ptxas";
    write(ptxas, "#!/bin/sh\nprintf 'ptxas V13.1.115\\n'\n");
    fs::permissions(ptxas, fs::perms::owner_all);
    auto output = (directory.path / "bundle").string();
    auto tool = ptxas.string();
    tensoracle_producer_options options{};
    options.struct_size = sizeof(options);
    options.output_directory = output.c_str();
    options.ptxas = tool.c_str();
    try {
        tensoracle::compile_bundle(nullptr, 0, options);
        FAIL("old ptxas accepted");
    } catch (tensoracle::error const &error) {
        REQUIRE(error.status() == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(std::string(error.what()).find(">=13.2,<14") != std::string::npos);
        REQUIRE(std::string(error.what()).find("13.1.115") != std::string::npos);
    }
    REQUIRE_FALSE(fs::exists(output));
}

TEST_CASE("C ABI clears bundle result handles on invalid arguments", "[SP-TENSORACLE-056]") {
    tensoracle_bundle_result *result = reinterpret_cast<tensoracle_bundle_result *>(1);
    tensoracle_error *diagnostic = nullptr;
    REQUIRE(tensoracle_inspect_bundle(nullptr, nullptr, &result, &diagnostic) ==
            TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(result == nullptr);
    REQUIRE(diagnostic != nullptr);
    REQUIRE(tensoracle_error_destroy(diagnostic) == TENSORACLE_SUCCESS);
    REQUIRE(tensoracle_bundle_result_destroy(nullptr) == TENSORACLE_SUCCESS);
}

TEST_CASE("C++ producer compiles CUDA source and verifies all three SMs without a GPU",
          "[SP-TENSORACLE-004][SP-TENSORACLE-038][SP-TENSORACLE-058]") {
    Directory directory;
    const char *root = TENSORACLE_TEST_CUDA_ROOT;
    if (!fs::exists(fs::path(root) / "lib/libnvrtc.so.13") ||
        !fs::exists(fs::path(root) / "bin/ptxas"))
        SKIP("CUDA 13.2 producer wheels are not installed in the supplied CUDA root");
    auto output = (directory.path / "bundle").string();
    tensoracle_kernel_argument x{sizeof(x), "x", "*fp32", 1};
    tensoracle_kernel_argument y{sizeof(y), "y", "*fp32", 1};
    tensoracle_kernel_argument n{sizeof(n), "n", "i32", 1};
    const tensoracle_kernel_argument *arguments[]{&x, &y, &n};
    tensoracle_compile_constant factor{sizeof(factor), "FACTOR", 2};
    const tensoracle_compile_constant *constants[]{&factor};
    tensoracle_kernel_spec spec{
        sizeof(spec),
        "copy",
        "extern \"C\" __global__ void copy_kernel(const float* x, float* y, int n) {"
        "int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]*FACTOR; }",
        "copy_kernel",
        arguments,
        3,
        constants,
        1,
        0};
    const tensoracle_kernel_spec *kernels[]{&spec};
    tensoracle_producer_options options{sizeof(options), output.c_str(), root,
                                        nullptr,         nullptr,        nullptr};
    auto result = tensoracle::compile_bundle(kernels, 1, options);
    REQUIRE(result.sha256().size() == 64);
    REQUIRE(result.manifest_json().find("hidden_trailing_parameters") == std::string::npos);
    for (int sm : {86, 89, 120}) {
        auto path = directory.path / "bundle/cubins" / ("sm" + std::to_string(sm)) / "copy.cubin";
        REQUIRE(fs::file_size(path) > 64);
    }
    REQUIRE(tensoracle::inspect_bundle(output.c_str()).sha256() == result.sha256());
}
