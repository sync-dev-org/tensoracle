#include "builder_fixture.hpp"
#include <tensoracle/cubin_plugin.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("The FP32 cubin preserves values that the TF32 control rounds away", "[SP-TENSORACLE-013][SP-TENSORACLE-014][gpu]") {
    using namespace builder_fixture;
    context gpu;
    int major{}, minor{};
    cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0));
    cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0));
    auto prefix = "sm" + std::to_string(major * 10 + minor);
    auto* ieee = tensoracle::cubin::find_bundle_image((prefix + ".dot.unspecialized").c_str());
    auto* rounded = tensoracle::cubin::find_bundle_image((prefix + ".dot.tf32-control").c_str());
    REQUIRE(ieee); REQUIRE(rounded);
    std::vector<float> input(16 * 32, 1.0001f), matrix(32 * 16, 1.0001f), output(16 * 16);
    device_buffer source(input.size() * sizeof(float)), weight(matrix.size() * sizeof(float)), target(output.size() * sizeof(float));
    stream queue;
    cuda_check(cuMemcpyHtoD(source.value, input.data(), input.size() * sizeof(float)));
    cuda_check(cuMemcpyHtoD(weight.value, matrix.data(), matrix.size() * sizeof(float)));
    int rows = 16;
    void* arguments[]{&source.value, &weight.value, &target.value, &rows};
    auto execute = [&](const tensoracle::cubin::KernelImage& image) {
        tensoracle::cubin::Kernel kernel(image);
        cuda_check(kernel.launch(1, 128, arguments, 4, queue.value));
        cuda_check(cuStreamSynchronize(queue.value));
        cuda_check(cuMemcpyDtoH(output.data(), target.value, output.size() * sizeof(float)));
        return output;
    };
    auto fp32 = execute(*ieee), tf32 = execute(*rounded);
    double expected = 32 * double(input[0]) * double(matrix[0]);
    for (size_t i = 0; i < fp32.size(); ++i) {
        REQUIRE(std::abs(fp32[i] - expected) < 1e-5);
        REQUIRE(tf32[i] == 32.0f);
        REQUIRE(std::abs(fp32[i] - tf32[i]) > 0.003f);
    }
}
