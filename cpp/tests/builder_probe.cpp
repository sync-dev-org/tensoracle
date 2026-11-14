#include "test_platform.hpp"
#include "builder_fixture.hpp"
#include "concurrent_builder.hpp"
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(_WIN32)
int wmain(int argc, wchar_t** values) {
#else
int main(int argc, char** values) {
#endif
    auto argv = test_platform::utf8_arguments(argc, values);
    if (argc == 7 && std::string(argv[1]) == "--concurrent")
        return concurrent_builder::open(std::filesystem::u8path(argv[2]), static_cast<test_channel::endpoint>(std::stoll(argv[3])), argv[4] == "force", argv[5], argv[6]);
    if (argc != 4) return 2;
    try {
        builder_fixture::context gpu;
        auto engine = tensoracle::open_engine(builder_fixture::options(std::filesystem::u8path(argv[1]), argv[2]),
            [](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { throw std::runtime_error("cached callback invoked after exec"); });
        auto values = builder_fixture::run(engine);
        std::ofstream output(std::filesystem::u8path(argv[3]), std::ios::binary); output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        return output ? 0 : 3;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
