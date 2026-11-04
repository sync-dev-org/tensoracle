#include "builder_fixture.hpp"
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--placeholder") {
        try {
            builder_fixture::context gpu;
            auto engine = tensoracle::open_engine(builder_fixture::options(argv[2]),
                [&](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) {
                    char token = 'x';
                    if (write(std::stoi(argv[3]), &token, 1) != 1 || read(std::stoi(argv[4]), &token, 1) != 1) _exit(4);
                    // A process exiting during construction leaves the real placeholder behind.
                    _exit(0);
                });
        } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
        return 3;
    }
    if (argc != 4) return 2;
    try {
        builder_fixture::context gpu;
        auto engine = tensoracle::open_engine(builder_fixture::options(argv[1], argv[2]),
            [](nvinfer1::INetworkDefinition&, const tensoracle::weights_view&) { throw std::runtime_error("cached callback invoked after exec"); });
        auto values = builder_fixture::run(engine);
        std::ofstream output(argv[3], std::ios::binary); output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        return output ? 0 : 3;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
