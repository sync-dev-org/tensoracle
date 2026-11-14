#pragma once
#include "builder_fixture.hpp"
#include "test_channel.hpp"
#include <cstdio>

namespace concurrent_builder {
using test_channel::send;
using test_channel::receive;
using test_channel::quiet;
// Each callback writes one append, including callbacks that subsequently fail.
inline void record_call(const std::filesystem::path& weights, const std::string& label) {
    auto path = weights.parent_path() / "calls";
#if defined(_WIN32)
    auto file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("callback record open failed");
    auto line = label + '\n'; DWORD count = 0;
    auto written = WriteFile(file, line.data(), DWORD(line.size()), &count, nullptr);
    CloseHandle(file);
    if (!written || count != line.size()) throw std::runtime_error("callback record write failed");
#else
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("callback record open failed");
    auto line = label + '\n';
    auto written = write(fd, line.data(), line.size());
    close(fd);
    if (written != ssize_t(line.size())) throw std::runtime_error("callback record write failed");
#endif
}
inline int open(const std::filesystem::path& weights, test_channel::endpoint channel, bool force,
                const std::string& name, const std::string& label) {
    using namespace builder_fixture;
    nlohmann::json result;
    try {
        context gpu;
        auto request = options(weights); request.force_rebuild = force; request.name = name;
        send(channel, 'o');
        auto engine = tensoracle::open_engine(request,
            [&](nvinfer1::INetworkDefinition& network, const tensoracle::weights_view& values) {
                record_call(weights, label);
                send(channel, 'b');
                auto command = receive(channel);
                if (command == 'f') throw std::runtime_error("controlled callback failure");
                if (command != 'g') throw std::runtime_error("invalid builder command");
                graph(network, values, "fp32");
            });
        result = {{"path", engine.path()}, {"values", run(engine)}};
    } catch (const tensoracle::error& failure) {
        result = {{"status", failure.status()}, {"error", failure.what()}};
    } catch (const std::exception& failure) {
        result = {{"error", failure.what()}};
    }
    auto successful = result.contains("path");
    std::ofstream(weights.parent_path() / (label + ".json")) << result;
    send(channel, successful ? 's' : 'e');
    return successful ? 0 : 1;
}
}
