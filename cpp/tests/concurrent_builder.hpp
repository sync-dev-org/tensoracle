#pragma once
#include "builder_fixture.hpp"
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>

namespace concurrent_builder {
inline void send(int channel, char value) {
    ssize_t count;
    do { count = ::send(channel, &value, 1, MSG_NOSIGNAL); } while (count < 0 && errno == EINTR);
    if (count != 1) throw std::runtime_error("builder channel write failed");
}
inline char receive(int channel, int timeout = 20000) {
    pollfd event{channel, POLLIN, 0};
    int result;
    do { result = poll(&event, 1, timeout); } while (result < 0 && errno == EINTR);
    if (result <= 0) throw std::runtime_error("builder channel timed out");
    char value;
    if (read(channel, &value, 1) != 1) throw std::runtime_error("builder channel closed");
    return value;
}
inline bool quiet(int channel) {
    pollfd event{channel, POLLIN, 0};
    return poll(&event, 1, 300) == 0;
}
// Each callback writes one append, including callbacks that subsequently fail.
inline void record_call(const std::filesystem::path& weights, const std::string& label) {
    auto path = weights.parent_path() / "calls";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("callback record open failed");
    auto line = label + '\n';
    auto written = write(fd, line.data(), line.size());
    close(fd);
    if (written != ssize_t(line.size())) throw std::runtime_error("callback record write failed");
}
inline int open(const std::filesystem::path& weights, int channel, bool force,
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
