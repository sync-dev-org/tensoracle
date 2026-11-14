#pragma once
#include "test_platform.hpp"
#include <array>
#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#endif
namespace test_channel {
#if defined(_WIN32)
using endpoint = intptr_t;
inline HANDLE handle(endpoint value) { return reinterpret_cast<HANDLE>(value); }
inline std::array<endpoint, 2> pair() {
    static std::atomic<unsigned> sequence{0};
    auto name = L"\\\\.\\pipe\\tensoracle-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(sequence++);
    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    auto server = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1, 4096, 4096, 0, &security);
    if (server == INVALID_HANDLE_VALUE) throw std::runtime_error("test channel create failed");
    auto client = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, &security, OPEN_EXISTING, 0, nullptr);
    if (client == INVALID_HANDLE_VALUE) { CloseHandle(server); throw std::runtime_error("test channel connect failed"); }
    if (!ConnectNamedPipe(server, nullptr) && GetLastError() != ERROR_PIPE_CONNECTED) {
        CloseHandle(server); CloseHandle(client); throw std::runtime_error("test channel accept failed");
    }
    SetHandleInformation(server, HANDLE_FLAG_INHERIT, 0);
    return {reinterpret_cast<intptr_t>(server), reinterpret_cast<intptr_t>(client)};
}
inline void close(endpoint value) { if (value >= 0) CloseHandle(handle(value)); }
inline void shutdown(endpoint value) { DisconnectNamedPipe(handle(value)); }
inline void send(endpoint value, char token) {
    DWORD count = 0;
    if (!WriteFile(handle(value), &token, 1, &count, nullptr) || count != 1) throw std::runtime_error("test channel write failed");
}
inline bool readable(endpoint value, int timeout) {
    auto deadline = GetTickCount64() + timeout;
    do {
        DWORD count = 0;
        if (!PeekNamedPipe(handle(value), nullptr, 0, nullptr, &count, nullptr)) throw std::runtime_error("test channel closed");
        if (count) return true;
        Sleep(5);
    } while (GetTickCount64() < deadline);
    return false;
}
inline char receive(endpoint value, int timeout = 20000) {
    if (!readable(value, timeout)) throw std::runtime_error("builder channel timed out");
    DWORD count = 0; char token;
    if (!ReadFile(handle(value), &token, 1, &count, nullptr) || count != 1) throw std::runtime_error("test channel read failed");
    return token;
}
inline bool quiet(endpoint value) { return !readable(value, 300); }
#else
using endpoint = int;
inline std::array<endpoint, 2> pair() {
    std::array<int, 2> values{};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, values.data())) throw std::runtime_error("socketpair failed");
    fcntl(values[0], F_SETFD, FD_CLOEXEC);
    return values;
}
inline void close(endpoint value) { ::close(value); }
inline void shutdown(endpoint value) { ::shutdown(value, SHUT_RDWR); }
inline void send(endpoint channel, char value) {
    ssize_t count;
    do { count = ::send(channel, &value, 1, MSG_NOSIGNAL); } while (count < 0 && errno == EINTR);
    if (count != 1) throw std::runtime_error("builder channel write failed");
}
inline char receive(endpoint channel, int timeout = 20000) {
    pollfd event{channel, POLLIN, 0};
    int result;
    do { result = poll(&event, 1, timeout); } while (result < 0 && errno == EINTR);
    if (result <= 0) throw std::runtime_error("builder channel timed out");
    char value;
    if (read(channel, &value, 1) != 1) throw std::runtime_error("builder channel closed");
    return value;
}
inline bool quiet(endpoint channel) { pollfd event{channel, POLLIN, 0}; return poll(&event, 1, 300) == 0; }
#endif
}
