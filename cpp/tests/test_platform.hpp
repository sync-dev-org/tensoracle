#pragma once
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace test_platform {
template<class Char> std::vector<std::string> utf8_arguments(int argc, Char** argv) {
    std::vector<std::string> result;
    for (int i = 0; i < argc; ++i) {
        if constexpr (std::is_same_v<Char, wchar_t>) result.push_back(std::filesystem::path(argv[i]).u8string());
        else result.emplace_back(argv[i]);
    }
    return result;
}
#if defined(_WIN32)
inline constexpr auto os_name = "Windows";
#else
inline constexpr auto os_name = "Linux";
#endif
inline std::filesystem::path make_directory() {
    static std::atomic<unsigned> sequence{0};
    auto parent = std::filesystem::temp_directory_path();
    for (;;) {
        auto path = parent / ("tensoracle-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++));
        if (std::filesystem::create_directory(path)) return path;
    }
}
inline void set_environment(const char* name, const std::string& value) {
#if defined(_WIN32)
    auto wide_name = std::filesystem::u8path(name).wstring();
    auto wide_value = std::filesystem::u8path(value).wstring();
    if (_wputenv_s(wide_name.c_str(), wide_value.c_str())) throw std::runtime_error("environment update failed");
    // CreateProcess inherits the OS environment, while getenv uses the CRT copy.
    if (!SetEnvironmentVariableW(wide_name.c_str(), wide_value.c_str())) throw std::runtime_error("OS environment update failed");
#else
    if (setenv(name, value.c_str(), 1)) throw std::runtime_error("environment update failed");
#endif
}
inline std::optional<std::string> get_environment(const char* name) {
#if defined(_WIN32)
    auto* value = _wgetenv(std::filesystem::u8path(name).c_str());
    if (value) return std::filesystem::path(value).u8string();
#else
    auto* value = std::getenv(name);
    if (value) return value;
#endif
    return std::nullopt;
}
inline void unset_environment(const char* name) {
#if defined(_WIN32)
    auto wide_name = std::filesystem::u8path(name).wstring();
    _wputenv_s(wide_name.c_str(), L""); SetEnvironmentVariableW(wide_name.c_str(), nullptr);
#else
    unsetenv(name);
#endif
}
#if defined(_WIN32)
inline std::wstring command_line(const std::vector<std::string>& arguments) {
    std::wstring result;
    for (const auto& argument : arguments) {
        if (!result.empty()) result += L' ';
        result += L'"'; size_t slashes = 0;
        for (auto c : std::filesystem::u8path(argument).wstring()) {
            if (c == L'\\') { ++slashes; continue; }
            result.append(slashes * (c == L'"' ? 2 : 1), L'\\');
            if (c == L'"') result += L'\\';
            result += c; slashes = 0;
        }
        result.append(slashes * 2, L'\\'); result += L'"';
    }
    return result;
}
#endif
class process {
public:
    explicit process(const std::vector<std::string>& arguments) {
#if defined(_WIN32)
        auto command = command_line(arguments);
        STARTUPINFOW startup{}; startup.cb = sizeof(startup);
        PROCESS_INFORMATION child{};
        auto executable = std::filesystem::u8path(arguments.at(0)).wstring();
        if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child))
            throw std::runtime_error("test child launch failed: " + std::to_string(GetLastError()));
        handle_ = child.hProcess; CloseHandle(child.hThread);
#else
        std::vector<char*> values;
        for (const auto& argument : arguments) values.push_back(const_cast<char*>(argument.c_str()));
        values.push_back(nullptr);
        if (posix_spawn(&pid_, arguments.at(0).c_str(), nullptr, nullptr, values.data(), environ)) throw std::runtime_error("test child launch failed");
#endif
    }
    process(const process&) = delete;
    process& operator=(const process&) = delete;
    ~process() {
#if defined(_WIN32)
        if (handle_) { TerminateProcess(handle_, 137); WaitForSingleObject(handle_, INFINITE); CloseHandle(handle_); }
#else
        if (pid_ > 0) { kill(pid_, SIGKILL); waitpid(pid_, nullptr, 0); }
#endif
    }
    void terminate() {
#if defined(_WIN32)
        if (!TerminateProcess(handle_, 137)) throw std::runtime_error("child termination failed");
#else
        if (kill(pid_, SIGKILL)) throw std::runtime_error("child termination failed");
#endif
    }
    int wait() {
#if defined(_WIN32)
        if (WaitForSingleObject(handle_, 180000) != WAIT_OBJECT_0) throw std::runtime_error("test child timed out");
        DWORD status; if (!GetExitCodeProcess(handle_, &status)) throw std::runtime_error("test child status failed");
        CloseHandle(handle_); handle_ = nullptr; return int(status);
#else
        int status{};
        if (waitpid(pid_, &status, 0) != pid_) throw std::runtime_error("test child wait failed");
        pid_ = -1; return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
    }
private:
#if defined(_WIN32)
    HANDLE handle_ = nullptr;
#else
    pid_t pid_ = -1;
#endif
};
inline int run(const std::vector<std::string>& arguments) { return process(arguments).wait(); }
inline void* load_library(const std::filesystem::path& path) {
#if defined(_WIN32)
    return LoadLibraryW(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}
inline void close_library(void* library) {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(library));
#else
    dlclose(library);
#endif
}
inline void* symbol(void* library, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}
inline std::filesystem::path executable_path() {
#if defined(_WIN32)
    std::vector<wchar_t> path(32768);
    auto count = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    if (!count || count >= path.size()) throw std::runtime_error("executable path failed");
    return std::wstring(path.data(), count);
#else
    return "/proc/self/exe";
#endif
}
}
