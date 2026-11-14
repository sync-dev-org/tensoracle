#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
int wmain(int argc, wchar_t** values) {
    std::vector<std::string> argv;
    for (int i = 0; i < argc; ++i) argv.push_back(std::filesystem::path(values[i]).u8string());
#else
int main(int argc, char** values) {
    std::vector<std::string> argv(values, values + argc);
#endif
    if (argc < 3) return 2;
#if defined(_WIN32)
    auto module = LoadLibraryW(std::filesystem::u8path(argv[1]).c_str());
    if (!module) { std::fprintf(stderr, "module load failed: %lu\n", GetLastError()); return 3; }
    auto model = reinterpret_cast<int(*)(const char*)>(GetProcAddress(module, "embedded_model"));
#else
    auto* module = dlopen(argv[1].c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) { std::fprintf(stderr, "%s\n", dlerror()); return 3; }
    auto model = reinterpret_cast<int(*)(const char*)>(dlsym(module, "embedded_model"));
#endif
    if (!model) return 4;
    std::filesystem::current_path(std::filesystem::u8path(argv[2]));
    try { return model(argc > 3 ? argv[3].c_str() : nullptr); }
    catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 1; }
}
