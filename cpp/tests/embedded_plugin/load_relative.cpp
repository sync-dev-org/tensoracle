#include <cstdio>
#include <dlfcn.h>
#include <exception>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    auto* module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!module) { std::fprintf(stderr, "%s\n", dlerror()); return 3; }
    auto model = reinterpret_cast<int(*)(const char*)>(dlsym(module, "embedded_model"));
    if (!model || chdir(argv[2])) return 4;
    try { return model(argc > 3 ? argv[3] : nullptr); }
    catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 1; }
}
