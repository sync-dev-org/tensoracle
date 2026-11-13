#include <cstdio>
#include <exception>
extern "C" int embedded_model(const char* weights);
int main(int argc, char** argv) {
    try { return embedded_model(argc > 1 ? argv[1] : nullptr); }
    catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 1; }
}
