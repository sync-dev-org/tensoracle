#include "cache.hpp"
#include "test_platform.hpp"
#include <tensoracle/cache.hpp>
int probe(int argc, const std::vector<std::string>& argv) {
    if (argc == 4 && argv[1] == "--list") {
        try {
            auto listing = tensoracle::list_cache();
            if (std::filesystem::u8path(listing.summary().directory) != std::filesystem::u8path(argv[2])) return 7;
            auto entries = listing.entries();
            return entries.size() == 1 && entries[0].file == argv[3] ? 0 : 8;
        } catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 9; }
    }
    if (argc == 4 && std::string(argv[1]) == "--engine") {
        using namespace tensoracle::detail;
        try {
            cache_request request;
            request.name = "sum"; request.graph_version = "1"; request.weights = {std::filesystem::u8path(argv[2])};
            request.inputs = {{"left", {3, 2}, {3, 2}, {3, 2}}, {"right", {3, 2}, {3, 2}, {3, 2}}};
            request.io = {{"left", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                          {"right", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                          {"result", TENSORACLE_FLOAT32, {3, 2}, false, {}, {}}};
            request.optimization_level = 0;
            auto engine = open_cached_engine(request, []() -> built_plan { throw std::runtime_error("cached graph called"); });
            if (engine.path != std::filesystem::u8path(argv[3])) return 5;
            return engine.engine.create_execution().get_tensor_shape("result") == std::vector<int64_t>{3, 2} ? 0 : 6;
        } catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 4; }
    }
    if (argc != 3) return 2;
    try { return tensoracle::detail::weight_sha256(std::filesystem::u8path(argv[1]), false) == argv[2] ? 0 : 1; }
    catch (...) { return 3; }
}
#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    return probe(argc, test_platform::utf8_arguments(argc, argv));
}
#else
int main(int argc, char** argv) { return probe(argc, test_platform::utf8_arguments(argc, argv)); }
#endif
