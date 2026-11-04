#include "cache.hpp"
int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--engine") {
        using namespace tensoracle::detail;
        try {
            cache_request request;
            request.name = "sum"; request.graph_version = "1"; request.weights = {argv[2]};
            request.inputs = {{"left", {3, 2}, {3, 2}, {3, 2}}, {"right", {3, 2}, {3, 2}, {3, 2}}};
            request.io = {{"left", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                          {"right", TENSORACLE_FLOAT32, {3, 2}, true, {}, {}},
                          {"result", TENSORACLE_FLOAT32, {3, 2}, false, {}, {}}};
            request.optimization_level = 0;
            auto engine = open_cached_engine(request, []() -> built_plan { throw std::runtime_error("cached graph called"); });
            if (engine.path != argv[3]) return 5;
            return engine.engine.create_execution().get_tensor_shape("result") == std::vector<int64_t>{3, 2} ? 0 : 6;
        } catch (const std::exception& failure) { std::fprintf(stderr, "%s\n", failure.what()); return 4; }
    }
    if (argc != 3) return 2;
    try { return tensoracle::detail::weight_sha256(argv[1], false) == argv[2] ? 0 : 1; }
    catch (...) { return 3; }
}
