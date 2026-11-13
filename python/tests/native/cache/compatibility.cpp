#include "cache.hpp"
#include "sha256.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        using namespace tensoracle::detail;
        namespace fs = std::filesystem;
        auto input = nlohmann::json::parse(std::cin);
        fs::path directory = argv[2];
        cache_request request;
        request.name = input.at("name");
        request.graph_version = input.at("graph_version");
        request.weights = {directory / "weights"};
        request.plugins = {directory / "plugin"};
        request.inputs = {{"image", {1, 3, 16, 16}, {2, 3, 32, 32}, {4, 3, 64, 64}}};
        request.optimization_level = input.at("optimization_level");
        auto conditions = make_conditions(request, {sha256("plugin bytes", 12)},
            input.at("gpu"), "11.2.1.2", "Linux", "x86_64");
        auto path = cache_path(conditions);
        std::optional<uint64_t> workspace;
        if (!input.at("workspace_limit").is_null()) workspace = input.at("workspace_limit").get<uint64_t>();
        auto outputs = input.at("shape_tensor_outputs").get<std::vector<std::string>>();
        if (std::string(argv[1]) == "write")
            write_engine_file(path, conditions, {'p', 'l', 'a', 'n'}, workspace, outputs);
        else if (std::string(argv[1]) != "read") return 2;
        auto loaded = read_engine_file(path, conditions);
        if (loaded.plan != std::vector<uint8_t>{'p', 'l', 'a', 'n'}) return 3;
        std::cout << nlohmann::json{{"filename", path.filename().string()},
            {"header", loaded.header}, {"json_version", NLOHMANN_JSON_VERSION_MAJOR * 10000
                + NLOHMANN_JSON_VERSION_MINOR * 100 + NLOHMANN_JSON_VERSION_PATCH}} << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
