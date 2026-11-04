#include "cache.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        auto input = nlohmann::json::parse(std::cin);
        std::optional<uint64_t> workspace;
        if (input.contains("workspace_limit") && !input["workspace_limit"].is_null())
            workspace = input["workspace_limit"].get<uint64_t>();
        tensoracle::detail::write_engine_file(argv[1], input.at("conditions"),
            {'p', 'l', 'a', 'n'}, workspace, input.value("shape_tensor_outputs", std::vector<std::string>{}));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
