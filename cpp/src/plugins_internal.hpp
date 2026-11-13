#pragma once
#include <tensoracle/plugins.h>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

struct tensoracle_plugin_libraries {
    std::vector<std::pair<std::filesystem::path, std::string>> identities;
};
