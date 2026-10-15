// SP-TENSORACLE-004: retain typed fields in order and compute SHA-256.
#include "tensoracle/cubin_plugin.hpp"
#include <cstring>
using namespace tensoracle::cubin;
using namespace nvinfer1;
int main() {
    if (sha256("abc", 3) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 1;
    float weight[2] = {1.25f, 2.5f};
    int version = 7;
    PluginField input[] = {
        {"weight", weight, PluginFieldType::kFLOAT32, 2},
        {"version", &version, PluginFieldType::kINT32, 1},
    };
    PluginFieldCollection source{2, input};
    Fields fields;
    if (!Fields::parse(&source, fields)) return 2;
    auto const* saved = fields.collection();
    if (saved->nbFields != 2 ||
        std::strcmp(saved->fields[0].name, "weight") != 0 ||
        std::strcmp(saved->fields[1].name, "version") != 0 ||
        saved->fields[0].type != PluginFieldType::kFLOAT32 ||
        saved->fields[0].length != 2 ||
        std::memcmp(saved->fields[0].data, weight, sizeof(weight)) != 0) return 3;
    weight[0] = 9.0f;
    if (std::memcmp(saved->fields[0].data, weight, sizeof(weight)) == 0) return 4;
    Fields copy = fields;
    if (std::memcmp(copy.collection()->fields[0].data, saved->fields[0].data,
                    sizeof(weight)) != 0) return 5;
    return 0;
}
