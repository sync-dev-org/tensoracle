#include "tensoracle/cubin_plugin.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <utility>

using namespace nvinfer1;
using namespace tensoracle::cubin;

class Probe final : public PluginBase {
public:
    explicit Probe(PluginIO io = {}) : PluginBase("probe", "tests", {}, std::move(io)) {}
    IPluginV3* clone() noexcept override { return nullptr; }
    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const*,
        void* const*, void*, cudaStream_t) noexcept override { return -1; }

protected:
    int32_t validate_shape(Dims const* inputs, int32_t count,
        Dims const& output) const noexcept override
    {
        for (int32_t i = 0; i < count; ++i) {
            if (inputs[i].nbDims != output.nbDims) return -1;
            for (int32_t j = 0; j < output.nbDims; ++j)
                if (inputs[i].d[j] != output.d[j]) return -1;
        }
        return 0;
    }
};

PluginIO mixed_io()
{
    return {{{DataType::kHALF, TensorFormat::kHWC8},
             {DataType::kFLOAT, TensorFormat::kLINEAR}},
            {DataType::kINT32, TensorFormat::kLINEAR}, 1};
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    Probe plugin(mixed_io());
    DynamicPluginTensorDesc descriptors[3]{};
    DataType types[] = {DataType::kHALF, DataType::kFLOAT, DataType::kINT32};
    TensorFormat formats[] = {TensorFormat::kHWC8, TensorFormat::kLINEAR,
                             TensorFormat::kLINEAR};
    for (int i = 0; i < 3; ++i) {
        descriptors[i].desc.type = types[i];
        descriptors[i].desc.format = formats[i];
        descriptors[i].desc.dims = Dims4(1, 48, 3, 5);
    }
    if (std::strcmp(argv[1], "formats") == 0) {
        for (int i = 0; i < 3; ++i) {
            assert(plugin.supportsFormatCombination(i, descriptors, 2, 1));
            descriptors[i].desc.type = DataType::kINT8;
            assert(!plugin.supportsFormatCombination(i, descriptors, 2, 1));
            descriptors[i].desc.type = types[i];
            descriptors[i].desc.format = TensorFormat::kCHW32;
            assert(!plugin.supportsFormatCombination(i, descriptors, 2, 1));
            descriptors[i].desc.format = formats[i];
        }
        // TensorRT has not initialized descriptors beyond the requested position.
        descriptors[1].desc.type = DataType::kINT8;
        descriptors[2].desc.format = TensorFormat::kCHW32;
        assert(plugin.supportsFormatCombination(0, descriptors, 2, 1));
        assert(!plugin.supportsFormatCombination(-1, descriptors, 2, 1));
        assert(!plugin.supportsFormatCombination(3, descriptors, 2, 1));
        assert(!plugin.supportsFormatCombination(0, nullptr, 2, 1));
    } else if (std::strcmp(argv[1], "counts") == 0) {
        DataType output{};
        PluginTensorDesc runtime[] = {descriptors[0].desc, descriptors[1].desc};
        assert(plugin.getNbOutputs() == 1);
        for (int count : {-1, 0, 1, 3}) {
            assert(!plugin.supportsFormatCombination(0, descriptors, count, 1));
            assert(plugin.getOutputDataTypes(&output, 1, types, count) != 0);
            assert(plugin.configurePlugin(descriptors, count, descriptors + 2, 1) != 0);
            assert(plugin.onShapeChange(runtime, count, &descriptors[2].desc, 1) != 0);
        }
        for (int count : {-1, 0, 2}) {
            assert(!plugin.supportsFormatCombination(0, descriptors, 2, count));
            assert(plugin.getOutputDataTypes(&output, count, types, 2) != 0);
            assert(plugin.configurePlugin(descriptors, 2, descriptors + 2, count) != 0);
            assert(plugin.onShapeChange(runtime, 2, &descriptors[2].desc, count) != 0);
        }
        assert(plugin.getOutputDataTypes(nullptr, 1, types, 2) != 0);
        assert(plugin.getOutputDataTypes(&output, 1, nullptr, 2) != 0);
        assert(plugin.configurePlugin(nullptr, 2, descriptors + 2, 1) != 0);
        assert(plugin.configurePlugin(descriptors, 2, nullptr, 1) != 0);
        assert(plugin.onShapeChange(nullptr, 2, &descriptors[2].desc, 1) != 0);
        assert(plugin.onShapeChange(runtime, 2, nullptr, 1) != 0);
    } else if (std::strcmp(argv[1], "types") == 0) {
        DataType output{};
        assert(plugin.getOutputDataTypes(&output, 1, types, 2) == 0);
        assert(output == DataType::kINT32);
        for (int i = 0; i < 2; ++i) {
            auto previous = types[i];
            types[i] = DataType::kINT8;
            assert(plugin.getOutputDataTypes(&output, 1, types, 2) != 0);
            types[i] = previous;
        }
    } else if (std::strcmp(argv[1], "shapes") == 0) {
        assert(plugin.configurePlugin(descriptors, 2, descriptors + 2, 1) == 0);
        PluginTensorDesc runtime[] = {descriptors[0].desc, descriptors[1].desc};
        assert(plugin.onShapeChange(runtime, 2, &descriptors[2].desc, 1) == 0);
        descriptors[1].desc.dims.d[2] = 7;
        runtime[1] = descriptors[1].desc;
        assert(plugin.configurePlugin(descriptors, 2, descriptors + 2, 1) != 0);
        assert(plugin.onShapeChange(runtime, 2, &descriptors[2].desc, 1) != 0);
        descriptors[1].desc.dims.d[2] = 3;
        runtime[1] = descriptors[1].desc;
        for (int i = 0; i < 3; ++i) {
            descriptors[i].desc.type = DataType::kINT8;
            assert(plugin.configurePlugin(descriptors, 2, descriptors + 2, 1) != 0);
            descriptors[i].desc.type = types[i];
            descriptors[i].desc.format = TensorFormat::kCHW32;
            assert(plugin.configurePlugin(descriptors, 2, descriptors + 2, 1) != 0);
            descriptors[i].desc.format = formats[i];
            auto saved = i < 2 ? runtime[i] : descriptors[2].desc;
            auto& changed = i < 2 ? runtime[i] : descriptors[2].desc;
            changed.type = DataType::kINT8;
            assert(plugin.onShapeChange(runtime, 2, &descriptors[2].desc, 1) != 0);
            changed = saved;
            changed.format = TensorFormat::kCHW32;
            assert(plugin.onShapeChange(runtime, 2, &descriptors[2].desc, 1) != 0);
            changed = saved;
        }
    } else if (std::strcmp(argv[1], "defaults") == 0) {
        Probe default_plugin;
        descriptors[0].desc.type = descriptors[1].desc.type = DataType::kFLOAT;
        descriptors[0].desc.format = descriptors[1].desc.format = TensorFormat::kLINEAR;
        assert(default_plugin.getNbOutputs() == 1);
        assert(default_plugin.supportsFormatCombination(0, descriptors, 1, 1));
        assert(default_plugin.supportsFormatCombination(1, descriptors, 1, 1));
        assert(!default_plugin.supportsFormatCombination(0, descriptors, 2, 1));
        DataType input = DataType::kFLOAT, output{};
        assert(default_plugin.getOutputDataTypes(&output, 1, &input, 1) == 0);
        assert(output == DataType::kFLOAT);
        assert(default_plugin.configurePlugin(descriptors, 1, descriptors + 1, 1) == 0);
        assert(default_plugin.onShapeChange(&descriptors[0].desc, 1,
            &descriptors[1].desc, 1) == 0);
        descriptors[0].desc.type = DataType::kHALF;
        assert(!default_plugin.supportsFormatCombination(0, descriptors, 1, 1));
    } else if (std::strcmp(argv[1], "invalid") == 0) {
        for (int shape_input : {-1, 2}) {
            auto io = mixed_io();
            io.shape_input_index = shape_input;
            bool rejected = false;
            try { Probe invalid(io); }
            catch (std::invalid_argument const&) { rejected = true; }
            assert(rejected);
        }
        auto io = mixed_io();
        io.inputs.clear();
        bool rejected = false;
        try { Probe invalid(io); }
        catch (std::invalid_argument const&) { rejected = true; }
        assert(rejected);
    } else { return 1; }
}
