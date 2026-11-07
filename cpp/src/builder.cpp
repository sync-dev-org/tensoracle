#include <tensoracle/builder.h>
#include "builder_internal.hpp"
#include "engine_internal.hpp"
#include "error.hpp"
#include "safetensors.hpp"
#include "structure.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {
using namespace tensoracle::detail;
void require(bool value, const std::string& message) { if (!value) throw std::invalid_argument(message); }
void fields(const void* value, size_t minimum) { require(value && structure_size(value) >= minimum, "builder structure is missing required fields"); }
std::string text(const char* value) { require(value && *value, "builder names and paths must not be empty"); return value; }
std::vector<int64_t> shape(const int64_t* values, size_t rank) {
    require(rank <= 8 && (!rank || values), "tensor shape requires dimensions and rank <= 8");
    if (!rank) return {};
    return {values, values + rank};
}
cache_request request_from(const tensoracle_builder_options* options) {
    fields(options, offsetof(tensoracle_builder_options, io_count) + sizeof(size_t));
    cache_request request;
    request.name = text(options->name); request.graph_version = text(options->graph_version);
    require(options->weight_count && options->weight_files, "at least one weight file is required");
    for (size_t i = 0; i < options->weight_count; ++i) request.weights.emplace_back(text(options->weight_files[i]));
    require(!options->input_count || options->inputs, "input profiles are required");
    for (size_t i = 0; i < options->input_count; ++i) {
        const auto* profile = options->inputs[i];
        fields(profile, offsetof(tensoracle_input_profile, maximum) + sizeof(profile->maximum));
        request.inputs.push_back({text(profile->name), shape(profile->minimum, profile->rank),
            shape(profile->optimum, profile->rank), shape(profile->maximum, profile->rank)});
    }
    require(!options->io_count || options->io, "IO declarations are required");
    std::set<std::string> names;
    for (size_t i = 0; i < options->io_count; ++i) {
        const auto* spec = options->io[i];
        fields(spec, offsetof(tensoracle_tensor_spec, input) + sizeof(spec->input));
        auto name = text(spec->name);
        require(names.insert(name).second && spec->input <= 1, "IO names must be unique and input must be 0 or 1");
        tensoracle::detail::tensor_spec value{name, spec->dtype, shape(spec->dimensions, spec->rank), spec->input != 0, {}, {}};
        for (auto dimension : value.shape) require(dimension >= -1, "IO dimensions must be >= -1");
        require(spec->dtype >= TENSORACLE_BOOL && spec->dtype <= TENSORACLE_E8M0, "unknown IO dtype");
        auto size = structure_size(spec);
        auto has_location = read_field<uint32_t>(spec, size, offsetof(tensoracle_tensor_spec, has_location));
        auto has_format = read_field<uint32_t>(spec, size, offsetof(tensoracle_tensor_spec, has_format));
        require(has_location <= 1 && has_format <= 1, "IO optional field flags must be 0 or 1");
        if (has_location) {
            fields(spec, offsetof(tensoracle_tensor_spec, location) + sizeof(spec->location)); value.location = spec->location;
            require(*value.location == 0 || *value.location == 1, "unknown IO location");
        }
        if (has_format) { fields(spec, offsetof(tensoracle_tensor_spec, format) + sizeof(spec->format)); value.format = spec->format; }
        request.io.push_back(std::move(value));
    }
    auto size = structure_size(options);
    auto count = read_field<size_t>(options, size, offsetof(tensoracle_builder_options, plugin_count));
    auto plugins = read_field<const char* const*>(options, size, offsetof(tensoracle_builder_options, plugins));
    require(!count || plugins, "plugin paths are required");
    for (size_t i = 0; i < count; ++i) request.plugins.emplace_back(text(plugins[i]));
    auto has_limit = read_field<uint32_t>(options, size, offsetof(tensoracle_builder_options, has_workspace_limit));
    require(has_limit <= 1, "workspace presence must be 0 or 1");
    if (has_limit) {
        fields(options, offsetof(tensoracle_builder_options, workspace_limit) + sizeof(int64_t));
        require(options->workspace_limit > 0, "workspace limit must be positive"); request.workspace_limit = uint64_t(options->workspace_limit);
    }
    if (size >= offsetof(tensoracle_builder_options, optimization_level) + sizeof(int32_t)) request.optimization_level = options->optimization_level;
    auto force = read_field<uint32_t>(options, size, offsetof(tensoracle_builder_options, force_rebuild));
    require(force <= 1, "force_rebuild must be 0 or 1"); request.force_rebuild = force != 0;
    (void)validate_cache_request(request);
    return request;
}
built_plan build_plan(const cache_request& request, const tensoracle_network_builder& callback) {
    auto weights = load_safetensors(request.weights);
    auto& logger = process_engine_logger();
    // TensorRT requires the timing cache to outlive the builder using it.
    std::unique_ptr<nvinfer1::ITimingCache> timing;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("TensorRT failed to create a builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) throw std::runtime_error("TensorRT failed to create a network");
    const char* message = nullptr;
    auto status = callback.build(callback.context, network.get(), weights.get(), &message);
    if (status != TENSORACLE_SUCCESS) throw tensoracle::error(status, message ? message : "network callback failed without a message");
    for (int32_t i = 0; i < network->getNbOutputs(); ++i)
        require(!network->getOutput(i)->isShapeTensor(), std::string("shape inference output is not supported: ") + network->getOutput(i)->getName());
    auto config = configure_builder(*builder, request);
    require(size_t(network->getNbInputs()) == request.inputs.size(), "declared input names must match the graph inputs");
    nvinfer1::IOptimizationProfile* profile = nullptr;
    std::set<std::string> inputs;
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
        auto* tensor = network->getInput(i);
        auto found = std::find_if(request.inputs.begin(), request.inputs.end(), [&](const auto& input) { return input.name == tensor->getName(); });
        require(found != request.inputs.end() && inputs.emplace(found->name).second, "declared input names must match the graph inputs");
        auto dimensions = tensor->getDimensions();
        require(dimensions.nbDims == int32_t(found->minimum.size()), "declared rank differs from graph input " + found->name);
        bool dynamic = false;
        for (int32_t j = 0; j < dimensions.nbDims; ++j) {
            auto d = dimensions.d[j]; dynamic = dynamic || d == -1;
            require(d == -1 || (d == found->minimum[j] && d == found->optimum[j] && d == found->maximum[j]),
                    "declared dimensions differ from graph input " + found->name);
        }
        if (dynamic) {
            if (!profile) profile = builder->createOptimizationProfile();
            if (!profile) throw std::runtime_error("TensorRT failed to create an optimization profile");
            const std::vector<int64_t>* ranges[] = {&found->minimum, &found->optimum, &found->maximum};
            for (int selector = 0; selector < 3; ++selector) {
                nvinfer1::Dims dims{}; dims.nbDims = dimensions.nbDims;
                std::copy(ranges[selector]->begin(), ranges[selector]->end(), dims.d);
                require(profile->setDimensions(found->name.c_str(), nvinfer1::OptProfileSelector(selector), dims),
                        "TensorRT rejected profile for input " + found->name);
            }
        }
    }
    if (profile) require(profile->isValid() && config->addOptimizationProfile(profile) >= 0, "TensorRT rejected the optimization profile");
    timing.reset(config->createTimingCache(nullptr, 0));
    if (!timing) throw std::runtime_error("TensorRT failed to create a timing cache");
    if (!config->setTimingCache(*timing, false)) throw std::runtime_error("TensorRT rejected the timing cache");
    // A cold TensorRT build can serialize large zero-filled buffers. Rebuild
    // with the populated in-memory cache before persisting the plan, releasing
    // the first plan to avoid retaining both serialized engines at once.
    {
        std::unique_ptr<nvinfer1::IHostMemory> warmup(builder->buildSerializedNetwork(*network, *config));
        if (!warmup) throw std::runtime_error("TensorRT failed to build the serialized engine");
    }
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("TensorRT failed to build the serialized engine");
    auto* first = static_cast<const uint8_t*>(plan->data());
    return {{first, first + plan->size()}, {}};
}
void write_weight(const tensoracle::detail::weight_record& value, tensoracle_weight* output) {
    fields(output, sizeof(size_t)); auto size = structure_size(output);
#define FIELD(name, item) write_field(output, size, offsetof(tensoracle_weight, name), item)
    FIELD(name, value.name.c_str()); FIELD(dtype, value.dtype); FIELD(rank, value.shape.size());
    FIELD(dimensions, value.shape.data()); FIELD(data, value.data()); FIELD(element_count, value.count); FIELD(byte_size, value.bytes);
#undef FIELD
}
void write_metadata(const std::pair<const std::string, std::string>& entry, tensoracle_weights_metadata_entry* output) {
    fields(output, sizeof(size_t)); auto size = structure_size(output);
    write_field(output, size, offsetof(tensoracle_weights_metadata_entry, key), entry.first.c_str());
    write_field(output, size, offsetof(tensoracle_weights_metadata_entry, key_size), entry.first.size());
    write_field(output, size, offsetof(tensoracle_weights_metadata_entry, value), entry.second.c_str());
    write_field(output, size, offsetof(tensoracle_weights_metadata_entry, value_size), entry.second.size());
}
}
extern "C" tensoracle_status tensoracle_weights_open(const char* path, tensoracle_weights** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(output, "weights output is required");
        auto weights = tensoracle::detail::load_safetensors({text(path)});
        *output = weights.release(); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_weights_destroy(tensoracle_weights* weights) noexcept {
    delete weights;
    return TENSORACLE_SUCCESS;
}
extern "C" tensoracle_status tensoracle_weights_metadata_count(const tensoracle_weights* weights, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        require(weights && count, "weights and metadata count output are required"); *count = weights->metadata.size(); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_weights_metadata_at(const tensoracle_weights* weights, size_t index,
    tensoracle_weights_metadata_entry* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(weights && index < weights->metadata.size(), "metadata index is out of range");
        auto entry = weights->metadata.begin(); std::advance(entry, index);
        write_metadata(*entry, output); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_weights_metadata_find(const tensoracle_weights* weights, const char* key,
    tensoracle_weights_metadata_entry* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(weights && key, "weights and metadata key are required");
        auto entry = weights->metadata.find(key); require(entry != weights->metadata.end(), std::string("unknown weight metadata ") + key);
        write_metadata(*entry, output); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_weights_count(const tensoracle_weights* weights, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        require(weights && count, "weights and count output are required"); *count = weights->tensors.size(); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_weights_at(const tensoracle_weights* weights, size_t index, tensoracle_weight* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] { require(weights && index < weights->tensors.size(), "weight index is out of range"); write_weight(weights->tensors[index], output); return TENSORACLE_SUCCESS; });
}
extern "C" tensoracle_status tensoracle_weights_find(const tensoracle_weights* weights, const char* name, tensoracle_weight* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(weights && name, "weights and tensor name are required");
        auto found = weights->names.find(name); require(found != weights->names.end(), std::string("unknown weight tensor ") + name);
        write_weight(weights->tensors[found->second], output); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_open_engine(const tensoracle_builder_options* options, const tensoracle_network_builder* callback,
    tensoracle_engine** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(output, "engine output is required");
        fields(callback, offsetof(tensoracle_network_builder, build) + sizeof(callback->build));
        require(callback->build, "network callback is required");
        auto request = request_from(options);
        bool built = false;
        struct cleanup {
            bool& built;
            ~cleanup() {
#if defined(__GLIBC__)
                if (built) (void)malloc_trim(0);
#endif
            }
        } release{built};
        auto cached = tensoracle::detail::open_cached_engine(request, [&] { built = true; return build_plan(request, *callback); });
        cached.engine.get()->path = cached.path.string();
        *output = cached.engine.release(); return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_engine_path(const tensoracle_engine* engine, const char** path, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (path) *path = nullptr; require(engine && path, "engine and path output are required");
        *path = engine->path.c_str(); return TENSORACLE_SUCCESS;
    });
}
