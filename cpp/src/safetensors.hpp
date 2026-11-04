#pragma once
#include <tensoracle/builder.h>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tensoracle::detail {
struct mapped_weights_file {
    void* mapping = nullptr;
    size_t size = 0;
    ~mapped_weights_file();
};
struct weight_record {
    std::string name;
    std::filesystem::path source;
    tensoracle_dtype dtype;
    std::vector<int64_t> shape;
    const void* mapped_data;
    uint64_t count, bytes;
    std::vector<uint64_t> aligned;
    const void* data() const { return aligned.empty() ? mapped_data : aligned.data(); }
};
}
struct tensoracle_weights {
    std::vector<std::unique_ptr<tensoracle::detail::mapped_weights_file>> files;
    std::vector<tensoracle::detail::weight_record> tensors;
    std::map<std::string, size_t> names;
};
namespace tensoracle::detail {
std::unique_ptr<tensoracle_weights> load_safetensors(const std::vector<std::filesystem::path>& paths);
}
