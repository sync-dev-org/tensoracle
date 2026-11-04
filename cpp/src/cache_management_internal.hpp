#pragma once
#include "cache.hpp"
#include <tensoracle/cache.h>
#include <sys/stat.h>

namespace tensoracle::detail {
struct cache_entry_record {
    std::filesystem::path path;
    std::string file;
    std::optional<struct stat> observed;
    int32_t status = TENSORACLE_CACHE_UNRECOGNIZED;
    nlohmann::json header;
    std::string header_json, weight_paths_json, reason;
};
struct cache_snapshot {
    std::string directory;
    std::filesystem::path absolute_directory;
    std::vector<cache_entry_record> entries;
    uint64_t total_size = 0;
    bool complete = true;
};
}

struct tensoracle_cache_result {
    tensoracle::detail::cache_snapshot snapshot;
    int32_t phase = TENSORACLE_CACHE_LIST;
    bool dry_run = false;
    std::vector<size_t> targets, excluded;
    std::vector<tensoracle::detail::cache_entry_record> removed, failed, missing;
    std::optional<int64_t> max_size;
    uint64_t removed_size = 0, remaining_size = 0, projected_size = 0;
    bool reached = true, complete = true, succeeded = true;
};

namespace tensoracle::detail {
std::unique_ptr<tensoracle_cache_result> list_cache_files();
std::unique_ptr<tensoracle_cache_result> plan_cache_prune(
    const tensoracle_cache_prune_options* options, int64_t now_ns);
inline std::unique_ptr<tensoracle_cache_result> plan_cache_prune(
    const tensoracle_cache_prune_options& options, int64_t now_ns) {
    return plan_cache_prune(&options, now_ns);
}
std::unique_ptr<tensoracle_cache_result> apply_cache_prune(
    const tensoracle_cache_result& plan, bool dry_run);
}
