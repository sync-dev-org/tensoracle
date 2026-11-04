#include "cache_management_internal.hpp"
#include "error.hpp"
#include "structure.hpp"
#include <chrono>

namespace {
using namespace tensoracle::detail;
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
size_t count_entries(const tensoracle_cache_result& result, int32_t collection) {
    switch (collection) {
    case TENSORACLE_CACHE_ENTRIES: return result.snapshot.entries.size();
    case TENSORACLE_CACHE_TARGETS: return result.targets.size();
    case TENSORACLE_CACHE_EXCLUDED: return result.excluded.size();
    case TENSORACLE_CACHE_REMOVED: return result.removed.size();
    case TENSORACLE_CACHE_FAILED: return result.failed.size();
    case TENSORACLE_CACHE_MISSING: return result.missing.size();
    default: throw std::invalid_argument("unknown cache result collection");
    }
}
const cache_entry_record& get_entry(const tensoracle_cache_result& result, int32_t collection, size_t index) {
    require(index < count_entries(result, collection), "cache entry index is out of range");
    switch (collection) {
    case TENSORACLE_CACHE_ENTRIES: return result.snapshot.entries[index];
    case TENSORACLE_CACHE_TARGETS: return result.snapshot.entries[result.targets[index]];
    case TENSORACLE_CACHE_EXCLUDED: return result.snapshot.entries[result.excluded[index]];
    case TENSORACLE_CACHE_REMOVED: return result.removed[index];
    case TENSORACLE_CACHE_FAILED: return result.failed[index];
    default: return result.missing[index];
    }
}
}
extern "C" tensoracle_status tensoracle_cache_list(tensoracle_cache_result** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(output, "cache result output is required");
        *output = tensoracle::detail::list_cache_files().release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_plan_prune(const tensoracle_cache_prune_options* options,
    tensoracle_cache_result** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(output, "cache prune plan output is required");
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        *output = tensoracle::detail::plan_cache_prune(options, now).release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_apply_prune(const tensoracle_cache_result* plan, uint32_t dry_run,
    tensoracle_cache_result** output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (output) *output = nullptr;
        require(plan && output, "cache prune plan and result output are required");
        require(dry_run <= 1, "dry_run must be zero or one");
        *output = tensoracle::detail::apply_cache_prune(*plan, dry_run != 0).release();
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_result_summary(const tensoracle_cache_result* result,
    tensoracle_cache_summary* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(result && output && structure_size(output) >= sizeof(size_t), "cache summary requires a result and struct_size");
        auto size = structure_size(output);
#define WRITE(field, value) write_field(output, size, offsetof(tensoracle_cache_summary, field), value)
        WRITE(directory, result->snapshot.directory.c_str());
        WRITE(phase, result->phase);
        WRITE(dry_run, uint32_t(result->dry_run));
        WRITE(total_size_bytes, result->snapshot.total_size);
        WRITE(removed_size_bytes, result->removed_size);
        WRITE(remaining_size_bytes, result->remaining_size);
        WRITE(projected_size_bytes, result->projected_size);
        WRITE(has_max_size, uint32_t(result->max_size.has_value()));
        WRITE(max_size_bytes, result->max_size.value_or(0));
        WRITE(max_size_reached, uint32_t(result->reached));
        WRITE(sizes_complete, uint32_t(result->complete));
        WRITE(succeeded, uint32_t(result->succeeded));
#undef WRITE
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_result_count(const tensoracle_cache_result* result,
    int32_t collection, size_t* count, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        if (count) *count = 0;
        require(result && count, "cache result and count output are required");
        *count = count_entries(*result, collection);
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_result_entry(const tensoracle_cache_result* result,
    int32_t collection, size_t index, tensoracle_cache_entry* output, tensoracle_error** error) noexcept {
    return tensoracle::detail::guard(error, [&] {
        require(result && output && structure_size(output) >= sizeof(size_t), "cache entry requires a result and struct_size");
        const auto& entry = get_entry(*result, collection, index);
        auto size = structure_size(output);
#define WRITE(field, value) write_field(output, size, offsetof(tensoracle_cache_entry, field), value)
        WRITE(file, entry.file.c_str());
        WRITE(status, entry.status);
        WRITE(has_stat, uint32_t(entry.observed.has_value()));
        WRITE(size_bytes, entry.observed ? uint64_t(entry.observed->st_size) : uint64_t(0));
        WRITE(recorded_at_seconds, entry.observed ? int64_t(entry.observed->st_mtim.tv_sec) : int64_t(0));
        WRITE(recorded_at_nanoseconds, entry.observed ? uint32_t(entry.observed->st_mtim.tv_nsec) : uint32_t(0));
        WRITE(header_json, entry.header_json.empty() ? static_cast<const char*>(nullptr) : entry.header_json.c_str());
        WRITE(weight_paths_json, entry.weight_paths_json.empty() ? static_cast<const char*>(nullptr) : entry.weight_paths_json.c_str());
        WRITE(reason, entry.reason.empty() ? static_cast<const char*>(nullptr) : entry.reason.c_str());
#undef WRITE
        return TENSORACLE_SUCCESS;
    });
}
extern "C" tensoracle_status tensoracle_cache_result_destroy(tensoracle_cache_result* result) noexcept {
    delete result;
    return TENSORACLE_SUCCESS;
}
