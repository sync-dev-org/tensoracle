#include "cache_management_internal.hpp"
#include "structure.hpp"
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <unistd.h>

namespace tensoracle::detail {
namespace fs = std::filesystem;
namespace {
bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size
        && a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec
        && a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
uint64_t size_of(const cache_entry_record& entry) {
    return entry.observed ? static_cast<uint64_t>(entry.observed->st_size) : 0;
}
uint64_t add_size(uint64_t sum, uint64_t size) {
    if (size > std::numeric_limits<uint64_t>::max() - sum) throw std::overflow_error("cache size exceeds uint64 range");
    return sum + size;
}
std::vector<fs::path> paths_in(const fs::path& directory) {
    std::error_code error;
    fs::directory_iterator entries(directory, error);
    if (error == std::errc::no_such_file_or_directory) return {};
    if (error) throw fs::filesystem_error("cannot scan cache directory", directory, error);
    std::vector<fs::path> paths;
    for (const auto& entry : entries) paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    return paths;
}
[[noreturn]] void os_error(const fs::path& path) {
    throw std::system_error(errno, std::generic_category(), path.string());
}
cache_entry_record inspect(const fs::path& path) {
    cache_entry_record result;
    result.path = fs::absolute(path).lexically_normal(); result.file = path.filename().string();
    try {
        struct stat observed{};
        if (lstat(path.c_str(), &observed)) os_error(path);
        if (observed.st_size < 0) throw std::runtime_error("file size is negative");
        result.observed = observed;
        if (!S_ISREG(observed.st_mode)) result.reason = "not a regular file";
        else if (!observed.st_size) result.status = TENSORACLE_CACHE_BUILDING_OR_LEFTOVER;
        else {
            struct file_owner { int fd; ~file_owner() { if (fd >= 0) ::close(fd); } } file{
                ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
            if (file.fd < 0) os_error(path);
            struct stat opened{};
            if (fstat(file.fd, &opened)) os_error(path);
            if (!same_file(opened, observed)) throw std::runtime_error("file changed during inspection");
            auto metadata = read_engine_header(file.fd, path, static_cast<uint64_t>(observed.st_size));
            if (fstat(file.fd, &opened)) os_error(path);
            if (!same_file(opened, observed)) throw std::runtime_error("file changed during inspection");
            result.header = std::move(metadata.header);
            result.header_json = result.header.dump(-1, ' ', true);
            result.status = TENSORACLE_CACHE_RECOGNIZED;
        }
    } catch (const corrupt_engine_error& error) { result.reason = error.what();
    } catch (const std::system_error& error) { result.reason = error.what();
    } catch (const std::runtime_error& error) { result.reason = error.what(); }
    return result;
}
cache_snapshot scan(const fs::path& directory) {
    cache_snapshot snapshot;
    snapshot.directory = directory.string();
    snapshot.absolute_directory = fs::absolute(directory).lexically_normal();
    for (const auto& path : paths_in(snapshot.absolute_directory)) {
        if (path.extension() != ".engine") continue;
        auto entry = inspect(path);
        snapshot.total_size = add_size(snapshot.total_size, size_of(entry));
        snapshot.complete = snapshot.complete && entry.observed.has_value();
        snapshot.entries.push_back(std::move(entry));
    }
    return snapshot;
}
void add_weight_paths(cache_snapshot& snapshot) {
    std::map<std::string, std::set<std::string>> by_digest;
    for (const auto& path : paths_in(snapshot.absolute_directory)) {
        if (path.filename().string().rfind("weight-", 0) != 0 || path.extension() != ".json") continue;
        if (auto record = read_weight_record(path))
            by_digest[record->at("sha256").get<std::string>()].insert(record->at("path").get<std::string>());
    }
    for (auto& entry : snapshot.entries) {
        if (entry.status != TENSORACLE_CACHE_RECOGNIZED) continue;
        auto paths = nlohmann::json::array();
        auto weights = entry.header["conditions"].find("weights");
        if (weights != entry.header["conditions"].end() && weights->is_array()) {
            for (const auto& value : *weights) {
                if (!value.is_string()) continue;
                auto digest = value.get<std::string>();
                paths.push_back({{"sha256", digest}, {"paths", by_digest[digest]}});
            }
        }
        entry.weight_paths_json = paths.dump(-1, ' ', true);
    }
}
struct selection {
    std::optional<int64_t> unused_for_ns, max_size;
};
selection validate(const tensoracle_cache_prune_options* options) {
    if (!options || structure_size(options) < sizeof(size_t)) throw std::invalid_argument("prune options require struct_size and exactly one criterion");
    auto size = structure_size(options);
    auto kind = read_field<uint32_t>(options, size, offsetof(tensoracle_cache_prune_options, selection));
    selection result;
    if (kind == TENSORACLE_CACHE_UNUSED_FOR) {
        if (size < offsetof(tensoracle_cache_prune_options, unused_for_ns) + sizeof(int64_t))
            throw std::invalid_argument("unused-for criterion requires a complete duration");
        auto value = read_field<int64_t>(options, size, offsetof(tensoracle_cache_prune_options, unused_for_ns));
        if (value < 0) throw std::invalid_argument("unused-for duration must be nonnegative");
        result.unused_for_ns = value;
    } else if (kind == TENSORACLE_CACHE_MAX_SIZE) {
        if (size < offsetof(tensoracle_cache_prune_options, max_size_bytes) + sizeof(int64_t))
            throw std::invalid_argument("max-size criterion requires a complete byte limit");
        auto value = read_field<int64_t>(options, size, offsetof(tensoracle_cache_prune_options, max_size_bytes));
        if (value < 0) throw std::invalid_argument("max-size bytes must be nonnegative");
        result.max_size = value;
    } else throw std::invalid_argument("prune requires exactly one of unused-for or max-size");
    return result;
}
}

std::unique_ptr<tensoracle_cache_result> list_cache_files() {
    auto result = std::make_unique<tensoracle_cache_result>();
    result->snapshot = scan(cache_directory());
    add_weight_paths(result->snapshot);
    result->remaining_size = result->projected_size = result->snapshot.total_size;
    result->complete = result->snapshot.complete;
    return result;
}
std::unique_ptr<tensoracle_cache_result> plan_cache_prune(const tensoracle_cache_prune_options* options, int64_t now) {
    auto criterion = validate(options); // Validate before inspecting the filesystem.
    auto result = std::make_unique<tensoracle_cache_result>();
    result->phase = TENSORACLE_CACHE_PLAN;
    result->snapshot = scan(cache_directory()); result->max_size = criterion.max_size;
    result->remaining_size = result->projected_size = result->snapshot.total_size;
    result->complete = result->snapshot.complete;
    std::vector<size_t> candidates;
    for (size_t i = 0; i < result->snapshot.entries.size(); ++i) {
        if (result->snapshot.entries[i].status == TENSORACLE_CACHE_RECOGNIZED) candidates.push_back(i);
        else result->excluded.push_back(i);
    }
    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        const auto& x = result->snapshot.entries[a]; const auto& y = result->snapshot.entries[b];
        return std::make_tuple(x.observed->st_mtim.tv_sec, x.observed->st_mtim.tv_nsec, x.file)
             < std::make_tuple(y.observed->st_mtim.tv_sec, y.observed->st_mtim.tv_nsec, y.file);
    });
    std::pair<int64_t, int64_t> cutoff;
    if (criterion.unused_for_ns) {
        cutoff = {now / 1000000000 - *criterion.unused_for_ns / 1000000000,
                  now % 1000000000 - *criterion.unused_for_ns % 1000000000};
        while (cutoff.second < 0) { --cutoff.first; cutoff.second += 1000000000; }
    }
    for (auto index : candidates) {
        const auto& entry = result->snapshot.entries[index];
        bool selected = criterion.unused_for_ns
            ? std::make_pair(int64_t(entry.observed->st_mtim.tv_sec), int64_t(entry.observed->st_mtim.tv_nsec)) < cutoff
            : result->projected_size > static_cast<uint64_t>(*criterion.max_size);
        if (selected) { result->targets.push_back(index); result->projected_size -= size_of(entry); }
    }
    result->reached = !result->max_size || result->projected_size <= static_cast<uint64_t>(*result->max_size);
    result->succeeded = result->complete && result->reached;
    return result;
}
std::unique_ptr<tensoracle_cache_result> apply_cache_prune(const tensoracle_cache_result& plan, bool dry_run) {
    if (plan.phase != TENSORACLE_CACHE_PLAN) throw std::invalid_argument("pruning requires a prune plan");
    auto result = std::make_unique<tensoracle_cache_result>(plan);
    result->phase = TENSORACLE_CACHE_APPLIED; result->dry_run = dry_run;
    if (!dry_run) for (auto index : plan.targets) {
        auto entry = plan.snapshot.entries.at(index);
        struct stat current{};
        if (lstat(entry.path.c_str(), &current)) {
            if (errno == ENOENT) { entry.reason = "file disappeared before removal"; result->missing.push_back(std::move(entry)); }
            else { entry.reason = std::generic_category().message(errno); result->failed.push_back(std::move(entry)); }
        } else if (!same_file(current, *entry.observed)) {
            entry.reason = "file changed since inspection; not removed"; result->failed.push_back(std::move(entry));
        } else if (::unlink(entry.path.c_str())) {
            if (errno == ENOENT) { entry.reason = "file disappeared before removal"; result->missing.push_back(std::move(entry)); }
            else { entry.reason = std::generic_category().message(errno); result->failed.push_back(std::move(entry)); }
        } else {
            result->removed_size = add_size(result->removed_size, size_of(entry));
            result->removed.push_back(std::move(entry));
        }
    }
    auto remaining = scan(plan.snapshot.absolute_directory);
    result->remaining_size = remaining.total_size;
    result->complete = result->snapshot.complete && remaining.complete;
    if (!dry_run) result->projected_size = result->remaining_size;
    result->reached = !result->max_size || result->projected_size <= static_cast<uint64_t>(*result->max_size);
    result->succeeded = result->failed.empty() && result->reached && result->complete;
    return result;
}
}
