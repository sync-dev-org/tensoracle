#include "test_platform.hpp"
#include <tensoracle/cache.hpp>
#include "cache.hpp"
#include "cache_management_internal.hpp"
#include "sha256.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
void set_recorded_time(const fs::path& path, int64_t seconds, int64_t fraction) {
#if defined(_WIN32)
    auto ticks = uint64_t(seconds + 11644473600LL) * 10000000 + uint64_t(fraction / 100);
    FILETIME time{DWORD(ticks), DWORD(ticks >> 32)}, observed{};
    auto handle = CreateFileW(path.c_str(), FILE_WRITE_ATTRIBUTES | FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(handle != INVALID_HANDLE_VALUE);
    auto written = SetFileTime(handle, nullptr, nullptr, &time);
    auto read = GetFileTime(handle, nullptr, nullptr, &observed);
    CloseHandle(handle);
    REQUIRE(written);
    REQUIRE(read);
    REQUIRE(observed.dwHighDateTime == time.dwHighDateTime);
    REQUIRE(observed.dwLowDateTime == time.dwLowDateTime);
#else
    timespec times[2]{{seconds, fraction}, {seconds, fraction}};
    REQUIRE(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0);
    struct stat observed{};
    REQUIRE(::stat(path.c_str(), &observed) == 0);
    REQUIRE(observed.st_mtim.tv_sec == seconds);
    REQUIRE(observed.st_mtim.tv_nsec == fraction);
#endif
}
struct fixture {
    fs::path root, directory;
    std::string previous;
    bool had_previous;
    fixture() {
        root = test_platform::make_directory(); directory = root / "tensoracle";
        auto old = test_platform::get_environment("XDG_CACHE_HOME");
        had_previous = old.has_value(); if (old) previous = *old;
        test_platform::set_environment("XDG_CACHE_HOME", root.u8string());
        fs::create_directory(directory);
    }
    ~fixture() {
        std::error_code ignored;
        fs::permissions(directory, fs::perms::owner_all, fs::perm_options::add, ignored);
        fs::remove_all(root, ignored);
        if (had_previous) test_platform::set_environment("XDG_CACHE_HOME", previous.c_str()); else test_platform::unset_environment("XDG_CACHE_HOME");
    }
    fs::path save(const char* name, int64_t time_ns = 1000000000, uint64_t size = 1024,
                  json conditions = nullptr) {
        auto path = directory / fs::u8path(name);
        if (conditions.is_null()) conditions = {{"name", name}, {"weights", json::array({std::string(64, 'a')})}};
        tensoracle::detail::write_engine_file(path, conditions, {'p'}, 4096, {"shape"});
        fs::resize_file(path, size);
#if defined(_WIN32)
        constexpr int64_t epoch = 116444736000000000LL;
        auto ticks = uint64_t(time_ns / 100 + epoch);
        FILETIME time{DWORD(ticks), DWORD(ticks >> 32)};
        auto handle = CreateFileW(path.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(handle != INVALID_HANDLE_VALUE);
        auto written = SetFileTime(handle, nullptr, nullptr, &time); CloseHandle(handle); REQUIRE(written);
#else
        auto time = std::chrono::nanoseconds(time_ns);
        auto seconds = std::chrono::floor<std::chrono::seconds>(time);
        auto fraction = (time - seconds).count();
        timespec times[2]{{seconds.count(), fraction}, {seconds.count(), fraction}};
        REQUIRE(utimensat(AT_FDCWD, path.u8string().c_str(), times, 0) == 0);
#endif
        return path;
    }
};
std::string contents(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
tensoracle::detail::file_time mtime(const fs::path& path) {
    return tensoracle::detail::status_of(path).mtime;
}
tensoracle_cache_prune_options maximum(int64_t bytes) {
    return {sizeof(tensoracle_cache_prune_options), TENSORACLE_CACHE_MAX_SIZE, 0, bytes};
}
std::vector<std::string> names(const tensoracle::cache_result& result, int32_t group) {
    std::vector<std::string> output;
    for (const auto& entry : result.entries(group)) output.emplace_back(entry.file);
    return output;
}
}

TEST_CASE("Cache listing reads only headers and preserves exact times without weights or a GPU", "[SP-TENSORACLE-042][SP-TENSORACLE-043][SP-TENSORACLE-047][SP-TENSORACLE-049]") {
    fixture files;
    json conditions{{"name", "encoder"}, {"graph_version", "v2"}, {"weights", json::array({std::string(64, 'a')})},
        {"plugins", json::array({"plugin-digest"})}, {"inputs", {{"x", {{"minimum", {1}}, {"optimum", {2}}, {"maximum", {3}}}}}},
        {"gpu", {{"name", "GPU"}, {"total_memory", 24000000000ULL}}}, {"tensorrt_version", "11.4.0"},
        {"os", "Linux"}, {"architecture", "x86_64"}, {"builder_settings", {{"optimization_level", 3}, {"tf32", false}}}};
#if defined(_WIN32)
    constexpr int64_t recorded = 1000000100;
#else
    constexpr int64_t recorded = 1000000123;
#endif
    auto path = files.save("large.engine", recorded, uint64_t(4) * 1024 * 1024 * 1024, conditions);
    std::ofstream(files.directory / "empty.engine");
    std::ofstream(files.directory / "bad.engine") << "bad";
    fs::create_symlink(path, files.directory / "link.engine");
    // The common parser leaves the OS descriptor exactly at the plan's start.
    int fd = tensoracle::detail::open_file(path);
    REQUIRE(fd >= 0);
    auto header = tensoracle::detail::read_engine_header(fd, path, fs::file_size(path));
#if defined(_WIN32)
    auto offset = _lseeki64(fd, 0, SEEK_CUR);
#else
    auto offset = lseek(fd, 0, SEEK_CUR);
#endif
    tensoracle::detail::close_file(fd);
    REQUIRE(uint64_t(offset) == header.plan_offset);
    REQUIRE(header.plan_size > 3ULL * 1024 * 1024 * 1024);
    auto listing = tensoracle::list_cache();
    auto entries = listing.entries();
    REQUIRE(entries.size() == 4);
    REQUIRE(std::string(entries[0].file) == "bad.engine");
    REQUIRE(entries[0].status == TENSORACLE_CACHE_UNRECOGNIZED);
    REQUIRE(entries[0].reason);
    REQUIRE(entries[1].status == TENSORACLE_CACHE_BUILDING_OR_LEFTOVER);
    REQUIRE(entries[2].status == TENSORACLE_CACHE_RECOGNIZED);
    REQUIRE(entries[2].size_bytes == 4ULL * 1024 * 1024 * 1024);
    REQUIRE(entries[2].recorded_at_seconds == 1);
    REQUIRE(entries[2].recorded_at_nanoseconds == recorded % 1000000000);
    REQUIRE(json::parse(entries[2].header_json)["workspace_limit"] == 4096);
    REQUIRE(json::parse(entries[2].header_json)["conditions"] == conditions);
    REQUIRE(json::parse(entries[2].header_json)["shape_tensor_outputs"] == json::array({"shape"}));
    REQUIRE(json::parse(entries[2].weight_paths_json)[0]["paths"].empty());
    REQUIRE(entries[3].status == TENSORACLE_CACHE_UNRECOGNIZED);
    REQUIRE(mtime(path) == tensoracle::detail::unix_time_parts(recorded));
}

TEST_CASE("Cache listing trusts only complete checked matching weight records", "[SP-TENSORACLE-043][SP-TENSORACLE-049]") {
    fixture files;
    files.save("one.engine");
    json payload{{"version", 1}, {"path", (files.root / "absent.safetensors").u8string()},
                 {"size", 1}, {"mtime_ns", 1}, {"inode", 2}, {"sha256", std::string(64, 'a')}};
    auto save_record = [&](json value, bool valid_checksum = true) {
        auto encoded = value.dump(-1, ' ', true);
        value["checksum"] = valid_checksum ? tensoracle::detail::sha256(encoded.data(), encoded.size()) : std::string(64, '0');
        std::ofstream(files.directory / "weight-record.json") << value.dump();
    };
    save_record(payload);
    auto listing = tensoracle::list_cache();
    auto paths = json::parse(listing.entries()[0].weight_paths_json);
    REQUIRE(paths[0]["paths"] == json::array({payload["path"]}));
    REQUIRE_FALSE(fs::exists(payload["path"].get<std::string>()));
    for (int damage = 0; damage < 5; ++damage) {
        auto invalid = payload;
        if (damage == 0) invalid["version"] = 2;
        if (damage == 1) invalid["size"] = true;
        if (damage == 2) invalid["sha256"] = "invalid";
        if (damage == 3) invalid["sha256"] = std::string(64, 'b');
        save_record(invalid, damage != 4);
        auto changed = tensoracle::list_cache();
        REQUIRE(json::parse(changed.entries()[0].weight_paths_json)[0]["paths"].empty());
    }
}

TEST_CASE("Prune plans select strictly older times and choose oldest names first for size", "[SP-TENSORACLE-044][SP-TENSORACLE-045][SP-TENSORACLE-050]") {
    fixture files;
    constexpr int64_t day = 86400000000000;
    files.save("old.engine", 100 * day - 2 * day - 100);
    files.save("boundary.engine", 100 * day - 2 * day);
    files.save("new.engine", 100 * day - day);
    tensoracle_cache_prune_options age{sizeof(age), TENSORACLE_CACHE_UNUSED_FOR, 2 * day, 0};
    auto exact = tensoracle::detail::plan_cache_prune(age, 100 * day);
    REQUIRE(exact->targets.size() == 1);
    REQUIRE(exact->snapshot.entries[exact->targets[0]].path.filename() == "old.engine");
    auto options = maximum(1024);
    auto plan = tensoracle::plan_cache_prune(options);
    REQUIRE(names(plan, TENSORACLE_CACHE_TARGETS) == std::vector<std::string>{"old.engine", "boundary.engine"});
    REQUIRE(fs::exists(files.directory / "old.engine")); // Preview can be displayed before deletion.
    auto before = contents(files.directory / "old.engine");
    auto time = mtime(files.directory / "old.engine");
    auto preview = tensoracle::apply_cache_prune(plan, true);
    REQUIRE(preview.summary().removed_size_bytes == 0);
    REQUIRE(preview.summary().projected_size_bytes == 1024);
    REQUIRE(preview.summary().remaining_size_bytes == 3072);
    REQUIRE(contents(files.directory / "old.engine") == before);
    REQUIRE(mtime(files.directory / "old.engine") == time);
    auto result = tensoracle::apply_cache_prune(plan);
    REQUIRE(names(result, TENSORACLE_CACHE_REMOVED) == names(plan, TENSORACLE_CACHE_TARGETS));
    REQUIRE(result.summary().removed_size_bytes == 2048);
    REQUIRE(result.summary().remaining_size_bytes == 1024);
    REQUIRE(result.summary().succeeded == 1);
}

TEST_CASE("Pruning excludes unknown files and reports missing changed and denied removals", "[SP-TENSORACLE-044][SP-TENSORACLE-050][SP-TENSORACLE-051]") {
    fixture files;
    auto missing = files.save("a-missing.engine");
    auto changed = files.save("b-changed.engine");
    auto removed = files.save("c-removed.engine");
    std::ofstream(files.directory / "unknown.engine") << "unknown";
    std::ofstream(files.directory / "empty.engine");
    auto plan = tensoracle::plan_cache_prune(maximum(0));
    fs::remove(missing);
    std::ofstream(changed) << "new contents";
    auto result = tensoracle::apply_cache_prune(plan);
    REQUIRE(names(result, TENSORACLE_CACHE_REMOVED) == std::vector<std::string>{"c-removed.engine"});
    REQUIRE(names(result, TENSORACLE_CACHE_MISSING) == std::vector<std::string>{"a-missing.engine"});
    REQUIRE(names(result, TENSORACLE_CACHE_FAILED) == std::vector<std::string>{"b-changed.engine"});
    REQUIRE(result.entries(TENSORACLE_CACHE_FAILED)[0].reason);
    REQUIRE(names(result, TENSORACLE_CACHE_EXCLUDED) == std::vector<std::string>{"empty.engine", "unknown.engine"});
    REQUIRE(result.summary().removed_size_bytes == 1024);
    REQUIRE(result.summary().remaining_size_bytes == 19);
    REQUIRE(result.summary().max_size_reached == 0);
    REQUIRE(result.summary().succeeded == 0);
#if defined(_WIN32)
    {
        auto denied = files.save(u8"削除-\U0001f310.engine");
        auto denial_plan = tensoracle::plan_cache_prune(maximum(0));
        auto held = CreateFileW(denied.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(held != INVALID_HANDLE_VALUE);
        auto denial = tensoracle::apply_cache_prune(denial_plan);
        CloseHandle(held);
        REQUIRE(fs::exists(denied));
        REQUIRE_FALSE(denial.entries(TENSORACLE_CACHE_FAILED).empty());
        REQUIRE(std::string(denial.entries(TENSORACLE_CACHE_FAILED)[0].reason).find(u8"削除-\U0001f310.engine") != std::string::npos);
        REQUIRE(denial.summary().removed_size_bytes == 0);
    }
#else
    if (geteuid() != 0) {
        auto denied = files.save("denied.engine");
        auto denial_plan = tensoracle::plan_cache_prune(maximum(0));
        fs::permissions(files.directory, fs::perms::owner_read | fs::perms::owner_exec);
        auto denial = tensoracle::apply_cache_prune(denial_plan);
        fs::permissions(files.directory, fs::perms::owner_all);
        REQUIRE(fs::exists(denied));
        REQUIRE_FALSE(denial.entries(TENSORACLE_CACHE_FAILED).empty());
        REQUIRE(denial.summary().removed_size_bytes == 0);
    }
#endif
}

TEST_CASE("Prune argument and ABI errors leave the cache untouched", "[SP-TENSORACLE-045][SP-TENSORACLE-050][SP-TENSORACLE-055][SP-TENSORACLE-056]") {
    fixture files;
    auto path = files.save("one.engine");
    for (const auto& options : {tensoracle_cache_prune_options{sizeof(tensoracle_cache_prune_options), 0, 0, 0},
             tensoracle_cache_prune_options{sizeof(tensoracle_cache_prune_options), 3, 1, 1},
             tensoracle_cache_prune_options{sizeof(tensoracle_cache_prune_options), TENSORACLE_CACHE_UNUSED_FOR, -1, 0}, maximum(-1)}) {
        tensoracle_cache_result* result = reinterpret_cast<tensoracle_cache_result*>(1);
        tensoracle_error* diagnostic = nullptr;
        REQUIRE(tensoracle_cache_plan_prune(&options, &result, &diagnostic) == TENSORACLE_INVALID_ARGUMENT);
        REQUIRE(result == nullptr);
        REQUIRE(diagnostic);
        tensoracle_error_destroy(diagnostic);
        REQUIRE(fs::exists(path));
    }
    auto listing = tensoracle::list_cache();
    tensoracle_cache_result* result = nullptr;
    REQUIRE(tensoracle_cache_apply_prune(listing.get(), 0, &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    struct prefix { size_t struct_size; const char* directory; unsigned char trailing[8]; } output;
    std::memset(&output, 0xa5, sizeof(output)); output.struct_size = offsetof(prefix, trailing);
    REQUIRE(tensoracle_cache_result_summary(listing.get(), reinterpret_cast<tensoracle_cache_summary*>(&output), nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(output.directory) == files.directory.u8string());
    for (auto byte : output.trailing) REQUIRE(byte == 0xa5);
    struct entry_prefix { size_t struct_size; const char* file; unsigned char trailing[8]; } entry;
    std::memset(&entry, 0xa5, sizeof(entry)); entry.struct_size = offsetof(entry_prefix, trailing);
    REQUIRE(tensoracle_cache_result_entry(listing.get(), TENSORACLE_CACHE_ENTRIES, 0,
        reinterpret_cast<tensoracle_cache_entry*>(&entry), nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(std::string(entry.file) == "one.engine");
    for (auto byte : entry.trailing) REQUIRE(byte == 0xa5);
    size_t count = 100;
    REQUIRE(tensoracle_cache_result_count(listing.get(), 99, &count, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(count == 0);
    tensoracle_cache_entry invalid{sizeof(invalid)};
    REQUIRE(tensoracle_cache_result_entry(listing.get(), TENSORACLE_CACHE_ENTRIES, 1, &invalid, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    auto short_options = maximum(0);
    short_options.struct_size = offsetof(tensoracle_cache_prune_options, max_size_bytes);
    REQUIRE(tensoracle_cache_plan_prune(&short_options, &result, nullptr) == TENSORACLE_INVALID_ARGUMENT);
    REQUIRE(tensoracle_cache_result_destroy(nullptr) == TENSORACLE_SUCCESS);
    REQUIRE(fs::exists(path));
}

TEST_CASE("A missing cache stays absent during listing planning and preview", "[SP-TENSORACLE-010][SP-TENSORACLE-043][SP-TENSORACLE-045]") {
    fixture files;
    fs::remove(files.directory);
    auto listing = tensoracle::list_cache();
    REQUIRE(listing.entries().empty());
    REQUIRE(listing.summary().total_size_bytes == 0);
    auto plan = tensoracle::plan_cache_prune(maximum(0));
    auto preview = tensoracle::apply_cache_prune(plan, true);
    REQUIRE(preview.summary().succeeded == 1);
    REQUIRE_FALSE(fs::exists(files.directory));
}

TEST_CASE("Listing and reuse reject the same invalid header schemas without changing mtime", "[SP-TENSORACLE-009][SP-TENSORACLE-042][SP-TENSORACLE-049]") {
    fixture files;
    auto path = files.directory / "bad.engine";
    auto put_header = [&](const json& header) {
        auto bytes = header.dump();
        std::ofstream output(path, std::ios::binary);
        output.write("TENSORACLE\2", 10);
        char length[8];
        for (size_t i = 0; i < 8; ++i) length[i] = char(uint64_t(bytes.size()) >> (56 - i * 8));
        output.write(length, 8); output << bytes << "plan";
    };
    for (const auto& invalid : {json::array(), json::object(), json{{"conditions", json::array()}},
              json{{"conditions", json::object()}, {"plan_sha256", std::string(64, 'a')}, {"shape_tensor_outputs", json::array({1})}}}) {
        put_header(invalid);
        auto time = mtime(path);
        auto listed = tensoracle::list_cache();
        REQUIRE(listed.entries()[0].status == TENSORACLE_CACHE_UNRECOGNIZED);
        REQUIRE_THROWS_AS(tensoracle::detail::read_engine_file(path, json::object()), tensoracle::detail::corrupt_engine_error);
        REQUIRE(mtime(path) == time);
    }
    put_header({{"conditions", json::object()}, {"plan_sha256", std::string(64, '0')}});
    auto listed = tensoracle::list_cache();
    REQUIRE(listed.entries()[0].status == TENSORACLE_CACHE_RECOGNIZED);
    REQUIRE_THROWS_AS(tensoracle::detail::read_engine_file(path, json::object()), tensoracle::detail::corrupt_engine_error);
}

TEST_CASE("Size pruning orders equal timestamps by filename and retains digest records", "[SP-TENSORACLE-044][SP-TENSORACLE-050]") {
    fixture files;
    for (int i = 19; i >= 0; --i) {
        auto name = std::string(i < 10 ? "0" : "") + std::to_string(i) + ".engine";
        files.save(name.c_str());
    }
    auto record = files.directory / "weight-not-a-target.json";
    std::ofstream(record) << "record";
    auto plan = tensoracle::plan_cache_prune(maximum(10 * 1024));
    auto targets = names(plan, TENSORACLE_CACHE_TARGETS);
    REQUIRE(targets.size() == 10);
    for (int i = 0; i < 10; ++i) REQUIRE(targets[i] == "0" + std::to_string(i) + ".engine");
    auto removed = tensoracle::apply_cache_prune(plan);
    REQUIRE(removed.summary().remaining_size_bytes == 10 * 1024);
    REQUIRE(fs::exists(record));
}

TEST_CASE("A prune plan retains its selected cache when the environment changes", "[SP-TENSORACLE-044][SP-TENSORACLE-050][SP-TENSORACLE-051]") {
    fixture files;
    files.save("old.engine", 100);
    files.save("new.engine", 200);
    auto plan = tensoracle::plan_cache_prune(maximum(1024));
    auto other = files.root / "other-cache-root";
    fs::create_directory(other);
    test_platform::set_environment("XDG_CACHE_HOME", other.u8string());
    auto result = tensoracle::apply_cache_prune(plan);
    REQUIRE(result.summary().remaining_size_bytes == 1024);
    REQUIRE(result.summary().removed_size_bytes == 1024);
    REQUIRE(result.summary().succeeded == 1);
    REQUIRE_FALSE(fs::exists(files.directory / "old.engine"));
    REQUIRE(fs::exists(files.directory / "new.engine"));
}

TEST_CASE("Unknown engine bytes can make a prune maximum unreachable without removal failures", "[SP-TENSORACLE-045][SP-TENSORACLE-050]") {
    fixture files;
    auto path = files.directory / "unknown.engine";
    std::ofstream(path) << "unrecognized engine";
    auto before = contents(path);
    auto time = mtime(path);
    auto plan = tensoracle::plan_cache_prune(maximum(0));
    REQUIRE(plan.entries(TENSORACLE_CACHE_TARGETS).empty());
    for (bool dry_run : {true, false}) {
        auto result = tensoracle::apply_cache_prune(plan, dry_run);
        REQUIRE(result.entries(TENSORACLE_CACHE_FAILED).empty());
        REQUIRE(result.summary().removed_size_bytes == 0);
        REQUIRE(result.summary().remaining_size_bytes == before.size());
        REQUIRE(result.summary().max_size_reached == 0);
        REQUIRE(result.summary().succeeded == 0);
        REQUIRE(contents(path) == before);
        REQUIRE(mtime(path) == time);
    }
}

TEST_CASE("Cache entries report negative file times as signed seconds and nonnegative fractions",
          "[SP-TENSORACLE-043][SP-TENSORACLE-046][SP-TENSORACLE-055]") {
    fixture files;
#if defined(_WIN32)
    constexpr int64_t time = -100; // FILETIME has 100 ns resolution.
    constexpr uint32_t fraction = 999999900;
#else
    constexpr int64_t time = -1;
    constexpr uint32_t fraction = 999999999;
#endif
    auto path = files.save("before-epoch.engine", time);
    REQUIRE(mtime(path) == tensoracle::detail::unix_time_parts(time));
    auto listing = tensoracle::list_cache();
    auto entries = listing.entries();
    REQUIRE(entries.size() == 1);
    REQUIRE(entries[0].recorded_at_seconds == -1);
    REQUIRE(entries[0].recorded_at_nanoseconds == fraction);
}

TEST_CASE("Age pruning compares normalized file times on both sides of the Unix epoch",
          "[SP-TENSORACLE-043][SP-TENSORACLE-044][SP-TENSORACLE-046][SP-TENSORACLE-055]") {
    fixture files;
    files.save("earliest.engine", -1200000000);
    files.save("older.engine", -700000000);
    files.save("boundary.engine", -500000000);
    files.save("newer.engine", -100000000);
    files.save("epoch.engine", 0);
    tensoracle_cache_prune_options age{sizeof(age), TENSORACLE_CACHE_UNUSED_FOR, 500000000, 0};
    auto plan = tensoracle::detail::plan_cache_prune(age, 0);
    std::vector<std::string> selected;
    for (auto index : plan->targets) selected.push_back(plan->snapshot.entries[index].file);
    REQUIRE(selected == std::vector<std::string>{"earliest.engine", "older.engine"});
}

TEST_CASE("Cache entries preserve file times beyond the signed nanosecond range",
          "[SP-TENSORACLE-043][SP-TENSORACLE-046][SP-TENSORACLE-055]") {
    fixture files;
    auto path = files.save("future.engine");
    for (auto time : {std::pair<int64_t, int64_t>{9223372036, 854775900},
                     std::pair<int64_t, int64_t>{10413792000, 123456700}}) {
        set_recorded_time(path, time.first, time.second);
        auto listing = tensoracle::list_cache();
        auto entries = listing.entries();
        REQUIRE(entries.size() == 1);
        REQUIRE(entries[0].status == TENSORACLE_CACHE_RECOGNIZED);
        CHECK(entries[0].recorded_at_seconds == time.first);
        CHECK(entries[0].recorded_at_nanoseconds == time.second);
    }
}

TEST_CASE("Age pruning retains future engines beyond the signed nanosecond range",
          "[SP-TENSORACLE-043][SP-TENSORACLE-044][SP-TENSORACLE-046][SP-TENSORACLE-055]") {
    fixture files;
    auto old = files.save("old.engine");
    auto future = files.save("future.engine");
    set_recorded_time(future, 10413792000, 123456700);
    tensoracle_cache_prune_options age{sizeof(age), TENSORACLE_CACHE_UNUSED_FOR, 86400000000000, 0};
    auto plan = tensoracle::plan_cache_prune(age);
    CHECK(names(plan, TENSORACLE_CACHE_TARGETS) == std::vector<std::string>{"old.engine"});
    auto removed = tensoracle::apply_cache_prune(plan);
    CHECK(names(removed, TENSORACLE_CACHE_REMOVED) == std::vector<std::string>{"old.engine"});
    CHECK(removed.summary().remaining_size_bytes == 1024);
    CHECK(fs::exists(future));
    CHECK_FALSE(fs::exists(old));
}

TEST_CASE("Size pruning orders engines beyond the signed nanosecond range by seconds and fraction",
          "[SP-TENSORACLE-043][SP-TENSORACLE-044][SP-TENSORACLE-046][SP-TENSORACLE-055]") {
    fixture files;
    auto past = files.save("c-past.engine");
    auto earlier = files.save("b-earlier.engine");
    auto later = files.save("a-later.engine");
    set_recorded_time(past, 7258118400, 0); // 2200-01-01 UTC
    set_recorded_time(earlier, 10413792000, 123456700); // 2300-01-01 UTC
    set_recorded_time(later, 10413792000, 123456800);
    auto plan = tensoracle::plan_cache_prune(maximum(1024));
    CHECK(names(plan, TENSORACLE_CACHE_TARGETS) == std::vector<std::string>{"c-past.engine", "b-earlier.engine"});
    auto removed = tensoracle::apply_cache_prune(plan);
    CHECK(names(removed, TENSORACLE_CACHE_REMOVED) == std::vector<std::string>{"c-past.engine", "b-earlier.engine"});
    CHECK(removed.summary().remaining_size_bytes == 1024);
    CHECK(fs::exists(later));
    CHECK_FALSE(fs::exists(earlier));
    CHECK_FALSE(fs::exists(past));
}

TEST_CASE("Weight digest records retain their version one nanosecond values for wide file times",
          "[SP-TENSORACLE-009][SP-TENSORACLE-055]") {
    fixture files;
    auto path = files.root / "weights";
    for (auto time : {std::make_tuple(int64_t(-1), int64_t(999999900), int64_t(-100)),
                     std::make_tuple(int64_t(1), int64_t(123456700), int64_t(1123456700)),
                     std::make_tuple(int64_t(10413792000), int64_t(123456700), int64_t(-8032952073586094916))}) {
        std::ofstream(path, std::ios::binary) << "original";
        set_recorded_time(path, std::get<0>(time), std::get<1>(time));
        auto digest = tensoracle::detail::weight_sha256(path, true);
        REQUIRE(digest == tensoracle::detail::sha256("original", 8));
        auto record_path = fs::directory_iterator(files.directory)->path();
        auto record = tensoracle::detail::read_weight_record(record_path);
        REQUIRE(record);
        json expected{{"version", 1}, {"path", tensoracle::detail::path_utf8(fs::absolute(path))},
            {"size", 8}, {"mtime_ns", std::get<2>(time)},
            {"inode", tensoracle::detail::status_of(path).inode}, {"sha256", digest}};
        auto encoded = expected.dump(-1, ' ', true);
        expected["checksum"] = tensoracle::detail::sha256(encoded.data(), encoded.size());
        REQUIRE(*record == expected);
        std::ofstream(path, std::ios::binary) << "modified";
        set_recorded_time(path, std::get<0>(time), std::get<1>(time));
        REQUIRE(tensoracle::detail::weight_sha256(path, false) == digest);
        REQUIRE(tensoracle::detail::weight_sha256(path, true) == tensoracle::detail::sha256("modified", 8));
    }
}

TEST_CASE("Cache listing and pruning expose filesystem paths as UTF-8",
          "[SP-TENSORACLE-015][SP-TENSORACLE-009][SP-TENSORACLE-043][SP-TENSORACLE-047]") {
    fixture files;
    auto root = files.root / fs::u8path(u8"保存-\U0001f310");
    files.directory = root / "tensoracle";
    fs::create_directories(files.directory);
    test_platform::set_environment("XDG_CACHE_HOME", root.u8string());
    auto path = files.directory / fs::u8path(u8"保存.engine");
    tensoracle::detail::write_engine_file(path, {{"name", u8"model"}}, {'p'}, {}, {});
    auto listing = tensoracle::list_cache();
    REQUIRE(fs::u8path(listing.summary().directory) == fs::absolute(files.directory));
    auto entries = listing.entries();
    REQUIRE(entries.size() == 1);
    REQUIRE(std::string(entries[0].file) == u8"保存.engine");
    REQUIRE(json::parse(entries[0].header_json)["conditions"]["name"] == "model");
    REQUIRE(test_platform::run({TENSORACLE_CACHE_PROBE, "--list", files.directory.u8string(), u8"保存.engine"}) == 0);
    auto plan = tensoracle::plan_cache_prune(maximum(0));
    REQUIRE(names(plan, TENSORACLE_CACHE_TARGETS) == std::vector<std::string>{u8"保存.engine"});
    auto removed = tensoracle::apply_cache_prune(plan);
    REQUIRE(removed.summary().succeeded);
    REQUIRE(names(removed, TENSORACLE_CACHE_REMOVED) == std::vector<std::string>{u8"保存.engine"});
    REQUIRE_FALSE(fs::exists(path));
}

TEST_CASE("Cache filesystem failures return UTF-8 directory diagnostics",
          "[SP-TENSORACLE-015][SP-TENSORACLE-043][SP-TENSORACLE-054]") {
    fixture files;
    auto root = files.root / fs::u8path(u8"保存-\U0001f310");
    fs::create_directory(root);
    std::ofstream(root / "tensoracle") << "not a directory";
    test_platform::set_environment("XDG_CACHE_HOME", root.u8string());
    try { auto listing = tensoracle::list_cache(); FAIL("a file was accepted as the cache directory"); }
    catch (const tensoracle::error& failure) {
        REQUIRE(std::string(failure.what()).find(u8"保存-\U0001f310") != std::string::npos);
    }
}

TEST_CASE("Unicode home paths select the default cache when XDG is absent or empty",
          "[SP-TENSORACLE-015][SP-TENSORACLE-010][SP-TENSORACLE-043]") {
    fixture files;
#if defined(_WIN32)
    constexpr auto home_name = "USERPROFILE";
#else
    constexpr auto home_name = "HOME";
#endif
    auto previous = test_platform::get_environment(home_name);
    struct restore_home {
        const char* name; std::optional<std::string> previous;
        ~restore_home() { if (previous) test_platform::set_environment(name, *previous); else test_platform::unset_environment(name); }
    } restore{home_name, previous};
    auto home = files.root / fs::u8path(u8"利用者-\U0001f310");
    auto directory = home / ".cache" / "tensoracle";
    fs::create_directories(directory);
    tensoracle::detail::write_engine_file(directory / "home.engine", {{"name", "home"}}, {'p'}, {}, {});
    test_platform::set_environment(home_name, home.u8string());
    for (bool empty : {false, true}) {
        if (empty) test_platform::set_environment("XDG_CACHE_HOME", "");
        else test_platform::unset_environment("XDG_CACHE_HOME");
        auto listing = tensoracle::list_cache();
        REQUIRE(fs::u8path(listing.summary().directory) == directory);
        REQUIRE(listing.entries().size() == 1);
        REQUIRE(std::string(listing.entries()[0].file) == "home.engine");
        REQUIRE(test_platform::run({TENSORACLE_CACHE_PROBE, "--list", directory.u8string(), "home.engine"}) == 0);
    }
}
