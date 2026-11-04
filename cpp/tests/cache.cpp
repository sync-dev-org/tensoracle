#include "cache.hpp"
#include "sha256.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <array>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace tensoracle::detail;

// Link wrappers replace only OS boundaries; cache storage and hashing stay real.
namespace {
std::string changing_weight;
int stat_calls = 0;
int writer_ready = -1, writer_release = -1;
}
extern "C" int __real_stat(const char*, struct stat*);
extern "C" int __wrap_stat(const char* path, struct stat* value) {
    if (path == changing_weight && ++stat_calls == 2) {
        struct stat before{};
        if (__real_stat(path, &before)) return -1;
        timespec times[]{before.st_atim, before.st_mtim};
        ++times[1].tv_sec;
        if (utimensat(AT_FDCWD, path, times, 0)) return -1;
    }
    return __real_stat(path, value);
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
    if (writer_ready >= 0) {
        char token = 'w';
        if (write(writer_ready, &token, 1) != 1 || read(writer_release, &token, 1) != 1) return -1;
    }
    return __real_fsync(fd);
}

namespace {
struct fixture {
    fs::path directory;
    std::string old_xdg;
    bool had_xdg;
    fixture() {
        char pattern[] = "/tmp/tensoracle-cache-XXXXXX";
        auto* path = mkdtemp(pattern);
        REQUIRE(path);
        directory = path;
        auto* old = std::getenv("XDG_CACHE_HOME");
        had_xdg = old != nullptr;
        if (old) old_xdg = old;
        setenv("XDG_CACHE_HOME", directory.c_str(), 1);
    }
    ~fixture() {
        if (had_xdg) setenv("XDG_CACHE_HOME", old_xdg.c_str(), 1);
        else unsetenv("XDG_CACHE_HOME");
        fs::remove_all(directory);
    }
};
void put(const fs::path& path, const std::string& contents) { std::ofstream(path, std::ios::binary) << contents; }
std::string get(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
}

TEST_CASE("Stored plans reject broken headers checksums and different conditions", "[SP-TENSORACLE-009][SP-TENSORACLE-011][SP-TENSORACLE-055]") {
    fixture files;
    auto path = files.directory / "test.engine";
    nlohmann::json conditions{{"name", "encoder"}, {"graph_version", "1"}};
    std::vector<uint8_t> plan{'p', 'l', 'a', 'n'};
    write_engine_file(path, conditions, plan, 1024, {});
    const auto original = get(path);
    auto loaded = read_engine_file(path, conditions);
    REQUIRE(loaded.plan == plan);
    REQUIRE(loaded.header["workspace_limit"] == 1024);
    REQUIRE(loaded.header["conditions"] == conditions);
    REQUIRE(original.substr(0, 10) == std::string("TENSORACLE\2", 10));
    REQUIRE_THROWS_AS(read_engine_file(path, {{"name", "decoder"}}), engine_condition_error);
    auto changed = original;
    changed.back() ^= 1;
    put(path, changed);
    REQUIRE_THROWS_AS(read_engine_file(path, conditions), corrupt_engine_error);
    for (const auto& malformed : {std::string{}, std::string("bad"), original.substr(0, 17),
                                  original.substr(0, original.size() - plan.size())}) {
        put(path, malformed);
        REQUIRE_THROWS_AS(read_engine_file(path, conditions), corrupt_engine_error);
    }
    write_engine_file(path, conditions, plan, std::nullopt, {"shape"});
    REQUIRE(read_engine_file(path, conditions).header["shape_tensor_outputs"] == nlohmann::json::array({"shape"}));
    REQUIRE(read_engine_file(path, conditions).header["workspace_limit"].is_null());
    REQUIRE(std::distance(fs::directory_iterator(files.directory), fs::directory_iterator{}) == 1);
}

TEST_CASE("Engine identity follows content roles shapes GPU and TensorRT but not workspace", "[SP-TENSORACLE-008][SP-TENSORACLE-009][SP-TENSORACLE-020][SP-TENSORACLE-052]") {
    fixture files;
    auto weight = files.directory / "weights";
    put(weight, "weight bytes");
    cache_request request;
    request.name = "encoder";
    request.graph_version = "1";
    request.weights = {weight};
    request.inputs = {{"x", {1, 2}, {3, 2}, {5, 2}}};
    auto conditions = make_conditions(request, {}, {{"name", "GPU A"}}, "11.4.0", "Linux", "x86_64");
    auto original = cache_path(conditions);
    request.workspace_limit = 123;
    REQUIRE(cache_path(make_conditions(request, {}, {{"name", "GPU A"}}, "11.4.0", "Linux", "x86_64")) == original);
    REQUIRE_FALSE(conditions.contains("precision"));
    for (const auto& key : {"name", "graph_version", "gpu", "tensorrt_version", "inputs", "weights", "plugins", "os", "architecture", "builder_settings"}) {
        auto changed = conditions;
        changed[key] = "different";
        REQUIRE(cache_path(changed) != original);
    }
    auto copy = files.directory / "other-name";
    put(copy, "weight bytes");
    request.weights = {copy};
    REQUIRE(make_conditions(request, {}, {{"name", "GPU A"}}, "11.4.0", "Linux", "x86_64") == conditions);
    request.inputs[0].minimum = {4, 2};
    REQUIRE_THROWS_AS(make_conditions(request, {}, {}, "11.4.0", "Linux", "x86_64"), std::invalid_argument);
}

TEST_CASE("Weight records are checked refreshed and shared across processes", "[SP-TENSORACLE-009][SP-TENSORACLE-011]") {
    fixture files;
    auto weight = files.directory / "weights";
    put(weight, "first");
    auto original = weight_sha256(weight, false);
    REQUIRE(original == sha256("first", 5));
    auto time = fs::last_write_time(weight);
    put(weight, "other");
    fs::last_write_time(weight, time);
    REQUIRE(weight_sha256(weight, false) == original);
    // A new OS process must read the disk record, not an in-memory digest.
    auto pid = fork();
    REQUIRE(pid >= 0);
    if (!pid) {
        execl(TENSORACLE_CACHE_PROBE, TENSORACLE_CACHE_PROBE, weight.c_str(), original.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    REQUIRE(weight_sha256(weight, true) == sha256("other", 5));
    for (const auto& entry : fs::directory_iterator(cache_directory())) {
        if (entry.path().extension() != ".json") continue;
        auto record = nlohmann::json::parse(get(entry.path()));
        record["sha256"] = std::string(64, '0');
        put(entry.path(), record.dump());
    }
    REQUIRE(weight_sha256(weight, false) == sha256("other", 5));
    put(weight, "longer content");
    REQUIRE(weight_sha256(weight, false) == sha256("longer content", 14));
    auto replacement = files.directory / "replacement";
    put(replacement, "replacement");
    fs::rename(replacement, weight);
    REQUIRE(weight_sha256(weight, false) == sha256("replacement", 11));
}

TEST_CASE("Weight record write failure leaves hashing usable and cache paths obey XDG", "[SP-TENSORACLE-009][SP-TENSORACLE-010]") {
    fixture files;
    auto weight = files.directory / "weights";
    put(weight, "contents");
    REQUIRE(cache_directory() == files.directory / "tensoracle");
    // A file in place of the cache directory makes writes fail for every user.
    put(cache_directory(), "cannot create a directory here");
    REQUIRE(weight_sha256(weight, false) == sha256("contents", 8));
    auto* previous = std::getenv("HOME");
    std::string home = previous ? previous : "";
    unsetenv("XDG_CACHE_HOME");
    setenv("HOME", files.directory.c_str(), 1);
    REQUIRE(cache_directory() == files.directory / ".cache" / "tensoracle");
    if (previous) setenv("HOME", home.c_str(), 1); else unsetenv("HOME");
}

TEST_CASE("Recording a successful load never recreates a missing engine file", "[SP-TENSORACLE-042][SP-TENSORACLE-046][SP-TENSORACLE-048]") {
    fixture files;
    auto path = files.directory / "test.engine";
    put(path, "engine");
    auto old = fs::file_time_type::clock::now() - std::chrono::hours(48);
    fs::last_write_time(path, old);
    touch_engine(path);
    REQUIRE(fs::last_write_time(path) > old);
    fs::remove(path);
    REQUIRE_NOTHROW(touch_engine(path));
    REQUIRE_FALSE(fs::exists(path));
}

TEST_CASE("Changed plugin bytes cannot select conditions for an older loaded creator", "[SP-TENSORACLE-009][SP-TENSORACLE-011]") {
    fixture files;
    auto weight = files.directory / "weight";
    auto plugin = files.directory / "plugin";
    put(weight, "weights");
    put(plugin, "original");
    cache_request request;
    request.name = "test"; request.graph_version = "1"; request.weights = {weight}; request.plugins = {plugin};
    auto registered = sha256("original", 8);
    REQUIRE(make_conditions(request, {registered}, {}, "11.4.0", "Linux", "x86_64")["plugins"][0] == registered);
    put(plugin, "changed");
    REQUIRE_THROWS_AS(make_conditions(request, {registered}, {}, "11.4.0", "Linux", "x86_64"), std::invalid_argument);
}

TEST_CASE("A weight changed during hashing leaves no digest record until a stable read", "[SP-TENSORACLE-002][SP-TENSORACLE-009]") {
    fixture files;
    auto weight = files.directory / "weights";
    put(weight, "first");
    struct reset { ~reset() { changing_weight.clear(); } } restore;
    changing_weight = weight.string(); stat_calls = 0;
    REQUIRE(weight_sha256(weight, false) == sha256("first", 5));
    REQUIRE(stat_calls == 2);
    REQUIRE_FALSE(fs::exists(cache_directory()));
    changing_weight.clear();
    REQUIRE(weight_sha256(weight, false) == sha256("first", 5));
    REQUIRE(std::distance(fs::directory_iterator(cache_directory()), fs::directory_iterator{}) == 1);
    auto record = read_weight_record(fs::directory_iterator(cache_directory())->path());
    REQUIRE(record); REQUIRE(record->at("sha256") == sha256("first", 5));
}

TEST_CASE("Concurrent digest writers publish one complete reusable record", "[SP-TENSORACLE-002][SP-TENSORACLE-009]") {
    fixture files;
    auto weight = files.directory / "weights";
    put(weight, "contents");
    int ready[2], release[2];
    REQUIRE(pipe(ready) == 0); REQUIRE(pipe(release) == 0);
    std::array<pid_t, 2> children{};
    for (auto& pid : children) {
        pid = fork(); REQUIRE(pid >= 0);
        if (!pid) {
            close(ready[0]); close(release[1]);
            writer_ready = ready[1]; writer_release = release[0];
            try { _exit(weight_sha256(weight, true) == sha256("contents", 8) ? 0 : 1); }
            catch (...) { _exit(2); }
        }
    }
    close(ready[1]); close(release[0]);
    char token;
    // Both processes have fully written distinct temporary records before either publishes.
    for (int i = 0; i < 2; ++i) REQUIRE(read(ready[0], &token, 1) == 1);
    REQUIRE(write(release[1], "go", 2) == 2);
    close(ready[0]); close(release[1]);
    for (auto pid : children) {
        int status{}; REQUIRE(waitpid(pid, &status, 0) == pid);
        REQUIRE(WIFEXITED(status)); REQUIRE(WEXITSTATUS(status) == 0);
    }
    REQUIRE(std::distance(fs::directory_iterator(cache_directory()), fs::directory_iterator{}) == 1);
    auto record = read_weight_record(fs::directory_iterator(cache_directory())->path());
    REQUIRE(record); REQUIRE(record->at("sha256") == sha256("contents", 8));
    auto time = fs::last_write_time(weight);
    put(weight, "replaced"); fs::last_write_time(weight, time);
    // Equal attributes with different bytes prove reuse of the complete published record.
    REQUIRE(weight_sha256(weight, false) == sha256("contents", 8));
}

TEST_CASE("Relative weight paths in different working directories have separate records", "[SP-TENSORACLE-009]") {
    fixture files;
    struct cwd_guard { fs::path old = fs::current_path(); ~cwd_guard() { fs::current_path(old); } } restore;
    auto first = files.directory / "first", second = files.directory / "second";
    fs::create_directory(first); fs::create_directory(second);
    put(first / "weights", "first"); put(second / "weights", "other");
    fs::current_path(first); REQUIRE(weight_sha256("weights", false) == sha256("first", 5));
    fs::current_path(second); REQUIRE(weight_sha256("weights", false) == sha256("other", 5));
    REQUIRE(std::distance(fs::directory_iterator(cache_directory()), fs::directory_iterator{}) == 2);
}
