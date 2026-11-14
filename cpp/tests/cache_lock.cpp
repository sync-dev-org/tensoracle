#include "cache.hpp"
#include "test_channel.hpp"
#include <catch2/catch_test_macros.hpp>
#include <future>
namespace {
namespace fs = std::filesystem;
using namespace tensoracle::detail;
struct directory {
    fs::path path = test_platform::make_directory();
    ~directory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
struct locker {
    test_channel::endpoint channel;
    std::unique_ptr<test_platform::process> child;
    explicit locker(const fs::path& path) {
        auto pair = test_channel::pair(); channel = pair[0];
        try { child = std::make_unique<test_platform::process>(std::vector<std::string>{TENSORACLE_FILE_LOCK_PROBE, path.u8string(), std::to_string(pair[1])}); }
        catch (...) { test_channel::close(pair[0]); test_channel::close(pair[1]); throw; }
        test_channel::close(pair[1]);
    }
    ~locker() { child.reset(); test_channel::close(channel); }
    char next() { return test_channel::receive(channel); }
    int finish(char command) { test_channel::send(channel, command); return child->wait(); }
};
}
TEST_CASE("Engine locks leave the complete file readable and waiters follow atomic publication", "[SP-TENSORACLE-009][SP-TENSORACLE-067]") {
    directory files; auto path = files.path / "test.engine";
    auto first = std::make_unique<engine_file_lock>(path);
    REQUIRE(first->empty());
    auto waiting = std::async(std::launch::async, [&] {
        engine_file_lock second(path);
        if (second.empty()) return false;
        return read_engine_file(path, {{"name", "test"}}).plan == std::vector<uint8_t>{'p'};
    });
    CHECK(waiting.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout);
    write_engine_file(path, {{"name", "test"}}, {'p'}, {}, {});
    REQUIRE(read_engine_file(path, {{"name", "test"}}).plan == std::vector<uint8_t>{'p'});
    first.reset();
    REQUIRE(waiting.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    REQUIRE(waiting.get());
    REQUIRE(std::distance(fs::directory_iterator(files.path), fs::directory_iterator{}) == 1);
}
TEST_CASE("A failed empty engine owner passes construction to a waiting thread", "[SP-TENSORACLE-011][SP-TENSORACLE-067]") {
    directory files; auto path = files.path / "test.engine";
    auto first = std::make_unique<engine_file_lock>(path);
    auto waiting = std::async(std::launch::async, [&] { engine_file_lock second(path); return second.empty(); });
    CHECK(waiting.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout);
    first.reset();
    REQUIRE(waiting.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    REQUIRE(waiting.get());
    REQUIRE_FALSE(fs::exists(path));
}
TEST_CASE("A terminated engine owner passes construction to a waiting process", "[SP-TENSORACLE-011][SP-TENSORACLE-067]") {
    directory files; auto path = files.path / "test.engine";
    locker first(path); REQUIRE(first.next() == 'o'); REQUIRE(first.next() == 'b');
    locker second(path); REQUIRE(second.next() == 'o'); CHECK(test_channel::quiet(second.channel));
    first.child->terminate(); REQUIRE(first.child->wait() == 137);
    REQUIRE(second.next() == 'b'); REQUIRE(second.finish('p') == 0);
    REQUIRE(read_engine_file(path, {{"name", "lock-test"}}).plan == std::vector<uint8_t>{'p'});
    locker reused(path); REQUIRE(reused.next() == 'o'); REQUIRE(reused.next() == 'r'); REQUIRE(reused.finish('g') == 0);
}
