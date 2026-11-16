#include "test_platform.hpp"
#include "sha256.hpp"
#include <tensoracle/plugins.h>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <fstream>

namespace fs = std::filesystem;
namespace {
struct directory {
    fs::path path = test_platform::make_directory();
    ~directory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
struct event {
    HANDLE handle;
    explicit event(const std::string& name) : handle(CreateEventW(nullptr, TRUE, FALSE, fs::u8path(name).c_str())) {
        REQUIRE(handle);
    }
    ~event() { CloseHandle(handle); }
};
nlohmann::json read_report(const fs::path& path) {
    std::ifstream input(path);
    return nlohmann::json::parse(input);
}
std::string nt_path(const fs::path& path) {
    auto handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    REQUIRE(handle != INVALID_HANDLE_VALUE);
    std::vector<wchar_t> name(32768);
    auto length = GetFinalPathNameByHandleW(handle, name.data(), DWORD(name.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_NT);
    CloseHandle(handle);
    REQUIRE(length > 0);
    REQUIRE(length < name.size());
    return fs::path(std::wstring(name.data(), length)).u8string();
}
}

TEST_CASE("Embedded registration accepts another process's hardlink name for the same DLL",
          "[SP-TENSORACLE-077][SP-TENSORACLE-078]") {
    directory files;
    auto first = files.path / "first.dll", second = files.path / "second.dll";
    fs::copy_file(TENSORACLE_REGISTRATION_PLUGIN, first);
    fs::create_hard_link(first, second);
    REQUIRE(fs::equivalent(first, second));
    std::ifstream input(first, std::ios::binary);
    std::string bytes{std::istreambuf_iterator<char>(input), {}};
    auto expected = tensoracle::detail::sha256(bytes.data(), bytes.size());
    auto prefix = "Local\\tensoracle-hardlink-" + files.path.filename().u8string();
    event ready(prefix + "-ready"), release(prefix + "-release");
    auto first_report = files.path / "first.json", second_report = files.path / "second.json";
    test_platform::process holding({TENSORACLE_HARDLINK_PROBE, first.u8string(), first_report.u8string(),
        prefix + "-ready", prefix + "-release"});
    REQUIRE(WaitForSingleObject(ready.handle, 15000) == WAIT_OBJECT_0);
    auto registered_first = read_report(first_report);
    INFO(registered_first.dump(2));
    REQUIRE(registered_first.at("status") == TENSORACLE_SUCCESS);
    REQUIRE(registered_first.at("registered") == true);
    REQUIRE(registered_first.at("digests") == std::vector<std::string>{expected});
    REQUIRE(registered_first.at("mapped_path") == nt_path(first));
    REQUIRE(test_platform::run({TENSORACLE_HARDLINK_PROBE, second.u8string(), second_report.u8string()}) == 0);
    auto registered_second = read_report(second_report);
    INFO(registered_second.dump(2));
    // Require the shared image section's first name, so a passing test proves
    // registration handled distinct names rather than avoiding the regression.
    REQUIRE(registered_second.at("mapped_path") == registered_first.at("mapped_path"));
    REQUIRE(registered_second.at("mapped_path") != nt_path(second));
    REQUIRE(registered_second.at("status") == TENSORACLE_SUCCESS);
    REQUIRE(registered_second.at("registered") == true);
    REQUIRE(registered_second.at("digests") == std::vector<std::string>{expected});
    REQUIRE(SetEvent(release.handle));
    REQUIRE(holding.wait() == 0);
}
