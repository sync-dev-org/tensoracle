#include "sha256.hpp"
#include <nlohmann/json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utility>

TEST_CASE("SHA-256 preserves standard digests with automatic and portable processing", "[SP-TENSORACLE-055]") {
    using tensoracle::detail::sha256;
    using tensoracle::detail::sha256_backend;
    // hashlib.sha256 over bytes(i % 251 for i in range(size)), independent of either backend.
    const std::pair<std::size_t, const char*> vectors[] = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        {129, "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135"},
        {4 * 1024 * 1024 + 17, "7fa1366968edf9a8490fffac3eab7e0b1619480a152d8e75c7471f753a8c1231"},
    };
    for (auto backend : {sha256_backend::automatic, sha256_backend::portable}) {
        CAPTURE(static_cast<int>(backend));
        for (const auto& [size, expected] : vectors) {
            CAPTURE(size);
            // Offset by one byte to exercise unaligned block loads.
            std::string input(size + 1, '\0');
            for (std::size_t i = 0; i < size; ++i) input[i + 1] = static_cast<char>(i % 251);
            tensoracle::detail::sha256_state state(backend);
            state.update(input.data() + 1, size);
            REQUIRE(state.finish() == expected);
            if (backend == sha256_backend::automatic)
                REQUIRE(sha256(input.data() + 1, size) == expected);
        }
        for (const auto& [input, expected] : {
                std::pair{"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
                std::pair{"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"}}) {
            tensoracle::detail::sha256_state state(backend);
            state.update(input, std::strlen(input));
            REQUIRE(state.finish() == expected);
        }
    }
}
TEST_CASE("Internal JSON supports ordered serialization and UTF-8 metadata", "[SP-TENSORACLE-021]") {
    auto json = nlohmann::json::parse("{\"z\":[1,2],\"name\":\"重み\"}");
    REQUIRE(json["name"] == "重み");
    REQUIRE(json.dump(-1, ' ', true) == "{\"name\":\"\\u91cd\\u307f\",\"z\":[1,2]}");
}

TEST_CASE("Weight SHA-256 accepts chunks across padding and block boundaries", "[SP-TENSORACLE-009]") {
    using tensoracle::detail::sha256_backend;
    std::string input(1000000, 'a');
    for (auto backend : {sha256_backend::automatic, sha256_backend::portable}) {
        CAPTURE(static_cast<int>(backend));
        for (size_t chunk : {size_t(1), size_t(55), size_t(56), size_t(63), size_t(64), size_t(65), size_t(4096)}) {
            CAPTURE(chunk);
            tensoracle::detail::sha256_state state(backend);
            state.update(nullptr, 0);
            for (size_t i = 0; i < input.size(); i += chunk)
                state.update(input.data() + i, std::min(chunk, input.size() - i));
            REQUIRE(state.finish() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
            REQUIRE(state.finish() == tensoracle::detail::sha256(input.data(), input.size()));
        }
    }
}
