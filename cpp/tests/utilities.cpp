#include "sha256.hpp"
#include <nlohmann/json.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Shared SHA-256 matches published digests including multiple blocks", "[SP-TENSORACLE-004]") {
    using tensoracle::detail::sha256;
    REQUIRE(sha256("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(sha256("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string message = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    REQUIRE(sha256(message.data(),message.size()) ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}
TEST_CASE("Internal JSON supports ordered serialization and UTF-8 metadata", "[SP-TENSORACLE-021]") {
    auto json = nlohmann::json::parse("{\"z\":[1,2],\"name\":\"重み\"}");
    REQUIRE(json["name"] == "重み");
    REQUIRE(json.dump(-1, ' ', true) == "{\"name\":\"\\u91cd\\u307f\",\"z\":[1,2]}");
}

TEST_CASE("Weight SHA-256 accepts chunks across padding and block boundaries", "[SP-TENSORACLE-009]") {
    std::string input(1000000, 'a');
    for (size_t chunk : {size_t(1), size_t(55), size_t(56), size_t(63), size_t(64), size_t(65), size_t(4096)}) {
        tensoracle::detail::sha256_state state;
        state.update(nullptr, 0);
        for (size_t i = 0; i < input.size(); i += chunk)
            state.update(input.data() + i, std::min(chunk, input.size() - i));
        REQUIRE(state.finish() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        REQUIRE(state.finish() == tensoracle::detail::sha256(input.data(), input.size()));
    }
}
