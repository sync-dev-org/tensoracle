#include "native_exports.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <iterator>

TEST_CASE("Native library exports and constant bytes are inspected without loading a library",
          "[SP-TENSORACLE-004]") {
    std::ifstream stream(TENSORACLE_EXPORT_FIXTURE, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(stream)), {});
    auto names=tensoracle::detail::native_exports(bytes);
    REQUIRE(names.count("getCreators") == 1);
    REQUIRE(names.count("tensoracle_test_constant") == 1);
    REQUIRE(tensoracle::detail::exported_data(bytes,"tensoracle_test_constant",4)==std::string("\1\2\0\4",4));
    REQUIRE_THROWS_AS(tensoracle::detail::exported_data(bytes,"absent",4),tensoracle::error);
    REQUIRE_THROWS_AS(tensoracle::detail::exported_data(bytes,"tensoracle_test_constant",3),tensoracle::error);
    REQUIRE_THROWS_AS(tensoracle::detail::native_exports(bytes.substr(0,32)),tensoracle::error);
}

TEST_CASE("PE named exports and constants are inspected without executing a DLL", "[SP-TENSORACLE-004]") {
    std::string bytes(1024,'\0');
    auto put=[&](size_t offset,uint64_t value,size_t size){for(size_t i=0;i<size;++i)bytes[offset+i]=static_cast<char>(value>>(i*8));};
    bytes.replace(0,2,"MZ"); put(0x3c,0x80,4); bytes.replace(0x80,4,"PE\0\0",4);
    put(0x86,1,2); put(0x94,0xf0,2); put(0x98,0x20b,2); put(0x98+112,0x1000,4);
    put(0x188+8,0x200,4); put(0x188+12,0x1000,4); put(0x188+16,0x200,4); put(0x188+20,0x200,4);
    put(0x200+20,1,4); put(0x200+24,1,4); put(0x200+28,0x1030,4); put(0x200+32,0x1040,4); put(0x200+36,0x1048,4);
    put(0x230,0x1080,4); put(0x240,0x1050,4); put(0x248,0,2);
    bytes.replace(0x250,25,"tensoracle_bundle_sha256\0",25); bytes.replace(0x280,5,"abcd\0",5);
    REQUIRE(tensoracle::detail::native_exports(bytes)==std::set<std::string>{"tensoracle_bundle_sha256"});
    REQUIRE(tensoracle::detail::exported_data(bytes,"tensoracle_bundle_sha256",5)==std::string("abcd\0",5));
    REQUIRE_THROWS_AS(tensoracle::detail::exported_data(bytes,"tensoracle_bundle_sha256"),tensoracle::error);
    put(0x98+112,0,4);
    REQUIRE(tensoracle::detail::native_exports(bytes).empty());
}
