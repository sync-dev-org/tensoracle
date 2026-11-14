#include "cache.hpp"
#include "test_channel.hpp"
#include "test_platform.hpp"
#if defined(_WIN32)
int wmain(int argc, wchar_t** values) {
#else
int main(int argc, char** values) {
#endif
    auto argv = test_platform::utf8_arguments(argc, values);
    if (argc != 3) return 2;
    auto channel = static_cast<test_channel::endpoint>(std::stoll(argv[2]));
    try {
        test_channel::send(channel, 'o');
        tensoracle::detail::engine_file_lock lock(std::filesystem::u8path(argv[1]));
        test_channel::send(channel, lock.empty() ? 'b' : 'r');
        auto command = test_channel::receive(channel);
        if (command == 'p') tensoracle::detail::write_engine_file(std::filesystem::u8path(argv[1]), {{"name", "lock-test"}}, {'p'}, {}, {});
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
