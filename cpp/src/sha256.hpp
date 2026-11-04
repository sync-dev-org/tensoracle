#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
namespace tensoracle::detail {
inline std::uint32_t rotate_right(std::uint32_t value, unsigned int count) noexcept
{
    return (value >> count) | (value << (32U - count));
}
// Constant-memory SHA-256, shared with native plugin helpers and cache files.
class sha256_state {
public:
    void update(const void* bytes, std::size_t size) noexcept {
        if (!size) return;
        auto* input = static_cast<const std::uint8_t*>(bytes);
        length_ += size;
        if (used_) {
            auto take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, input, take);
            used_ += take; input += take; size -= take;
            if (used_ == block_.size()) { compress(block_.data()); used_ = 0; }
        }
        while (size >= block_.size()) {
            compress(input); input += block_.size(); size -= block_.size();
        }
        if (size) { std::memcpy(block_.data(), input, size); used_ = size; }
    }
    std::string finish() const {
        auto copy = *this;
        const auto bits = length_ * 8U;
        std::uint8_t padding[128]{};
        padding[0] = 0x80U;
        auto count = used_ < 56 ? 56 - used_ : 120 - used_;
        copy.update(padding, count);
        for (int i = 0; i < 8; ++i) padding[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        copy.update(padding, 8);
        constexpr char hex[] = "0123456789abcdef";
        std::string result(64U, '0');
        for (std::size_t word = 0; word < 8; ++word)
            for (std::size_t nibble = 0; nibble < 8; ++nibble)
                result[word * 8U + nibble] = hex[(copy.state_[word] >> (4U * (7U - nibble))) & 15U];
        return result;
    }
private:
    std::array<std::uint32_t, 8> state_{{
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U,
    }};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_ = 0;
    std::uint64_t length_ = 0;
    void compress(const std::uint8_t* message) noexcept {
        constexpr std::array<std::uint32_t, 64> constants{{
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
        }};
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            auto const position = index * 4U;
            words[index] = (static_cast<std::uint32_t>(message[position]) << 24U) |
                (static_cast<std::uint32_t>(message[position + 1U]) << 16U) |
                (static_cast<std::uint32_t>(message[position + 2U]) << 8U) |
                message[position + 3U];
        }
        for (std::size_t index = 16; index < 64; ++index) {
            auto const a = words[index - 15U];
            auto const b = words[index - 2U];
            words[index] = words[index - 16U] +
                (rotate_right(a, 7U) ^ rotate_right(a, 18U) ^ (a >> 3U)) +
                words[index - 7U] +
                (rotate_right(b, 17U) ^ rotate_right(b, 19U) ^ (b >> 10U));
        }
        auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (std::size_t index = 0; index < 64; ++index) {
            auto const s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            auto const choice = (e & f) ^ (~e & g);
            auto const t1 = h + s1 + choice + constants[index] + words[index];
            auto const s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            auto const majority = (a & b) ^ (a & c) ^ (b & c);
            auto const t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }
};
inline std::string sha256(void const* bytes, std::size_t size) {
    sha256_state state;
    state.update(bytes, size);
    return state.finish();
}
}
