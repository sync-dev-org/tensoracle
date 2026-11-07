#pragma once
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Other architectures and MSVC compile only the portable implementation.
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <cpuid.h>
#include <immintrin.h>
#define TENSORACLE_DETAIL_SHA256_X86
#endif

namespace tensoracle::detail {
inline std::uint32_t rotate_right(std::uint32_t value, unsigned int count) noexcept
{
    return (value >> count) | (value << (32U - count));
}
inline constexpr std::array<std::uint32_t, 64> sha256_constants{{
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
}};

inline void sha256_compress_portable(std::uint32_t* state, const std::uint8_t* message,
                                    std::size_t blocks) noexcept {
    while (blocks--) {
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
        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];
        for (std::size_t index = 0; index < 64; ++index) {
            auto const s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            auto const choice = (e & f) ^ (~e & g);
            auto const t1 = h + s1 + choice + sha256_constants[index] + words[index];
            auto const s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            auto const majority = (a & b) ^ (a & c) ^ (b & c);
            auto const t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
        message += 64;
    }
}

#ifdef TENSORACLE_DETAIL_SHA256_X86
// Keep SHA/SSSE3 instructions inside this function, including when the header is
// compiled by a consumer without CPU-specific build flags.
__attribute__((target("sha,ssse3")))
inline void sha256_compress_x86(std::uint32_t* state, const std::uint8_t* message,
                               std::size_t blocks) noexcept {
    // SHA256RNDS2 packs ABEF and CDGH from the high lane to the low lane.
    auto abef = _mm_set_epi32(state[0], state[1], state[4], state[5]);
    auto cdgh = _mm_set_epi32(state[2], state[3], state[6], state[7]);
    const auto endian = _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11,
                                    4, 5, 6, 7, 0, 1, 2, 3);
    while (blocks--) {
        const auto saved_abef = abef, saved_cdgh = cdgh;
        __m128i words[4];
        for (std::size_t i = 0; i < 4; ++i)
            words[i] = _mm_shuffle_epi8(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(message + 16 * i)), endian);
        // A four-vector ring holds W[t-16..t-1]. Each expansion generates four
        // new words; MSG2 includes the dependencies within those four words.
#if defined(__clang__)
#pragma clang loop unroll(full)
#else
#pragma GCC unroll 16
#endif
        for (std::size_t group = 0; group < 16; ++group) {
            const auto i = group % 4;
            if (group >= 4) {
                auto next = _mm_sha256msg1_epu32(words[i], words[(i + 1) % 4]);
                next = _mm_add_epi32(next,
                    _mm_alignr_epi8(words[(i + 3) % 4], words[(i + 2) % 4], 4));
                words[i] = _mm_sha256msg2_epu32(next, words[(i + 3) % 4]);
            }
            auto rounds = _mm_add_epi32(words[i], _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(sha256_constants.data() + 4 * group)));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, rounds);
            rounds = _mm_shuffle_epi32(rounds, 0x0e);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, rounds);
        }
        abef = _mm_add_epi32(abef, saved_abef);
        cdgh = _mm_add_epi32(cdgh, saved_cdgh);
        message += 64;
    }
    std::uint32_t first[4], second[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(first), abef);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(second), cdgh);
    state[0] = first[3]; state[1] = first[2]; state[2] = second[3]; state[3] = second[2];
    state[4] = first[1]; state[5] = first[0]; state[6] = second[1]; state[7] = second[0];
}
#endif

enum class sha256_backend { automatic, portable };
using sha256_compressor = void (*)(std::uint32_t*, const std::uint8_t*, std::size_t) noexcept;

inline sha256_compressor sha256_default_compressor() noexcept {
    // Resolve once per process rather than querying CPUID for each file or block.
    static const auto selected = []() noexcept -> sha256_compressor {
#ifdef TENSORACLE_DETAIL_SHA256_X86
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & bit_SSSE3) &&
            __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1U << 29)))
            return sha256_compress_x86;
#endif
        return sha256_compress_portable;
    }();
    return selected;
}

// Constant-memory SHA-256, shared with native plugin helpers and cache files.
class sha256_state {
public:
    explicit sha256_state(sha256_backend backend = sha256_backend::automatic) noexcept
        : compress_(backend == sha256_backend::portable ? sha256_compress_portable
                                                       : sha256_default_compressor()) {}
    void update(const void* bytes, std::size_t size) noexcept {
        if (!size) return;
        auto* input = static_cast<const std::uint8_t*>(bytes);
        length_ += size;
        if (used_) {
            auto take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, input, take);
            used_ += take; input += take; size -= take;
            if (used_ == block_.size()) { compress_(state_.data(), block_.data(), 1); used_ = 0; }
        }
        if (size >= block_.size()) {
            const auto blocks = size / block_.size();
            compress_(state_.data(), input, blocks);
            const auto consumed = blocks * block_.size();
            input += consumed; size -= consumed;
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
    sha256_compressor compress_;
    std::array<std::uint32_t, 8> state_{{
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U,
    }};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_ = 0;
    std::uint64_t length_ = 0;
};
inline std::string sha256(void const* bytes, std::size_t size) {
    sha256_state state;
    state.update(bytes, size);
    return state.finish();
}
}

#ifdef TENSORACLE_DETAIL_SHA256_X86
#undef TENSORACLE_DETAIL_SHA256_X86
#endif
