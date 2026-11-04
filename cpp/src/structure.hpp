#pragma once

#include <cstddef>
#include <cstring>

namespace tensoracle::detail {

inline size_t structure_size(const void* structure) noexcept {
    size_t size = 0;
    std::memcpy(&size, structure, sizeof(size));
    return size;
}

template <class T>
void write_field(void* structure, size_t size, size_t offset, T value) noexcept {
    if (size >= offset + sizeof(T))
        std::memcpy(static_cast<unsigned char*>(structure) + offset, &value, sizeof(T));
}

template <class T>
T read_field(const void* structure, size_t size, size_t offset) noexcept {
    T value{};
    if (size >= offset + sizeof(T))
        std::memcpy(&value, static_cast<const unsigned char*>(structure) + offset, sizeof(T));
    return value;
}

} // namespace tensoracle::detail
