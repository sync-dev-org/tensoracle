#pragma once
#include "path.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <utility>
#include <fcntl.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tensoracle::detail {
namespace fs = std::filesystem;
// Unix seconds and a nonnegative nanosecond fraction preserve the OS time range.
using file_time = std::pair<int64_t, int64_t>;
struct file_status {
    int64_t size;
    file_time mtime, ctime;
    uint64_t device, inode;
    bool regular;
};
#if defined(_WIN32)
[[noreturn]] inline void windows_file_error(const fs::path& path) {
    auto code = int(GetLastError());
    throw std::system_error(code, std::system_category(), path_utf8(path));
}
inline HANDLE file_handle(int fd) { return reinterpret_cast<HANDLE>(_get_osfhandle(fd)); }
inline int open_file(const fs::path& path, bool nofollow = false, bool create = false) {
    auto handle = CreateFileW(path.c_str(), create ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        create ? CREATE_NEW : OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | (nofollow ? FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS : 0), nullptr);
    if (handle == INVALID_HANDLE_VALUE) windows_file_error(path);
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | (create ? _O_RDWR : _O_RDONLY) | _O_NOINHERIT);
    if (fd < 0) { CloseHandle(handle); throw std::system_error(errno, std::generic_category(), path_utf8(path)); }
    return fd;
}
inline int close_file(int fd) { return _close(fd); }
inline int read_file_bytes(int fd, void* bytes, size_t size) { return _read(fd, bytes, unsigned(std::min(size, size_t(1024 * 1024)))); }
inline int write_file_bytes(int fd, const void* bytes, size_t size) { return _write(fd, bytes, unsigned(std::min(size, size_t(1024 * 1024)))); }
inline file_time filetime_parts(int64_t ticks) {
    auto seconds = ticks / 10000000 - 11644473600LL;
    auto fraction = (ticks % 10000000) * 100;
    if (fraction < 0) { --seconds; fraction += 1000000000; }
    return {seconds, fraction};
}
inline file_status status_of(int fd) {
    BY_HANDLE_FILE_INFORMATION info{};
    FILE_BASIC_INFO basic{};
    auto handle = file_handle(fd);
    if (!GetFileInformationByHandle(handle, &info) || !GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)))
        windows_file_error("file metadata");
    return {int64_t((uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow),
        filetime_parts(basic.LastWriteTime.QuadPart), filetime_parts(basic.ChangeTime.QuadPart),
        info.dwVolumeSerialNumber, (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow,
        !(info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))};
}
#else
inline int open_file(const fs::path& path, bool nofollow = false, bool create = false) {
    auto fd = ::open(path.c_str(), (create ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY) | O_CLOEXEC
        | (nofollow ? O_NOFOLLOW | O_NONBLOCK : 0), 0600);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), path_utf8(path));
    return fd;
}
inline int close_file(int fd) { return ::close(fd); }
inline ssize_t read_file_bytes(int fd, void* bytes, size_t size) { return ::read(fd, bytes, std::min(size, size_t(1024 * 1024))); }
inline ssize_t write_file_bytes(int fd, const void* bytes, size_t size) { return ::write(fd, bytes, std::min(size, size_t(1024 * 1024))); }
inline file_status status_from(const struct stat& info) {
    return {info.st_size, {info.st_mtim.tv_sec, info.st_mtim.tv_nsec},
        {info.st_ctim.tv_sec, info.st_ctim.tv_nsec},
        uint64_t(info.st_dev), uint64_t(info.st_ino), bool(S_ISREG(info.st_mode))};
}
inline file_status status_of(int fd) {
    struct stat info{};
    if (fstat(fd, &info)) throw std::system_error(errno, std::generic_category(), "file metadata");
    return status_from(info);
}
#endif
// Unix timestamps carry a signed second and a nonnegative fractional second.
inline file_time unix_time_parts(int64_t nanoseconds) {
    auto seconds = nanoseconds / 1000000000;
    auto fraction = nanoseconds % 1000000000;
    if (fraction < 0) { --seconds; fraction += 1000000000; }
    return {seconds, fraction};
}
struct file_descriptor {
    int value = -1;
    ~file_descriptor() { if (value >= 0) close_file(value); }
    file_descriptor(const file_descriptor&) = delete;
    file_descriptor& operator=(const file_descriptor&) = delete;
    explicit file_descriptor(int fd) : value(fd) {}
};
inline file_status status_of(const fs::path& path, bool nofollow = false) {
#if defined(_WIN32)
    file_descriptor fd{open_file(path, nofollow)};
    return status_of(fd.value);
#else
    struct stat info{};
    if ((nofollow ? ::lstat(path.c_str(), &info) : ::stat(path.c_str(), &info)))
        throw std::system_error(errno, std::generic_category(), path_utf8(path));
    return status_from(info);
#endif
}
inline bool same_file(const file_status& a, const file_status& b) { return a.device == b.device && a.inode == b.inode; }
inline bool same_snapshot(const file_status& a, const file_status& b) {
    return same_file(a, b) && a.size == b.size && a.mtime == b.mtime && a.ctime == b.ctime && a.regular == b.regular;
}
}
