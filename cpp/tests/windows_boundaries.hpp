#pragma once
// Test-only substitution of Win32 external boundaries. The library target never
// includes this file; files, hashing, cache state and synchronization remain real.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <functional>
namespace windows_boundary {
inline std::function<FARPROC(HMODULE, LPCSTR)> resolve;
inline std::function<void(HANDLE)> before_stat, before_flush;
inline std::function<bool(HANDLE)> deny_timestamp;
inline FARPROC WINAPI get_proc(HMODULE module, LPCSTR name) {
    if (resolve) return resolve(module, name);
    return ::GetProcAddress(module, name);
}
inline FARPROC real_get_proc(HMODULE module, LPCSTR name) { return ::GetProcAddress(module, name); }
inline BOOL WINAPI stat(HANDLE file, LPBY_HANDLE_FILE_INFORMATION value) {
    if (before_stat) before_stat(file);
    return ::GetFileInformationByHandle(file, value);
}
inline BOOL WINAPI flush(HANDLE file) {
    if (before_flush) before_flush(file);
    return ::FlushFileBuffers(file);
}
inline BOOL WINAPI timestamp(HANDLE file, const FILETIME* creation, const FILETIME* access, const FILETIME* write) {
    if (deny_timestamp && deny_timestamp(file)) { SetLastError(ERROR_ACCESS_DENIED); return FALSE; }
    return ::SetFileTime(file, creation, access, write);
}
}
#define GetProcAddress windows_boundary::get_proc
#define GetFileInformationByHandle windows_boundary::stat
#define FlushFileBuffers windows_boundary::flush
#define SetFileTime windows_boundary::timestamp
#endif
