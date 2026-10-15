"""Validate native plugin exports and retain TensorRT library handles."""

import hashlib
import struct
import sys
import threading
from collections.abc import Sequence
from pathlib import Path


class PluginLibraryError(ValueError):
    """A native plugin library cannot be registered for this process."""


_lock = threading.RLock()
_loaded_paths: dict[Path, str] = {}
_loaded_digests: dict[str, object] = {}


def _read_struct(data: bytes, offset: int, format_: str) -> tuple[int, ...]:
    size = struct.calcsize(format_)
    if offset < 0 or offset + size > len(data):
        raise PluginLibraryError("truncated native library")
    return struct.unpack_from(format_, data, offset)


def _elf_exports(data: bytes) -> set[str]:
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise PluginLibraryError("invalid ELF plugin library")
    bits, encoding = data[4:6]
    if bits not in (1, 2) or encoding not in (1, 2):
        raise PluginLibraryError("unsupported ELF plugin library")
    endian = "<" if encoding == 1 else ">"
    if bits == 2:
        (section_offset,) = _read_struct(data, 40, endian + "Q")
        section_size, section_count = _read_struct(data, 58, endian + "HH")
        section_format = endian + "IIQQQQIIQQ"
        symbol_format = endian + "IBBHQQ"
    else:
        (section_offset,) = _read_struct(data, 32, endian + "I")
        section_size, section_count = _read_struct(data, 46, endian + "HH")
        section_format = endian + "IIIIIIIIII"
        symbol_format = endian + "IIIBBH"
    if section_count == 0 or section_size < struct.calcsize(section_format):
        raise PluginLibraryError("missing ELF section table")
    sections = [
        _read_struct(data, section_offset + index * section_size, section_format)
        for index in range(section_count)
    ]
    exports = set()
    for section in sections:
        if section[1] != 11:  # SHT_DYNSYM
            continue
        string_index = section[6]
        if string_index >= len(sections):
            raise PluginLibraryError("invalid ELF symbol string table")
        strings = sections[string_index]
        string_data = data[strings[4] : strings[4] + strings[5]]
        if len(string_data) != strings[5]:
            raise PluginLibraryError("truncated ELF symbol string table")
        symbols_offset, symbols_size, symbol_stride = section[4], section[5], section[9]
        if symbol_stride < struct.calcsize(symbol_format) or symbols_size % symbol_stride:
            raise PluginLibraryError("invalid ELF symbol table")
        for offset in range(symbols_offset, symbols_offset + symbols_size, symbol_stride):
            symbol = _read_struct(data, offset, symbol_format)
            name_offset = symbol[0]
            if bits == 2:
                info, visibility, section_index = symbol[1:4]
            else:
                info, visibility, section_index = symbol[3:6]
            if section_index == 0 or info >> 4 not in (1, 2) or visibility & 3 not in (0, 3):
                continue
            if name_offset >= len(string_data):
                raise PluginLibraryError("invalid ELF symbol name")
            end = string_data.find(b"\0", name_offset)
            if end < 0:
                raise PluginLibraryError("unterminated ELF symbol name")
            exports.add(string_data[name_offset:end].decode("ascii", errors="replace"))
    return exports


def _pe_export_offsets(data: bytes) -> dict[str, int]:
    """Locate PE export values without executing any library initializers."""
    if data[:2] != b"MZ":
        raise PluginLibraryError("invalid PE plugin library")
    (header,) = _read_struct(data, 0x3C, "<I")
    if data[header : header + 4] != b"PE\0\0":
        raise PluginLibraryError("invalid PE plugin library")
    (section_count,) = _read_struct(data, header + 6, "<H")
    (optional_size,) = _read_struct(data, header + 20, "<H")
    optional = header + 24
    (magic,) = _read_struct(data, optional, "<H")
    if magic not in (0x10B, 0x20B):
        raise PluginLibraryError("unsupported PE plugin library")
    (export_rva,) = _read_struct(data, optional + (112 if magic == 0x20B else 96), "<I")
    sections = [
        _read_struct(data, optional + optional_size + index * 40 + 8, "<IIII")
        for index in range(section_count)
    ]

    def file_offset(rva: int) -> int:
        for virtual_size, virtual_address, raw_size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return raw_offset + rva - virtual_address
        raise PluginLibraryError("invalid PE export address")

    if export_rva == 0:
        return {}
    export = file_offset(export_rva)
    (count,) = _read_struct(data, export + 24, "<I")
    functions_rva, names_rva, ordinals_rva = _read_struct(data, export + 28, "<III")
    if count > 100_000:
        raise PluginLibraryError("invalid PE export count")
    names = {}
    for index in range(count):
        (name_rva,) = _read_struct(data, file_offset(names_rva) + index * 4, "<I")
        offset = file_offset(name_rva)
        end = data.find(b"\0", offset)
        if end < 0:
            raise PluginLibraryError("unterminated PE export name")
        (ordinal,) = _read_struct(data, file_offset(ordinals_rva) + index * 2, "<H")
        (value_rva,) = _read_struct(data, file_offset(functions_rva) + ordinal * 4, "<I")
        names[data[offset:end].decode("ascii", errors="replace")] = file_offset(value_rva)
    return names


def _pe_exports(data: bytes) -> set[str]:
    return set(_pe_export_offsets(data))


def _exports(data: bytes) -> set[str]:
    if sys.platform.startswith("linux"):
        return _elf_exports(data)
    if sys.platform == "win32":
        return _pe_exports(data)
    raise PluginLibraryError(f"native plugin inspection is unsupported on {sys.platform}")


def exported_data(data: bytes, name: str, expected_size: int | None = None) -> bytes:
    """Read an exported constant from a native library without loading the library."""
    if sys.platform == "win32":
        if expected_size is None or expected_size <= 0:
            raise PluginLibraryError("PE data inspection requires a positive size")
        offset = _pe_export_offsets(data).get(name)
        if offset is None or offset + expected_size > len(data):
            raise PluginLibraryError(f"native library does not export {name}")
        return data[offset : offset + expected_size]
    if sys.platform != "linux":
        raise PluginLibraryError(f"native data inspection is unsupported on {sys.platform}")
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise PluginLibraryError("expected a 64-bit little-endian ELF plugin library")
    (section_offset,) = _read_struct(data, 40, "<Q")
    section_size, section_count = _read_struct(data, 58, "<HH")
    if section_size < 64:
        raise PluginLibraryError("invalid ELF section table")
    sections = [
        _read_struct(data, section_offset + index * section_size, "<IIQQQQIIQQ")
        for index in range(section_count)
    ]
    for section in sections:
        if section[1] != 11:
            continue
        if section[6] >= len(sections):
            raise PluginLibraryError("invalid ELF symbol string table")
        strings = sections[section[6]]
        names = data[strings[4] : strings[4] + strings[5]]
        stride = section[9]
        if stride < 24:
            raise PluginLibraryError("invalid ELF symbol stride")
        for offset in range(section[4], section[4] + section[5], stride):
            name_offset, _, _, section_index, address, size = _read_struct(data, offset, "<IBBHQQ")
            if name_offset >= len(names) or section_index >= len(sections):
                continue
            end = names.find(b"\0", name_offset)
            if end < 0 or names[name_offset:end] != name.encode("ascii"):
                continue
            owner = sections[section_index]
            file_offset = owner[4] + address - owner[3]
            if size <= 0 or file_offset < 0 or file_offset + size > len(data):
                raise PluginLibraryError(f"invalid native data symbol {name}")
            if expected_size is not None and expected_size != size:
                raise PluginLibraryError(f"native data symbol {name} has the wrong size")
            return data[file_offset : file_offset + size]
    raise PluginLibraryError(f"native library does not export {name}")


def register_libraries(paths: Sequence[Path]) -> tuple[str, ...]:
    """Register each distinct library once and return the digests of its loaded bytes."""
    import tensorrt as trt

    digests = []
    with _lock:
        for candidate in paths:
            try:
                path = Path(candidate).resolve(strict=True)
                data = path.read_bytes()
            except OSError as error:
                raise PluginLibraryError(
                    f"plugin library is missing or unreadable: {candidate}"
                ) from error
            digest = hashlib.sha256(data).hexdigest()
            if path in _loaded_paths:
                if _loaded_paths[path] != digest:
                    raise PluginLibraryError(f"loaded plugin library changed: {path}")
                digests.append(digest)
                continue
            if "getCreators" not in _exports(data):
                raise PluginLibraryError(f"plugin library does not export getCreators: {path}")
            if digest not in _loaded_digests:
                try:
                    handle = trt.get_plugin_registry().load_library(str(path))
                except Exception as error:
                    raise PluginLibraryError(
                        f"TensorRT could not register plugin library: {path}"
                    ) from error
                if handle is None:
                    raise PluginLibraryError(f"TensorRT could not register plugin library: {path}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise PluginLibraryError(f"plugin library changed while loading: {path}")
                _loaded_digests[digest] = handle
            _loaded_paths[path] = digest
            digests.append(digest)
    return tuple(digests)
