"""List and prune saved TensorRT engines without importing TensorRT."""

import argparse
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ._cache_format import CorruptEngineError, _cache_directory, _read_header, _read_weight_record

_DAY_NS = 86_400_000_000_000


@dataclass
class _Entry:
    path: Path
    observed: os.stat_result | None
    details: dict


def _identity(observed: os.stat_result) -> tuple:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _inspect(path: Path) -> _Entry:
    observed = None
    details = {"file": path.name, "status": "unrecognized", "size_bytes": None}
    try:
        observed = path.lstat()
        details["size_bytes"] = observed.st_size
        details["recorded_at_ns"] = observed.st_mtime_ns
        try:
            details["recorded_at"] = datetime.fromtimestamp(
                observed.st_mtime_ns // 1_000_000_000, UTC
            ).isoformat()
        except (OverflowError, OSError, ValueError):
            details["recorded_at"] = None
        if not stat.S_ISREG(observed.st_mode):
            details["reason"] = "not a regular file"
        elif observed.st_size == 0:
            details["status"] = "building or leftover"
        else:
            with path.open("rb", buffering=0) as stream:
                header = _read_header(stream, path, observed.st_size)
                if _identity(os.fstat(stream.fileno())) != _identity(observed):
                    raise OSError("file changed during inspection")
            details["status"] = "recognized"
            details["conditions"] = header["conditions"]
    except (OSError, CorruptEngineError) as error:
        details["reason"] = str(error)
    return _Entry(path, observed, details)


def _scan(directory: Path) -> list[_Entry]:
    try:
        paths = sorted(directory.iterdir())
    except FileNotFoundError:
        return []
    return [_inspect(path) for path in paths if path.suffix == ".engine"]


def _total(entries: list[_Entry]) -> int:
    return sum(entry.details["size_bytes"] or 0 for entry in entries)


def _list(directory: Path, entries: list[_Entry]) -> int:
    paths_by_digest: dict[str, set[str]] = {}
    for path in sorted(directory.glob("weight-*.json")):
        record = _read_weight_record(path)
        if record is not None:
            paths_by_digest.setdefault(record["sha256"], set()).add(record["path"])
    for entry in entries:
        if entry.details["status"] == "recognized":
            weights = entry.details["conditions"].get("weights", [])
            entry.details["weight_paths"] = (
                [
                    {"sha256": digest, "paths": sorted(paths_by_digest.get(digest, []))}
                    for digest in weights
                    if isinstance(digest, str)
                ]
                if isinstance(weights, list)
                else []
            )
    print(
        json.dumps(
            {
                "directory": str(directory),
                "entries": [entry.details for entry in entries],
                "total_size_bytes": _total(entries),
            }
        )
    )
    return 0


def _prune(directory: Path, entries: list[_Entry], args: argparse.Namespace) -> int:
    excluded = [entry.details for entry in entries if entry.details["status"] != "recognized"]
    candidates = sorted(
        (entry for entry in entries if entry.details["status"] == "recognized"),
        key=lambda entry: (entry.observed.st_mtime_ns, entry.path.name),
    )
    targets = []
    projected_size = _total(entries)
    cutoff = (
        time.time_ns() - int(args.unused_for * _DAY_NS) if args.unused_for is not None else None
    )
    for entry in candidates:
        if cutoff is not None:
            selected = entry.observed.st_mtime_ns < cutoff
        else:
            selected = projected_size > args.max_size
        if selected:
            targets.append(entry)
            projected_size -= entry.observed.st_size
    print(
        json.dumps(
            {
                "event": "plan",
                "directory": str(directory),
                "dry_run": args.dry_run,
                "rebuild_notice": "Removed engines will be rebuilt the next time they are used.",
                "targets": [entry.details for entry in targets],
                "excluded": excluded,
                "total_size_bytes": _total(entries),
            }
        ),
        flush=True,
    )

    removed, failed, missing = [], [], []
    if not args.dry_run:
        for entry in targets:
            try:
                if _identity(entry.path.lstat()) != _identity(entry.observed):
                    raise OSError("file changed since inspection; not removed")
                entry.path.unlink()
            except FileNotFoundError:
                missing.append(
                    {"file": entry.path.name, "reason": "file disappeared before removal"}
                )
            except OSError as error:
                failed.append({"file": entry.path.name, "reason": str(error)})
            else:
                removed.append(entry.details)
    remaining = _scan(directory)
    remaining_size = _total(remaining)
    size_to_check = projected_size if args.dry_run else remaining_size
    reached = args.max_size is None or size_to_check <= args.max_size
    complete_sizes = all(entry.observed is not None for entry in entries + remaining)
    print(
        json.dumps(
            {
                "event": "result",
                "dry_run": args.dry_run,
                "removed": removed,
                "excluded": excluded,
                "failed": failed,
                "missing": missing,
                "removed_size_bytes": sum(entry["size_bytes"] for entry in removed),
                "remaining_size_bytes": remaining_size,
                "projected_size_bytes": projected_size if args.dry_run else remaining_size,
                "max_size_bytes": args.max_size,
                "max_size_reached": reached,
            }
        )
    )
    return int(bool(failed) or not reached or not complete_sizes)


def _days(value: str) -> Decimal:
    try:
        days = Decimal(value)
        if days.is_finite() and days >= 0:
            return days
    except InvalidOperation:
        pass
    raise argparse.ArgumentTypeError("days must be a finite, nonnegative number")


def _size(value: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB|TiB)?", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError(
            "use nonnegative bytes or B, KiB, MiB, GiB, TiB (e.g. 1.5GiB)"
        )
    units = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}
    return int(Decimal(match[1]) * units[(match[2] or "B").lower()])


def main(argv: list[str] | None = None) -> int:
    """Run the installed ``tensoracle cache`` commands; return 0, 1, or usage error 2."""
    parser = argparse.ArgumentParser(prog="tensoracle")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="inspect or prune saved TensorRT engines")
    actions = cache.add_subparsers(dest="action", required=True)
    actions.add_parser(
        "list", help="show conditions, sizes, recorded times, and known weight paths as JSON"
    )
    prune = actions.add_parser(
        "prune",
        help="remove old engines during maintenance",
        description=(
            "Print selected engines and a rebuild notice before removal, then report results as JSON. "
            "Use during maintenance without concurrent engine builds or opens; these are not locked. "
            "Empty and unrecognized files are kept and count toward the total .engine bytes."
        ),
    )
    criteria = prune.add_mutually_exclusive_group(required=True)
    criteria.add_argument(
        "--unused-for",
        type=_days,
        metavar="DAYS",
        help="remove engines recorded strictly before DAYS days ago (24 hours/day; fractions allowed)",
    )
    criteria.add_argument(
        "--max-size",
        type=_size,
        metavar="SIZE",
        help="remove oldest engines until total .engine bytes <= SIZE; bytes or B/KiB/MiB/GiB/TiB; fractions rounded down",
    )
    prune.add_argument(
        "--dry-run", action="store_true", help="show the same selection without removing files"
    )
    args = parser.parse_args(argv)
    directory = _cache_directory()
    try:
        entries = _scan(directory)
        return (
            _list(directory, entries) if args.action == "list" else _prune(directory, entries, args)
        )
    except OSError as error:
        print(f"tensoracle: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
