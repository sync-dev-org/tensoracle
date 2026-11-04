"""List and prune saved TensorRT engines without importing TensorRT."""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from decimal import Decimal, DecimalException, InvalidOperation

_DAY_NS = 86_400_000_000_000


def _details(entry: dict) -> dict:
    """Format metadata copied from a native result; never inspect a cache file here."""
    details = {key: entry[key] for key in ("file", "status", "size_bytes")}
    seconds = entry["recorded_at_seconds"]
    if seconds is not None:
        details["recorded_at_ns"] = seconds * 1_000_000_000 + entry["recorded_at_nanoseconds"]
        try:
            details["recorded_at"] = datetime.fromtimestamp(seconds, UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            details["recorded_at"] = None
    if "header" in entry:
        details["conditions"] = entry["header"]["conditions"]
        details["workspace_limit"] = entry["header"].get("workspace_limit", "unknown")
    for key in ("weight_paths", "reason"):
        if key in entry:
            details[key] = entry[key]
    return details


def _entries(result, collection="entries") -> list[dict]:
    return [_details(entry) for entry in result.entries(collection)]


def _list(result) -> int:
    summary = result.summary()
    print(
        json.dumps(
            {
                "directory": summary["directory"],
                "entries": _entries(result),
                "total_size_bytes": summary["total_size_bytes"],
            }
        )
    )
    return 0


def _prune(native, plan, *, dry_run: bool) -> int:
    summary = plan.summary()
    print(
        json.dumps(
            {
                "event": "plan",
                "directory": summary["directory"],
                "dry_run": dry_run,
                "rebuild_notice": "Removed engines will be rebuilt the next time they are used.",
                "targets": _entries(plan, "targets"),
                "excluded": _entries(plan, "excluded"),
                "total_size_bytes": summary["total_size_bytes"],
            }
        ),
        flush=True,
    )
    result = native.cache_apply_prune(plan, dry_run=dry_run)
    summary = result.summary()
    print(
        json.dumps(
            {
                "event": "result",
                "dry_run": summary["dry_run"],
                "removed": _entries(result, "removed"),
                "excluded": _entries(result, "excluded"),
                "failed": [
                    {"file": entry["file"], "reason": entry["reason"]}
                    for entry in result.entries("failed")
                ],
                "missing": [
                    {"file": entry["file"], "reason": entry["reason"]}
                    for entry in result.entries("missing")
                ],
                **{
                    key: summary[key]
                    for key in (
                        "removed_size_bytes",
                        "remaining_size_bytes",
                        "projected_size_bytes",
                        "max_size_bytes",
                        "max_size_reached",
                    )
                },
            }
        )
    )
    return int(not summary["succeeded"])


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
    try:
        from . import _native

        if args.action == "list":
            return _list(_native.cache_list())
        plan = _native.cache_plan_prune(
            unused_for_ns=None if args.unused_for is None else int(args.unused_for * _DAY_NS),
            max_size_bytes=args.max_size,
        )
        return _prune(_native, plan, dry_run=args.dry_run)
    except (ValueError, DecimalException, OverflowError) as error:
        parser.error(str(error))
    except (OSError, RuntimeError) as error:
        print(f"tensoracle: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
