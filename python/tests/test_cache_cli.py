"""Inspect and prune saved engines using headers and filesystem metadata alone."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from native_build import ROOT

DAY_NS = 86_400_000_000_000
OLD_NS = 1_000_000_000


@pytest.fixture(scope="module")
def cache_writer(tmp_path_factory):
    """Save test engines with the production C++ writer, without a CUDA device."""
    build = tmp_path_factory.mktemp("cache-writer")
    for command in (
        [
            "cmake",
            "-S",
            str(Path(__file__).parent / "native/cache"),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DTENSORACLE_CPP_ROOT={ROOT / 'cpp'}",
        ],
        ["cmake", "--build", str(build), "--parallel", "4"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    return Path((build / "artifact-Release.txt").read_text())


@pytest.fixture
def saved_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    directory = tmp_path / "cache" / "tensoracle"
    directory.mkdir(parents=True)
    return directory


def _save_engine(
    writer,
    directory,
    filename="one.engine",
    *,
    conditions=None,
    mtime=OLD_NS,
    size=None,
    metadata=None,
):
    if conditions is None:
        conditions = {
            "name": "encoder",
            "graph_version": "1",
            "weights": ["a" * 64],
            "plugins": [],
            "inputs": {"x": {"minimum": [1], "optimum": [2], "maximum": [3]}},
            "builder_settings": {"optimization_level": 3, "workspace_limit": None, "tf32": False},
            "gpu": {"name": "test GPU", "total_memory": 24 * 1024**3},
            "os": "Linux",
            "architecture": "x86_64",
            "tensorrt_version": "11.0",
        }
    path = directory / filename
    result = subprocess.run(
        [str(writer), str(path)],
        input=json.dumps({"conditions": conditions, **(metadata or {})}),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    if metadata == {}:
        # A valid header without a workspace record must display "unknown".
        contents = path.read_bytes()
        length = int.from_bytes(contents[10:18], "big")
        header = json.loads(contents[18 : 18 + length])
        del header["workspace_limit"]
        encoded = json.dumps(header).encode()
        path.write_bytes(
            contents[:10] + len(encoded).to_bytes(8, "big") + encoded + contents[18 + length :]
        )
    if size is not None:
        assert size >= path.stat().st_size
        with path.open("r+b") as stream:
            stream.truncate(size)
    os.utime(path, ns=(mtime, mtime))
    return path, conditions


@pytest.fixture
def save_engine(cache_writer):
    from functools import partial

    return partial(_save_engine, cache_writer)


def save_record(directory, *, path, digest="a" * 64, filename="weight-one.json", **changes):
    record = {
        "version": 1,
        "path": str(path),
        "size": 123,
        "mtime_ns": OLD_NS,
        "inode": 42,
        "sha256": digest,
        **changes,
    }
    record["checksum"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = directory / filename
    result.write_text(json.dumps(record), encoding="utf-8")
    return result


def call_cli(capsys, *args):
    from tensoracle._cli import main

    status = main(["cache", *args])
    captured = capsys.readouterr()
    return status, [json.loads(line) for line in captured.out.splitlines()], captured.err


@pytest.mark.req("SP-TENSORACLE-043")
@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.req("SP-TENSORACLE-042")
def test_list_distinguishes_all_conditions_and_preserves_recorded_time(
    save_engine, saved_cache, capsys
):
    """Entries with matching names retain every condition, size, and exact recorded time."""
    first, conditions = save_engine(saved_cache)
    second, _ = save_engine(
        saved_cache, "other-weight.engine", conditions={**conditions, "weights": ["b" * 64]}
    )
    third, _ = save_engine(
        saved_cache, "other-plugin.engine", conditions={**conditions, "plugins": ["c" * 64]}
    )
    status, (listing,), stderr = call_cli(capsys, "list")
    assert status == 0 and not stderr
    assert listing["directory"] == str(saved_cache)
    entries = {entry["file"]: entry for entry in listing["entries"]}
    for path in (first, second, third):
        entry = entries[path.name]
        assert entry["status"] == "recognized"
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["recorded_at_ns"] == OLD_NS
        assert entry["recorded_at"].startswith("1970-01-01T00:00:01")
        assert path.stat().st_mtime_ns == OLD_NS
    assert entries[first.name]["conditions"] == conditions
    assert entries[second.name]["conditions"]["weights"] == ["b" * 64]
    assert entries[third.name]["conditions"]["plugins"] == ["c" * 64]
    assert listing["total_size_bytes"] == sum(
        path.stat().st_size for path in (first, second, third)
    )


@pytest.mark.req("SP-TENSORACLE-043")
@pytest.mark.req("SP-TENSORACLE-052")
@pytest.mark.parametrize(
    "metadata, expected",
    [({"workspace_limit": None}, None), ({"workspace_limit": 4096}, 4096), ({}, "unknown")],
    ids=["tensorrt-default", "explicit-limit", "missing-record"],
)
def test_list_shows_recorded_workspace_limit_or_unknown(
    save_engine, saved_cache, capsys, metadata, expected
):
    """Listing distinguishes TensorRT's default, an explicit byte limit, and a missing record."""
    path, conditions = save_engine(saved_cache, metadata=metadata)
    contents = path.read_bytes()
    status, (listing,), stderr = call_cli(capsys, "list")
    assert status == 0 and not stderr
    entry = listing["entries"][0]
    assert entry["workspace_limit"] == expected
    assert entry["conditions"] == conditions
    assert path.read_bytes() == contents
    assert path.stat().st_mtime_ns == OLD_NS


@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.parametrize(
    "damage", ["missing", "json", "version", "type", "checksum", "digest", "mismatch", "valid"]
)
def test_list_uses_only_valid_matching_weight_records(
    save_engine, saved_cache, tmp_path, capsys, damage
):
    """A weight path is informational and appears only for an intact matching digest record."""
    save_engine(saved_cache)
    source = tmp_path / "removed-weights.safetensors"
    record = save_record(saved_cache, path=source)
    if damage == "missing":
        record.unlink()
    elif damage == "json":
        record.write_text("{", encoding="utf-8")
    elif damage in ("version", "type", "digest", "mismatch"):
        save_record(
            saved_cache,
            path=source,
            **{
                "version": {"version": 2},
                "type": {"size": True},
                "digest": {"digest": "invalid"},
                "mismatch": {"digest": "b" * 64},
            }[damage],
        )
    elif damage == "checksum":
        data = json.loads(record.read_text())
        data["path"] = str(tmp_path / "wrong")
        record.write_text(json.dumps(data), encoding="utf-8")
    status, (listing,), _ = call_cli(capsys, "list")
    assert status == 0
    assert listing["entries"][0]["weight_paths"] == [
        {"sha256": "a" * 64, "paths": [str(source)] if damage == "valid" else []}
    ]
    assert not source.exists()


@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.parametrize(
    "contents",
    [
        b"nonsense",
        b"TENSORACLE\x01",
        b"TENSORACLE\x01" + (2**63).to_bytes(8, "big"),
        b"TENSORACLE\x01" + (1).to_bytes(8, "big") + b"{p",
    ],
)
def test_list_identifies_placeholders_and_unrecognized_files(saved_cache, capsys, contents):
    """Empty files and unreadable headers have distinct statuses and remain available for inspection."""
    (saved_cache / "building.engine").write_bytes(b"")
    (saved_cache / "unknown.engine").write_bytes(contents)
    status, (listing,), _ = call_cli(capsys, "list")
    assert status == 0
    entries = {entry["file"]: entry for entry in listing["entries"]}
    assert entries["building.engine"]["status"] == "building or leftover"
    assert entries["unknown.engine"]["status"] == "unrecognized"
    assert entries["unknown.engine"]["reason"]


@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.req("SP-TENSORACLE-047")
def test_list_requires_no_gpu_tensor_runtime_weight_or_plan_body(
    save_engine, saved_cache, tmp_path
):
    """Native listing needs no tensor Python packages, missing weights, or memory for the plan."""
    engine, _ = save_engine(saved_cache, size=4 * 1024**3)
    save_record(saved_cache, path=tmp_path / "absent.safetensors")
    child = r"""
import importlib.abc
import os
import resource
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('tensorrt', 'numpy', 'safetensors', 'cuda', 'torch'):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from tensoracle import _native
# A sparse 4 GiB plan cannot fit in the remaining address space.
limit = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE') + 128 * 1024**2
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
from tensoracle._cli import main
raise SystemExit(main(['cache', 'list']))
"""
    result = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, check=False, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["entries"][0]["size_bytes"] == 4 * 1024**3
    assert engine.stat().st_mtime_ns == OLD_NS


@pytest.mark.req("SP-TENSORACLE-044")
@pytest.mark.req("SP-TENSORACLE-050")
def test_prune_unused_for_uses_native_time_and_reports_before_removal(
    save_engine, saved_cache, capsys, monkeypatch
):
    """Old engines are removed using the native clock after the plan and rebuild notice are flushed."""
    now = time.time_ns()
    before, _ = save_engine(saved_cache, "old.engine", mtime=now - 3 * DAY_NS)
    newer, _ = save_engine(saved_cache, "new.engine", mtime=now - DAY_NS)
    real_print = print

    def checked_print(*args, **kwargs):
        if '"event": "plan"' in str(args[0]):
            assert before.exists() and newer.exists()
            plan = json.loads(args[0])
            assert [entry["file"] for entry in plan["targets"]] == [before.name]
            assert any(word in plan["rebuild_notice"].lower() for word in ("rebuild", "rebuilt"))
            assert kwargs.get("flush") is True
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", checked_print)
    status, (plan, result), _ = call_cli(capsys, "prune", "--unused-for", "2")
    assert status == 0
    assert not before.exists() and newer.exists()
    assert result["removed_size_bytes"] == plan["targets"][0]["size_bytes"]
    assert [entry["file"] for entry in result["removed"]] == [before.name]
    assert result["remaining_size_bytes"] == newer.stat().st_size


@pytest.mark.req("SP-TENSORACLE-044")
@pytest.mark.req("SP-TENSORACLE-050")
def test_prune_max_size_removes_oldest_first_and_keeps_weight_records(
    save_engine, saved_cache, capsys
):
    """Size pruning removes engines by recorded time and never removes digest records or placeholders."""
    newest, _ = save_engine(saved_cache, "a-new.engine", mtime=3 * OLD_NS, size=1024)
    oldest, _ = save_engine(saved_cache, "z-old.engine", mtime=OLD_NS, size=1024)
    middle, _ = save_engine(saved_cache, "m-middle.engine", mtime=2 * OLD_NS, size=1024)
    (saved_cache / "building.engine").write_bytes(b"")
    (saved_cache / "unknown.engine").write_bytes(b"unknown")
    record = save_record(saved_cache, path=saved_cache / "missing-weights")
    status, (plan, result), _ = call_cli(capsys, "prune", "--max-size", "1.5KiB")
    assert status == 0
    assert [entry["file"] for entry in plan["targets"]] == [oldest.name, middle.name]
    assert [entry["file"] for entry in result["removed"]] == [oldest.name, middle.name]
    assert {entry["file"] for entry in result["excluded"]} == {"building.engine", "unknown.engine"}
    assert result["removed_size_bytes"] == 2048
    assert result["remaining_size_bytes"] == 1031
    assert newest.exists() and record.exists()
    assert newest.stat().st_mtime_ns == 3 * OLD_NS


@pytest.mark.req("SP-TENSORACLE-045")
@pytest.mark.req("SP-TENSORACLE-042")
@pytest.mark.parametrize("selection", [("--unused-for", "1"), ("--max-size", "0B")])
def test_dry_run_has_the_same_targets_and_preserves_bytes_and_mtime(
    save_engine, saved_cache, capsys, selection
):
    """Preview and removal select the same engines while preview preserves all file contents and mtimes."""
    save_engine(saved_cache)
    (saved_cache / "unknown.engine").write_bytes(b"unknown")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in saved_cache.iterdir()}
    status, (preview, result), _ = call_cli(capsys, "prune", *selection, "--dry-run")
    assert status in (0, 1)  # Unknown bytes can make a zero-byte target unattainable.
    assert result["removed"] == [] and result["removed_size_bytes"] == 0
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    _, (plan, _), _ = call_cli(capsys, "prune", *selection)
    assert preview["targets"] == plan["targets"]


@pytest.mark.req("SP-TENSORACLE-050")
@pytest.mark.parametrize(
    "selection",
    [
        (),
        ("--unused-for", "1", "--max-size", "1GiB"),
        ("--unused-for", "-1"),
        ("--unused-for", "nan"),
        ("--unused-for", "inf"),
        ("--max-size", "-1"),
        ("--max-size", "1XB"),
        ("--max-size", str(2**63)),
        ("--unused-for", "106752"),
        ("--unused-for", "1e1000000"),
    ],
)
def test_invalid_prune_arguments_are_usage_errors_without_changes(
    save_engine, saved_cache, selection
):
    """Missing, conflicting, and invalid criteria stop with a usage error before deleting anything."""
    from tensoracle._cli import main

    path, _ = save_engine(saved_cache)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(SystemExit) as caught:
        main(["cache", "prune", *selection])
    assert caught.value.code == 2
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.req("SP-TENSORACLE-050")
@pytest.mark.req("SP-TENSORACLE-051")
@pytest.mark.parametrize("change", ["denied", "disappeared", "changed"])
def test_prune_reports_concurrent_changes_and_counts_only_removed_bytes(
    save_engine, saved_cache, capsys, monkeypatch, change
):
    """Failed, changed, and disappeared files are reported separately from successful removals."""
    path, _ = save_engine(saved_cache)
    original_size = path.stat().st_size
    real_print = print

    def print_then_change(*args, **kwargs):
        real_print(*args, **kwargs)
        if '"event": "plan"' in str(args[0]):
            if change == "disappeared":
                path.unlink()
            elif change == "changed":
                path.write_bytes(b"new contents")

    monkeypatch.setattr("builtins.print", print_then_change)
    if change == "denied" and os.geteuid() == 0:
        pytest.skip("root bypasses directory write permissions")
    try:
        if change == "denied":
            saved_cache.chmod(0o500)
        status, (_, result), _ = call_cli(capsys, "prune", "--max-size", "0")
    finally:
        saved_cache.chmod(0o700)
    assert result["removed"] == [] and result["removed_size_bytes"] == 0
    if change == "disappeared":
        assert status == 0 and result["remaining_size_bytes"] == 0
        assert [entry["file"] for entry in result["missing"]] == [path.name]
    else:
        assert status == 1
        assert [entry["file"] for entry in result["failed"]] == [path.name]
        assert result["failed"][0]["reason"]
        assert result["remaining_size_bytes"] == (original_size if change == "denied" else 12)


@pytest.mark.req("SP-TENSORACLE-050")
def test_unrecognized_bytes_prevent_success_when_max_size_cannot_be_reached(saved_cache, capsys):
    """Pruning keeps unknown files and reports failure if their bytes exceed the requested maximum."""
    path = saved_cache / "unknown.engine"
    path.write_bytes(b"unrecognized engine")
    status, (_, result), _ = call_cli(capsys, "prune", "--max-size", "0")
    assert status == 1
    assert result["remaining_size_bytes"] == len(b"unrecognized engine")
    assert result["removed_size_bytes"] == 0
    assert [entry["file"] for entry in result["excluded"]] == [path.name]
    assert path.read_bytes() == b"unrecognized engine"


@pytest.mark.req("SP-TENSORACLE-010")
@pytest.mark.req("SP-TENSORACLE-043")
def test_missing_default_cache_is_listed_without_creating_it(tmp_path, monkeypatch, capsys):
    """An unset XDG_CACHE_HOME selects the user's .cache and a missing cache stays absent."""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    status, (listing,), _ = call_cli(capsys, "list")
    assert status == 0
    assert listing == {
        "directory": str(tmp_path / ".cache" / "tensoracle"),
        "entries": [],
        "total_size_bytes": 0,
    }
    assert not (tmp_path / ".cache").exists()


@pytest.mark.req("SP-TENSORACLE-047")
@pytest.mark.req("SP-TENSORACLE-056")
def test_native_result_copies_metadata_before_its_owner_is_destroyed(save_engine, saved_cache):
    """Copied headers, paths, timestamps, and summary strings survive destruction of their C result."""
    import gc

    from tensoracle import _native

    path, conditions = save_engine(saved_cache, mtime=OLD_NS + 123)
    source = saved_cache / "absent.safetensors"
    save_record(saved_cache, path=source)
    result = _native.cache_list()
    entries = result.entries()
    summary = result.summary()
    del result
    gc.collect()
    # Reuse allocator storage through several independently owned results.
    for _ in range(10):
        _native.cache_list()
    assert summary["directory"] == str(saved_cache)
    assert entries[0]["file"] == path.name
    assert entries[0]["header"]["conditions"] == conditions
    assert entries[0]["header"]["plan_sha256"] == hashlib.sha256(b"plan").hexdigest()
    assert entries[0]["weight_paths"] == [{"sha256": "a" * 64, "paths": [str(source)]}]
    assert entries[0]["recorded_at_seconds"] == 1
    assert entries[0]["recorded_at_nanoseconds"] == 123


@pytest.mark.req("SP-TENSORACLE-050")
@pytest.mark.parametrize("criterion", ["unused_for_ns", "max_size_bytes"])
def test_native_prune_rejects_integer_overflow_without_removing_files(
    save_engine, saved_cache, criterion
):
    """A criterion larger than the C ABI's signed integer range is a value error before removal."""
    from tensoracle import _native

    path, _ = save_engine(saved_cache)
    with pytest.raises(ValueError, match="64-bit"):
        _native.cache_plan_prune(**{criterion: 2**63})
    assert path.exists()


@pytest.mark.req("SP-TENSORACLE-049")
@pytest.mark.req("SP-TENSORACLE-055")
def test_list_treats_v1_engines_as_unrecognized(saved_cache, capsys):
    """A 1.x engine cannot be mistaken for a v2 engine or selected for removal."""
    header = json.dumps({"conditions": {}, "plan_sha256": "0" * 64}).encode()
    path = saved_cache / "v1.engine"
    contents = b"TENSORACLE\x01" + len(header).to_bytes(8, "big") + header + b"plan"
    path.write_bytes(contents)
    status, (listing,), _ = call_cli(capsys, "list")
    assert status == 0
    assert listing["entries"][0]["status"] == "unrecognized"
    status, (plan, result), _ = call_cli(capsys, "prune", "--max-size", "0")
    assert status == 1 and not plan["targets"]
    assert result["excluded"][0]["file"] == path.name
    assert path.read_bytes() == contents


@pytest.mark.req("SP-TENSORACLE-043")
@pytest.mark.req("SP-TENSORACLE-050")
@pytest.mark.parametrize("action", [("list",), ("prune", "--max-size", "0")])
def test_cache_io_errors_report_failure_without_traceback(tmp_path, monkeypatch, capsys, action):
    """A cache root that is a file yields a failing exit status with a useful diagnostic."""
    root = tmp_path / "not-a-directory"
    root.write_text("file")
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    status, events, stderr = call_cli(capsys, *action)
    assert status == 1 and not events
    assert "tensoracle:" in stderr and str(root) in stderr
