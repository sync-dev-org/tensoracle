"""Reuse weight digests without changing engine identity."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tensoracle._cache as cache
from tensoracle import EngineBuild, InputShape


def _conditions(weight: Path, *, plugin: Path | None = None, force_rebuild: bool = False):
    args = (
        "encoder",
        "1",
        (weight,),
        (plugin,) if plugin else (),
        EngineBuild({"features": InputShape.fixed((1, 3, 2))}),
    )
    if force_rebuild:
        return cache._conditions(*args, force_rebuild=True)
    return cache._conditions(*args)


def _records(root: Path) -> list[Path]:
    return list((root / "tensoracle").glob("*.json"))


@pytest.mark.req("SP-TENSORACLE-009")
def test_another_process_reuses_weight_digest_without_reading_weight(tmp_path, monkeypatch):
    """A later process uses the stored SHA-256 without opening an unchanged weight file."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"first weight contents")
    expected = _conditions(weight)["weights"][0]
    child = """
import json
import sys
from pathlib import Path
import tensoracle._cache as cache
from tensoracle import EngineBuild, InputShape
weight = Path(sys.argv[1])
cache._gpu_name = lambda: 'GPU A'
original_open = Path.open
def guarded_open(self, mode='r', *args, **kwargs):
    if self == weight and 'r' in mode:
        raise AssertionError('weight file was read again')
    return original_open(self, mode, *args, **kwargs)
Path.open = guarded_open
conditions = cache._conditions('encoder', '1', (weight,), (),
    EngineBuild({'features': InputShape.fixed((1, 3, 2))}))
print(json.dumps(conditions['weights']))
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(weight)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [expected]


@pytest.mark.req("SP-TENSORACLE-009")
def test_each_weight_file_attribute_change_recomputes_digest(tmp_path, monkeypatch):
    """A changed path, size, modification time, or inode triggers a fresh SHA-256."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"first")
    original_hash = cache._file_sha256
    reads = []

    def counted_hash(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(cache, "_file_sha256", counted_hash)
    first = _conditions(weight)["weights"][0]
    assert _conditions(weight)["weights"] == [first]
    assert len(reads) == 1

    other_path = tmp_path / "same.safetensors"
    other_path.write_bytes(weight.read_bytes())
    assert _conditions(other_path)["weights"] == [first]
    assert len(reads) == 2

    weight.write_bytes(b"second and longer")
    assert _conditions(weight)["weights"] == [hashlib.sha256(weight.read_bytes()).hexdigest()]
    assert len(reads) == 3

    before = weight.stat()
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert _conditions(weight)["weights"] == [hashlib.sha256(weight.read_bytes()).hexdigest()]
    assert len(reads) == 4

    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(weight.read_bytes())
    os.utime(replacement, ns=(before.st_atime_ns, weight.stat().st_mtime_ns))
    os.replace(replacement, weight)
    assert _conditions(weight)["weights"] == [hashlib.sha256(weight.read_bytes()).hexdigest()]
    assert len(reads) == 5


@pytest.mark.req("SP-TENSORACLE-009")
def test_relative_weight_path_is_bound_to_working_directory(tmp_path, monkeypatch):
    """The same relative spelling in two directories cannot reuse the other file's digest."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "weight.safetensors").write_bytes(b"first")
    (second_dir / "weight.safetensors").write_bytes(b"other")
    monkeypatch.chdir(first_dir)
    first = _conditions(Path("weight.safetensors"))["weights"][0]
    monkeypatch.chdir(second_dir)
    second = _conditions(Path("weight.safetensors"))["weights"][0]
    assert first == hashlib.sha256(b"first").hexdigest()
    assert second == hashlib.sha256(b"other").hexdigest()
    assert len(_records(tmp_path / "cache")) == 2


@pytest.mark.req("SP-TENSORACLE-009")
@pytest.mark.parametrize(
    "damage", ["removed", "invalid-json", "wrong-format", "wrong-digest", "unreadable"]
)
def test_missing_or_unusable_record_is_recomputed(tmp_path, monkeypatch, damage):
    """A missing, damaged, incompatible, or unreadable record never blocks hashing."""
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"contents")
    expected = _conditions(weight)["weights"][0]
    (record,) = _records(root)
    if damage == "removed":
        record.unlink()
    elif damage == "invalid-json":
        record.write_text("{", encoding="utf-8")
    elif damage == "wrong-format":
        record.write_text('{"version":999}', encoding="utf-8")
    elif damage == "wrong-digest":
        contents = json.loads(record.read_text(encoding="utf-8"))
        contents["sha256"] = "0" * 64
        record.write_text(json.dumps(contents), encoding="utf-8")
    else:
        original_read = Path.read_bytes

        def unreadable(self):
            if self == record:
                raise PermissionError("record cannot be read")
            return original_read(self)

        monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert _conditions(weight)["weights"] == [expected]


@pytest.mark.req("SP-TENSORACLE-009")
def test_unwritable_record_does_not_block_hashing(tmp_path, monkeypatch):
    """A failed record write leaves the computed digest usable and the next call recomputes."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"contents")
    original_replace = os.replace
    original_hash = cache._file_sha256
    reads = []

    def denied_replace(source, target):
        if str(target).endswith(".json"):
            raise PermissionError("record cannot be replaced")
        return original_replace(source, target)

    def counted_hash(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(os, "replace", denied_replace)
    monkeypatch.setattr(cache, "_file_sha256", counted_hash)
    expected = hashlib.sha256(b"contents").hexdigest()
    assert _conditions(weight)["weights"] == [expected]
    assert _conditions(weight)["weights"] == [expected]
    assert len(reads) == 2
    assert not _records(tmp_path / "cache")


@pytest.mark.req("SP-TENSORACLE-009")
def test_concurrent_record_writers_leave_a_complete_digest(tmp_path, monkeypatch):
    """Two processes writing a digest leave one complete record usable by a later process."""
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"contents")
    gate = tmp_path / "start"
    child = """
import sys
import time
from pathlib import Path
import tensoracle._cache as cache
from tensoracle import EngineBuild, InputShape
cache._gpu_name = lambda: 'GPU A'
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
print(cache._conditions('encoder', '1', (Path(sys.argv[1]),), (),
    EngineBuild({'features': InputShape.fixed((1, 3, 2))}))['weights'][0])
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", child, str(weight), str(gate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    gate.touch()
    expected = hashlib.sha256(b"contents").hexdigest()
    for process in children:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        assert stdout.strip() == expected
    (record,) = _records(root)
    assert json.loads(record.read_text(encoding="utf-8"))["sha256"] == expected
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    monkeypatch.setattr(cache, "_file_sha256", lambda path: pytest.fail("weight was reread"))
    assert _conditions(weight)["weights"] == [expected]


@pytest.mark.req("SP-TENSORACLE-002")
def test_weight_changed_during_hash_is_not_recorded(tmp_path, monkeypatch):
    """A file changed while hashing does not associate the old metadata with that digest."""
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"first")
    original_hash = cache._file_sha256
    reads = []

    def changing_hash(path):
        digest = original_hash(path)
        reads.append(path)
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return digest

    monkeypatch.setattr(cache, "_file_sha256", changing_hash)
    assert _conditions(weight)["weights"] == [hashlib.sha256(b"first").hexdigest()]
    assert not _records(root)
    monkeypatch.setattr(cache, "_file_sha256", original_hash)
    assert _conditions(weight)["weights"] == [hashlib.sha256(b"first").hexdigest()]
    assert len(_records(root)) == 1
    assert len(reads) == 1


@pytest.mark.req("SP-TENSORACLE-011")
def test_force_rebuild_refreshes_digest_even_when_attributes_match(tmp_path, monkeypatch):
    """A requested rebuild rereads content and replaces a stale digest record."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"first")
    original = _conditions(weight)["weights"][0]
    before = weight.stat()
    weight.write_bytes(b"other")
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert weight.stat().st_size == before.st_size
    assert weight.stat().st_ino == before.st_ino
    assert _conditions(weight)["weights"] == [original]
    refreshed = hashlib.sha256(b"other").hexdigest()
    assert _conditions(weight, force_rebuild=True)["weights"] == [refreshed]
    monkeypatch.setattr(cache, "_file_sha256", lambda path: pytest.fail("weight was reread"))
    assert _conditions(weight)["weights"] == [refreshed]


@pytest.mark.req("SP-TENSORACLE-010")
@pytest.mark.parametrize("xdg", [True, False])
def test_record_lives_in_user_cache_directory(tmp_path, monkeypatch, xdg):
    """A stored weight digest stays under the configured or default user cache."""
    if xdg:
        root = tmp_path / "xdg"
        monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    else:
        root = tmp_path / "home" / ".cache"
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    weight.write_bytes(b"contents")
    _conditions(weight)
    assert len(_records(root)) == 1


@pytest.mark.req("SP-TENSORACLE-002")
@pytest.mark.req("SP-TENSORACLE-009")
def test_engine_identity_matches_the_full_hash_calculation(tmp_path, monkeypatch):
    """Stored digests produce the same conditions and engine name as full file hashing."""
    root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    monkeypatch.setattr(cache, "_gpu_name", lambda: "GPU A")
    weight = tmp_path / "weights.safetensors"
    plugin = tmp_path / "plugin.so"
    weight.write_bytes(b"weight contents")
    plugin.write_bytes(b"plugin contents")
    expected = {
        "name": "encoder",
        "graph_version": "1",
        "weights": [hashlib.sha256(weight.read_bytes()).hexdigest()],
        "plugins": [hashlib.sha256(plugin.read_bytes()).hexdigest()],
        "inputs": {"features": {"minimum": [1, 3, 2], "optimum": [1, 3, 2], "maximum": [1, 3, 2]}},
        "builder_settings": {"workspace_limit": 1 << 30, "optimization_level": 3, "tf32": False},
        "gpu_name": "GPU A",
        "os": cache.platform.system(),
        "architecture": cache.platform.machine(),
        "tensorrt_version": cache.trt.__version__,
    }
    actual = _conditions(weight, plugin=plugin)
    assert actual == expected
    expected_digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert root / "tensoracle" / f"{expected_digest}.engine" == (
        root / "tensoracle" / f"{hashlib.sha256(cache._json_bytes(actual)).hexdigest()}.engine"
    )
    original_hash = cache._file_sha256
    reads = []

    def counted_hash(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(cache, "_file_sha256", counted_hash)
    assert _conditions(weight, plugin=plugin) == expected
    assert reads == [plugin]
