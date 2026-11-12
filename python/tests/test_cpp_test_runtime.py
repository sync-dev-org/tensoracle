"""CTest selects a full TensorRT runtime separately from the build SDK."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT
RUNTIME_SETUP = ROOT / "cpp/tests/runtime.cmake"


def runtime(directory, *, resources=True):
    directory.mkdir(parents=True)
    (directory / "libnvinfer.so.11").touch()
    if resources:
        (directory / "libnvinfer_builder_resource_sm120.so.11.2.1").touch()
    return directory


def configure(tmp_path, *, explicit=None, inherited=None):
    source = tmp_path / "repository/cpp/tests"
    source.mkdir(parents=True)
    shutil.copyfile(RUNTIME_SETUP, source / "runtime.cmake")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(test_runtime LANGUAGES NONE)\n"
        "enable_testing()\n"
        'include("${CMAKE_CURRENT_SOURCE_DIR}/runtime.cmake")\n'
        'add_test(NAME runtime COMMAND "${CMAKE_COMMAND}" -P\n'
        '  "${CMAKE_CURRENT_SOURCE_DIR}/parent.cmake")\n'
        "set_tests_properties(runtime PROPERTIES\n"
        '  ENVIRONMENT "${_tensoracle_test_environment}")\n',
        encoding="utf-8",
    )
    (source / "parent.cmake").write_text(
        'file(WRITE "parent.txt" "$ENV{LD_LIBRARY_PATH}")\n'
        'execute_process(COMMAND "${CMAKE_COMMAND}" -P\n'
        '  "${CMAKE_CURRENT_LIST_DIR}/child.cmake" COMMAND_ERROR_IS_FATAL ANY)\n',
        encoding="utf-8",
    )
    (source / "child.cmake").write_text(
        'file(WRITE "child.txt" "$ENV{LD_LIBRARY_PATH}")\n', encoding="utf-8"
    )
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    if inherited is not None:
        environment["LD_LIBRARY_PATH"] = str(inherited)
    command = ["cmake", "-S", str(source), "-B", str(tmp_path / "build")]
    if explicit is not None:
        command.append(f"-DTENSORACLE_TEST_TENSORRT_RUNTIME={explicit}")
    return subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False, timeout=10
    )


@pytest.mark.parametrize("selection", ["venv", "explicit", "without-python"])
def test_ctest_and_its_children_receive_the_selected_full_runtime(tmp_path, selection):
    """An explicit runtime overrides the venv and reaches the test's child process."""
    if selection != "without-python":
        automatic = runtime(
            tmp_path / "repository/python/.venv/lib/python3.13/site-packages/tensorrt_libs"
        )
    selected = automatic if selection == "venv" else runtime(tmp_path / "separate runtime")
    inherited = tmp_path / "other libraries"

    result = configure(
        tmp_path,
        explicit=selected if selection != "venv" else None,
        inherited=inherited,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The caller supplies no search path when CTest actually runs.
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    completed = subprocess.run(
        ["ctest", "--test-dir", str(tmp_path / "build"), "--output-on-failure"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for process in ("parent", "child"):
        assert (tmp_path / f"build/{process}.txt").read_text() == f"{selected}:{inherited}"


@pytest.mark.parametrize("supplied", ["absent", "sdk-only"])
def test_missing_full_runtime_explains_how_to_supply_it(tmp_path, supplied):
    """Absent runtimes and link-only SDKs fail during configure with a usable option."""
    sdk = runtime(tmp_path / "link SDK", resources=False) if supplied == "sdk-only" else None

    result = configure(tmp_path, explicit=sdk)

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "-DTENSORACLE_TEST_TENSORRT_RUNTIME=" in message
    assert "libnvinfer.so.11" in message
    assert "builder" in message
    if sdk is not None:
        assert str(sdk) in message
