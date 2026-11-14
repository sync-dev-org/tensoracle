"""Prepare locked native dependencies and build tools before running CMake."""

import argparse
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
CPP = DIRECTORY.parent
ROOT = CPP.parent
PROJECT = ROOT if (ROOT / "pyproject.toml").is_file() else ROOT / "python"


def profile_path(value=None):
    value = value or os.environ.get("TENSORACLE_CONAN_PROFILE")
    path = Path(value or ("windows-msvc194" if sys.platform == "win32" else "linux-gcc-toolset-13"))
    return path if path.is_absolute() else DIRECTORY / "profiles" / path


def cli():
    requirements = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))[
        "build-system"
    ]["requires"]
    pinned = next(value for value in requirements if value.startswith("conan=="))
    executable = os.environ.get("TENSORACLE_CONAN_EXECUTABLE") or shutil.which("conan")
    observed = "missing"
    if executable:
        try:
            result = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, check=False
            )
            observed = result.stdout.strip()
            if result.returncode == 0 and observed == f"Conan version {pinned.split('==')[1]}":
                return str(Path(executable).resolve())
        except OSError:
            pass
    raise RuntimeError(f"Expected {pinned}; found {observed!r}. Use uv tool install {pinned}")


def cache_arguments():
    cache = os.environ.get("TENSORACLE_CONAN_CACHE")
    return ["--core-conf", f"core.cache:storage_path={cache}"] if cache else []


def build_arguments():
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    return ["--conf:all", f"tools.build:jobs={jobs}"] if jobs else []


def profile_arguments(profile=None, build_profile=None):
    host = profile_path(profile)
    build = profile_path(build_profile or os.environ.get("TENSORACLE_CONAN_BUILD_PROFILE") or host)
    tools = DIRECTORY / "profiles/tools"
    return ["-pr:h", str(host), "-pr:h", str(tools), "-pr:b", str(build)]


def install(output, *, wheel=False, profile=None, build_profile=None, offline=None):
    """Resolve one graph for a native build or a wheel; never build CMake or Ninja."""
    executable = cli()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "CONAN_HOME" not in os.environ and not os.access(Path.home(), os.W_OK):
        os.environ["CONAN_HOME"] = str(output / "home")
    if offline is None:
        offline = os.environ.get("TENSORACLE_CONAN_OFFLINE", "").upper() in (
            "1",
            "ON",
            "TRUE",
        )
    if not offline:
        recipes = runpy.run_path(str(DIRECTORY / "recipes.py"))
        recipes["obtain_recipes"](DIRECTORY, executable, cache_arguments())
    policies = (
        ["--no-remote", "--build=never"]
        if offline
        else [
            "--remote",
            "conancenter",
            "--build=missing",
            *shlex.split(os.environ.get("TENSORACLE_CONAN_ARGS", "")),
            "--build=~cmake/*",
            "--build=~ninja/*",
        ]
    )
    command = [
        executable,
        "install",
        str(CPP),
        "--lockfile",
        str(CPP / "conan.lock"),
        *profile_arguments(profile, build_profile),
        "--output-folder",
        str(output),
        *policies,
        *cache_arguments(),
        *build_arguments(),
        "--format=json",
    ]
    if wheel:
        command.extend(["-c", "tools.cmake.cmaketoolchain:user_presets="])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        home = os.environ.get("CONAN_HOME", "Conan default")
        raise RuntimeError(
            result.stderr + "\nPrepare dependencies: python cpp/conan/prepare-tests.py "
            f"--profile {profile_path(profile)} (CONAN_HOME={home}; "
            f"TENSORACLE_CONAN_CACHE={os.environ.get('TENSORACLE_CONAN_CACHE', 'Conan default')}). "
            "CMake and Ninja require prebuilt binaries on the selected platform."
        )
    return json.loads(result.stdout)


def environment(output):
    """Read the complete activated build environment, including MSVC on Windows."""
    output = Path(output).resolve()
    code = "import json, os; print(json.dumps(dict(os.environ)))"
    if sys.platform == "win32":
        command = f'call "{output / "conanbuild.bat"}" >nul && ' + subprocess.list2cmdline(
            [sys.executable, "-c", code]
        )
        invocation = command
    else:
        invocation = [
            "/bin/bash",
            "-c",
            '. "$1" >/dev/null && "$2" -c "$3"',
            "tensoracle-build",
            str(output / "conanbuild.sh"),
            sys.executable,
            code,
        ]
    result = subprocess.run(
        invocation,
        capture_output=True,
        text=True,
        check=False,
        cwd=output,
        shell=sys.platform == "win32",
        executable=os.environ.get("COMSPEC", "cmd.exe") if sys.platform == "win32" else None,
    )
    if result.returncode:
        raise RuntimeError("Could not activate Conan's build environment: " + result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/cpp")
    parser.add_argument("--profile")
    parser.add_argument("--build-profile")
    parser.add_argument("--offline", action="store_true", default=None)
    args = parser.parse_args()
    try:
        install(
            args.output,
            profile=args.profile,
            build_profile=args.build_profile,
            offline=args.offline,
        )
        presets = json.loads((args.output.resolve() / "CMakePresets.json").read_text())
        print(f"Native preset: {presets['configurePresets'][0]['name']} ({args.output.resolve()})")
    except (RuntimeError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
