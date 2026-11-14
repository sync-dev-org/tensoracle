"""Prepare native test packages once before running offline tests."""

import argparse
import json
import subprocess
from pathlib import Path

import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=prepare.ROOT / ".nf/tmp/conan-tests")
    parser.add_argument("--profile")
    parser.add_argument("--build-profile")
    args = parser.parse_args()
    try:
        prepare.install(
            args.output,
            wheel=True,
            profile=args.profile,
            build_profile=args.build_profile,
            offline=False,
        )
        lock = prepare.DIRECTORY / "cache-compatibility.lock"
        reference = json.loads(lock.read_text())["requires"][0]
        subprocess.run(
            [
                prepare.cli(),
                "install",
                f"--requires={reference}",
                "--lockfile",
                str(lock),
                *prepare.profile_arguments(args.profile, args.build_profile),
                "--output-folder",
                str(args.output / "cache-compatibility"),
                "--remote",
                "conancenter",
                "--build=missing",
                "--build=~cmake/*",
                "--build=~ninja/*",
                *prepare.cache_arguments(),
                *prepare.build_arguments(),
            ],
            check=True,
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
