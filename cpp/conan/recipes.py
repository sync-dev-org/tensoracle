"""Acquire locked SDK recipes and copy them into source distributions."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def locked_references(directory):
    lock = json.loads((directory.parent / "conan.lock").read_text())
    return {
        ref.split("/")[0]: ref.split("%")[0]
        for ref in lock.get("requires", []) + lock.get("build_requires", [])
        if ref.split("/")[0] in ("tensorrt", "cuda-toolkit")
    }


def _conan(cli, *arguments, **kwargs):
    result = subprocess.run(
        [str(cli), *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(f"Conan {arguments[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _export(recipe, reference, cli, cache_args):
    version = reference.split("/")[1].split("#")[0]
    exported = json.loads(
        _conan(
            cli, "export", recipe, f"--version={version}", "--format=json", *cache_args
        )
    )
    actual = exported["reference"]
    if actual != reference:
        raise RuntimeError(f"Recipe {actual} differs from locked {reference}")


def _download(manifest, stage, cli):
    # The authenticated process has its own disposable home. Its output never
    # reaches the caller, including on failure. No credential enters an argv.
    with tempfile.TemporaryDirectory(prefix="tensoracle-auth-") as directory:
        home = Path(directory)
        home.chmod(0o700)
        token = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False
        )
        if token.returncode or not token.stdout.strip():
            raise RuntimeError("gh could not supply a recipe download credential")
        credentials = {
            "credentials": [
                {
                    "url": manifest["url"],
                    "token": token.stdout.strip(),
                    "headers": {"Accept": "application/octet-stream"},
                }
            ]
        }
        descriptor = os.open(
            home / "source_credentials.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w") as output:
            json.dump(credentials, output)
        recipe = stage / "download"
        recipe.mkdir()
        (recipe / "conanfile.py").write_text(
            "from conan import ConanFile\nfrom conan.tools.files import download\n"
            "class Archive(ConanFile):\n"
            "    def source(self):\n"
            f"        download(self, {manifest['url']!r}, 'recipes.tar.gz')\n"
        )
        result = subprocess.run(
            [str(cli), "source", str(recipe), "-vwarning"],
            env={**os.environ, "CONAN_HOME": str(home)},
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Recipe archive download failed; check host gh access")
    archive = recipe / "recipes.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Recipe archive SHA256 differs from the pinned digest")
    with tarfile.open(archive) as package:
        package.extractall(stage / "unpacked", filter="data")
    return stage / "unpacked/recipes"


def obtain_recipes(directory, cli="conan", cache_args=()):
    directory = Path(directory)
    references = locked_references(directory)
    missing = {}
    for name, reference in references.items():
        embedded = directory / "recipes" / name
        if embedded.is_dir():
            _export(embedded, reference, cli, cache_args)
            continue
        cached = subprocess.run(
            [str(cli), "cache", "path", reference, *cache_args],
            capture_output=True,
            text=True,
            check=False,
        )
        if cached.returncode:
            missing[name] = reference
    if missing:
        manifest = json.loads((directory / "recipes.json").read_text())
        with tempfile.TemporaryDirectory(prefix="tensoracle-recipe-fetch-") as stage:
            recipes = _download(manifest, Path(stage), cli)
            for name, reference in missing.items():
                _export(recipes / name, reference, cli, cache_args)
    return {
        name: Path(_conan(cli, "cache", "path", reference, *cache_args))
        for name, reference in references.items()
    }


def copy_recipes(directory, destination, cli="conan", cache_args=()):
    directory = Path(directory)
    references = locked_references(directory)
    for name, source in obtain_recipes(directory, cli, cache_args).items():
        target = Path(destination) / name
        exported_sources = Path(
            _conan(
                cli,
                "cache",
                "path",
                references[name],
                "--folder=export_source",
                *cache_args,
            )
        )
        for folder in (source, exported_sources):
            if folder.is_dir():
                shutil.copytree(
                    folder,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "conanmanifest.txt"
                    ),
                )


def copy_bundle(directory, destination, cli="conan"):
    """Retain the exact verified release input, without download credentials."""
    manifest = json.loads((Path(directory) / "recipes.json").read_text())
    with tempfile.TemporaryDirectory(prefix="tensoracle-recipe-input-") as temporary:
        stage = Path(temporary)
        _download(manifest, stage, cli)
        shutil.copyfile(stage / "download/recipes.tar.gz", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conan-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--conan", default="conan")
    parser.add_argument("--core-conf", action="append", default=[])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--copy-to", type=Path)
    selection.add_argument("--download-bundle", type=Path)
    args = parser.parse_args()
    cache_args = [
        argument for value in args.core_conf for argument in ("--core-conf", value)
    ]
    if args.download_bundle:
        copy_bundle(args.conan_dir, args.download_bundle, args.conan)
    elif args.copy_to:
        copy_recipes(args.conan_dir, args.copy_to, args.conan, cache_args)
    else:
        for name, path in obtain_recipes(
            args.conan_dir, args.conan, cache_args
        ).items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
