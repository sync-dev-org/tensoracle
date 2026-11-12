"""Check which logical CPUs the release build script gives to Docker."""

import importlib.util
import io
import itertools
import json
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent if PROJECT.name == "python" else PROJECT
SCRIPT = ROOT / "tools/build_distributions.py"


@pytest.fixture
def script():
    if not SCRIPT.is_file():
        pytest.skip("the public snapshot does not include development release tooling")
    spec = importlib.util.spec_from_file_location("build_distributions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture(scope="module")
def sdk_output_probe(tmp_path_factory):
    """Compile only the release consumer's output check, without CUDA or a GPU."""
    if not SCRIPT.is_file():
        pytest.skip("the public snapshot does not include development release tooling")
    work = tmp_path_factory.mktemp("sdk-output-probe")
    source = work / "main.cpp"
    source.write_text(
        '#include "sdk_consumer_output.hpp"\n#include <cstdlib>\n#include <iostream>\n'
        "int main(int argc, char** argv) try {\n"
        "  if (argc != 5) return 2;\n"
        "  float actual[4];\n"
        "  for (int i = 0; i < 4; ++i) actual[i] = std::strtof(argv[i + 1], nullptr);\n"
        "  const float expected[]{1.5f, 0, 6.25f, 3.25f};\n"
        "  sdk_consumer::write_checked_output(std::cout, actual, expected, 4);\n"
        "  return 0;\n"
        "} catch (const std::exception& error) { std::cerr << error.what(); return 1; }\n"
    )
    executable = work / "probe"
    result = subprocess.run(
        ["g++", "-std=c++17", "-I", str(ROOT / "tools"), str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "6.5"])
def test_sdk_consumer_rejects_nonfinite_and_inaccurate_outputs(sdk_output_probe, value):
    """Invalid inference values fail the release check without printing a successful output."""
    result = subprocess.run(
        [str(sdk_output_probe), "1.5", "0", value, "3.25"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    expected_error = (
        "non-finite inference output" if value != "6.5" else "incorrect inference output"
    )
    assert expected_error in result.stderr
    assert "OUTPUT" not in result.stdout


@pytest.mark.parametrize("value", ["1.5", "1.50000048"])
def test_sdk_consumer_prints_the_actual_output_within_tolerance(sdk_output_probe, value):
    """Accepted output reports the computed float values, including a difference within tolerance."""
    result = subprocess.run(
        [str(sdk_output_probe), value, "0", "6.25", "3.25"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    label, *values = result.stdout.split()
    assert label == "OUTPUT"
    expected_first = struct.unpack("f", struct.pack("f", float(value)))[0]
    assert list(map(float, values)) == pytest.approx(
        [expected_first, 0, 6.25, 3.25], rel=0, abs=1e-8
    )
    if value != "1.5":
        assert float(values[0]) != 1.5


class FakeHost:
    """Answer the Git and Docker calls that the build script makes."""

    def __init__(self, source):
        self.source = source
        self.commands = []

    def check_output(self, command, text=False):
        if command[:3] == ["docker", "image", "ls"]:
            return ""
        if command[:3] == ["docker", "image", "inspect"]:
            image = {"Id": "sha256:" + "1" * 64, "RepoDigests": [], "Architecture": "amd64"}
            return json.dumps([image])
        if "archive" in command:
            return tar_bytes({"wheel-build/Dockerfile": b"FROM scratch\n"})
        if "--git-common-dir" in command:
            return str(self.source / ".git")
        if "status" in command:
            return ""
        return "0" * 40

    def run(self, command, check, **kwargs):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0)


@pytest.fixture
def host(script, tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "python").mkdir(parents=True)
    (source / "python/pyproject.toml").write_text('[build-system]\nrequires = ["conan==2.34.0"]\n')
    (source / "noren-words.txt").write_text("private-example\n")
    fake = FakeHost(source)
    (source / "cpp/conan").mkdir(parents=True)
    (source / "cpp/conan.lock").write_text('{"version": "0.5", "requires": ["miniz/3.1.1#123"]}')
    (source / "cpp/conanfile.py").write_text("# consumer\n")
    (source / "cpp/conan/recipes.json").write_text(
        json.dumps({"tag": "v0.2.0", "sha256": "a" * 64})
    )
    subprocess_calls = SimpleNamespace(
        run=fake.run, check_output=fake.check_output, STDOUT=subprocess.STDOUT
    )
    monkeypatch.setattr(script, "subprocess", subprocess_calls)
    monkeypatch.setattr(script.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**4))
    fake.arguments = [
        "build_distributions.py",
        "--core-repository",
        str(tmp_path / "core"),
        "--core-tag",
        "v1.0.0",
        "--source",
        str(source),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    return fake


def docker_cpus(commands):
    """Return each Docker command with the CPU lists passed to --cpuset-cpus."""
    steps = []
    for command in commands:
        if command[0] != "docker":
            continue
        step = "offline run" if "--network" in command else command[1]
        cpus = [value for flag, value in itertools.pairwise(command) if flag == "--cpuset-cpus"]
        steps.append((step, cpus))
    return steps


@pytest.mark.parametrize("cpus", ["46-53", "1,3,8-12"])
def test_docker_uses_the_logical_cpus_given_on_the_command_line(script, host, monkeypatch, cpus):
    """The image build and all containers use the logical CPUs that the release lead chose."""
    monkeypatch.setattr(sys, "argv", host.arguments + ["--cpuset-cpus", cpus])

    script.main()

    assert docker_cpus(host.commands) == [
        ("build", [cpus]),
        ("run", [cpus]),
        ("run", [cpus]),
        ("offline run", [cpus]),
    ]


def test_release_stages_the_recipe_in_a_host_sdist_before_starting_containers(
    script, host, monkeypatch
):
    """Containers receive a host-built sdist and no host credentials or recipe download process."""
    monkeypatch.setattr(sys, "argv", host.arguments)

    script.main()

    sdist = next(command for command in host.commands if command[:2] == ["uv", "build"])
    assert "--sdist" in sdist
    containers = [command for command in host.commands if command[:2] == ["docker", "run"]]
    assert host.commands.index(sdist) < host.commands.index(containers[0])
    for command in containers:
        assert not any("src=" + str(host.source) in value for value in command)
        assert not any("TOKEN" in value or "source_credentials" in value for value in command)


def test_docker_uses_logical_cpus_0_to_7_without_the_option(script, host, monkeypatch):
    """Without a CPU list, the image build and both containers keep the logical CPUs 0-7."""
    monkeypatch.setattr(sys, "argv", host.arguments)

    script.main()

    assert docker_cpus(host.commands) == [
        ("build", ["0-7"]),
        ("run", ["0-7"]),
        ("run", ["0-7"]),
        ("offline run", ["0-7"]),
    ]


@pytest.mark.parametrize(
    "cpus", ["", "7-0", "0-", "-1", "0,,1", "0-7,", " 0-7", "a-b", "\u0660-\u0667"]
)
def test_cpu_lists_that_docker_does_not_accept_stop_before_any_command(
    script, host, monkeypatch, capsys, cpus
):
    """A CPU list outside the --cpuset-cpus form stops the script before Git or Docker runs."""
    monkeypatch.setattr(sys, "argv", host.arguments + ["--cpuset-cpus", cpus])

    with pytest.raises(SystemExit) as stopped:
        script.main()

    assert stopped.value.code == 2
    assert "--cpuset-cpus" in capsys.readouterr().err
    assert host.commands == []


def test_release_containers_select_source_built_dependencies_and_record_inputs(
    script, host, monkeypatch
):
    """All stages use the release profile, writable Conan paths, and a memory limit."""
    monkeypatch.setattr(sys, "argv", host.arguments)

    script.main()

    containers = [command for command in host.commands if command[:2] == ["docker", "run"]]
    image_build = next(command for command in host.commands if command[:2] == ["docker", "build"])
    assert image_build[image_build.index("--memory") + 1] == "48g"
    assert len(containers) == 3
    for command in containers[1:]:
        assert not any("src=" + str(host.source) in value for value in command)
    for command in containers:
        environment = [value for flag, value in itertools.pairwise(command) if flag == "--env"]
        assert "TENSORACLE_CONAN_PROFILE=linux-gcc14" in environment
        assert "TENSORACLE_CONAN_ARGS=--build=*" in environment
        assert "CONAN_HOME=/output/_work/conan-home-original" in environment
        assert "TENSORACLE_CONAN_CACHE=/output/_work/conan-packages-original" in environment
        assert command[command.index("--memory") + 1] == "48g"
        assert "CMAKE_BUILD_PARALLEL_LEVEL=8" in environment
        assert not any("TENSORACLE_SDK_ROOT=" in value for value in environment)
        assert not any("dst=/sdk" in value for value in command)
    manifest = json.loads((Path(host.arguments[-1]) / "inputs.json").read_text())
    assert manifest["conan"]["cli"] == "conan==2.34.0"
    assert manifest["conan"]["lock"]["requires"] == ["miniz/3.1.1#123"]
    assert "cpp/conan/recipes.json" in manifest["conan"]["files"]
    assert manifest["conan"]["recipes"]["tag"] == "v0.2.0"


@pytest.mark.parametrize("extension", [".so", ".cmake"])
def test_release_inspection_rejects_build_and_conan_paths(script, tmp_path, extension):
    """A wheel with a path into its build or Conan cache fails release inspection."""
    artifact = tmp_path / ("library" + extension)
    artifact.write_bytes(b"prefix\x00/output/_work/conan-packages-sdist/lib/libminiz.a\x00suffix")

    with pytest.raises(AssertionError, match="build path"):
        script.check_build_paths(artifact)


def test_sdist_selection_ignores_the_sdk_and_recipe_input(script, tmp_path):
    """The source build selects PKG-INFO even when release SDK archives share its directory."""
    for name, files in {
        "tensoracle-2.1.2.tar.gz": {"tensoracle-2.1.2/PKG-INFO": b"Name: tensoracle\n"},
        "tensoracle-sdk-2.1.2-linux-x86_64.tar.gz": {"tensoracle/cpp/include/header.h": b""},
        "recipe-input.tar.gz": {"recipes/tensoracle/conanfile.py": b""},
    }.items():
        (tmp_path / name).write_bytes(tar_bytes(files))

    assert script.sdist_archive(tmp_path).name == "tensoracle-2.1.2.tar.gz"


def test_sdk_archive_preserves_the_sdk_and_records_the_recipe_update(script, tmp_path, monkeypatch):
    """SDK generation copies wheel content, restores relative links, and records the SDK digest."""
    wheel = tmp_path / "tensoracle-2.1.2-cp312-abi3-linux_x86_64.whl"
    files = {
        "tensoracle/cpp/include/tensoracle/tensoracle.h": b"public ABI\n",
        "tensoracle/cpp/lib/libtensoracle.so.2.1.2": b"ELF from the wheel build",
        "tensoracle/cpp/lib/libtensoracle.so.2": b"ELF from the wheel build",
        "tensoracle/cpp/lib/libtensoracle.so": b"ELF from the wheel build",
        "tensoracle/cpp/src/cubin_plugin.cpp": b"helper source\n",
        "tensoracle/cpp/lib/cmake/tensoracle/tensoracleConfig.cmake": b"relative config\n",
        "tensoracle/cpp/licenses/miniz/LICENSE": b"third party\n",
        "tensoracle-2.1.2.dist-info/licenses/LICENSE": b"MIT\n",
        "tensoracle/_native.abi3.so": b"Python binding must stay in the wheel",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    commands = []
    monkeypatch.setattr(script, "run", lambda *args, **kwargs: commands.append(args))

    sdk = script.create_cpp_sdk(wheel, tmp_path / "stage", "2.1.2")

    assert commands == [
        (
            "patchelf",
            "--set-rpath",
            "$ORIGIN",
            tmp_path / "stage/sdk/tensoracle/cpp/lib/libtensoracle.so.2.1.2",
        )
    ]
    with tarfile.open(sdk) as archive:
        names = set(archive.getnames())
        assert "tensoracle/_native.abi3.so" not in names
        for name, contents in files.items():
            if name.startswith("tensoracle/cpp/") and ".so" not in name:
                assert archive.extractfile(name).read() == contents
        assert archive.extractfile("tensoracle/cpp/licenses/tensoracle/LICENSE").read() == b"MIT\n"
        assert archive.getmember("tensoracle/cpp/lib/libtensoracle.so").issym()
        assert (
            archive.getmember("tensoracle/cpp/lib/libtensoracle.so.2").linkname
            == "libtensoracle.so.2.1.2"
        )
    import hashlib

    release = json.loads((wheel.parent / "sdk-release.json").read_text())
    assert release == {
        "version": "2.1.2",
        "url": f"https://github.com/sync-dev-org/tensoracle/releases/download/v2.1.2/{sdk.name}",
        "sha256": hashlib.sha256(sdk.read_bytes()).hexdigest(),
    }
    assert sorted(path.name for path in wheel.parent.glob("*.tar.gz")) == [sdk.name]


# The private words are assembled at run time so the published test source does not contain them.
PRIVATE_REPOSITORY = "tensoracle" + "-dev"
PRIVATE_HOST = "moog" + "-wsl"


@pytest.mark.parametrize(
    "name,contents",
    [
        (
            "tensoracle/cpp/notice",
            f"https://github.com/sync-dev-org/{PRIVATE_REPOSITORY}".encode(),
        ),
        ("tensoracle/cpp/config.cmake", b"/output/_work/private/include"),
        ("tensoracle/cpp/notice", "\uff2d\uff2f\uff2f\uff27-WSL".encode()),
    ],
)
def test_public_archives_reject_private_words_and_build_paths(script, tmp_path, name, contents):
    """Release archives use the same normalized forbidden words as the public snapshot."""
    artifact = tmp_path / "sdk.tar.gz"
    artifact.write_bytes(tar_bytes({name: contents}))

    with pytest.raises(AssertionError):
        script.inspect_public_archive(
            artifact, tmp_path / "inspect", [PRIVATE_REPOSITORY, PRIVATE_HOST]
        )


def test_sdk_elf_rejects_the_wheel_tensorrt_runpath(script, tmp_path, monkeypatch):
    """An SDK must use host TensorRT lookup instead of a sibling Python package."""
    library = tmp_path / "libtensoracle.so.2.1.2"
    library.write_bytes(b"\x7fELF")

    def readelf(*args, **kwargs):
        return (
            "(NEEDED) Shared library: [libnvinfer.so.11]\n"
            "(SONAME) Library soname: [libtensoracle.so.2]\n"
            "(RUNPATH) Library runpath: [$ORIGIN/../../../tensorrt_libs]\n"
        )

    monkeypatch.setattr(script.subprocess, "check_output", readelf)
    with pytest.raises(AssertionError):
        script.inspect_elf(library, "2.1.2", sdk=True)


def test_shared_recipe_revision_is_locked_with_a_new_sdk_version(script, tmp_path):
    """The SDK stays independent of TensorRT while its test consumer supplies TensorRT."""
    import hashlib
    import runpy
    import shutil

    specification = importlib.util.spec_from_file_location(
        "check_cpp_sdk", ROOT / "tools/check_cpp_sdk.py"
    )
    checker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(checker)
    sdk_root = tmp_path / "sdk-root/tensoracle/cpp"
    (sdk_root / "lib").mkdir(parents=True)
    (sdk_root / "include").mkdir()
    (sdk_root / "lib/cmake/tensoracle").mkdir(parents=True)
    (sdk_root / "lib/libtensoracle.so.2.3.0").write_bytes(b"prebuilt library\n")
    (sdk_root / "lib/libtensoracle.so.2").symlink_to("libtensoracle.so.2.3.0")
    sdk = tmp_path / "sdk.tar.gz"
    with tarfile.open(sdk, "w:gz") as archive:
        archive.add(sdk_root.parents[1] / "tensoracle", arcname="tensoracle")
    recipes = tmp_path / "recipe-root"
    acquisition = runpy.run_path(str(ROOT / "cpp/conan/recipes.py"))
    manifest = json.loads((ROOT / "cpp/conan/recipes.json").read_text())
    (tmp_path / "shared").mkdir()
    shared = acquisition["_download"](manifest, tmp_path / "shared", "conan")
    shutil.copytree(shared / "tensoracle", recipes / "tensoracle")
    (recipes / "tensorrt").mkdir()
    (recipes / "tensorrt/conanfile.py").write_text(
        "from conan import ConanFile\nfrom conan.tools.files import copy\nimport os\nclass SDK(ConanFile):\n"
        '    name = "tensorrt"\n    version = "11.2.1.2"\n'
        '    package_type = "shared-library"\n    settings = "os", "arch"\n'
        '    exports_sources = "NvInfer.h", "libnvinfer.so"\n'
        "    def package(self):\n"
        '        copy(self, "NvInfer.h", self.source_folder, os.path.join(self.package_folder, "include"))\n'
        '        copy(self, "libnvinfer.so", self.source_folder, os.path.join(self.package_folder, "lib"))\n'
        "    def package_info(self):\n"
        '        self.cpp_info.libs = ["nvinfer"]\n'
        '        self.cpp_info.set_property("cmake_file_name", "TensorRT")\n'
        '        self.cpp_info.set_property("cmake_target_name", "TensorRT::nvinfer")\n'
    )
    (recipes / "tensorrt/NvInfer.h").write_text('extern "C" int trt_fixture();\n')
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(recipes / "tensorrt/libnvinfer.so")],
        input="int trt_fixture(void) { return 7; }\n",
        capture_output=True,
        text=True,
        check=True,
    )
    url = "https://github.com/sync-dev-org/tensoracle/releases/download/v2.3.0/sdk.tar.gz"
    release = {
        "version": "2.3.0",
        "url": url,
        "sha256": hashlib.sha256(sdk.read_bytes()).hexdigest(),
    }
    # Only the external TensorRT package boundary is substituted. The tensoracle
    # recipe is the unchanged implementation from the pinned shared bundle.
    recipe_input = tmp_path / "recipe-input.tar.gz"
    with tarfile.open(recipe_input, "w:gz") as archive:
        for directory in recipes.iterdir():
            archive.add(directory, arcname="recipes/" + directory.name)
    manifest["sha256"] = hashlib.sha256(recipe_input.read_bytes()).hexdigest()
    package, generators, record = checker.acquire_sdk(
        sdk,
        recipe_input,
        release,
        manifest,
        tmp_path / "verification",
        ROOT / "cpp/conan/profiles/linux-gcc13",
    )
    assert record["recipe_bundle"] == manifest
    assert record["source_recipe_reference"].startswith("tensoracle/2.2.1#")
    assert record["recipe_references"]["tensoracle"].startswith("tensoracle/2.3.0#")
    assert record["source_recipe_reference"] != record["recipe_references"]["tensoracle"]
    nodes = list(record["graph"]["nodes"].values())
    node = next(node for node in nodes if node.get("name") == "tensoracle")
    assert node["settings"] == {"os": "Linux", "arch": "x86_64"}
    assert not any(
        dependency["ref"].startswith("tensorrt/") for dependency in node["dependencies"].values()
    )
    root = next(node for node in nodes if node.get("name") is None)
    dependency = next(
        dependency
        for dependency in root["dependencies"].values()
        if dependency["ref"].startswith("tensorrt/")
    )
    assert dependency["direct"] and dependency["test"]
    assert record["sdk_url"] == url
    assert record["changed_digest_exit"] != 0
    assert (package / "tensoracle/cpp/lib/libtensoracle.so.2").read_bytes() == b"prebuilt library\n"
    assert all(
        reference in {item.split("%")[0] for item in record["lock"]["requires"]}
        for reference in record["recipe_references"].values()
    )
    consumer = tmp_path / "native-consumer"
    consumer.mkdir()
    (consumer / "main.cpp").write_text(
        "#include <NvInfer.h>\nint main() { return trt_fixture() != 7; }\n"
    )
    (consumer / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\nproject(consumer LANGUAGES CXX)\n"
        "find_package(TensorRT CONFIG REQUIRED)\nadd_executable(consumer main.cpp)\n"
        "target_link_libraries(consumer PRIVATE TensorRT::nvinfer)\n"
    )
    build = tmp_path / "native-build"
    for command in (
        [
            "cmake",
            "-S",
            str(consumer),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_TOOLCHAIN_FILE={generators / 'conan_toolchain.cmake'}",
        ],
        ["cmake", "--build", str(build)],
        [str(build / "consumer")],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def test_sdk_acquisition_rejects_a_changed_shared_bundle(script, tmp_path):
    """A changed recipe input fails its pinned digest before any recipe can be exported."""
    specification = importlib.util.spec_from_file_location(
        "check_cpp_sdk", ROOT / "tools/check_cpp_sdk.py"
    )
    checker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(checker)
    bundle = tmp_path / "recipe-input.tar.gz"
    bundle.write_bytes(b"changed bundle")
    with pytest.raises(AssertionError, match="recipe bundle SHA256"):
        checker.acquire_sdk(
            tmp_path / "sdk.tar.gz",
            bundle,
            {"version": "2.3.0", "url": "https://example.invalid/sdk", "sha256": "a" * 64},
            {"tag": "fixture", "sha256": "0" * 64},
            tmp_path / "verification",
            ROOT / "cpp/conan/profiles/linux-gcc13",
        )
    assert not (tmp_path / "verification/conan-home").exists()
