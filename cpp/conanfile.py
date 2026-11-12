"""Resolve the same native dependencies for standalone, test, and wheel builds."""

import os
from typing import ClassVar

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain
from conan.tools.files import save


class TensoracleDependencies(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    default_options: ClassVar[dict] = {"*:shared": False, "*:fPIC": True}
    default_build_options: ClassVar[dict] = {"cuda-toolkit/*:compiler": False}

    def requirements(self):
        self.requires("nlohmann_json/3.11.3")
        self.requires("miniz/3.1.1")
        self.requires("catch2/3.16.1")
        self.requires("tensorrt/11.2.1.2")

    def build_requirements(self):
        self.tool_requires("cuda-toolkit/13.2.2")

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeToolchain(self)
        toolchain.user_presets_path = False
        toolchain.variables["CUDAToolkit_ROOT"] = self.dependencies.build[
            "cuda-toolkit"
        ].conf_info.get("user.cuda:root")
        toolchain.generate()
        licenses = []
        for name, filename in (("nlohmann_json", "LICENSE.MIT"), ("miniz", "LICENSE")):
            path = os.path.join(
                self.dependencies[name].package_folder, "licenses", filename
            )
            licenses.append(
                f'set(TENSORACLE_{name.upper()}_LICENSE "{path.replace(chr(92), "/")}")'
            )
        save(self, "tensoracle_licenses.cmake", "\n".join(licenses) + "\n")
