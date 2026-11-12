"""Package the miniz release used by tensoracle as a private static library."""

import os
from typing import ClassVar

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get


class Miniz(ConanFile):
    name = "miniz"
    version = "3.0.2"
    license = "MIT"
    homepage = "https://github.com/richgel999/miniz"
    description = "Deflate and ZIP compression library"
    package_type = "static-library"
    settings = "os", "arch", "compiler", "build_type"
    options: ClassVar[dict] = {"fPIC": [True, False]}
    default_options: ClassVar[dict] = {"fPIC": True}
    exports_sources = "CMakeLists.txt"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def source(self):
        get(
            self,
            **self.conan_data["sources"][self.version],
            strip_root=True,
            filename=f"miniz-{self.version}.tar.gz",
        )

    def generate(self):
        CMakeToolchain(self).generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_script_folder=self.export_sources_folder)
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE",
            self.source_folder,
            os.path.join(self.package_folder, "licenses"),
        )
        CMake(self).install()

    def package_info(self):
        self.cpp_info.libs = ["miniz"]
        self.cpp_info.set_property("cmake_file_name", "miniz")
        self.cpp_info.set_property("cmake_target_name", "miniz::miniz")
