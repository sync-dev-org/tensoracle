"""Download and package the released tensoracle C and C++ SDK."""

import os

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import copy, get


class TensoracleSDK(ConanFile):
    name = "tensoracle"
    description = "Prebuilt tensoracle C ABI, C++ headers, and CMake package"
    homepage = "https://github.com/sync-dev-org/tensoracle"
    license = "MIT"
    package_type = "shared-library"
    settings = "os", "arch"
    options = {"shared": [True]}  # noqa: RUF012
    default_options = {"shared": True}  # noqa: RUF012
    exports = "recipes.txt", "LICENSE"

    def set_version(self):
        if not self.version:
            self.version = next(iter(self.conan_data["sources"]))

    def validate(self):
        if self.settings.os != "Linux" or self.settings.arch != "x86_64":
            raise ConanInvalidConfiguration("This tensoracle SDK supports Linux x86_64 only")

    def requirements(self):
        self.requires("tensorrt/11.2.1.2", transitive_headers=True, transitive_libs=True)

    def build(self):
        data = self.conan_data["sources"][str(self.version)][str(self.settings.os)][
            str(self.settings.arch)
        ]
        get(self, **data)

    def package(self):
        copy(
            self,
            "*",
            os.path.join(self.build_folder, "tensoracle"),
            os.path.join(self.package_folder, "tensoracle"),
        )

    def package_info(self):
        self.cpp_info.includedirs = ["tensoracle/cpp/include"]
        self.cpp_info.libdirs = ["tensoracle/cpp/lib"]
        self.cpp_info.bindirs = []
        self.cpp_info.libs = ["tensoracle"]
        self.cpp_info.builddirs = ["tensoracle/cpp/lib/cmake/tensoracle"]
        self.cpp_info.set_property("cmake_find_mode", "none")
