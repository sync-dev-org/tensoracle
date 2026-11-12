"""Assemble NVIDIA's Linux TensorRT SDK from digest-verified build inputs."""

import os
import shutil
import zipfile

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.files import check_sha256, copy, download


class TensorRTSDK(ConanFile):
    name = "tensorrt"
    version = "11.2.1.2"
    description = "TensorRT headers and libnvinfer for linking Linux x86_64 builds"
    homepage = "https://developer.nvidia.com/tensorrt"
    license = "Apache-2.0 AND LicenseRef-NVIDIA-TensorRT"
    package_type = "shared-library"
    settings = "os", "arch"
    exports = "LICENSE.*"
    no_copy_source = True

    def validate(self):
        if self.settings.os != "Linux" or self.settings.arch != "x86_64":
            raise ConanInvalidConfiguration("This TensorRT SDK supports Linux x86_64 only")

    def source(self):
        data = self.conan_data["sources"][self.version]
        for name, digest in data["headers"].items():
            download(self, f"{data['headers_url']}/{name}", f"include/{name}", sha256=digest)
        download(self, filename="LICENSE.OSS", **data["oss_license"])
        wheel = os.path.join(self.source_folder, "tensorrt.whl")
        download(self, filename=wheel, **data["wheel"])
        try:
            os.makedirs("lib", exist_ok=True)
            with zipfile.ZipFile(wheel) as archive:
                with (
                    archive.open("tensorrt_libs/libnvinfer.so.11") as source,
                    open("lib/libnvinfer.so.11", "wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
                with (
                    archive.open(data["wheel_license"]) as source,
                    open("LICENSE.NVIDIA", "wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
            check_sha256(self, "lib/libnvinfer.so.11", data["library_sha256"])
        finally:
            os.remove(wheel)

    def package(self):
        copy(
            self,
            "*.h",
            os.path.join(self.source_folder, "include"),
            os.path.join(self.package_folder, "include"),
        )
        copy(
            self,
            "libnvinfer.so.11",
            os.path.join(self.source_folder, "lib"),
            os.path.join(self.package_folder, "lib"),
        )
        os.symlink("libnvinfer.so.11", os.path.join(self.package_folder, "lib/libnvinfer.so"))
        copy(self, "LICENSE.*", self.source_folder, os.path.join(self.package_folder, "licenses"))

    def package_info(self):
        self.cpp_info.libs = ["nvinfer"]
        self.cpp_info.set_property("cmake_file_name", "TensorRT")
        self.cpp_info.set_property("cmake_target_name", "TensorRT::nvinfer")
