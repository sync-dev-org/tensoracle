"""Run native fixtures with Conan's tools and complete compiler environment."""

import os

import pytest

from native_build import preparation


@pytest.fixture(scope="session", autouse=True)
def native_tools(tmp_path_factory):
    prepare = preparation()
    output = tmp_path_factory.mktemp("native-tools")
    prepare["install"](output, wheel=True, offline=True)
    activated = prepare["environment"](output)
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(activated)
    os.environ["CMAKE_GENERATOR"] = "Ninja"
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)
