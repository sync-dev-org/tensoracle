"""Run a small safetensors network through the caller's own C++ binding."""

from ._native import infer

__all__ = ["infer"]
