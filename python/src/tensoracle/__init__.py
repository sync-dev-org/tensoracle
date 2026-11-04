"""Compare outputs, produce and inspect cubins, manage caches, and convert checkpoints."""

from importlib import import_module as _import_module

__all__ = [
    "OutputComparison",
    "compare_outputs",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(_import_module("._comparison", __name__), name)
    globals()[name] = value
    return value
