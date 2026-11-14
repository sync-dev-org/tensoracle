"""Compare outputs, produce and inspect cubins, manage caches, and convert checkpoints."""

import os as _os
from importlib import import_module as _import_module
from pathlib import Path as _Path

if _os.name == "nt":
    # Keep the registration alive for bindings imported later in this process.
    # Editable installs keep Python sources and native files in different package paths.
    _dll_root = next(
        _Path(location) / "cpp" / "bin"
        for location in __path__
        if (_Path(location) / "cpp" / "bin").is_dir()
    )
    _dll_directory = _os.add_dll_directory(str(_dll_root))

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
