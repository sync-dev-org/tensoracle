"""Build, store, reuse, and execute TensorRT engines from safetensors weights."""

from importlib import import_module

__all__ = [
    "BuilderSettings",
    "CorruptEngineError",
    "EngineBuild",
    "EngineCacheError",
    "EngineConditionError",
    "EngineExecution",
    "EngineHandle",
    "EngineIO",
    "EngineIOError",
    "InputShape",
    "OutputComparison",
    "PluginLibraryError",
    "ShapeTensorOutputWarning",
    "TensorSpec",
    "build_serialized_engine",
    "compare_outputs",
    "open_engine",
]

_MODULES = {
    **dict.fromkeys(
        (
            "BuilderSettings",
            "EngineBuild",
            "InputShape",
            "ShapeTensorOutputWarning",
            "build_serialized_engine",
        ),
        "._build",
    ),
    **dict.fromkeys(
        (
            "CorruptEngineError",
            "EngineCacheError",
            "EngineConditionError",
            "EngineHandle",
            "EngineIO",
            "EngineIOError",
            "TensorSpec",
            "open_engine",
        ),
        "._cache",
    ),
    **dict.fromkeys(("OutputComparison", "compare_outputs"), "._comparison"),
    "PluginLibraryError": "._plugins",
    "EngineExecution": "._execution",
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
