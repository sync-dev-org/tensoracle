"""Python callers use utilities while engine operations belong to the C++ API."""

import importlib.util

import pytest


@pytest.mark.req("SP-TENSORACLE-060")
def test_python_exports_comparison_without_engine_operations():
    """The package exports comparison and has no engine building, opening, execution, or plugins."""
    import tensoracle

    assert set(tensoracle.__all__) == {"OutputComparison", "compare_outputs"}
    for name in (
        "open_engine",
        "EngineHandle",
        "EngineExecution",
        "BuilderSettings",
        "InputShape",
        "EngineBuild",
        "build_serialized_engine",
        "ShapeTensorOutputWarning",
        "EngineIO",
        "TensorSpec",
        "EngineIOError",
        "CorruptEngineError",
        "EngineCacheError",
        "EngineConditionError",
        "PluginLibraryError",
        "register_libraries",
    ):
        assert not hasattr(tensoracle, name), name
    for module in ("_build", "_cache", "_cache_format", "_execution", "_plugins", "_memory"):
        assert importlib.util.find_spec(f"tensoracle.{module}") is None, module


@pytest.mark.req("SP-TENSORACLE-060")
def test_python_retains_production_inspection_conversion_and_comparison():
    """The supported Python utilities remain importable through their established modules."""
    from tensoracle import OutputComparison, compare_outputs
    from tensoracle.bundle import inspect_bundle, inspect_wheel
    from tensoracle.convert import pth_to_safetensors
    from tensoracle.producer import KernelSpec, compile_bundle

    assert all(
        callable(value)
        for value in (
            OutputComparison,
            compare_outputs,
            inspect_bundle,
            inspect_wheel,
            pth_to_safetensors,
            KernelSpec,
            compile_bundle,
        )
    )
