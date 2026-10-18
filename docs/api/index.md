# Engine execution and comparison API

The following names are available directly from `tensoracle`.

## Build declarations

::: tensoracle._build.InputShape

::: tensoracle._build.BuilderSettings

::: tensoracle._build.EngineBuild

::: tensoracle._build.build_serialized_engine

::: tensoracle._build.ShapeTensorOutputWarning

## Saved engines

::: tensoracle._cache.TensorSpec

::: tensoracle._cache.EngineIO

::: tensoracle._cache.EngineHandle

::: tensoracle._cache.open_engine

::: tensoracle._cache.EngineCacheError

::: tensoracle._cache.CorruptEngineError

::: tensoracle._cache.EngineConditionError

::: tensoracle._cache.EngineIOError

## Engine execution

Call `EngineHandle.create_execution()` after `open_engine`. Each unit owns an independent context; the caller owns device buffers and streams and confirms completion before closing it.

::: tensoracle._execution.EngineExecution

## Plugin loading

::: tensoracle._plugins.PluginLibraryError

## Output comparison

::: tensoracle._comparison.OutputComparison

::: tensoracle._comparison.compare_outputs
