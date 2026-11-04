#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <tensoracle/cache.hpp>
#include <tensoracle/tensoracle.hpp>

#include <vector>

namespace nb = nanobind;

void bind_cubin(nb::module_& module);
namespace {
tensoracle_dtype array_dtype(nb::handle dtype) {
    const auto kind = nb::cast<std::string>(dtype.attr("kind"));
    const auto size = nb::cast<size_t>(dtype.attr("itemsize"));
    if (kind == "b" && size == 1) return TENSORACLE_BOOL;
    if (kind == "i" || kind == "u") {
        const bool signed_type = kind == "i";
        switch (size) {
            case 1: return signed_type ? TENSORACLE_INT8 : TENSORACLE_UINT8;
            case 2: return signed_type ? TENSORACLE_INT16 : TENSORACLE_UINT16;
            case 4: return signed_type ? TENSORACLE_INT32 : TENSORACLE_UINT32;
            case 8: return signed_type ? TENSORACLE_INT64 : TENSORACLE_UINT64;
        }
    }
    if (kind == "f") {
        if (size == 2) return TENSORACLE_FLOAT16;
        if (size == 4) return TENSORACLE_FLOAT32;
        if (size == 8) return TENSORACLE_FLOAT64;
        if (size == sizeof(long double)) return TENSORACLE_LONG_DOUBLE;
    }
    return 0; // The shared library rejects unsupported numeric kinds.
}

struct cpu_array {
    nb::object storage;
    std::vector<size_t> shape;
    Py_buffer buffer{};
    tensoracle_array value{};

    cpu_array(nb::handle numpy, nb::handle input) {
        auto dtype = input.attr("dtype");
        const auto type = array_dtype(dtype);
        for (auto dimension : nb::borrow<nb::tuple>(input.attr("shape")))
            shape.push_back(nb::cast<size_t>(dimension));
        storage = numpy.attr("ascontiguousarray")(
            input, nb::arg("dtype") = dtype.attr("newbyteorder")("="));
        if (PyObject_GetBuffer(storage.ptr(), &buffer, PyBUF_SIMPLE) != 0)
            throw nb::python_error();
        value = {sizeof(value), type, shape.size(), shape.data(), buffer.buf};
    }
    cpu_array(const cpu_array&) = delete;
    cpu_array& operator=(const cpu_array&) = delete;
    ~cpu_array() { if (buffer.obj) PyBuffer_Release(&buffer); }
};

double tolerance_value(nb::handle input) {
    try {
        return static_cast<double>(nb::float_(input));
    } catch (const nb::python_error& error) {
        if (error.matches(PyExc_TypeError) || error.matches(PyExc_ValueError))
            throw nb::value_error("atol and rtol must be finite nonnegative numbers");
        throw;
    }
}

template <class T>
nb::object optional_value(const tensoracle_output_comparison& result, uint32_t flag, T value) {
    return result.available & flag ? nb::cast(value) : nb::none();
}

nb::tuple compare_outputs(nb::handle candidate, nb::handle reference, nb::handle atol, nb::handle rtol) {
    auto numpy = nb::module_::import_("numpy");
    for (auto input : {candidate, reference}) {
        const auto is_array = PyObject_IsInstance(input.ptr(), numpy.attr("ndarray").ptr());
        if (is_array < 0) throw nb::python_error();
        if (!is_array) throw nb::type_error("candidate and reference must be NumPy arrays");
    }
    cpu_array a(numpy, candidate), b(numpy, reference);
    tensoracle_comparison_options options{sizeof(options), !atol.is_none(), !rtol.is_none(), 0, 0};
    // Presence and numeric validation belong to C++; these calls only convert
    // Python values when both have been supplied.
    if (options.has_atol && options.has_rtol) {
        options.atol = tolerance_value(atol);
        options.rtol = tolerance_value(rtol);
    }
    tensoracle_output_comparison result{};
    result.struct_size = sizeof(result);
    tensoracle_error* diagnostic = nullptr;
    tensoracle_status status;
    {
        nb::gil_scoped_release release;
        status = tensoracle_compare_outputs(&a.value, &b.value, &options, &result, &diagnostic);
    }
    try {
        tensoracle::check(status, diagnostic);
    } catch (const tensoracle::error& error) {
        if (error.status() == TENSORACLE_INVALID_ARGUMENT) throw nb::value_error(error.what());
        throw;
    }
    return nb::make_tuple(
        result.candidate_nonfinite_count, result.reference_nonfinite_count,
        optional_value(result, TENSORACLE_HAS_MAX_ABSOLUTE_ERROR, result.max_absolute_error),
        optional_value(result, TENSORACLE_HAS_RELATIVE_L2_ERROR, result.relative_l2_error),
        optional_value(result, TENSORACLE_HAS_COSINE_SIMILARITY, result.cosine_similarity),
        optional_value(result, TENSORACLE_HAS_TOLERANCE, result.outside_tolerance_count),
        optional_value(result, TENSORACLE_HAS_TOLERANCE, result.outside_tolerance_fraction),
        optional_value(result, TENSORACLE_HAS_MISMATCH, result.mismatch_count),
        optional_value(result, TENSORACLE_HAS_MISMATCH, result.mismatch_fraction));
}

template <class F> auto cache_operation(F&& operation) -> decltype(operation()) {
    try {
        nb::gil_scoped_release release;
        return operation();
    } catch (const tensoracle::error& error) {
        if (error.status() == TENSORACLE_INVALID_ARGUMENT) throw nb::value_error(error.what());
        throw;
    }
}

int64_t cache_integer(nb::handle input) {
    auto value = PyLong_AsLongLong(input.ptr());
    if (PyErr_Occurred()) {
        nb::python_error error;
        if (error.matches(PyExc_OverflowError))
            throw nb::value_error("cache prune criterion must fit in a signed 64-bit integer");
        throw error;
    }
    return static_cast<int64_t>(value);
}

nb::dict cache_summary(const tensoracle::cache_result& owner) {
    const auto summary = owner.summary();
    nb::dict result;
    result["directory"] = nb::str(summary.directory);
    result["phase"] = summary.phase;
    result["dry_run"] = bool(summary.dry_run);
    result["total_size_bytes"] = summary.total_size_bytes;
    result["removed_size_bytes"] = summary.removed_size_bytes;
    result["remaining_size_bytes"] = summary.remaining_size_bytes;
    result["projected_size_bytes"] = summary.projected_size_bytes;
    result["max_size_bytes"] = summary.has_max_size ? nb::cast(summary.max_size_bytes) : nb::none();
    result["max_size_reached"] = bool(summary.max_size_reached);
    result["sizes_complete"] = bool(summary.sizes_complete);
    result["succeeded"] = bool(summary.succeeded);
    return result;
}

nb::list cache_entries(const tensoracle::cache_result& owner, const std::string& collection) {
    int32_t kind;
    if (collection == "entries") kind = TENSORACLE_CACHE_ENTRIES;
    else if (collection == "targets") kind = TENSORACLE_CACHE_TARGETS;
    else if (collection == "excluded") kind = TENSORACLE_CACHE_EXCLUDED;
    else if (collection == "removed") kind = TENSORACLE_CACHE_REMOVED;
    else if (collection == "failed") kind = TENSORACLE_CACHE_FAILED;
    else if (collection == "missing") kind = TENSORACLE_CACHE_MISSING;
    else throw nb::value_error("unknown cache result collection");
    auto loads = nb::module_::import_("json").attr("loads");
    nb::list output;
    for (const auto& entry : owner.entries(kind)) {
        nb::dict result;
        result["file"] = nb::str(entry.file);
        result["status"] = entry.status == TENSORACLE_CACHE_RECOGNIZED ? "recognized"
            : entry.status == TENSORACLE_CACHE_BUILDING_OR_LEFTOVER ? "building or leftover"
            : "unrecognized";
        result["size_bytes"] = entry.has_stat ? nb::cast(entry.size_bytes) : nb::none();
        result["recorded_at_seconds"] = entry.has_stat ? nb::cast(entry.recorded_at_seconds) : nb::none();
        result["recorded_at_nanoseconds"] = entry.has_stat ? nb::cast(entry.recorded_at_nanoseconds) : nb::none();
        // All C strings borrow owner. Copy/parse them while its handle is alive.
        if (entry.header_json) result["header"] = loads(nb::str(entry.header_json));
        if (entry.weight_paths_json) result["weight_paths"] = loads(nb::str(entry.weight_paths_json));
        if (entry.reason) result["reason"] = nb::str(entry.reason);
        output.append(result);
    }
    return output;
}
}

NB_MODULE(_native, module) {
    bind_cubin(module);
    module.doc() = "Thin bindings to tensoracle's shared C ABI";
    module.def("version", [] {
        auto version = tensoracle::version();
        return nb::make_tuple(version.major, version.minor, version.patch);
    });
    module.def("tensorrt_version", [] {
        auto version = tensoracle::tensorrt_version();
        return nb::make_tuple(version.major, version.minor, version.patch);
    });
    module.def("compare_outputs", &compare_outputs, nb::arg("candidate"), nb::arg("reference"),
               nb::kw_only(), nb::arg("atol") = nb::none(), nb::arg("rtol") = nb::none());
    nb::class_<tensoracle::cache_result>(module, "_CacheResult")
        .def("summary", &cache_summary)
        .def("entries", &cache_entries, nb::arg("collection") = "entries");
    module.def("cache_list", [] {
        return cache_operation([] { return tensoracle::list_cache(); });
    });
    module.def("cache_plan_prune", [](nb::object unused_for, nb::object max_size) {
        tensoracle_cache_prune_options options{sizeof(options), 0, 0, 0};
        if (!unused_for.is_none()) {
            options.selection |= TENSORACLE_CACHE_UNUSED_FOR;
            options.unused_for_ns = cache_integer(unused_for);
        }
        if (!max_size.is_none()) {
            options.selection |= TENSORACLE_CACHE_MAX_SIZE;
            options.max_size_bytes = cache_integer(max_size);
        }
        return cache_operation([&] { return tensoracle::plan_cache_prune(options); });
    }, nb::kw_only(), nb::arg("unused_for_ns") = nb::none(), nb::arg("max_size_bytes") = nb::none());
    module.def("cache_apply_prune", [](const tensoracle::cache_result& plan, bool dry_run) {
        return cache_operation([&] { return tensoracle::apply_cache_prune(plan, dry_run); });
    }, nb::arg("plan"), nb::kw_only(), nb::arg("dry_run") = false);
}
