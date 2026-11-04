#include <algorithm>
#include <map>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <tensoracle/cubin.hpp>
#include <vector>

namespace nb = nanobind;
namespace {
struct Spec {
    std::string name, source, entry;
    std::vector<std::string> argument_names, argument_types, constant_names;
    std::vector<tensoracle_kernel_argument> arguments;
    std::vector<const tensoracle_kernel_argument *> argument_pointers;
    std::vector<tensoracle_compile_constant> constants;
    std::vector<const tensoracle_compile_constant *> constant_pointers;
    tensoracle_kernel_spec value{};
    explicit Spec(nb::dict input) {
        name = nb::cast<std::string>(input["name"]);
        source = nb::cast<std::string>(input["source"]);
        entry = nb::cast<std::string>(input["entry_symbol"]);
        auto signature = nb::cast<nb::tuple>(input["signature"]);
        auto divisors = nb::cast<nb::dict>(input["divisible_by"]);
        auto constexprs = nb::cast<nb::dict>(input["constexprs"]);
        for (auto item : signature) {
            auto pair = nb::cast<nb::tuple>(item);
            argument_names.push_back(nb::cast<std::string>(pair[0]));
            argument_types.push_back(nb::cast<std::string>(pair[1]));
        }
        for (size_t i = 0; i < argument_names.size(); ++i) {
            auto key = nb::str(argument_names[i].c_str());
            uint32_t divisor = divisors.contains(key) ? nb::cast<uint32_t>(divisors[key]) : 1;
            arguments.push_back({sizeof(tensoracle_kernel_argument), argument_names[i].c_str(),
                                 argument_types[i].c_str(), divisor});
        }
        for (auto [key, value] : divisors) {
            auto name = nb::cast<std::string>(key);
            if (std::find(argument_names.begin(), argument_names.end(), name) ==
                argument_names.end())
                throw nb::value_error("specialization names must refer to public arguments");
        }
        for (auto [key, value] : constexprs)
            constant_names.push_back(nb::cast<std::string>(key));
        for (auto const &name : constant_names)
            constants.push_back({sizeof(tensoracle_compile_constant), name.c_str(),
                                 nb::cast<int64_t>(constexprs[nb::str(name.c_str())])});
        for (auto const &argument : arguments)
            argument_pointers.push_back(&argument);
        for (auto const &constant : constants)
            constant_pointers.push_back(&constant);
        value = {sizeof(value),
                 name.c_str(),
                 source.c_str(),
                 entry.c_str(),
                 argument_pointers.data(),
                 arguments.size(),
                 constant_pointers.data(),
                 constants.size(),
                 nb::cast<uint32_t>(input["dynamic_shared_memory_bytes"])};
    }
};
template <class F> auto translate(const nb::dict &errors, F &&operation) -> decltype(operation()) {
    try {
        return operation();
    } catch (tensoracle::error const &error) {
        auto status = error.status();
        if (status == TENSORACLE_INVALID_ARGUMENT) {
            PyErr_SetString(PyExc_ValueError, error.what());
        } else if (status == TENSORACLE_TOOL_NOT_FOUND) {
            PyErr_SetString(PyExc_ImportError, error.what());
        } else {
            auto key = nb::int_(status);
            nb::object type =
                errors.contains(key)
                    ? nb::borrow<nb::object>(errors[key])
                    : nb::borrow<nb::object>(
                          errors[nb::int_(static_cast<int>(TENSORACLE_BUNDLE_ERROR))]);
            PyErr_SetString(type.ptr(), error.what());
        }
        throw nb::python_error();
    }
}
} // namespace
void bind_cubin(nb::module_ &module) {
    nb::dict errors;
    const std::pair<int, const char *> classes[] = {
        {TENSORACLE_BUNDLE_ERROR, "BundleError"},
        {TENSORACLE_MISSING_CUBIN, "MissingCubinError"},
        {TENSORACLE_CORRUPT_CUBIN, "CorruptCubinError"},
        {TENSORACLE_CUBIN_ARCHITECTURE_ERROR, "CubinArchitectureError"},
        {TENSORACLE_PLUGIN_BUNDLE_MISMATCH, "PluginBundleMismatchError"},
        {TENSORACLE_COMPILATION_ERROR, "CompilationError"}};
    nb::object base = nb::borrow<nb::object>(PyExc_ValueError);
    for (auto [status, name] : classes) {
        auto qualified = std::string("tensoracle._native.") + name;
        nb::object type =
            nb::steal<nb::object>(PyErr_NewException(qualified.c_str(), base.ptr(), nullptr));
        module.attr(name) = type;
        errors[nb::int_(status)] = type;
        if (status == TENSORACLE_BUNDLE_ERROR)
            base = type;
    }
    module.def("cubin_sm", [errors](nb::bytes bytes) {
        return translate(errors, [&] {
            return "sm" + std::to_string(tensoracle::cubin_sm(bytes.c_str(), bytes.size()));
        });
    });
    module.def("bundle_sha256", [errors](const std::string &json) {
        return translate(errors, [&] { return tensoracle::bundle_digest(json.c_str()).sha256(); });
    });
    module.def(
        "inspect_bundle",
        [errors](const std::string &root, nb::object plugin) {
            auto path = plugin.is_none() ? std::string() : nb::cast<std::string>(plugin);
            return translate(errors, [&] {
                return tensoracle::inspect_bundle(root.c_str(),
                                                  plugin.is_none() ? nullptr : path.c_str())
                    .sha256();
            });
        },
        nb::arg("path"), nb::arg("plugin").none());
    module.def(
        "inspect_wheel",
        [errors](const std::string &wheel, nb::object plugin) {
            auto path = plugin.is_none() ? std::string() : nb::cast<std::string>(plugin);
            return translate(errors, [&] {
                return tensoracle::inspect_wheel(wheel.c_str(),
                                                 plugin.is_none() ? nullptr : path.c_str())
                    .sha256();
            });
        },
        nb::arg("path"), nb::arg("plugin").none());
    module.def(
        "compile_bundle",
        [errors](nb::list inputs, const std::string &output, const std::string &root,
                 nb::object ptxas, nb::object cuobjdump, nb::object nvrtc) {
            auto ptx = ptxas.is_none() ? std::string() : nb::cast<std::string>(ptxas);
            auto obj = cuobjdump.is_none() ? std::string() : nb::cast<std::string>(cuobjdump);
            auto rtc = nvrtc.is_none() ? std::string() : nb::cast<std::string>(nvrtc);
            std::vector<std::unique_ptr<Spec>> storage;
            std::vector<const tensoracle_kernel_spec *> specs;
            for (auto item : inputs) {
                storage.push_back(std::make_unique<Spec>(nb::cast<nb::dict>(item)));
                specs.push_back(&storage.back()->value);
            }
            tensoracle_producer_options options{sizeof(options),
                                                output.c_str(),
                                                root.c_str(),
                                                ptxas.is_none() ? nullptr : ptx.c_str(),
                                                cuobjdump.is_none() ? nullptr : obj.c_str(),
                                                nvrtc.is_none() ? nullptr : rtc.c_str()};
            return translate(errors, [&] {
                nb::gil_scoped_release release;
                return tensoracle::compile_bundle(specs.data(), specs.size(), options)
                    .manifest_json();
            });
        },
        nb::arg("inputs"), nb::arg("output"), nb::arg("root"), nb::arg("ptxas").none(),
        nb::arg("cuobjdump").none(), nb::arg("nvrtc").none());
}
