#include "bundle_internal.hpp"
#include "error.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <iomanip>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <tuple>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace tensoracle::detail {
namespace {
std::string run(const std::vector<std::string> &arguments) {
#if defined(_WIN32)
    reject(TENSORACLE_COMPILATION_ERROR,
           "cubin production on Windows requires the later Windows v2 release");
#else
    int descriptors[2];
    require(pipe(descriptors) == 0, "cannot create CUDA tool pipe", TENSORACLE_COMPILATION_ERROR);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    std::vector<char *> argv;
    for (auto const &value : arguments)
        argv.push_back(const_cast<char *>(value.c_str()));
    argv.push_back(nullptr);
    pid_t child = 0;
    int launched =
        posix_spawn(&child, arguments[0].c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(descriptors[1]);
    if (launched != 0) {
        close(descriptors[0]);
        reject(TENSORACLE_TOOL_NOT_FOUND, "could not execute CUDA tool: " + arguments[0]);
    }
    std::string output;
    std::array<char, 8192> buffer;
    for (;;) {
        auto bytes = read(descriptors[0], buffer.data(), buffer.size());
        if (bytes > 0)
            output.append(buffer.data(), static_cast<size_t>(bytes));
        else if (bytes < 0 && errno == EINTR)
            continue;
        else
            break;
    }
    close(descriptors[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0)
        if (errno != EINTR)
            reject(TENSORACLE_COMPILATION_ERROR, "could not wait for CUDA tool");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "CUDA tool failed: " + arguments[0] + "\n" + output, TENSORACLE_COMPILATION_ERROR);
    return output;
#endif
}
struct Temporary {
    fs::path path;
    Temporary() {
        static std::atomic<uint64_t> sequence{0};
        for (unsigned attempt = 0; attempt < 100; ++attempt) {
            auto name =
                "tensoracle-producer-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                std::to_string(sequence.fetch_add(1));
            path = fs::temp_directory_path() / name;
            if (fs::create_directory(path))
                return;
        }
        reject(TENSORACLE_COMPILATION_ERROR, "could not create producer temporary directory");
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};
struct Nvrtc {
#if defined(_WIN32)
    HMODULE library = nullptr;
#else
    void *library = nullptr;
#endif
    using Program = void *;
    int (*create)(Program *, const char *, const char *, int, const char *const *,
                  const char *const *) = nullptr;
    int (*compile)(Program, int, const char *const *) = nullptr;
    int (*ptx_size)(Program, size_t *) = nullptr;
    int (*ptx)(Program, char *) = nullptr;
    int (*log_size)(Program, size_t *) = nullptr;
    int (*log)(Program, char *) = nullptr;
    int (*destroy)(Program *) = nullptr;
    int (*version)(int *, int *) = nullptr;
    template <class T> void symbol(T &target, const char *name) {
#if defined(_WIN32)
        target = reinterpret_cast<T>(GetProcAddress(library, name));
#else
        target = reinterpret_cast<T>(dlsym(library, name));
#endif
        require(target != nullptr, std::string("NVRTC lacks ") + name,
                TENSORACLE_COMPILATION_ERROR);
    }
    explicit Nvrtc(const fs::path &path) {
#if defined(_WIN32)
        library = LoadLibraryW(path.wstring().c_str());
#else
        library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        require(library != nullptr,
                "NVRTC is unavailable at " + path.string() +
                    "; install tensoracle[producer] (nvidia-cuda-nvrtc)",
                TENSORACLE_TOOL_NOT_FOUND);
        try {
            symbol(create, "nvrtcCreateProgram");
            symbol(compile, "nvrtcCompileProgram");
            symbol(ptx_size, "nvrtcGetPTXSize");
            symbol(ptx, "nvrtcGetPTX");
            symbol(log_size, "nvrtcGetProgramLogSize");
            symbol(log, "nvrtcGetProgramLog");
            symbol(destroy, "nvrtcDestroyProgram");
            symbol(version, "nvrtcVersion");
        } catch (...) {
            close();
            throw;
        }
    }
    void close() {
#if defined(_WIN32)
        if (library)
            FreeLibrary(library);
#else
        if (library)
            dlclose(library);
#endif
        library = nullptr;
    }
    ~Nvrtc() { close(); }
    std::string compile_source(const std::string &source, int sm, const fs::path &root) {
        Program program = nullptr;
        require(create(&program, source.c_str(), "kernel.cu", 0, nullptr, nullptr) == 0,
                "NVRTC could not create program", TENSORACLE_COMPILATION_ERROR);
        struct Owner {
            Nvrtc &api;
            Program &p;
            ~Owner() { api.destroy(&p); }
        } owner{*this, program};
        std::vector<std::string> options{"--gpu-architecture=compute_" + std::to_string(sm),
                                         "--std=c++17", "--fmad=false",
                                         "--include-path=" + (root / "include").string()};
        std::vector<const char *> values;
        for (auto const &option : options)
            values.push_back(option.c_str());
        if (compile(program, static_cast<int>(values.size()), values.data()) != 0) {
            size_t size = 0;
            log_size(program, &size);
            std::string message(size, '\0');
            if (size)
                log(program, message.data());
            reject(TENSORACLE_COMPILATION_ERROR,
                   "NVRTC compilation failed for sm" + std::to_string(sm) + ": " + message);
        }
        size_t size = 0;
        require(ptx_size(program, &size) == 0 && size > 0, "NVRTC did not produce PTX",
                TENSORACLE_COMPILATION_ERROR);
        std::string result(size, '\0');
        require(ptx(program, result.data()) == 0, "NVRTC PTX retrieval failed",
                TENSORACLE_COMPILATION_ERROR);
        result.pop_back();
        return result;
    }
};
fs::path default_root() {
#if defined(_WIN32)
    return fs::path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&tensoracle_compile_bundle), &info) && info.dli_fname) {
        auto root =
            fs::absolute(info.dli_fname).parent_path().parent_path().parent_path().parent_path();
        return root / "nvidia/cu13";
    }
    return fs::path();
#endif
}
std::string tool(const char *explicit_path, const fs::path &root, const char *relative,
                 const char *package, const char *argument) {
    auto path = explicit_path ? fs::path(explicit_path) : root / relative;
    require(fs::is_regular_file(path),
            std::string(package) + " tool is missing at " + path.string() +
                "; install tensoracle[producer] or pass " + argument + "=... explicitly",
            TENSORACLE_TOOL_NOT_FOUND);
    return fs::absolute(path).string();
}
void validate_ptx_signature(const std::string &ptx, const std::string &symbol,
                            const json &signature) {
    std::smatch entry;
    require(
        std::regex_search(ptx, entry, std::regex("\\.entry\\s+" + symbol + "\\s*\\(([^)]*)\\)")),
        "CUDA C++ entry symbol is absent: " + symbol, TENSORACLE_COMPILATION_ERROR);
    auto parameters = entry[1].str();
    std::regex pattern("\\.param\\s+\\.([a-z][0-9]+)");
    std::vector<std::string> types;
    for (std::sregex_iterator it(parameters.begin(), parameters.end(), pattern), end; it != end;
         ++it)
        types.push_back((*it)[1]);
    require(types.size() == signature.size(),
            "CUDA C++ public argument layout differs from signature: " + symbol,
            TENSORACLE_COMPILATION_ERROR);
    for (size_t i = 0; i < types.size(); ++i) {
        auto type = signature[i]["type"].get<std::string>();
        bool matches = type[0] == '*' ? (types[i] == "u64" || types[i] == "b64")
                       : type == "fp32"
                           ? types[i] == "f32"
                           : (types[i] == "u32" || types[i] == "s32" || types[i] == "b32");
        require(matches,
                "CUDA C++ public argument type differs from signature at ordinal " +
                    std::to_string(i) + ": " + symbol,
                TENSORACLE_COMPILATION_ERROR);
    }
}
std::string parameter_block(const std::string &elf, const std::string &symbol) {
    auto marker = "\n.nv.info." + symbol + "\n";
    auto start = elf.find(marker);
    require(start != std::string::npos, "cubin lacks kernel metadata for " + symbol,
            TENSORACLE_COMPILATION_ERROR);
    auto end = elf.find("\n.", start + marker.size());
    return elf.substr(start, end == std::string::npos ? end : end - start);
}
json resources(const std::string &cuobjdump, const fs::path &path, const json &signature,
               const std::string &symbol, uint32_t dynamic) {
    auto elf = parameter_block(run({cuobjdump, "-elf", path.string()}), symbol);
    std::regex pattern("Ordinal\\s*:\\s*0x([0-9a-fA-F]+)\\s+Offset\\s*:\\s*0x([0-9a-fA-F]+)\\s+"
                       "Size\\s*:\\s*0x([0-9a-fA-F]+)");
    std::vector<std::tuple<size_t, size_t, size_t>> actual, expected;
    for (std::sregex_iterator it(elf.begin(), elf.end(), pattern), end; it != end; ++it)
        actual.emplace_back(std::stoull((*it)[1], nullptr, 16), std::stoull((*it)[2], nullptr, 16),
                            std::stoull((*it)[3], nullptr, 16));
    size_t offset = 0;
    for (size_t i = 0; i < signature.size(); ++i) {
        auto type = signature[i]["type"].get<std::string>();
        size_t size = type[0] == '*' ? 8 : 4;
        offset = (offset + size - 1) / size * size;
        expected.emplace_back(i, offset, size);
        offset += size;
    }
    std::sort(actual.begin(), actual.end());
    require(actual == expected, "CUDA C++ public argument layout differs from signature: " + symbol,
            TENSORACLE_COMPILATION_ERROR);
    auto usage = run({cuobjdump, "--dump-resource-usage", path.string()});
    auto start = usage.find("Function " + symbol + ":");
    require(start != std::string::npos, "cuobjdump did not report function " + symbol,
            TENSORACLE_COMPILATION_ERROR);
    auto end = usage.find("Function ", start + 9);
    auto block = usage.substr(start, end == std::string::npos ? end : end - start);
    std::smatch registers, shared;
    require(std::regex_search(block, registers, std::regex("REG:([0-9]+)")) &&
                std::regex_search(block, shared, std::regex("SHARED:([0-9]+)")),
            "cuobjdump did not report kernel resources", TENSORACLE_COMPILATION_ERROR);
    return {{"registers_per_thread", std::stoul(registers[1])},
            {"compiled_shared_memory_bytes", std::stoul(shared[1])},
            {"launch_dynamic_shared_memory_bytes", dynamic}};
}
std::string generated_cpp(const json &manifest, const std::vector<std::string> &payloads) {
    std::ostringstream cpp;
    cpp << "// Generated by tensoracle C++ producer; do not edit.\n#include "
           "\"tensoracle/cubin_plugin.hpp\"\n#include <cstring>\n"
           "#if defined(_WIN32)\n#define TENSORACLE_EXPORT __declspec(dllexport)\n#else\n"
           "#define TENSORACLE_EXPORT __attribute__((visibility(\"default\")))\n#endif\n";
    cpp << "extern \"C\" TENSORACLE_EXPORT char const tensoracle_bundle_sha256[] = \""
        << manifest["bundle_sha256"].get<std::string>() << "\";\n";
    auto const &entries = manifest["entries"];
    for (size_t i = 0; i < entries.size(); ++i) {
        auto symbol = entries[i]["embedded_symbol"].get<std::string>();
        cpp << "extern \"C\" TENSORACLE_EXPORT alignas(16) unsigned char const " << symbol
            << "[] = {\n";
        size_t column = 0;
        for (unsigned char byte : payloads[i]) {
            cpp << "0x" << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte) << ",";
            if (++column % 16 == 0)
                cpp << "\n";
        }
        cpp << std::dec << "\n};\nextern \"C\" TENSORACLE_EXPORT unsigned long long const "
            << symbol << "_size = sizeof(" << symbol << ");\n";
    }
    cpp << "namespace {\nstruct BundleEntry { char const* id; tensoracle::cubin::KernelImage "
           "image; };\n";
    for (auto const &entry : entries) {
        cpp << "tensoracle::cubin::ArgumentRule const "
            << entry["embedded_symbol"].get<std::string>() << "_arguments[] = {\n";
        for (auto const &argument : entry["signature"]) {
            auto type = argument["type"].get<std::string>();
            auto kind = type[0] == '*' ? "kPointer" : type == "i32" ? "kInt32" : "kFloat32";
            auto divisors = entry["specialization"]["divisible_by"];
            cpp << "{tensoracle::cubin::ArgumentKind::" << kind << ","
                << divisors.value(argument["name"].get<std::string>(), 1) << "},\n";
        }
        cpp << "};\n";
    }
    cpp << "BundleEntry const entries[] = {\n";
    for (auto const &entry : entries) {
        auto symbol = entry["embedded_symbol"].get<std::string>();
        cpp << "{\"" << entry["id"].get<std::string>() << "\",{" << symbol << "," << symbol
            << "_size,\"" << entry["sha256"].get<std::string>() << "\",\""
            << entry["entry_symbol"].get<std::string>() << "\","
            << entry["target_sm"].get<std::string>().substr(2) << ","
            << entry["resources"]["registers_per_thread"] << ","
            << entry["resources"]["launch_dynamic_shared_memory_bytes"] << ","
            << entry["signature"].size() << "," << symbol << "_arguments}},\n";
    }
    cpp << "};\n}\nnamespace tensoracle::cubin {\nKernelImage const* find_bundle_image(char const* "
           "id) noexcept {\n"
           "if (!id) return nullptr;\nfor(auto const& entry:entries) "
           "if(std::strcmp(entry.id,id)==0) return &entry.image;\n"
           "return nullptr;\n}\n}\n";
    return cpp.str();
}
} // namespace
} // namespace tensoracle::detail
using namespace tensoracle::detail;
extern "C" tensoracle_status tensoracle_compile_bundle(const tensoracle_kernel_spec *const *specs,
                                                       size_t count,
                                                       const tensoracle_producer_options *options,
                                                       tensoracle_bundle_result **result,
                                                       tensoracle_error **error) noexcept {
    return guard(error, [&]() -> tensoracle_status {
        if (result)
            *result = nullptr;
        auto bytes = options ? structure_size(options) : 0;
        require(options && result &&
                    bytes >= offsetof(tensoracle_producer_options, output_directory) +
                                 sizeof(const char *),
                "producer options require struct_size, output directory and result",
                TENSORACLE_INVALID_ARGUMENT);
#define OPTION(name)                                                                               \
    read_field<decltype(options->name)>(options, bytes, offsetof(tensoracle_producer_options, name))
        auto output = OPTION(output_directory);
        require(present(output), "producer requires an output directory",
                TENSORACLE_INVALID_ARGUMENT);
        auto root = OPTION(cuda_root) ? fs::path(OPTION(cuda_root)) : default_root();
        auto ptxas = tool(OPTION(ptxas), root, "bin/ptxas", "nvidia-cuda-nvcc", "ptxas");
        auto version_output = run({ptxas, "--version"});
        std::smatch match;
        require(
            std::regex_search(version_output, match, std::regex("V([0-9]+)\\.([0-9]+)\\.([0-9]+)")),
            "ptxas did not report its version; required >=13.2,<14", TENSORACLE_INVALID_ARGUMENT);
        auto version = match.str(0).substr(1);
        require(std::stoi(match[1]) == 13 && std::stoi(match[2]) >= 2,
                "ptxas >=13.2,<14 is required, got " + version, TENSORACLE_INVALID_ARGUMENT);
        require(specs && count > 0, "kernel names must be unique and nonempty",
                TENSORACLE_INVALID_ARGUMENT);
        std::set<std::string> names, embedded_names;
        json kernels = json::array();
        std::vector<std::string> sources;
        for (size_t i = 0; i < count; ++i) {
            auto spec = specs[i];
            auto size = spec ? structure_size(spec) : 0;
            require(spec &&
                        size >= offsetof(tensoracle_kernel_spec, argument_count) + sizeof(size_t),
                    "kernel spec requires struct_size and arguments", TENSORACLE_INVALID_ARGUMENT);
            require(present(spec->name) &&
                        std::regex_match(spec->name, std::regex("[a-zA-Z0-9_.-]+")) &&
                        names.insert(spec->name).second,
                    "kernel names must be unique and nonempty", TENSORACLE_INVALID_ARGUMENT);
            require(present(spec->source) && present(spec->entry_symbol),
                    "kernel requires CUDA C++ source and entry symbol",
                    TENSORACLE_INVALID_ARGUMENT);
            json signature = json::array(), divisors = json::object(), constants = json::object();
            require(spec->arguments && spec->argument_count > 0 && spec->argument_count <= 14,
                    "unsupported public argument count", TENSORACLE_INVALID_ARGUMENT);
            for (size_t j = 0; j < spec->argument_count; ++j) {
                auto argument = spec->arguments[j];
                auto length = argument ? structure_size(argument) : 0;
                require(argument &&
                            length >=
                                offsetof(tensoracle_kernel_argument, type) + sizeof(const char *) &&
                            present(argument->name) && present(argument->type),
                        "invalid public argument", TENSORACLE_INVALID_ARGUMENT);
                signature.push_back({{"name", argument->name}, {"type", argument->type}});
                auto divisor = uint32_t{1};
                if (length >= offsetof(tensoracle_kernel_argument, divisible_by) + sizeof(uint32_t))
                    divisor = read_field<uint32_t>(
                        argument, length, offsetof(tensoracle_kernel_argument, divisible_by));
                require(divisor > 0, "invalid specialization divisor", TENSORACLE_INVALID_ARGUMENT);
                if (divisor != 1)
                    divisors[argument->name] = divisor;
            }
            auto constant_count =
                read_field<size_t>(spec, size, offsetof(tensoracle_kernel_spec, constant_count));
            auto constant_array = read_field<decltype(spec->constants)>(
                spec, size, offsetof(tensoracle_kernel_spec, constants));
            require(constant_count == 0 || constant_array, "missing compile constants",
                    TENSORACLE_INVALID_ARGUMENT);
            std::string source;
            for (size_t j = 0; j < constant_count; ++j) {
                auto constant = constant_array[j];
                require(
                    constant &&
                        structure_size(constant) >=
                            offsetof(tensoracle_compile_constant, value) + sizeof(int64_t) &&
                        present(constant->name) &&
                        std::regex_match(constant->name, std::regex("[a-zA-Z_][a-zA-Z0-9_]*")) &&
                        !constants.contains(constant->name),
                    "invalid compile constant", TENSORACLE_INVALID_ARGUMENT);
                constants[constant->name] = constant->value;
                source += "#define " + std::string(constant->name) + " " +
                          std::to_string(constant->value) + "\n";
            }
            source += spec->source;
            auto dynamic = read_field<uint32_t>(
                spec, size, offsetof(tensoracle_kernel_spec, dynamic_shared_memory_bytes));
            json entry = {{"signature", signature},
                          {"specialization", {{"divisible_by", divisors}}},
                          {"entry_symbol", spec->entry_symbol},
                          {"constants", constants},
                          {"resources",
                           {{"compiled_shared_memory_bytes", 0},
                            {"launch_dynamic_shared_memory_bytes", dynamic},
                            {"registers_per_thread", 1}}},
                          {"sha256", ""},
                          {"file", ""},
                          {"bytes", 1},
                          {"target_sm", "sm86"},
                          {"name", spec->name},
                          {"source_sha256", sha256(source.data(), source.size())}};
            validate_entry(entry);
            kernels.push_back(entry);
            sources.push_back(std::move(source));
        }
        auto cuobjdump =
            tool(OPTION(cuobjdump), root, "bin/cuobjdump", "nvidia-cuda-cuobjdump", "cuobjdump");
        auto nvrtc_path =
            tool(OPTION(nvrtc), root, "lib/libnvrtc.so.13", "nvidia-cuda-nvrtc", "nvrtc");
#undef OPTION
        Nvrtc nvrtc(nvrtc_path);
        int major = 0, minor = 0;
        require(nvrtc.version(&major, &minor) == 0, "NVRTC version unavailable",
                TENSORACLE_COMPILATION_ERROR);
        Temporary temporary;
        json manifest = {
            {"schema_version", 2},
            {"producer",
             {{"nvrtc", std::to_string(major) + "." + std::to_string(minor)}, {"ptxas", version}}},
            {"entries", json::array()}};
        std::vector<std::string> payloads;
        for (int sm : {86, 89, 120}) {
            for (size_t i = 0; i < kernels.size(); ++i) {
                auto entry = kernels[i];
                auto name = entry["name"].get<std::string>();
                entry.erase("name");
                auto relative = "cubins/sm" + std::to_string(sm) + "/" + name + ".cubin";
                auto path = temporary.path / relative;
                auto ptx = nvrtc.compile_source(sources[i], sm, root);
                validate_ptx_signature(ptx, entry["entry_symbol"].get<std::string>(),
                                       entry["signature"]);
                auto ptx_path = temporary.path / "kernel.ptx";
                write_file(ptx_path, ptx);
                fs::create_directories(path.parent_path());
                auto symbol = entry["entry_symbol"].get<std::string>();
                run({ptxas, "--gpu-name=sm_" + std::to_string(sm), "--entry", symbol,
                     ptx_path.string(), "--output-file", path.string()});
                auto data = read_file(path);
                require(architecture(data) == static_cast<uint32_t>(sm),
                        "compiled cubin has wrong SM", TENSORACLE_CUBIN_ARCHITECTURE_ERROR);
                auto embedded = "tensoracle_cubin_sm" + std::to_string(sm) + "_" +
                                std::regex_replace(name, std::regex("[^a-zA-Z0-9_]"), "_");
                require(embedded_names.insert(embedded).second,
                        "kernel names collide as embedded C++ symbols",
                        TENSORACLE_INVALID_ARGUMENT);
                entry["id"] = "sm" + std::to_string(sm) + "." + name;
                entry["file"] = relative;
                entry["target_sm"] = "sm" + std::to_string(sm);
                entry["embedded_symbol"] = embedded;
                entry["sha256"] = sha256(data.data(), data.size());
                entry["bytes"] = data.size();
                entry["resources"] = resources(
                    cuobjdump, path, entry["signature"], symbol,
                    entry["resources"]["launch_dynamic_shared_memory_bytes"].get<uint32_t>());
                validate_entry(entry);
                manifest["entries"].push_back(std::move(entry));
                payloads.push_back(std::move(data));
            }
        }
        manifest["bundle_sha256"] = digest(manifest);
        auto text = manifest.dump(2, ' ', true) + "\n";
        inspect(text, [&](auto const &name) { return read_file(temporary.path / name); }, nullptr);
        auto generated = generated_cpp(manifest, payloads);
        // All compilation and validation finish before touching the destination.
        auto owner = std::make_unique<tensoracle_bundle_result>(result_for(manifest));
        for (size_t i = 0; i < payloads.size(); ++i)
            write_file(fs::path(output) / manifest["entries"][i]["file"].get<std::string>(),
                       payloads[i]);
        write_file(fs::path(output) / "bundle.cpp", generated);
        write_file(fs::path(output) / "bundle.json", text);
        *result = owner.release();
        return TENSORACLE_SUCCESS;
    });
}
