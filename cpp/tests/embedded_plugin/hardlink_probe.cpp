#include "../test_platform.hpp"
#include <tensoracle/plugins.hpp>
#include <NvInferRuntime.h>
#include <nlohmann/json.hpp>
#include <psapi.h>
#include <fstream>

int wmain(int argc, wchar_t** argv) {
    if (argc != 3 && argc != 5) return 2;
    nlohmann::json report;
    try {
        auto module = LoadLibraryW(argv[1]);
        if (!module) throw std::runtime_error("module load failed: " + std::to_string(GetLastError()));
        auto anchor = GetProcAddress(module, "fixture_creators");
        if (!anchor) throw std::runtime_error("fixture creators are missing");
        auto set_namespace = reinterpret_cast<void(*)(const char*)>(GetProcAddress(module, "fixture_namespace"));
        if (!set_namespace) throw std::runtime_error("fixture namespace is missing");
        set_namespace("tensoracle.tests.hardlink");
        std::vector<wchar_t> mapped(32768);
        auto length = K32GetMappedFileNameW(GetCurrentProcess(), module, mapped.data(), DWORD(mapped.size()));
        if (!length || length >= mapped.size()) throw std::runtime_error("mapped file name is missing");
        report["mapped_path"] = std::filesystem::path(std::wstring(mapped.data(), length)).u8string();
        auto get = reinterpret_cast<nvinfer1::IPluginCreatorInterface* const*(*)(int32_t*)>(anchor);
        int32_t count = 0;
        auto source = get(&count);
        std::vector<void*> creators(source, source + count);
        auto registration = tensoracle::register_embedded_plugins(reinterpret_cast<const void*>(anchor),
            creators, "tensoracle.tests.hardlink");
        report["digests"] = registration.digests();
        report["registered"] = getPluginRegistry()->getCreator("RegistrationFirst", "1", "tensoracle.tests.hardlink") == source[0];
        report["status"] = TENSORACLE_SUCCESS;
    } catch (const tensoracle::error& failure) {
        report["status"] = failure.status();
        report["error"] = failure.what();
    } catch (const std::exception& failure) {
        report["error"] = failure.what();
    }
    { std::ofstream output{std::filesystem::path(argv[2])}; output << report.dump(2); if (!output.good()) return 3; }
    if (argc == 5) {
        auto ready = OpenEventW(EVENT_MODIFY_STATE, FALSE, argv[3]);
        auto release = OpenEventW(SYNCHRONIZE, FALSE, argv[4]);
        if (!ready || !release) return 4;
        SetEvent(ready);
        auto result = WaitForSingleObject(release, 30000);
        CloseHandle(ready); CloseHandle(release);
        if (result != WAIT_OBJECT_0) return 5;
    }
    return 0;
}
