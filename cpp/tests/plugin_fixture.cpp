#if defined(_WIN32)
#define TENSORACLE_FIXTURE_EXPORT __declspec(dllexport)
#else
#define TENSORACLE_FIXTURE_EXPORT __attribute__((visibility("default")))
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
static void loaded() {
#if defined(_WIN32)
    if (const wchar_t* path = _wgetenv(L"TENSORACLE_TEST_PLUGIN_MARKER")) {
        if (auto* file = _wfopen(path, L"a")) { std::fputc('x', file); std::fclose(file); }
    }
#else
    if (const char* path = std::getenv("TENSORACLE_TEST_PLUGIN_MARKER")) {
        if (auto* file = std::fopen(path, "a")) { std::fputc('x', file); std::fclose(file); }
    }
#endif
}
namespace { struct initialization { initialization() { loaded(); } } initialized; }
#if defined(TENSORACLE_PLUGIN_VALID)
#include <NvInferRuntime.h>
#if !defined(TENSORACLE_PLUGIN_NO_LOGGER)
extern "C" TENSORACLE_FIXTURE_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}
#endif
#if defined(TENSORACLE_PLUGIN_LEGACY)
class creator : public nvinfer1::IPluginCreator {
public:
    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) noexcept override { return nullptr; }
    nvinfer1::IPluginV2* deserializePlugin(const char*, const void*, size_t) noexcept override { return nullptr; }
    void setPluginNamespace(const char*) noexcept override {}
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    const char* getPluginName() const noexcept override { return "TensoracleLegacyFixture"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return "tensoracle.tests.legacy"; }
private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
#else
class creator : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection*, nvinfer1::TensorRTPhase) noexcept override { return nullptr; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    const char* getPluginName() const noexcept override { return "TensoracleRegistrationFixture"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return "tensoracle.tests"; }
private:
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
#endif
extern "C" TENSORACLE_FIXTURE_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t* count) {
    static creator instance;
    static nvinfer1::IPluginCreatorInterface* creators[] = {&instance};
    *count = 1;
    return creators;
}
#endif
