#include <cstdint>
#include <cstdio>
#include <cstdlib>
__attribute__((constructor)) static void loaded() {
    if (const char* path = std::getenv("TENSORACLE_TEST_PLUGIN_MARKER")) {
        if (auto* file = std::fopen(path, "a")) { std::fputc('x', file); std::fclose(file); }
    }
}
#if defined(TENSORACLE_PLUGIN_VALID)
#include <NvInferRuntime.h>
#if !defined(TENSORACLE_PLUGIN_NO_LOGGER)
extern "C" __attribute__((visibility("default"))) void setLoggerFinder(nvinfer1::ILoggerFinder*) noexcept {}
#endif
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
extern "C" __attribute__((visibility("default"))) nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t* count) {
    static creator instance;
    static nvinfer1::IPluginCreatorInterface* creators[] = {&instance};
    *count = 1;
    return creators;
}
#endif
