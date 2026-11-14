#if defined(_WIN32)
#define TENSORACLE_FIXTURE_EXPORT __declspec(dllexport)
#else
#define TENSORACLE_FIXTURE_EXPORT __attribute__((visibility("default")))
#endif
#include <NvInferRuntime.h>
#include <string>

namespace {
std::string plugin_namespace = TENSORACLE_FIXTURE_NAMESPACE;
class creator : public nvinfer1::IPluginCreatorV3One {
public:
    explicit creator(const char* name) : name_(name) {}
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection*, nvinfer1::TensorRTPhase) noexcept override { return nullptr; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    const char* getPluginName() const noexcept override { return name_; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const char* getPluginNamespace() const noexcept override { return plugin_namespace.c_str(); }
private:
    const char* name_;
    nvinfer1::PluginFieldCollection fields_{0, nullptr};
};
creator first("RegistrationFirst"), second("RegistrationSecond"), conflict("RegistrationFirst");
unsigned char bss[256 * 1024];
nvinfer1::ILoggerFinder* logger_finder = nullptr;
}
#define EXPORT extern "C" TENSORACLE_FIXTURE_EXPORT
EXPORT void fixture_namespace(const char* value) { plugin_namespace = value; }
EXPORT nvinfer1::IPluginCreatorInterface* const* fixture_creators(int32_t* count) {
    static nvinfer1::IPluginCreatorInterface* creators[]{&first, &second};
    *count = 2; return creators;
}
EXPORT nvinfer1::IPluginCreatorInterface* fixture_conflict() { return &conflict; }
EXPORT void* fixture_bss() { return bss + sizeof(bss) - 1; }
#if !defined(TENSORACLE_NO_GET_CREATORS)
EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder* value) noexcept { logger_finder = value; }
EXPORT bool fixture_logger_initialized() { return logger_finder != nullptr; }
EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t* count) {
    return fixture_creators(count);
}
#endif
