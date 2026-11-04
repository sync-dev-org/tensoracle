#pragma once
#include "tensoracle.hpp"
#include "cache.h"
#include <vector>

namespace tensoracle {
class cache_result {
public:
    cache_result() = default;
    explicit cache_result(tensoracle_cache_result* result) : handle_(result) {}
    tensoracle_cache_result* get() const noexcept { return handle_.get(); }
    // String fields in returned structures borrow this result's lifetime.
    tensoracle_cache_summary summary() const {
        tensoracle_cache_summary output{sizeof(output)};
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_cache_result_summary(get(), &output, &diagnostic);
        check(status, diagnostic);
        return output;
    }
    std::vector<tensoracle_cache_entry> entries(int32_t collection = TENSORACLE_CACHE_ENTRIES) const {
        size_t count = 0;
        tensoracle_error* diagnostic = nullptr;
        auto status = tensoracle_cache_result_count(get(), collection, &count, &diagnostic);
        check(status, diagnostic);
        std::vector<tensoracle_cache_entry> output(count);
        for (size_t i = 0; i < count; ++i) {
            output[i].struct_size = sizeof(tensoracle_cache_entry);
            status = tensoracle_cache_result_entry(get(), collection, i, &output[i], &diagnostic);
            check(status, diagnostic);
        }
        return output;
    }
private:
    unique_handle<tensoracle_cache_result, tensoracle_cache_result_destroy> handle_;
};
inline cache_result list_cache() {
    tensoracle_cache_result* output = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_cache_list(&output, &diagnostic);
    check(status, diagnostic);
    return cache_result(output);
}
inline cache_result plan_cache_prune(const tensoracle_cache_prune_options& options) {
    tensoracle_cache_result* output = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_cache_plan_prune(&options, &output, &diagnostic);
    check(status, diagnostic);
    return cache_result(output);
}
inline cache_result apply_cache_prune(const cache_result& plan, bool dry_run = false) {
    tensoracle_cache_result* output = nullptr;
    tensoracle_error* diagnostic = nullptr;
    auto status = tensoracle_cache_apply_prune(plan.get(), dry_run ? 1 : 0, &output, &diagnostic);
    check(status, diagnostic);
    return cache_result(output);
}
}
