#ifndef TENSORACLE_CACHE_H
#define TENSORACLE_CACHE_H
#include "tensoracle.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensoracle_cache_result tensoracle_cache_result;
enum {
    TENSORACLE_CACHE_RECOGNIZED = 1,
    TENSORACLE_CACHE_BUILDING_OR_LEFTOVER = 2,
    TENSORACLE_CACHE_UNRECOGNIZED = 3,
    TENSORACLE_CACHE_UNUSED_FOR = 1,
    TENSORACLE_CACHE_MAX_SIZE = 2,
    TENSORACLE_CACHE_ENTRIES = 1,
    TENSORACLE_CACHE_TARGETS = 2,
    TENSORACLE_CACHE_EXCLUDED = 3,
    TENSORACLE_CACHE_REMOVED = 4,
    TENSORACLE_CACHE_FAILED = 5,
    TENSORACLE_CACHE_MISSING = 6,
    TENSORACLE_CACHE_LIST = 1,
    TENSORACLE_CACHE_PLAN = 2,
    TENSORACLE_CACHE_APPLIED = 3
};

/* Exactly one selection flag is required. The selected signed value must be
 * nonnegative and completely present; other fields are ignored. Durations are
 * exact nanoseconds (one day = 86400000000000 ns). Unknown fields are ignored. */
typedef struct tensoracle_cache_prune_options {
    size_t struct_size;
    uint32_t selection;
    int64_t unused_for_ns;
    int64_t max_size_bytes;
} tensoracle_cache_prune_options;

/* All returned strings are borrowed until the result is destroyed. Outputs use
 * the common struct_size contract: only complete fields that fit are written.
 * header_json preserves the full stored header; NULL means unrecognized/empty.
 * weight_paths_json is [{"sha256":...,"paths":[...]}] for recognized list entries.
 * A path is informational, from a checked digest record; weights are not read.
 * has_stat == 0 means size/time are unknown, rather than zero. Timestamps use
 * signed Unix seconds plus a nonnegative nanosecond fraction, without rounding. */
typedef struct tensoracle_cache_entry {
    size_t struct_size;
    const char* file;
    int32_t status;
    uint32_t has_stat;
    uint64_t size_bytes;
    int64_t recorded_at_seconds;
    uint32_t recorded_at_nanoseconds;
    const char* header_json;
    const char* weight_paths_json;
    const char* reason;
} tensoracle_cache_entry;

typedef struct tensoracle_cache_summary {
    size_t struct_size;
    const char* directory;
    int32_t phase;
    uint32_t dry_run;
    uint64_t total_size_bytes;
    uint64_t removed_size_bytes;
    uint64_t remaining_size_bytes;
    uint64_t projected_size_bytes;
    uint32_t has_max_size;
    int64_t max_size_bytes;
    uint32_t max_size_reached;
    uint32_t sizes_complete;
    uint32_t succeeded;
} tensoracle_cache_summary;

/* Uses XDG_CACHE_HOME/tensoracle or the default user cache. Missing directories
 * stay absent. Listing reads only headers and metadata, never plans or weights,
 * does not initialize CUDA, and does not change recorded mtimes. */
TENSORACLE_API tensoracle_status tensoracle_cache_list(
    tensoracle_cache_result** result, tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Planning does not remove anything. Display TARGETS and a notice that removed
 * engines must be rebuilt on next use before calling apply. Unknown/empty files
 * are excluded from removal but count toward total bytes. Ties use filenames. */
TENSORACLE_API tensoracle_status tensoracle_cache_plan_prune(
    const tensoracle_cache_prune_options* options, tensoracle_cache_result** plan,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
/* Maintenance only: no concurrent builds/opens in this cache. No lock protects
 * them from pruning. Files changed/missing since planning are reported and not
 * counted as removed. dry_run is 0 or 1 and preserves content and mtimes.
 *
 * SUCCESS means the result was produced, including partial removal. Use
 * summary.succeeded for the CLI exit state: failed removals, an unmet maximum,
 * or unknown sizes make it false. FAILED/MISSING retain reasons. The snapshot
 * remains owned by plan; applying does not mutate or consume it. Entries in
 * FAILED/MISSING retain the original metadata from planning. The plan keeps
 * its cache location even if the environment or working directory changes. */
TENSORACLE_API tensoracle_status tensoracle_cache_apply_prune(
    const tensoracle_cache_result* plan, uint32_t dry_run,
    tensoracle_cache_result** result, tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_cache_result_summary(
    const tensoracle_cache_result* result, tensoracle_cache_summary* summary,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_cache_result_count(
    const tensoracle_cache_result* result, int32_t collection, size_t* count,
    tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_cache_result_entry(
    const tensoracle_cache_result* result, int32_t collection, size_t index,
    tensoracle_cache_entry* entry, tensoracle_error** error) TENSORACLE_NOEXCEPT;
TENSORACLE_API tensoracle_status tensoracle_cache_result_destroy(
    tensoracle_cache_result* result) TENSORACLE_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#endif
