`cache-3.11.3.json` contains engine filenames and complete file bytes captured
from the production cache functions built with nlohmann_json 3.11.3. The `engine`
strings represent bytes through Latin-1, including the binary length prefix;
they are not reserialized JSON expectations. The plan payload is `plan`.

The cases cover ordinary conditions, escaped Unicode and control characters,
absent and present workspace limits, shape outputs, and unsigned 64-bit limits.
The compatibility test builds the same cache functions with 3.11.3 and 3.12.0,
checks both against these fixed bytes, and reads each writer's engine and weight
record with the other reader. `cmake -P cpp/conan/prepare-tests.cmake` prepares
both JSON packages; the test itself uses the prepared Conan cache offline.

The 3.11.3 build uses `cpp/conan.lock` without selecting another recipe revision.
The 3.12.0 recipe revision is fixed in `cpp/conan/cache-compatibility.lock`, which
both preparation and the test read. Only the JSON entry changes for that build;
the other dependencies retain the entries from `cpp/conan.lock`.
