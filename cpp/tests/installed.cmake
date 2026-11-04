file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
# Check relocation and build without the source headers or TensorRT SDK.
file(RENAME "${prefix}" "${WORK_DIR}/relocated")
set(prefix "${WORK_DIR}/relocated")
file(COPY "${CONSUMER_SOURCE}/" DESTINATION "${WORK_DIR}/source")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${WORK_DIR}/source" -B "${WORK_DIR}/build"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_CXX_COMPILER=${CXX}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${WORK_DIR}/build" --config "${CONFIG}" --parallel 2
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${WORK_DIR}/build" --output-on-failure -C "${CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
