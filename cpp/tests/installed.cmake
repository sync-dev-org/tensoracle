file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${prefix}" --config "${CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
# Shared installs need only the TensorRT link library; static installs also
# resolve their external link dependencies with the toolchain that configured
# the library build, whether bootstrap generated it or the caller passed it.
file(RENAME "${prefix}" "${WORK_DIR}/relocated")
set(prefix "${WORK_DIR}/relocated")
file(COPY "${CONSUMER_SOURCE}/" DESTINATION "${WORK_DIR}/source")
set(_dependency_options)
if(NOT SHARED)
    if(NOT TOOLCHAIN_FILE OR NOT EXISTS "${TOOLCHAIN_FILE}")
        message(FATAL_ERROR "Static installed-consumer test needs the library build's CMake toolchain file; got '${TOOLCHAIN_FILE}'")
    endif()
    list(APPEND _dependency_options
        "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
        "-DCMAKE_BUILD_TYPE=${CONFIG}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=LD_LIBRARY_PATH --unset=LD_PRELOAD
    "${CMAKE_COMMAND}" -S "${WORK_DIR}/source" -B "${WORK_DIR}/build"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_CXX_COMPILER=${CXX}"
    "-DTENSORACLE_TENSORRT_LIBRARY=${TENSORRT_LIBRARY}"
    ${_dependency_options}
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=LD_LIBRARY_PATH --unset=LD_PRELOAD
    "${CMAKE_COMMAND}" --build "${WORK_DIR}/build" --config "${CONFIG}" --parallel 2
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${WORK_DIR}/build" --output-on-failure -C "${CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
