cmake_minimum_required(VERSION 3.20)
project(dllm_openexr_harness LANGUAGES CXX)

set(DLLM_MODE "" CACHE STRING "Reviewed harness instrumentation mode")
set(OPENEXR_INSTALL_ROOT "" CACHE PATH "Pinned instrumented OpenEXR install")
set(DLLM_HARNESS_SOURCE "/work/harness.cc" CACHE FILEPATH "Harness source to build")
set(
  DLLM_DEEP_PARSE_SOURCE
  "/opt/dllm/runtime/deep_parse_check.cc"
  CACHE FILEPATH "Deep parse checker source to build"
)
set(DLLM_OUTPUT_DIRECTORY "/work" CACHE PATH "Directory for the built harness binary")
list(PREPEND CMAKE_PREFIX_PATH "${OPENEXR_INSTALL_ROOT}")

find_package(OpenEXR CONFIG REQUIRED)

add_executable(dllm_harness "${DLLM_HARNESS_SOURCE}")
target_compile_features(dllm_harness PRIVATE cxx_std_17)
target_compile_options(
  dllm_harness
  PRIVATE -O1 -gline-tables-only -fno-omit-frame-pointer
)
target_link_libraries(
  dllm_harness
  PRIVATE OpenEXR::OpenEXR OpenEXR::OpenEXRUtil
)

if(DLLM_MODE STREQUAL "asan")
  target_compile_options(
    dllm_harness
    PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined
  )
  target_link_options(
    dllm_harness
    PRIVATE -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined
  )
  set_target_properties(dllm_harness PROPERTIES OUTPUT_NAME fuzz_target)

  add_executable(dllm_deep_parse_check "${DLLM_DEEP_PARSE_SOURCE}")
  target_compile_features(dllm_deep_parse_check PRIVATE cxx_std_17)
  target_compile_options(
    dllm_deep_parse_check
    PRIVATE
      -O1
      -gline-tables-only
      -fno-omit-frame-pointer
      -fsanitize=address,undefined
      -fno-sanitize-recover=undefined
  )
  target_link_options(
    dllm_deep_parse_check
    PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=undefined
  )
  target_link_libraries(
    dllm_deep_parse_check
    PRIVATE OpenEXR::OpenEXR OpenEXR::OpenEXRUtil
  )
  set_target_properties(
    dllm_deep_parse_check
    PROPERTIES
      OUTPUT_NAME openexr-deep-parse-check
      RUNTIME_OUTPUT_DIRECTORY "${DLLM_OUTPUT_DIRECTORY}"
  )
elseif(DLLM_MODE STREQUAL "coverage")
  target_compile_options(
    dllm_harness
    PRIVATE -fsanitize=fuzzer -fprofile-instr-generate -fcoverage-mapping
  )
  target_link_options(
    dllm_harness
    PRIVATE -fsanitize=fuzzer -fprofile-instr-generate -fcoverage-mapping
  )
  set_target_properties(dllm_harness PROPERTIES OUTPUT_NAME coverage_target)
else()
  message(FATAL_ERROR "DLLM_MODE must be asan or coverage")
endif()

set_target_properties(
  dllm_harness
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${DLLM_OUTPUT_DIRECTORY}"
)
