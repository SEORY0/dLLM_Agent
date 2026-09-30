# `openexr` below is a named BuildKit build context, not a build stage --
# supply a real git checkout of OpenEXR pinned to the exact commit below
# (the RUN block verifies `git rev-parse HEAD`/tree hash/clean worktree,
# so a source tarball without `.git` metadata will fail that check):
#
#   git clone https://github.com/AcademySoftwareFoundation/openexr.git /tmp/openexr-src
#   git -C /tmp/openexr-src checkout 777d231a179de1711d2a942810d9559216ab3f4c
#   DOCKER_BUILDKIT=1 docker build --build-context openexr=/tmp/openexr-src \
#     -f tools/openexr_runtime/Dockerfile -t dllm-openexr-runtime .
FROM cybergym/oss-fuzz@sha256:e0b7469d54b9a687cb88d3d11861d09e28b67ad498c93ac91139dcbe6bee4c02

LABEL org.opencontainers.image.source="https://github.com/AcademySoftwareFoundation/openexr.git" \
      org.opencontainers.image.revision="777d231a179de1711d2a942810d9559216ab3f4c" \
      org.opencontainers.image.openexr.tree="962a0b40842bc4bc73a81294af5bef6dc974272a" \
      org.opencontainers.image.openexr.security-sha256="a29644c43663d8cad3a3f2f22fa064f1d138df9873e32e410265756d25755087" \
      org.opencontainers.image.imath.revision="1e480d11cb98b032a2dece9b9a8730512effc7f6" \
      org.opencontainers.image.version="v3.4.14"

# Deliberate deviation, recorded in the image label below (and to be reported in
# the methods): the ASan/UBSan search build disables ONLY UBSan's `nonnull-attribute`
# check. With -fno-sanitize-recover=undefined that single check aborts the process
# at Imf::OpaqueAttribute's ctor (ImfOpaqueAttribute.cpp:35, a NULL passed to a
# `nonnull` parameter for a zero-length copy) after ~2000 executions on nearly
# every corpus, so every search cell degenerated into a ~140-restart loop and no
# fuzzer made progress (measured: 1393 hits, one site, 10/10 cells). It is a
# benign-class nit, not a memory-safety report; every other UBSan check and all of
# ASan remain enabled and fatal.
LABEL org.opencontainers.image.dllm.ubsan-disabled-checks="nonnull-attribute"

COPY --from=openexr . /opt/openexr-src
COPY tools/openexr_runtime/CMakeLists.txt /opt/dllm/runtime/CMakeLists.txt
COPY tools/openexr_runtime/poc_probe.cc /opt/dllm/runtime/poc_probe.cc
COPY tools/openexr_runtime/deep_parse_check.cc /opt/dllm/runtime/deep_parse_check.cc
COPY tools/openexr_runtime/build-fuzz /opt/dllm/bin/build-fuzz
COPY tools/openexr_runtime/build-coverage /opt/dllm/bin/build-coverage

LABEL org.opencontainers.image.poc-probe.source-sha256="4cf61b34216777285bf0ec2deba85611d04208d42151cf07bab4962dc4400f8f" \
      org.opencontainers.image.poc-probe.evidence="/opt/dllm/evidence/poc-probe-hashes.env"

RUN printf '%s  %s\n' "a29644c43663d8cad3a3f2f22fa064f1d138df9873e32e410265756d25755087" /opt/openexr-src/SECURITY.md \
      | sha256sum --check --strict - \
    && test "$(git -C /opt/openexr-src rev-parse HEAD)" = "777d231a179de1711d2a942810d9559216ab3f4c" \
    && test "$(git -C /opt/openexr-src rev-parse 'HEAD^{tree}')" = "962a0b40842bc4bc73a81294af5bef6dc974272a" \
    && test -z "$(git -C /opt/openexr-src status --porcelain=v1 --untracked-files=all)" \
    && unset CFLAGS CXXFLAGS \
    && cmake -S /opt/openexr-src -B /tmp/openexr-asan -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/openexr-asan \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_FLAGS_DEBUG="-O1 -gline-tables-only -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize=nonnull-attribute -fsanitize=fuzzer-no-link -fno-sanitize-recover=undefined" \
      -DCMAKE_CXX_FLAGS_DEBUG="-O1 -gline-tables-only -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize=nonnull-attribute -fsanitize=fuzzer-no-link -fno-sanitize-recover=undefined" \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_TESTING=OFF \
      -DOPENEXR_BUILD_TOOLS=OFF \
      -DOPENEXR_BUILD_EXAMPLES=OFF \
      -DBUILD_WEBSITE=OFF \
      -DOPENEXR_BUILD_PYTHON=OFF \
      -DOPENEXR_FORCE_INTERNAL_IMATH=ON \
      -DOPENEXR_IMATH_TAG="1e480d11cb98b032a2dece9b9a8730512effc7f6" \
      -DOPENEXR_FORCE_INTERNAL_DEFLATE=ON \
      -DOPENEXR_FORCE_INTERNAL_OPENJPH=ON \
    && cmake --build /tmp/openexr-asan --target install --parallel "$(nproc)" --verbose \
    && cmake -S /opt/openexr-src -B /tmp/openexr-coverage -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/openexr-coverage \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_FLAGS_DEBUG="-O1 -gline-tables-only -fno-omit-frame-pointer -fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping" \
      -DCMAKE_CXX_FLAGS_DEBUG="-O1 -gline-tables-only -fno-omit-frame-pointer -fsanitize=fuzzer-no-link -fprofile-instr-generate -fcoverage-mapping" \
      -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_TESTING=OFF \
      -DOPENEXR_BUILD_TOOLS=OFF \
      -DOPENEXR_BUILD_EXAMPLES=OFF \
      -DBUILD_WEBSITE=OFF \
      -DOPENEXR_BUILD_PYTHON=OFF \
      -DOPENEXR_FORCE_INTERNAL_IMATH=ON \
      -DOPENEXR_IMATH_TAG="1e480d11cb98b032a2dece9b9a8730512effc7f6" \
      -DOPENEXR_FORCE_INTERNAL_DEFLATE=ON \
      -DOPENEXR_FORCE_INTERNAL_OPENJPH=ON \
    && cmake --build /tmp/openexr-coverage --target install --parallel "$(nproc)" --verbose \
    && mkdir -p /opt/dllm/evidence \
    && cp /tmp/openexr-asan/compile_commands.json /opt/dllm/evidence/openexr-asan-compile-commands.json \
    && cp /tmp/openexr-coverage/compile_commands.json /opt/dllm/evidence/openexr-coverage-compile-commands.json \
    && cmake -S /opt/dllm/runtime -B /tmp/poc-probe-asan -G "Unix Makefiles" \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDLLM_MODE=asan \
      -DOPENEXR_INSTALL_ROOT=/opt/openexr-asan \
      -DDLLM_HARNESS_SOURCE=/opt/dllm/runtime/poc_probe.cc \
      -DDLLM_OUTPUT_DIRECTORY=/opt/dllm/bin \
    && cmake --build /tmp/poc-probe-asan --target dllm_harness --parallel 1 --verbose \
    && mv /opt/dllm/bin/fuzz_target /opt/dllm/bin/poc_probe-asan \
    && cmake --build /tmp/poc-probe-asan --target dllm_deep_parse_check --parallel 1 --verbose \
    && cmake -S /opt/dllm/runtime -B /tmp/poc-probe-coverage -G "Unix Makefiles" \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DDLLM_MODE=coverage \
      -DOPENEXR_INSTALL_ROOT=/opt/openexr-coverage \
      -DDLLM_HARNESS_SOURCE=/opt/dllm/runtime/poc_probe.cc \
      -DDLLM_OUTPUT_DIRECTORY=/opt/dllm/bin \
    && cmake --build /tmp/poc-probe-coverage --target dllm_harness --parallel 1 --verbose \
    && mv /opt/dllm/bin/coverage_target /opt/dllm/bin/poc_probe-coverage \
    && cp /tmp/poc-probe-asan/compile_commands.json /opt/dllm/evidence/runtime-asan-compile-commands.json \
    && cp /tmp/poc-probe-coverage/compile_commands.json /opt/dllm/evidence/runtime-coverage-compile-commands.json \
    && SEARCH_BINARY_SHA256="$(sha256sum /opt/dllm/bin/poc_probe-asan | awk '{print $1}')" \
    && COVERAGE_BINARY_SHA256="$(sha256sum /opt/dllm/bin/poc_probe-coverage | awk '{print $1}')" \
    && DEEP_PARSE_BINARY_SHA256="$(sha256sum /opt/dllm/bin/openexr-deep-parse-check | awk '{print $1}')" \
    && DEEP_PARSE_SOURCE_SHA256="$(sha256sum /opt/dllm/runtime/deep_parse_check.cc | awk '{print $1}')" \
    && OPENEXR_ASAN_COMMANDS_SHA256="$(sha256sum /opt/dllm/evidence/openexr-asan-compile-commands.json | awk '{print $1}')" \
    && OPENEXR_COVERAGE_COMMANDS_SHA256="$(sha256sum /opt/dllm/evidence/openexr-coverage-compile-commands.json | awk '{print $1}')" \
    && RUNTIME_ASAN_COMMANDS_SHA256="$(sha256sum /opt/dllm/evidence/runtime-asan-compile-commands.json | awk '{print $1}')" \
    && RUNTIME_COVERAGE_COMMANDS_SHA256="$(sha256sum /opt/dllm/evidence/runtime-coverage-compile-commands.json | awk '{print $1}')" \
    && export SEARCH_BINARY_SHA256 COVERAGE_BINARY_SHA256 DEEP_PARSE_BINARY_SHA256 \
      DEEP_PARSE_SOURCE_SHA256 OPENEXR_ASAN_COMMANDS_SHA256 \
      OPENEXR_COVERAGE_COMMANDS_SHA256 RUNTIME_ASAN_COMMANDS_SHA256 \
      RUNTIME_COVERAGE_COMMANDS_SHA256 \
    && python3 -c 'import json,os,pathlib; names=(("/opt/dllm/bin/poc_probe-asan","SEARCH_BINARY_SHA256"),("/opt/dllm/bin/poc_probe-coverage","COVERAGE_BINARY_SHA256"),("/opt/dllm/bin/openexr-deep-parse-check","DEEP_PARSE_BINARY_SHA256"),("/opt/dllm/runtime/deep_parse_check.cc","DEEP_PARSE_SOURCE_SHA256"),("/opt/dllm/evidence/openexr-asan-compile-commands.json","OPENEXR_ASAN_COMMANDS_SHA256"),("/opt/dllm/evidence/openexr-coverage-compile-commands.json","OPENEXR_COVERAGE_COMMANDS_SHA256"),("/opt/dllm/evidence/runtime-asan-compile-commands.json","RUNTIME_ASAN_COMMANDS_SHA256"),("/opt/dllm/evidence/runtime-coverage-compile-commands.json","RUNTIME_COVERAGE_COMMANDS_SHA256")); payload={"coverage_instrumentation":["fuzzer-no-link","profile-instr-generate","coverage-mapping"],"deep_parse_checker_sha256":os.environ["DEEP_PARSE_BINARY_SHA256"],"files":[{"path":path,"sha256":os.environ[name]} for path,name in names],"schema_version":"dllm-openexr-runtime-build/v1","search_sanitizers":["address","undefined"],"target_commit":"777d231a179de1711d2a942810d9559216ab3f4c","undefined_recovery":False}; pathlib.Path("/opt/dllm/evidence/build-manifest.json").write_bytes(json.dumps(payload,separators=(",",":"),sort_keys=True).encode())' \
    && { \
      printf 'poc_probe_source_sha256=%s\n' "$(sha256sum /opt/dllm/runtime/poc_probe.cc | awk '{print $1}')"; \
      printf 'poc_probe_asan_sha256=%s\n' "$(sha256sum /opt/dllm/bin/poc_probe-asan | awk '{print $1}')"; \
      printf 'poc_probe_coverage_sha256=%s\n' "$(sha256sum /opt/dllm/bin/poc_probe-coverage | awk '{print $1}')"; \
      printf 'deep_parse_source_sha256=%s\n' "$DEEP_PARSE_SOURCE_SHA256"; \
      printf 'deep_parse_binary_sha256=%s\n' "$DEEP_PARSE_BINARY_SHA256"; \
      printf 'build_manifest_sha256=%s\n' "$(sha256sum /opt/dllm/evidence/build-manifest.json | awk '{print $1}')"; \
    } > /opt/dllm/evidence/poc-probe-hashes.env \
    && chmod 0555 /opt/dllm/bin/build-fuzz /opt/dllm/bin/build-coverage /opt/dllm/bin/poc_probe-asan /opt/dllm/bin/poc_probe-coverage /opt/dllm/bin/openexr-deep-parse-check \
    && chmod -R a+rX /opt/dllm /opt/openexr-src /opt/openexr-asan /opt/openexr-coverage \
    && rm -rf /tmp/openexr-asan /tmp/openexr-coverage /tmp/poc-probe-asan /tmp/poc-probe-coverage

ENTRYPOINT []
CMD []
