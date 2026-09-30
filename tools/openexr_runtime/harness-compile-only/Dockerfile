# Compile-only variant of dllm-openexr-runtime: the harness source comes from a
# /work volume mount at `docker run` time, not a COPY baked into the image.
#
# The base image's own CMakeLists.txt (tools/openexr_runtime/CMakeLists.txt) already
# supports an arbitrary -DDLLM_HARNESS_SOURCE, but tools/openexr_runtime/Dockerfile's
# build-fuzz/build-coverage always point it at the COPY'd poc_probe.cc -- unusable
# once the harness source changes every synthesis iteration (spec §6's noted
# limitation, docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md).
#
# Reuses the base image's already-built ASan/coverage OpenEXR installs rather than
# recompiling OpenEXR itself (a 10-20 min build, see tools/rebuild_and_finalize.sh) --
# only the harness recompiles, and only when a container actually runs.
#
#   docker build -f tools/openexr_runtime/harness-compile-only/Dockerfile \
#     -t dllm-openexr-harness-compile-only tools/openexr_runtime/harness-compile-only
#   docker run --rm -v "$PWD/work:/work" dllm-openexr-harness-compile-only coverage
ARG BASE_IMAGE=dllm-openexr-runtime
FROM ${BASE_IMAGE}

COPY build-harness /opt/dllm/bin/build-harness
RUN chmod 0555 /opt/dllm/bin/build-harness

ENTRYPOINT ["/opt/dllm/bin/build-harness"]
