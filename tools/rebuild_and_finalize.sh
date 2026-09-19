#!/usr/bin/env bash
# Rebuild the pinned OpenEXR ASan/coverage image, extract the two binaries and the
# build manifest out of it, and regenerate the finalized one-day config with the
# new image digest / binary hashes.
#
# Needed whenever tools/openexr_runtime/{Dockerfile,CMakeLists.txt} change: a new
# build is a new image digest and new binary hashes, all of which the config pins.
#
#   tools/rebuild_and_finalize.sh
#   OPENEXR_SRC=/path/to/openexr CONFIG_OUT=... tools/rebuild_and_finalize.sh
set -euo pipefail

cd "$(dirname "$0")/.."

OPENEXR_SRC="${OPENEXR_SRC:-/tmp/openexr-src}"
OPENEXR_COMMIT="777d231a179de1711d2a942810d9559216ab3f4c"
IMAGE_TAG="${IMAGE_TAG:-dllm-openexr-runtime}"
ARTIFACTS="${ARTIFACTS:-artifacts/openexr-runtime}"
TEMPLATE="${TEMPLATE:-configs/openexr-v3.4.14-one-day-1gpu.json}"
CONFIG_OUT="${CONFIG_OUT:-/data/dllm-agent/openexr-one-day-1gpu.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/dllm-agent/openexr-one-day-1gpu}"
CONFIG_ID="${CONFIG_ID:-openexr-v3.4.14-one-day-1gpu}"
REMOTE_CODE="${REMOTE_CODE:-/data/dllm-agent/cache/modeling_stable_diffcoder.py}"

# The Dockerfile verifies `git rev-parse HEAD`, the tree hash and a clean worktree,
# so the build context must be a real checkout of exactly the pinned commit.
if [ ! -d "$OPENEXR_SRC/.git" ]; then
  git clone https://github.com/AcademySoftwareFoundation/openexr.git "$OPENEXR_SRC"
fi
git -C "$OPENEXR_SRC" checkout -q "$OPENEXR_COMMIT"

echo "== building $IMAGE_TAG (this compiles OpenEXR twice; ~10-20 min)"
DOCKER_BUILDKIT=1 docker build --build-context openexr="$OPENEXR_SRC" \
  -f tools/openexr_runtime/Dockerfile -t "$IMAGE_TAG" .

DIGEST="$(docker inspect --format='{{.Id}}' "$IMAGE_TAG")"
echo "== image digest: $DIGEST"

mkdir -p "$ARTIFACTS"
for pair in bin/poc_probe-asan bin/poc_probe-coverage evidence/build-manifest.json; do
  out="$ARTIFACTS/$(basename "$pair")"
  docker run --rm "$IMAGE_TAG" cat "/opt/dllm/$pair" > "$out"
  # `docker run ... cat <missing>` yields an empty file, not an error.
  [ -s "$out" ] || { echo "empty extract: $out" >&2; exit 1; }
done

rm -f "$CONFIG_OUT"
uv run python tools/run_openexr_one_day.py finalize-config \
  --template "$TEMPLATE" \
  --output "$CONFIG_OUT" \
  --target-harness-path tools/openexr_runtime/poc_probe.cc \
  --prompt-path prompts/openexr-system.md \
  --remote-code-path "$REMOTE_CODE" \
  --search-binary-path "$ARTIFACTS/poc_probe-asan" \
  --search-image-digest "$DIGEST" \
  --coverage-binary-path "$ARTIFACTS/poc_probe-coverage" \
  --coverage-image-digest "$DIGEST" \
  --coverage-build-manifest-path "$ARTIFACTS/build-manifest.json" \
  --c0-source-path fixtures/openexr/c0-validity-preserving-zero.hex \
  --dictionary-path configs/openexr-v3.4.14.dict \
  --artifact-root "$ARTIFACT_ROOT" \
  --config-id "$CONFIG_ID" \
  --qualification-state qualified > /dev/null
echo "== wrote $CONFIG_OUT"
echo "== ubsan-disabled-checks label: $(docker inspect --format='{{index .Config.Labels "org.opencontainers.image.dllm.ubsan-disabled-checks"}}' "$IMAGE_TAG")"
echo "next: tools/start_smoke.sh"
