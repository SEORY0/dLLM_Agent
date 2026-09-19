#!/usr/bin/env bash
# Stamp a fresh start time into the finalized one-day config and launch the
# qualification smoke run in a tmux session named smoke-run.
#
# main_start_monotonic_seconds is a *run-time* value (time.monotonic() resets on
# reboot and is unrelated to wall-clock), so it can't live in a git-tracked
# template. The smoke cells must launch just AFTER it (remaining <= 300s), so
# stamp it a few seconds ahead of the model-prep phase.
#
#   tools/start_smoke.sh            # uses the defaults below
#   CONFIG=... SMOKE_ROOT=... tools/start_smoke.sh
set -euo pipefail

CONFIG="${CONFIG:-/data/dllm-agent/openexr-one-day-1gpu.json}"
SMOKE_ROOT="${SMOKE_ROOT:-/data/dllm-agent/openexr-one-day-1gpu/smoke}"
LOG="${LOG:-/data/dllm-agent/logs/smoke-run.log}"
LEAD_SECONDS="${LEAD_SECONDS:-20}"

cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$LOG")"

tmux kill-session -t smoke-run 2>/dev/null || true
rm -rf "$SMOKE_ROOT"

uv run python - "$CONFIG" "$LEAD_SECONDS" <<'PY'
import json
import sys
from pathlib import Path
from time import monotonic

from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256

path, lead = Path(sys.argv[1]), float(sys.argv[2])
cfg = json.loads(path.read_text())
cfg["main_start_monotonic_seconds"] = monotonic() + lead
cfg["config_sha256"] = config_payload_sha256(cfg)
path.write_text(json.dumps(cfg, sort_keys=True, separators=(",", ":")))
print("stamped main_start_monotonic_seconds =", cfg["main_start_monotonic_seconds"])
PY

tmux new-session -d -s smoke-run \
  "cd $PWD && env -u CUDA_VISIBLE_DEVICES uv run python tools/run_openexr_one_day.py smoke-run --config $CONFIG --smoke-root $SMOKE_ROOT 2>&1 | tee $LOG"
echo "launched: tmux attach -t smoke-run   (log: $LOG)"
