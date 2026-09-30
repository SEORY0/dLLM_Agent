#!/usr/bin/env bash
# Build the Dream-Coder venv: transformers 4.46.2 (this file's requirements.txt)
# layered on top of the project .venv's torch via a .pth, instead of installing its
# own torch. See docs/superpowers/plans/2026-09-29-trajectory-signal-harness-synthesis.md
# Task 1 and requirements.txt's header comment for why.
#
#   tools/dream_runtime/setup.sh                     # builds tools/dream_runtime/venv
#   DREAM_VENV=/path/to/venv tools/dream_runtime/setup.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PROJECT_SITE_PACKAGES="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
DREAM_VENV="${DREAM_VENV:-tools/dream_runtime/venv}"

uv venv -q --python 3.12 "$DREAM_VENV"

# Write the .pth BEFORE installing: accelerate depends on torch, and without this,
# pip/uv sees no torch in this fresh venv and tries to download its own (huge, and
# the wrong build for this workstation's index setup) instead of reusing the
# project's. The .pth extends this venv's sys.path, so pip's own dependency check
# (which runs inside this same interpreter) sees the project's torch as satisfied.
SITE_PACKAGES="$("$DREAM_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$PROJECT_SITE_PACKAGES" > "$SITE_PACKAGES/zz_project_torch.pth"

uv pip install -q --python "$DREAM_VENV/bin/python" -r tools/dream_runtime/requirements.txt

echo "== verifying torch (shared) + transformers (own) import together"
"$DREAM_VENV/bin/python" -c '
import torch, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
'
echo "== wrote $DREAM_VENV, sharing torch from $PROJECT_SITE_PACKAGES"
