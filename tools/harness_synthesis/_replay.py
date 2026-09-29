"""Shared plumbing between the Dream GPU scripts.

Subprocess invocation, trajectory replay through the production
``TrajectoryRecorder``, and code-block extraction — split out of
``dream_backend_smoke.py`` so ``kb_diversity_experiment.py`` (Task 8) does not
duplicate it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dllm_agent.harness_synthesis.trajectory_hook import CommitObservation, TrajectoryRecorder

_REPO_ROOT = Path(__file__).resolve().parents[2]
DREAM_VENV_PYTHON = _REPO_ROOT / "tools/dream_runtime/venv/bin/python"
DREAM_GENERATE = Path(__file__).with_name("_dream_generate.py")
MODEL_PATH = "/data/dllm-agent/cache-dream-coder"


def run_dream_generate(  # noqa: PLR0913 - one CLI-adapter boundary, not worth splitting.
    prompt: str,
    seed: int,
    out_path: Path,
    *,
    gpu: int | None = None,
    new_tokens: int = 256,
    steps: int = 128,
) -> dict[str, Any]:
    """Invoke ``_dream_generate.py`` under the Dream venv and load its JSON output."""
    env = None
    if gpu is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    subprocess.run(  # noqa: S603 - fixed argv (this repo's own script + venv), shell=False.
        [
            str(DREAM_VENV_PYTHON),
            str(DREAM_GENERATE),
            "--model-path",
            MODEL_PATH,
            "--prompt",
            prompt,
            "--seed",
            str(seed),
            "--new-tokens",
            str(new_tokens),
            "--steps",
            str(steps),
            "--out",
            str(out_path),
        ],
        check=True,
        env=env,
    )
    return json.loads(out_path.read_text())


def replay(payload: dict[str, Any]) -> TrajectoryRecorder:
    """Rebuild a real generation's trajectory through the production recorder.

    Raises ``TrajectoryRecorderError`` (propagated, not caught) if the raw events do
    not match the token-id snapshots -- that would mean either this script's own event
    bookkeeping or the production module has a bug, and a diversity/fitness result
    computed on top of a broken trajectory would be worse than no result.
    """
    mask_token_id = int(payload["mask_token_id"])
    recorder = TrajectoryRecorder(mask_token_id=mask_token_id)
    events_by_step: dict[int, list[CommitObservation]] = {}
    for event in payload["events"]:
        events_by_step.setdefault(event["step"], []).append(
            CommitObservation(
                position=event["position"],
                token_id=event["token_id"],
                confidence=event["confidence"],
                entropy=event["entropy"],
                margin=event["margin"],
            )
        )
    for entry in payload["steps_log"]:
        step = entry["step"]
        recorder.observe(step, entry["token_ids"], events_by_step.get(step, []))
    return recorder


def decode_code_block(payload: dict[str, Any]) -> tuple[str, list[tuple[int, int]]]:
    """Return the ```` ```cpp ```` block's text and each covering token's char span."""
    pieces: list[str] = payload["generated_token_pieces"]
    text, offsets = "", []
    for piece in pieces:
        offsets.append((len(text), len(text) + len(piece)))
        text += piece
    end = len(text)
    for stop in ("```", "<|"):
        index = text.find(stop)
        if index != -1:
            end = min(end, index)
    code_offsets = [offset for offset in offsets if offset[0] < end]
    return text[:end], code_offsets
