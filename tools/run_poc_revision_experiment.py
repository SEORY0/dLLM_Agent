#!/usr/bin/env -S uv run --script
# ruff: noqa: D100,D103
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import poc_revision_experiment_core as _core

ARMS = _core.ARMS
BOOTSTRAP_SEED = _core.BOOTSTRAP_SEED
IMAGE_PIN = _core.IMAGE_PIN
MAX_MODEL_NFE = _core.MAX_MODEL_NFE
MIN_PSEUDO_PARTICLES = _core.MIN_PSEUDO_PARTICLES
MODE = _core.MODE
MODEL_PIN = _core.MODEL_PIN
NATIVE_REFERENCE = _core.NATIVE_REFERENCE
TARGET_PIN = _core.TARGET_PIN
Schedule = _core.Schedule
build_schedule = _core.build_schedule
cyclic_derangement = _core.cyclic_derangement
reference_manifest = _core.reference_manifest
write_fixture_experiment = _core.write_fixture_experiment

__all__ = (
    "ARMS",
    "BOOTSTRAP_SEED",
    "IMAGE_PIN",
    "MAX_MODEL_NFE",
    "MIN_PSEUDO_PARTICLES",
    "MODE",
    "MODEL_PIN",
    "NATIVE_REFERENCE",
    "TARGET_PIN",
    "Schedule",
    "build_schedule",
    "cyclic_derangement",
    "main",
    "reference_manifest",
    "write_fixture_experiment",
)


def main(argv: list[str]) -> int:
    from tools.poc_revision_experiment_cli import main as cli_main  # noqa: PLC0415

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
