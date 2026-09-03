# ruff: noqa: D100,D103

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools import analyze_poc_revision_experiment as analyzer
from tools import poc_revision_experiment_live as live
from tools.poc_revision_experiment_budget import DEFAULT_LIVE_BUDGET, LiveBudget
from tools.poc_revision_experiment_core import build_schedule, write_fixture_experiment
from tools.poc_revision_experiment_json import stable_json


class _Args(argparse.Namespace):
    dry_run_fixture: bool
    live: bool
    output_root: Path
    agent_executable: Path
    base_config: Path
    source: Path
    runtime_python: Path
    cache_dir: Path
    timeout_seconds: int
    repetitions: int
    particles: int
    epochs: int
    initial_steps: int
    revision_steps: int
    probe_seconds: int
    proof_seconds: int

    def __init__(self) -> None:
        super().__init__()
        self.dry_run_fixture = False
        self.live = False
        self.output_root = Path()
        self.agent_executable = Path(".venv/bin/dllm-agent")
        self.base_config = Path("configs/openexr-v3.4.14-poc-revision.json")
        self.source = Path("vendor/openexr")
        self.runtime_python = Path(".venv/bin/python")
        self.cache_dir = Path(".cache/poc-revision-live")
        self.timeout_seconds = 900
        self.repetitions = 5
        self.particles = 4
        self.epochs = DEFAULT_LIVE_BUDGET.epochs
        self.initial_steps = DEFAULT_LIVE_BUDGET.initial_steps
        self.revision_steps = DEFAULT_LIVE_BUDGET.revision_steps
        self.probe_seconds = DEFAULT_LIVE_BUDGET.probe_seconds
        self.proof_seconds = DEFAULT_LIVE_BUDGET.proof_seconds


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    defaults = _Args()
    _ = parser.add_argument("--dry-run-fixture", action="store_true")
    _ = parser.add_argument("--live", action="store_true")
    _ = parser.add_argument("--output-root", type=Path, required=True)
    _ = parser.add_argument("--agent-executable", type=Path, default=defaults.agent_executable)
    _ = parser.add_argument("--base-config", type=Path, default=defaults.base_config)
    _ = parser.add_argument("--source", type=Path, default=defaults.source)
    _ = parser.add_argument("--runtime-python", type=Path, default=defaults.runtime_python)
    _ = parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    _ = parser.add_argument(
        "--timeout-seconds", type=_positive_int, default=defaults.timeout_seconds
    )
    _ = parser.add_argument("--repetitions", type=_positive_int, default=defaults.repetitions)
    _ = parser.add_argument("--particles", type=_positive_int, default=defaults.particles)
    _ = parser.add_argument("--epochs", type=_positive_int, default=defaults.epochs)
    _ = parser.add_argument("--initial-steps", type=_positive_int, default=defaults.initial_steps)
    _ = parser.add_argument("--revision-steps", type=_positive_int, default=defaults.revision_steps)
    _ = parser.add_argument("--probe-seconds", type=_positive_int, default=defaults.probe_seconds)
    _ = parser.add_argument("--proof-seconds", type=_positive_int, default=defaults.proof_seconds)
    args = parser.parse_args(argv[1:], namespace=_Args())
    if args.live == args.dry_run_fixture:
        parser.error("pass exactly one of --dry-run-fixture or --live")
    if args.live:
        summary = live.run_live_smoke(
            live.LivePaths(
                agent_executable=args.agent_executable,
                base_config=args.base_config,
                source=args.source,
                runtime_python=args.runtime_python,
                cache_dir=args.cache_dir,
                output_root=args.output_root,
                timeout_seconds=args.timeout_seconds,
            ),
            build_schedule(args.repetitions, args.particles),
            budget=LiveBudget(
                epochs=args.epochs,
                initial_steps=args.initial_steps,
                revision_steps=args.revision_steps,
                probe_seconds=args.probe_seconds,
                proof_seconds=args.proof_seconds,
            ),
        )
        summary = analyzer.analyze_experiment(args.output_root)
    else:
        summary = write_fixture_experiment(args.output_root, args.repetitions, args.particles)
    _ = sys.stdout.write(stable_json(summary).decode("utf-8") + "\n")
    return 0


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        msg = f"{raw!r} is not a positive integer"
        raise argparse.ArgumentTypeError(msg) from error
    if value <= 0:
        msg = f"{raw!r} is not a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return value
