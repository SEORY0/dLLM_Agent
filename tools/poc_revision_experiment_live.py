# ruff: noqa: D100,D101,D103,TC003

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_budget import DEFAULT_LIVE_BUDGET, LiveBudget
from tools.poc_revision_experiment_core import ARMS, Schedule
from tools.poc_revision_experiment_json import (
    JsonObject,
    ints,
    parse_json_object,
    require_object,
    stable_json,
    strings,
)
from tools.poc_revision_experiment_metrics import live_metrics_from_revision
from tools.poc_revision_experiment_process import (
    CommandRunner,
    CommandTimeoutError,
    SubprocessCommandRunner,
)

LIVE_MODE: Final = "live-smoke"
SMOKE_TIMEOUT_SECONDS: Final = 900


class LiveExperimentError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class LivePaths:
    agent_executable: Path
    base_config: Path
    source: Path
    runtime_python: Path
    cache_dir: Path
    output_root: Path
    timeout_seconds: int = SMOKE_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class PlannedCell:
    arm: str
    repetition: int
    config_path: Path
    cache_dir: Path
    artifact_dir: Path
    output_path: Path
    stdout_path: Path
    stderr_path: Path
    argv: tuple[str, ...]
    budget: LiveBudget


def plan_live_smoke(
    paths: LivePaths,
    schedule: tuple[Schedule, ...],
    budget: LiveBudget = DEFAULT_LIVE_BUDGET,
) -> tuple[PlannedCell, ...]:
    _require(len(ARMS) == 5, "live experiment requires exact five-arm design")  # noqa: PLR2004
    base_config = parse_json_object(paths.base_config.read_bytes())
    cells: list[PlannedCell] = []
    for item in schedule:
        cells.extend(_write_cell_plan(paths, base_config, item, arm, budget) for arm in ARMS)
    return tuple(cells)


def run_live_smoke(
    paths: LivePaths,
    schedule: tuple[Schedule, ...],
    budget: LiveBudget = DEFAULT_LIVE_BUDGET,
    command_runner: CommandRunner | None = None,
) -> JsonObject:
    runner = SubprocessCommandRunner() if command_runner is None else command_runner
    reports: list[JsonObject] = []
    for cell in plan_live_smoke(paths, schedule, budget):
        try:
            result = runner.run(
                cell.argv, cell.stdout_path, cell.stderr_path, paths.timeout_seconds
            )
        except CommandTimeoutError as error:
            raise LiveExperimentError(str(error)) from error
        if result.returncode != 0:
            msg = (
                f"{cell.arm} rep {cell.repetition:03d} failed with return code {result.returncode}"
            )
            raise LiveExperimentError(msg)
        reports.append(_write_arm_report(cell, schedule[cell.repetition]))
    raw = b"".join(stable_json(report) + b"\n" for report in reports)
    _ = (paths.output_root / "raw-arm-reports.jsonl").write_bytes(raw)
    summary: JsonObject = {
        "schema_version": "dllm-poc-revision-live-index/v1",
        "mode": LIVE_MODE,
        "arms": strings(ARMS),
        "raw_reports_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_json(paths.output_root / "live-index.json", summary)
    return summary


def _write_cell_plan(
    paths: LivePaths, base_config: JsonObject, schedule: Schedule, arm: str, budget: LiveBudget
) -> PlannedCell:
    root = paths.output_root / f"rep-{schedule.repetition:03d}" / arm
    artifact_dir = root / "artifacts"
    log_dir = root / "logs"
    for directory in (artifact_dir, log_dir, paths.cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config_path = root / "command-config.json"
    output_path = root / "revision-report.json"
    config = _live_config(base_config, schedule, arm, budget)
    _write_json(config_path, config)
    argv = (
        str(paths.agent_executable),
        "poc-revision",
        "--config",
        str(config_path),
        "--source",
        str(paths.source),
        "--runtime-python",
        str(paths.runtime_python),
        "--cache-dir",
        str(paths.cache_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--output",
        str(output_path),
    )
    return PlannedCell(
        arm=arm,
        repetition=schedule.repetition,
        config_path=config_path,
        cache_dir=paths.cache_dir,
        artifact_dir=artifact_dir,
        output_path=output_path,
        stdout_path=log_dir / "stdout.txt",
        stderr_path=log_dir / "stderr.txt",
        argv=argv,
        budget=budget,
    )


def _live_config(
    base_config: JsonObject, schedule: Schedule, arm: str, budget: LiveBudget
) -> JsonObject:
    config = parse_json_object(stable_json(base_config))
    revision = require_object(config, "revision")
    run = require_object(config, "run")
    revision["particle_count"] = len(schedule.particle_seeds)
    revision["epochs"] = budget.epochs
    revision["initial_steps"] = budget.initial_steps
    revision["revision_steps"] = budget.revision_steps
    revision["probe_seconds"] = budget.probe_seconds
    revision["fuzzer_seeds"] = ints(schedule.fuzzer_seeds)
    revision["proof_seconds"] = budget.proof_seconds
    run["arm"] = arm
    run["pseudo_feedback_rotation_seed"] = schedule.permutation_seed
    run["particle_seed_base"] = schedule.particle_seeds[0]
    run["proof_seed"] = schedule.proof_seed
    return config


def _write_arm_report(cell: PlannedCell, schedule: Schedule) -> JsonObject:
    revision = parse_json_object(cell.output_path.read_bytes())
    cli_manifest = cell.artifact_dir / "run-manifest.json"
    _require(cli_manifest.is_file(), f"{cell.arm} rep {cell.repetition:03d} missing CLI manifest")
    outer_manifest = _outer_manifest(cell, schedule, cli_manifest)
    _write_json(cell.output_path.parent / "run-manifest.json", outer_manifest)
    report: JsonObject = {
        "schema_version": "dllm-poc-revision-arm-report/v1",
        "mode": LIVE_MODE,
        "arm": cell.arm,
        "repetition": cell.repetition,
        "run_manifest": _ref(
            cell.output_path.parent, cell.output_path.parent / "run-manifest.json"
        ),
        "revision_report": _revision_ref(cell),
        "evidence_tree": _tree_ref(cell.output_path.parent, cell.artifact_dir),
        "metrics": live_metrics_from_revision(revision),
        "budget": require_object(outer_manifest, "budget"),
        "schedule": require_object(outer_manifest, "schedule"),
    }
    _write_json(cell.output_path.parent / "arm-report.json", report)
    return report


def _outer_manifest(cell: PlannedCell, schedule: Schedule, cli_manifest: Path) -> JsonObject:
    return {
        "schema_version": "dllm-poc-revision-run-manifest/v1",
        "mode": LIVE_MODE,
        "arm": cell.arm,
        "repetition": cell.repetition,
        "schedule": _schedule_manifest(schedule),
        "budget": cell.budget.manifest(len(schedule.particle_seeds), cell.arm),
        "cli_argv": strings(cell.argv),
        "cli_run_manifest": _ref(cell.output_path.parent, cli_manifest),
    }


def _schedule_manifest(schedule: Schedule) -> JsonObject:
    return {
        "retrieval_seed": _seed_record(
            schedule.retrieval_seed, consumed=False, component="not-applicable"
        ),
        "permutation_seed": _seed_record(
            schedule.permutation_seed,
            consumed=True,
            component="run.pseudo_feedback_rotation_seed",
        ),
        "particle_seed_base": _seed_record(
            schedule.particle_seeds[0],
            consumed=True,
            component="run.particle_seed_base",
        ),
        "proof_seed": _seed_record(
            schedule.proof_seed,
            consumed=True,
            component="run.proof_seed",
        ),
        "particle_seeds": {
            "values": ints(schedule.particle_seeds),
            "consumed": True,
            "component": "derived:run.particle_seed_base",
        },
        "fuzzer_seeds": {
            "values": ints(schedule.fuzzer_seeds),
            "consumed": True,
            "component": "revision.fuzzer_seeds",
        },
        "pseudo_permutation": {"values": ints(schedule.pseudo_permutation), "consumed": True},
    }


def _seed_record(value: int, *, consumed: bool, component: str) -> JsonObject:
    return {"value": value, "consumed": consumed, "component": component}


def _revision_ref(cell: PlannedCell) -> JsonObject:
    return {
        "path": cell.output_path.name,
        "sha256": _file_sha256(cell.output_path),
        "schema_version": "dllm-poc-revision-report/v1",
    }


def _ref(base: Path, target: Path) -> JsonObject:
    return {"path": target.relative_to(base).as_posix(), "sha256": _file_sha256(target)}


def _tree_ref(base: Path, target: Path) -> JsonObject:
    return {"path": target.relative_to(base).as_posix(), "sha256": evidence_tree_sha256(target)}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: JsonObject) -> None:
    _ = path.write_bytes(stable_json(payload) + b"\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _fail(message: str) -> NoReturn:
    raise LiveExperimentError(message)
