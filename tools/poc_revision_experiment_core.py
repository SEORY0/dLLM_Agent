#!/usr/bin/env -S uv run --script
# ruff: noqa: D100,D101,D103
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_json import (
    JsonObject,
    ints,
    require_object,
    stable_json,
    strings,
)

ARMS: Final = (
    "libfuzzer-only",
    "structured-one-shot",
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
)
NATIVE_REFERENCE: Final = "native-hunt-reference"
BOOTSTRAP_SEED: Final = 202608209999
MODE: Final = "fixture-validation-not-model-efficacy"
TARGET_PIN: Final = "AcademySoftwareFoundation/openexr@v3.4.14"
MODEL_PIN: Final = (
    "ByteDance-Seed/Stable-DiffCoder-8B-Instruct@10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
)
IMAGE_PIN: Final = "sha256:7591279f51256ef5e80341c788d31145ecf1dd0f7a64db7f0cdeaba527ff5add"
MIN_PSEUDO_PARTICLES: Final = 2
MAX_MODEL_NFE: Final = 704


@dataclass(frozen=True, slots=True)
class Schedule:
    repetition: int
    particle_seeds: tuple[int, ...]
    fuzzer_seeds: tuple[int, int]
    retrieval_seed: int
    permutation_seed: int
    proof_seed: int
    pseudo_permutation: tuple[int, ...]


def build_schedule(repetitions: int = 5, particles: int = 4) -> tuple[Schedule, ...]:
    if repetitions <= 0 or particles < MIN_PSEUDO_PARTICLES:
        msg = "repetitions must be positive and particles must be at least two"
        raise ValueError(msg)
    return tuple(
        Schedule(
            repetition=repetition,
            particle_seeds=tuple(
                202608200000 + (1000 * repetition) + particle for particle in range(particles)
            ),
            fuzzer_seeds=(202608210000 + (1000 * repetition), 202608210001 + (1000 * repetition)),
            retrieval_seed=202608220000 + repetition,
            permutation_seed=202608230000 + repetition,
            proof_seed=202608240000 + repetition,
            pseudo_permutation=cyclic_derangement(particles, 202608230000 + repetition),
        )
        for repetition in range(repetitions)
    )


def cyclic_derangement(size: int, seed: int) -> tuple[int, ...]:
    if size < MIN_PSEUDO_PARTICLES:
        msg = "pseudo-feedback requires at least two particles"
        raise ValueError(msg)
    offset = (seed % (size - 1)) + 1
    return tuple((index + offset) % size for index in range(size))


def write_fixture_experiment(
    output_root: Path, repetitions: int = 5, particles: int = 4
) -> JsonObject:
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[JsonObject] = [
        _write_arm_fixture(output_root, schedule, arm)
        for schedule in build_schedule(repetitions, particles)
        for arm in ARMS
    ]
    raw = b"".join(stable_json(report) + b"\n" for report in reports)
    _ = (output_root / "raw-arm-reports.jsonl").write_bytes(raw)
    summary: JsonObject = {
        "schema_version": "dllm-poc-revision-fixture-index/v1",
        "mode": MODE,
        "repetitions": repetitions,
        "particles": particles,
        "arms": strings(ARMS),
        "raw_reports_sha256": hashlib.sha256(raw).hexdigest(),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    _write_json(output_root / "fixture-index.json", summary)
    return summary


def reference_manifest(schedule: Schedule) -> JsonObject:
    return _manifest(schedule, NATIVE_REFERENCE)


def _manifest(schedule: Schedule, arm: str) -> JsonObject:
    arm_dir = Path("ARTIFACT_ROOT") / f"rep-{schedule.repetition:03d}" / arm
    model_nfe = 0 if arm in ("libfuzzer-only", NATIVE_REFERENCE) else MAX_MODEL_NFE
    return {
        "schema_version": "dllm-poc-revision-run-manifest/v1",
        "mode": MODE,
        "arm": arm,
        "repetition": schedule.repetition,
        "reference_kind": "non-isomorphic" if arm == NATIVE_REFERENCE else None,
        "target_pin": TARGET_PIN,
        "model_pin": MODEL_PIN,
        "image_pin": IMAGE_PIN,
        "schedule": {
            "retrieval_seed": schedule.retrieval_seed,
            "permutation_seed": schedule.permutation_seed,
            "proof_seed": schedule.proof_seed,
            "particle_seeds": ints(schedule.particle_seeds),
            "fuzzer_seeds": ints(schedule.fuzzer_seeds),
            "pseudo_permutation": ints(schedule.pseudo_permutation),
        },
        "budget": {
            "particles": len(schedule.particle_seeds),
            "initial_nfe_per_particle": 128,
            "epochs": 3,
            "revision_nfe_per_particle_epoch": 16,
            "max_model_nfe": model_nfe,
            "probe_seconds": 3,
            "probes_per_snapshot": 2,
            "requested_probe_seconds": 96,
            "requested_fuzz_seconds": 126,
            "proof_seconds": 30,
            "requested_fuzz_proof_seconds": 126,
            "setup_seconds": 0,
            "inclusive_requested_seconds": 126,
        },
        "cli_argv": strings(
            (
                "dllm-agent",
                "poc-revision",
                "--config",
                "configs/openexr-v3.4.14-poc-experiment.json",
                "--arm",
                arm,
                "--repetition",
                str(schedule.repetition),
                "--output",
                str(arm_dir),
            )
        ),
    }


def _write_arm_fixture(output_root: Path, schedule: Schedule, arm: str) -> JsonObject:
    arm_dir = output_root / f"rep-{schedule.repetition:03d}" / arm
    evidence_dir = arm_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(schedule, arm)
    revision = _revision_report(schedule, arm)
    _write_json(arm_dir / "run-manifest.json", manifest)
    _write_json(arm_dir / "revision-report.json", revision)
    _ = (evidence_dir / "probe.log").write_text(
        f"{arm} rep {schedule.repetition}\n", encoding="utf-8"
    )
    manifest_hash = _file_sha256(arm_dir / "run-manifest.json")
    report_hash = _file_sha256(arm_dir / "revision-report.json")
    evidence_hash = evidence_tree_sha256(evidence_dir)
    budget = require_object(manifest, "budget")
    schedule_payload = require_object(manifest, "schedule")
    run_manifest_ref: JsonObject = {"path": "run-manifest.json", "sha256": manifest_hash}
    revision_report_ref: JsonObject = {
        "path": "revision-report.json",
        "sha256": report_hash,
        "schema_version": "dllm-poc-revision-report/v1",
    }
    evidence_tree_ref: JsonObject = {"path": "evidence", "sha256": evidence_hash}
    report: JsonObject = {
        "schema_version": "dllm-poc-revision-arm-report/v1",
        "mode": MODE,
        "arm": arm,
        "repetition": schedule.repetition,
        "run_manifest": run_manifest_ref,
        "revision_report": revision_report_ref,
        "evidence_tree": evidence_tree_ref,
        "metrics": _metrics(schedule.repetition, arm),
        "budget": budget,
        "schedule": schedule_payload,
    }
    _write_json(arm_dir / "arm-report.json", report)
    return report


def _revision_report(schedule: Schedule, arm: str) -> JsonObject:
    status = "proved" if _metrics(schedule.repetition, arm)["proof_observed"] else "skipped"
    return {
        "schema_version": "dllm-poc-revision-report/v1",
        "config_hash_sha256": _hash_text(f"{arm}:{schedule.repetition}:config"),
        "snapshots": [],
        "probe_feedback": [],
        "field_attributions": [],
        "remask_decisions": [],
        "revision_events_sha256": _hash_text(f"{arm}:{schedule.repetition}:events"),
        "accounting_rejected": 0,
        "accounting_dedup": 0,
        "proof": {"oracle_status": status, "input_sha256": None, "proof_log_sha256": None},
    }


def _metrics(repetition: int, arm: str) -> JsonObject:
    success = arm == "selective-revision"
    proof_time = 18 + (repetition % 5) if success else None
    target_branches = 3 + repetition if success else repetition % 3
    return {
        "validated_success": success,
        "target_branches": target_branches,
        "proof_observed": success,
        "proof_time_seconds": proof_time,
        "proof_censored_seconds": 30 if proof_time is None else None,
    }


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: JsonObject) -> None:
    _ = path.write_bytes(stable_json(payload) + b"\n")
