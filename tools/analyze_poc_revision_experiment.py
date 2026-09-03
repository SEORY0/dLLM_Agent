#!/usr/bin/env -S uv run --script
# ruff: noqa: D100,D101,D103
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_core import ARMS
from tools.poc_revision_experiment_json import (
    JsonObject,
    parse_json_object,
    require_bool,
    require_int,
    require_object,
    require_str,
    stable_json,
)
from tools.poc_revision_experiment_metrics import live_metrics_from_revision
from tools.poc_revision_experiment_seed_validation import live_seed_binding_error
from tools.poc_revision_experiment_stats import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    contrast,
    proof_time,
    verdict,
    verdict_reasons,
)

PRIMARY_CONTRAST: Final = "selective-revision_vs_pseudo-feedback"
CONTRASTS: Final = (
    ("selective-revision", "pseudo-feedback"),
    ("selective-revision", "structured-one-shot"),
    ("selective-revision", "feedback-no-remask"),
)
EXPECTED_ARMS: Final = frozenset(ARMS)


class ExperimentAnalysisError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ArmReport:
    arm: str
    repetition: int
    mode: str
    metrics: JsonObject


def analyze_experiment(root: Path) -> JsonObject:
    reports = tuple(_load_reports(root))
    if not reports:
        _fail("no arm reports found")
    summary = analyze(reports)
    _validate_raw_index(root)
    _ = (root / "paired-summary.json").write_bytes(stable_json(summary) + b"\n")
    return summary


def analyze(reports: tuple[ArmReport, ...]) -> JsonObject:
    by_arm = _paired_reports(reports)
    contrasts: JsonObject = {
        f"{left}_vs_{right}": contrast(by_arm, left, right) for left, right in CONTRASTS
    }
    primary = require_object(contrasts, PRIMARY_CONTRAST)
    return {
        "schema_version": "dllm-poc-revision-paired-summary/v1",
        "mode": reports[0].mode,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "contrasts": contrasts,
        "proof_time": proof_time(reports),
        "go_no_go": verdict(primary),
        "verdict_reasons": verdict_reasons(primary),
        "native_hunt_reference": {
            "label": "non-isomorphic engineering reference",
            "included_in_primary_contrast": False,
        },
    }


def _load_reports(root: Path) -> list[ArmReport]:
    return [
        _arm_report(path, _read_json(path))
        for path in sorted(root.glob("rep-*/**/arm-report.json"))
    ]


def _arm_report(path: Path, payload: JsonObject) -> ArmReport:
    _require(
        require_str(payload, "schema_version") == "dllm-poc-revision-arm-report/v1",
        path,
        "schema_version",
    )
    arm = require_str(payload, "arm")
    repetition = require_int(payload, "repetition")
    mode = require_str(payload, "mode")
    budget = require_object(payload, "budget")
    schedule = require_object(payload, "schedule")
    manifest = _validate_ref(path, payload, "run_manifest", "dllm-poc-revision-run-manifest/v1")
    revision = _validate_ref(path, payload, "revision_report", "dllm-poc-revision-report/v1")
    _require(require_str(manifest, "arm") == arm, path, "manifest arm mismatch")
    _require(
        require_int(manifest, "repetition") == repetition, path, "manifest repetition mismatch"
    )
    _require(require_str(manifest, "mode") == mode, path, "manifest mode mismatch")
    _require(require_object(manifest, "budget") == budget, path, "manifest budget mismatch")
    _require(require_object(manifest, "schedule") == schedule, path, "manifest schedule mismatch")
    if mode == "live-smoke":
        cli_manifest = _validate_nested_ref(
            path.parent / require_str(require_object(payload, "run_manifest"), "path"),
            manifest,
            "cli_run_manifest",
            "dllm-poc-revision-run-manifest/v1",
        )
        seed_error = live_seed_binding_error(schedule, cli_manifest)
        _require(seed_error is None, path, seed_error or "live seed binding")
        _validate_live_budget(path, budget, cli_manifest)
    _validate_evidence(path, payload)
    metrics = require_object(payload, "metrics")
    _ = require_bool(metrics, "validated_success")
    if mode == "live-smoke":
        derived = live_metrics_from_revision(revision)
        _require(metrics == derived, path, "live metrics mismatch")
        metrics = derived
    return ArmReport(arm=arm, repetition=repetition, mode=mode, metrics=metrics)


def _validate_live_budget(path: Path, budget: JsonObject, cli_manifest: JsonObject) -> None:
    particles = require_int(budget, "particles")
    particle_match = require_int(cli_manifest, "particle_count") == particles
    _require(particle_match, path, "particle_count mismatch")
    for field in ("epochs", "initial_steps", "revision_steps", "probe_seconds", "proof_seconds"):
        field_match = require_int(cli_manifest, field) == require_int(budget, field)
        _require(field_match, path, f"{field} mismatch")
    step_nfe = require_int(budget, "initial_steps")
    step_nfe += require_int(budget, "epochs") * require_int(budget, "revision_steps")
    expected_nfe = particles * step_nfe
    _require(require_int(budget, "max_model_nfe") == expected_nfe, path, "max_model_nfe mismatch")


def _validate_raw_index(root: Path) -> None:
    for name, schema in (
        ("fixture-index.json", "dllm-poc-revision-fixture-index/v1"),
        ("live-index.json", "dllm-poc-revision-live-index/v1"),
    ):
        index_path = root / name
        if index_path.is_file():
            index = _read_json(index_path)
            _require(require_str(index, "schema_version") == schema, index_path, "schema_version")
            raw_path = root / "raw-arm-reports.jsonl"
            _require(raw_path.is_file(), index_path, "raw-arm-reports path")
            actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            expected = require_str(index, "raw_reports_sha256")
            _require(actual == expected, index_path, "raw-arm-reports sha256 mismatch")
            if name == "fixture-index.json":
                _validate_fixture_raw_rows(root, raw_path)


def _validate_fixture_raw_rows(root: Path, raw_path: Path) -> None:
    for line in raw_path.read_bytes().splitlines():
        row = parse_json_object(line)
        report_path = root / f"rep-{require_int(row, 'repetition'):03d}" / require_str(row, "arm")
        report_path /= "arm-report.json"
        _require(report_path.is_file(), raw_path, "fixture raw row report path")
        expected = stable_json(row) + b"\n"
        actual = report_path.read_bytes()
        _require(actual == expected, report_path, "fixture raw row mismatch")


def _validate_ref(path: Path, payload: JsonObject, field: str, expected_schema: str) -> JsonObject:
    reference = require_object(payload, field)
    target = path.parent / require_str(reference, "path")
    _require(target.is_file(), path, f"{field} path")
    expected_hash = require_str(reference, "sha256")
    actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    _require(actual_hash == expected_hash, path, f"{field} sha256 mismatch")
    referenced = _read_json(target)
    _require(
        require_str(referenced, "schema_version") == expected_schema,
        target,
        "referenced schema_version",
    )
    return referenced


def _validate_nested_ref(
    source: Path, payload: JsonObject, field: str, expected_schema: str
) -> JsonObject:
    reference = require_object(payload, field)
    target = source.parent / require_str(reference, "path")
    _require(target.is_file(), source, f"{field} path")
    expected_hash = require_str(reference, "sha256")
    actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    _require(actual_hash == expected_hash, source, f"{field} sha256 mismatch")
    referenced = _read_json(target)
    _require(
        require_str(referenced, "schema_version") == expected_schema,
        target,
        "referenced schema_version",
    )
    return referenced


def _validate_evidence(path: Path, payload: JsonObject) -> None:
    evidence = require_object(payload, "evidence_tree")
    evidence_path = path.parent / require_str(evidence, "path")
    _require(evidence_path.is_dir(), path, "evidence_tree path")
    expected = require_str(evidence, "sha256")
    actual = evidence_tree_sha256(evidence_path)
    _require(actual == expected, path, f"evidence_tree sha256 mismatch: {actual} != {expected}")


def _paired_reports(reports: tuple[ArmReport, ...]) -> dict[str, dict[int, ArmReport]]:
    by_arm: dict[str, dict[int, ArmReport]] = {}
    modes = {report.mode for report in reports}
    _require_text(len(modes) == 1, "mixed experiment modes")
    for report in reports:
        cells = by_arm.setdefault(report.arm, {})
        if report.repetition in cells:
            _fail("duplicate arm/repetition cell")
        cells[report.repetition] = report
    _require_text(frozenset(by_arm) == EXPECTED_ARMS, "expected exact five-arm set")
    repetition_sets = {arm: frozenset(cells) for arm, cells in by_arm.items()}
    first = repetition_sets[ARMS[0]]
    for arm, repetitions in repetition_sets.items():
        _require_text(repetitions == first, f"repetition set mismatch for {arm}")
    return by_arm


def _read_json(path: Path) -> JsonObject:
    try:
        return parse_json_object(path.read_bytes())
    except OSError as error:
        raise ExperimentAnalysisError(path, "read failed") from error


def _require(condition: bool, path: Path, detail: str) -> None:
    if not condition:
        raise ExperimentAnalysisError(path, detail)


def _require_text(condition: bool, detail: str) -> None:
    if not condition:
        _fail(detail)


def _fail(message: str) -> NoReturn:
    raise ExperimentAnalysisError(message)


class _Args(argparse.Namespace):
    experiment_root: Path

    def __init__(self) -> None:
        super().__init__()
        self.experiment_root = Path()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args(argv[1:], namespace=_Args())
    try:
        summary = analyze_experiment(args.experiment_root)
    except ExperimentAnalysisError as error:
        _ = sys.stderr.write(f"{error}\n")
        return 2
    digest = hashlib.sha256(stable_json(summary)).hexdigest()
    _ = sys.stdout.write(stable_json({"paired_summary_sha256": digest}).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
