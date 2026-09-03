"""PoC revision evidence tree hash contract tests."""

from pathlib import Path

import pytest
from tools import analyze_poc_revision_experiment as analyzer
from tools import poc_revision_experiment_live as live
from tools import run_poc_revision_experiment as runner
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_json import (
    JsonObject,
    parse_json_object,
    require_array,
    require_int,
    require_object,
    require_str,
    stable_json,
)

ROOT = Path(__file__).parents[1]
BASE_COMMAND_CONFIG = ROOT / "configs/openexr-v3.4.14-poc-revision.json"


class _Completed:
    returncode: int

    def __init__(self) -> None:
        self.returncode = 0


class _RevisionRunner:
    def run(
        self,
        argv: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> _Completed:
        _ = timeout_seconds
        output_path = Path(argv[argv.index("--output") + 1])
        artifact_dir = Path(argv[argv.index("--artifact-dir") + 1])
        config = parse_json_object(Path(argv[argv.index("--config") + 1]).read_bytes())
        arm = require_str(require_object(config, "run"), "arm")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _ = stdout_path.write_text(f"{arm}\n", encoding="utf-8")
        _ = stderr_path.write_text("", encoding="utf-8")
        _write_revision(output_path, arm)
        _write_manifest(artifact_dir / "run-manifest.json", arm, argv)
        return _Completed()


def test_fixture_arm_report_uses_public_evidence_tree_hash_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    """Given fixture output, when evidence is hashed, then the public tool matches."""
    root = tmp_path / "fixture"

    _ = runner.write_fixture_experiment(root, repetitions=1, particles=4)
    analyzed = analyzer.analyze_experiment(root)

    arm_dir = root / "rep-000/selective-revision"
    evidence_root = arm_dir / "evidence"
    report = parse_json_object((arm_dir / "arm-report.json").read_bytes())
    evidence = require_object(report, "evidence_tree")
    assert analyzed["mode"] == runner.MODE
    assert require_str(evidence, "sha256") == evidence_tree_sha256(evidence_root)
    _ = (evidence_root / "tamper.txt").write_text("tamper\n", encoding="utf-8")
    with pytest.raises(analyzer.ExperimentAnalysisError, match="evidence_tree sha256"):
        _ = analyzer.analyze_experiment(root)


def test_live_arm_report_uses_public_evidence_tree_hash_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    """Given live output, when evidence is hashed, then the public tool matches."""
    paths = _paths(tmp_path)

    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_RevisionRunner(),
    )
    analyzed = analyzer.analyze_experiment(paths.output_root)

    arm_dir = paths.output_root / "rep-000/selective-revision"
    evidence_root = arm_dir / "artifacts"
    report = parse_json_object((arm_dir / "arm-report.json").read_bytes())
    evidence = require_object(report, "evidence_tree")
    assert analyzed["mode"] == "live-smoke"
    assert require_str(evidence, "sha256") == evidence_tree_sha256(evidence_root)
    _ = (evidence_root / "tamper.txt").write_text("tamper\n", encoding="utf-8")
    with pytest.raises(analyzer.ExperimentAnalysisError, match="evidence_tree sha256"):
        _ = analyzer.analyze_experiment(paths.output_root)


def _paths(tmp_path: Path) -> live.LivePaths:
    return live.LivePaths(
        agent_executable=ROOT / ".venv/bin/dllm-agent",
        base_config=BASE_COMMAND_CONFIG,
        source=ROOT / "vendor/openexr",
        runtime_python=ROOT / ".venv/bin/python",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "live",
    )


def _write_revision(output_path: Path, arm: str) -> None:
    _ = output_path.write_bytes(
        stable_json(
            {
                "schema_version": "dllm-poc-revision-report/v1",
                "config_hash_sha256": "0" * 64,
                "snapshots": [],
                "probe_feedback": [],
                "field_attributions": [],
                "remask_decisions": [],
                "revision_events_sha256": "1" * 64,
                "accounting_rejected": 0,
                "accounting_dedup": 0,
                "proof": {
                    "oracle_status": "proved" if arm == "selective-revision" else "skipped",
                    "input_sha256": None,
                    "proof_log_sha256": None,
                },
            }
        )
        + b"\n"
    )


def _write_manifest(path: Path, arm: str, argv: tuple[str, ...]) -> None:
    config = parse_json_object(Path(argv[argv.index("--config") + 1]).read_bytes())
    revision = require_object(config, "revision")
    run = require_object(config, "run")
    initial_steps, revision_steps = _model_steps(arm, revision)
    _ = path.write_bytes(
        stable_json(
            {
                "schema_version": "dllm-poc-revision-run-manifest/v1",
                "arm": arm,
                "fuzzer_seeds": require_array(revision, "fuzzer_seeds"),
                "particle_count": require_int(revision, "particle_count"),
                "epochs": require_int(revision, "epochs"),
                "initial_steps": initial_steps,
                "revision_steps": revision_steps,
                "probe_seconds": require_int(revision, "probe_seconds"),
                "particle_seed_base": require_int(run, "particle_seed_base"),
                "proof_seed": require_int(run, "proof_seed"),
                "proof_seconds": require_int(revision, "proof_seconds"),
            }
        )
        + b"\n"
    )


def _model_steps(arm: str, revision: JsonObject) -> tuple[int, int]:
    if arm == "libfuzzer-only":
        return (0, 0)
    return (require_int(revision, "initial_steps"), require_int(revision, "revision_steps"))
