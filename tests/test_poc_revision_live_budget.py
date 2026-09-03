"""Configurable live-budget regression tests."""

import hashlib
from pathlib import Path

import pytest
from tools import analyze_poc_revision_experiment as analyzer
from tools import poc_revision_experiment_cli as cli
from tools import poc_revision_experiment_live as live
from tools import run_poc_revision_experiment as runner
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_budget import LiveBudget
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


class _FakeCompleted:
    returncode: int

    def __init__(self) -> None:
        self.returncode = 0


class _BudgetRunner:
    def run(
        self,
        argv: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> _FakeCompleted:
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
        return _FakeCompleted()


def test_pilot_budget_reaches_configs_and_outer_manifests(tmp_path: Path) -> None:
    """Given pilot budget flags, when planned and run, then configs and manifests match."""
    paths = _paths(tmp_path)
    budget = LiveBudget(
        epochs=3,
        initial_steps=128,
        revision_steps=16,
        probe_seconds=3,
        proof_seconds=30,
    )

    plan = live.plan_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        budget=budget,
    )
    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        budget=budget,
        command_runner=_BudgetRunner(),
    )

    for cell in plan:
        revision = require_object(parse_json_object(cell.config_path.read_bytes()), "revision")
        assert require_int(revision, "epochs") == 3
        assert require_int(revision, "initial_steps") == 128
        assert require_int(revision, "revision_steps") == 16
        assert require_int(revision, "probe_seconds") == 3
        assert require_int(revision, "proof_seconds") == 30
    manifest = parse_json_object(
        (paths.output_root / "rep-000/selective-revision/run-manifest.json").read_bytes()
    )
    outer_budget = require_object(manifest, "budget")
    assert require_int(outer_budget, "epochs") == 3
    assert require_int(outer_budget, "initial_steps") == 128
    assert require_int(outer_budget, "revision_steps") == 16
    assert require_int(outer_budget, "probe_seconds") == 3
    assert require_int(outer_budget, "proof_seconds") == 30
    assert require_int(outer_budget, "max_model_nfe") == 704


def test_libfuzzer_baseline_records_zero_model_budget(tmp_path: Path) -> None:
    """Given pilot budget, when live CLI reports baseline steps, then outer budget matches."""
    paths = _paths(tmp_path)
    budget = LiveBudget(
        epochs=3,
        initial_steps=128,
        revision_steps=16,
        probe_seconds=3,
        proof_seconds=30,
    )

    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        budget=budget,
        command_runner=_BudgetRunner(),
    )
    analyzed = analyzer.analyze_experiment(paths.output_root)

    baseline = _outer_budget(paths, "libfuzzer-only")
    selective = _outer_budget(paths, "selective-revision")
    assert analyzed["mode"] == "live-smoke"
    assert require_int(baseline, "particles") == 4
    assert require_int(baseline, "epochs") == 3
    assert require_int(baseline, "initial_steps") == 0
    assert require_int(baseline, "revision_steps") == 0
    assert require_int(baseline, "probe_seconds") == 3
    assert require_int(baseline, "proof_seconds") == 30
    assert require_int(baseline, "max_model_nfe") == 0
    assert require_int(selective, "initial_steps") == 128
    assert require_int(selective, "revision_steps") == 16
    assert require_int(selective, "max_model_nfe") == 704


def test_cli_rejects_non_positive_live_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given an invalid live budget flag, when parsed, then the CLI rejects it."""
    with pytest.raises(SystemExit) as error:
        _ = cli.main(["run", "--live", "--output-root", str(tmp_path), "--epochs", "0"])

    assert error.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_analyzer_rejects_hash_valid_nested_budget_drift(tmp_path: Path) -> None:
    """Given hash-valid nested budget drift, when analyzed, then it fails."""
    paths = _paths(tmp_path)
    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_BudgetRunner(),
    )
    arm_dir = paths.output_root / "rep-000/selective-revision"
    cli_manifest_path = arm_dir / "artifacts/run-manifest.json"
    cli_manifest = parse_json_object(cli_manifest_path.read_bytes())
    cli_manifest["proof_seconds"] = 1
    _ = cli_manifest_path.write_bytes(stable_json(cli_manifest) + b"\n")
    outer_manifest_path = arm_dir / "run-manifest.json"
    outer_manifest = parse_json_object(outer_manifest_path.read_bytes())
    require_object(outer_manifest, "cli_run_manifest")["sha256"] = _sha256(cli_manifest_path)
    _ = outer_manifest_path.write_bytes(stable_json(outer_manifest) + b"\n")
    arm_report_path = arm_dir / "arm-report.json"
    arm_report = parse_json_object(arm_report_path.read_bytes())
    require_object(arm_report, "run_manifest")["sha256"] = _sha256(outer_manifest_path)
    require_object(arm_report, "evidence_tree")["sha256"] = _tree_sha256(arm_dir / "artifacts")
    _ = arm_report_path.write_bytes(stable_json(arm_report) + b"\n")

    with pytest.raises(analyzer.ExperimentAnalysisError, match="proof_seconds mismatch"):
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


def _outer_budget(paths: live.LivePaths, arm: str) -> JsonObject:
    manifest = parse_json_object(
        (paths.output_root / "rep-000" / arm / "run-manifest.json").read_bytes()
    )
    return require_object(manifest, "budget")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    return evidence_tree_sha256(root)
