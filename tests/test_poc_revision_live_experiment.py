"""Live-smoke PoC revision experiment tests."""

import hashlib
from pathlib import Path

import pytest
from tools import analyze_poc_revision_experiment as analyzer
from tools import poc_revision_experiment_live as live
from tools import run_poc_revision_experiment as runner
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.poc_revision_experiment_json import (
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
    def __init__(self, returncode: int) -> None:
        self.returncode: int = returncode


class _FakeRevisionRunner:
    def __init__(self, failing_arm: str | None = None) -> None:
        self.argvs: list[tuple[str, ...]] = []
        self.failing_arm: str | None = failing_arm

    def run(
        self,
        argv: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> _FakeCompleted:
        _ = timeout_seconds
        self.argvs.append(argv)
        output_path = Path(argv[argv.index("--output") + 1])
        artifact_dir = Path(argv[argv.index("--artifact-dir") + 1])
        config = parse_json_object(Path(argv[argv.index("--config") + 1]).read_bytes())
        arm = require_str(require_object(config, "run"), "arm")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _ = stdout_path.write_text(f"{arm}\n", encoding="utf-8")
        _ = stderr_path.write_text("", encoding="utf-8")
        if arm == self.failing_arm:
            return _FakeCompleted(7)
        _write_fake_revision(output_path, arm)
        _write_fake_manifest(artifact_dir / "run-manifest.json", arm, argv)
        return _FakeCompleted(0)


def test_live_smoke_plan_writes_five_isolated_commands_and_configs(tmp_path: Path) -> None:
    """Given live smoke inputs, when planned, then five fixed argv commands are emitted."""
    paths = _paths(tmp_path)

    plan = live.plan_live_smoke(paths, runner.build_schedule(repetitions=1, particles=4))

    assert tuple(cell.arm for cell in plan) == runner.ARMS
    assert len({cell.artifact_dir for cell in plan}) == 5
    assert {cell.cache_dir for cell in plan} == {paths.cache_dir}
    assert len({cell.output_path for cell in plan}) == 5
    for cell in plan:
        config = parse_json_object(cell.config_path.read_bytes())
        revision = require_object(config, "revision")
        run = require_object(config, "run")
        assert require_str(run, "arm") == cell.arm
        assert require_int(run, "pseudo_feedback_rotation_seed") == 202608230000
        assert require_int(run, "particle_seed_base") == 202608200000
        assert require_int(run, "proof_seed") == 202608240000
        assert require_int(revision, "particle_count") == 4
        assert require_int(revision, "epochs") == 1
        assert require_int(revision, "initial_steps") == 8
        assert require_int(revision, "revision_steps") == 2
        assert require_int(revision, "probe_seconds") == 1
        assert tuple(require_array(revision, "fuzzer_seeds")) == (202608210000, 202608210001)
        assert require_int(revision, "proof_seconds") == 10
        assert cell.argv == _argv(paths, cell)


def test_live_smoke_fake_subprocess_reports_are_analyzed(tmp_path: Path) -> None:
    """Given a fake subprocess, when live smoke runs, then analyzer consumes real files."""
    paths = _paths(tmp_path)
    fake = _FakeRevisionRunner()

    summary = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=fake,
    )
    analyzed = analyzer.analyze_experiment(paths.output_root)

    assert len(fake.argvs) == 5
    assert summary["mode"] == "live-smoke"
    assert analyzed["mode"] == "live-smoke"
    assert analyzed["go_no_go"] == "inconclusive"
    assert "insufficient_paired_repetitions" in require_array(analyzed, "verdict_reasons")
    assert (paths.output_root / "paired-summary.json").is_file()
    assert (paths.output_root / "rep-000/selective-revision/arm-report.json").is_file()
    assert require_object(analyzed, "contrasts")["selective-revision_vs_pseudo-feedback"]


def test_analyzer_rejects_live_arm_report_metric_tamper(tmp_path: Path) -> None:
    """Given tampered live metrics, when analyzed, then revision-report derivation wins."""
    paths = _paths(tmp_path)
    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_FakeRevisionRunner(),
    )
    report_path = paths.output_root / "rep-000/selective-revision/arm-report.json"
    report = parse_json_object(report_path.read_bytes())
    require_object(report, "metrics")["validated_success"] = False
    _ = report_path.write_bytes(stable_json(report) + b"\n")

    with pytest.raises(analyzer.ExperimentAnalysisError, match="metrics mismatch"):
        _ = analyzer.analyze_experiment(paths.output_root)


def test_analyzer_rejects_live_raw_reports_hash_tamper(tmp_path: Path) -> None:
    """Given a tampered raw report stream, when analyzed, then index binding rejects it."""
    paths = _paths(tmp_path)
    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_FakeRevisionRunner(),
    )
    _ = (paths.output_root / "raw-arm-reports.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(analyzer.ExperimentAnalysisError, match="raw-arm-reports"):
        _ = analyzer.analyze_experiment(paths.output_root)


def test_live_schedule_records_only_actually_consumed_seed_fields(tmp_path: Path) -> None:
    """Given current command schema, when live smoke runs, then seed use is explicit."""
    paths = _paths(tmp_path)

    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_FakeRevisionRunner(),
    )

    manifest = parse_json_object(
        (paths.output_root / "rep-000/selective-revision/run-manifest.json").read_bytes()
    )
    schedule = require_object(manifest, "schedule")
    assert require_object(schedule, "fuzzer_seeds")["consumed"] is True
    assert require_object(schedule, "permutation_seed")["consumed"] is True
    assert require_object(schedule, "retrieval_seed")["consumed"] is False
    particle_seed = require_object(schedule, "particle_seed_base")
    proof_seed = require_object(schedule, "proof_seed")
    assert particle_seed["consumed"] is True
    assert particle_seed["component"] == "run.particle_seed_base"
    assert proof_seed["consumed"] is True
    assert proof_seed["component"] == "run.proof_seed"


def test_analyzer_rejects_live_nested_seed_mismatch_after_hash_update(tmp_path: Path) -> None:
    """Given hash-valid nested CLI manifest seed drift, when analyzed, then it fails."""
    paths = _paths(tmp_path)
    _ = live.run_live_smoke(
        paths,
        runner.build_schedule(repetitions=1, particles=4),
        command_runner=_FakeRevisionRunner(),
    )
    arm_dir = paths.output_root / "rep-000/selective-revision"
    cli_manifest_path = arm_dir / "artifacts/run-manifest.json"
    cli_manifest = parse_json_object(cli_manifest_path.read_bytes())
    cli_manifest["proof_seed"] = 1
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

    with pytest.raises(analyzer.ExperimentAnalysisError, match="proof seed mismatch"):
        _ = analyzer.analyze_experiment(paths.output_root)


def test_live_smoke_fails_closed_on_subprocess_error(tmp_path: Path) -> None:
    """Given a subprocess failure, when live smoke runs, then no success is reported."""
    with pytest.raises(live.LiveExperimentError, match="feedback-no-remask"):
        _ = live.run_live_smoke(
            _paths(tmp_path),
            runner.build_schedule(repetitions=1, particles=4),
            command_runner=_FakeRevisionRunner(failing_arm="feedback-no-remask"),
        )


def _paths(tmp_path: Path) -> live.LivePaths:
    return live.LivePaths(
        agent_executable=ROOT / ".venv/bin/dllm-agent",
        base_config=BASE_COMMAND_CONFIG,
        source=ROOT / "vendor/openexr",
        runtime_python=ROOT / ".venv/bin/python",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "live",
    )


def _argv(paths: live.LivePaths, cell: live.PlannedCell) -> tuple[str, ...]:
    return (
        str(paths.agent_executable),
        "poc-revision",
        "--config",
        str(cell.config_path),
        "--source",
        str(paths.source),
        "--runtime-python",
        str(paths.runtime_python),
        "--cache-dir",
        str(cell.cache_dir),
        "--artifact-dir",
        str(cell.artifact_dir),
        "--output",
        str(cell.output_path),
    )


def _write_fake_revision(output_path: Path, arm: str) -> None:
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


def _write_fake_manifest(path: Path, arm: str, argv: tuple[str, ...]) -> None:
    config = parse_json_object(Path(argv[argv.index("--config") + 1]).read_bytes())
    revision = require_object(config, "revision")
    run = require_object(config, "run")
    initial_steps = 0 if arm == "libfuzzer-only" else require_int(revision, "initial_steps")
    revision_steps = 0 if arm == "libfuzzer-only" else require_int(revision, "revision_steps")
    _ = path.write_bytes(
        stable_json(
            {
                "schema_version": "dllm-poc-revision-run-manifest/v1",
                "arm": arm,
                "config_path": str(Path(argv[argv.index("--config") + 1])),
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    return evidence_tree_sha256(root)
