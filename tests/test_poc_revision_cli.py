"""Installed PoC revision CLI/config tests."""

import hashlib
import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Literal, Self

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from dllm_agent import cli
from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.command import (
    PocRevisionCommandConfig,
    RunManifest,
    VerifiedPocRevisionCommand,
    WorkerReadyProvenance,
    build_run_manifest,
    load_poc_revision_command,
)
from dllm_agent.revision.controller_models import (
    ProbeRunner,
    ProofRunner,
    RevisionRunRequest,
    RevisionSampler,
)
from dllm_agent.revision.models import PocRevisionReport, RevisionConfig
from dllm_agent.runtime.revision_process import revision_worker_command
from dllm_agent.targets.models import TargetEvidence
from dllm_agent.targets.openexr.probe import ProbeConfig

ROOT = Path(__file__).parents[1]
RUNNER = CliRunner()
IMAGE_ID = "sha256:7591279f51256ef5e80341c788d31145ecf1dd0f7a64db7f0cdeaba527ff5add"
HARNESS_SHA256 = "4cf61b34216777285bf0ec2deba85611d04208d42151cf07bab4962dc4400f8f"
TARGET_COMMIT = "777d231a179de1711d2a942810d9559216ab3f4c"
SECURITY_SHA256 = "a29644c43663d8cad3a3f2f22fa064f1d138df9873e32e410265756d25755087"
PARTICLE_SEED_BASE = 202_608_200_000
PROOF_SEED = 202_608_240_000
type JsonScalar = str | int | float | bool
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue] | None
type ConfigMutation = Literal["extra", "image", "commit", "security", "escape"]
RUN_MANIFEST_ADAPTER = TypeAdapter(RunManifest)
ZERO_TREE = "0" * 40


def test_poc_revision_help_exposes_required_options() -> None:
    # Given / When
    result = RUNNER.invoke(cli.app, ["poc-revision", "--help"])

    # Then
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--source" in result.stdout
    assert "--runtime-python" in result.stdout
    assert "--cache-dir" in result.stdout
    assert "--artifact-dir" in result.stdout
    assert "--output" in result.stdout


def test_poc_revision_rejects_bad_config_before_artifacts(tmp_path: Path) -> None:
    # Given
    runtime = _executable(tmp_path)
    source = _source_tree(tmp_path)
    config = tmp_path / "bad.json"
    _ = config.write_text("{", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "report.json"

    # When
    result = RUNNER.invoke(
        cli.app,
        [
            "poc-revision",
            "--config",
            str(config),
            "--source",
            str(source),
            "--runtime-python",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--artifact-dir",
            str(artifacts),
            "--output",
            str(output),
        ],
    )

    # Then
    assert result.exit_code == 2
    assert "poc revision config" in result.stderr
    assert not artifacts.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("extra", "extra"),
        ("image", "image digest"),
        ("commit", "target commit"),
        ("security", "SECURITY"),
        ("escape", "escapes"),
    ],
)
def test_poc_revision_rejects_untrusted_config_before_artifacts(
    tmp_path: Path, mutation: ConfigMutation, expected: str
) -> None:
    # Given
    runtime = _executable(tmp_path)
    source = (
        Path("/home/seory0/.cache/dllm-agent/targets/openexr-v3.4.14-pinned")
        if mutation == "security"
        else _source_tree(tmp_path)
    )
    config = tmp_path / "config.json"
    _ = config.write_text(json.dumps(_config(tmp_path, mutation)), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "report.json"

    # When
    result = RUNNER.invoke(
        cli.app,
        [
            "poc-revision",
            "--config",
            str(config),
            "--source",
            str(source),
            "--runtime-python",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--artifact-dir",
            str(artifacts),
            "--output",
            str(output),
        ],
    )

    # Then
    assert result.exit_code == 2
    assert expected in result.stderr
    assert not artifacts.exists()
    assert not output.exists()


def test_poc_revision_writes_report_and_run_manifest_with_injected_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    runtime = _executable(tmp_path)
    source = _source_tree(tmp_path)
    config = tmp_path / "config.json"
    _ = config.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "report.json"
    calls: list[tuple[Path, Path, Path]] = []

    def fake_run(
        config_path: Path,
        source_path: Path,
        runtime_path: Path,
        *,
        cache_dir: Path,
        artifact_dir: Path,
    ) -> tuple[
        PocRevisionReport, VerifiedPocRevisionCommand, WorkerReadyProvenance, tuple[str, ...]
    ]:
        del cache_dir, artifact_dir
        calls.append((config_path, source_path, runtime_path))
        return _report(), _verified_command(), _ready(), ("python", "worker.py")

    monkeypatch.setattr(cli, "run_poc_revision_command", fake_run)

    # When
    result = RUNNER.invoke(
        cli.app,
        [
            "poc-revision",
            "--config",
            str(config),
            "--source",
            str(source),
            "--runtime-python",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--artifact-dir",
            str(artifacts),
            "--output",
            str(output),
        ],
    )

    # Then
    assert result.exit_code == 0
    assert calls == [(config, source, runtime)]
    assert PocRevisionReport.model_validate_json(output.read_bytes()).schema_version == (
        "dllm-poc-revision-report/v1"
    )
    run_manifest = RUN_MANIFEST_ADAPTER.validate_json(
        (artifacts / "run-manifest.json").read_bytes()
    )
    assert run_manifest.schema_version == "dllm-poc-revision-run-manifest/v1"
    assert run_manifest.report_sha256 == sha256_bytes(output.read_bytes())
    assert run_manifest.worker_ready is not None
    assert run_manifest.worker_ready.load_count == 1
    assert run_manifest.worker_argv == ("python", "worker.py")


def test_poc_revision_manifest_binds_verified_source_prompt_budget_and_worker() -> None:
    # Given
    report = _report()
    command = _verified_command()
    ready = _ready()
    worker_args = revision_worker_command(
        Path("/runtime/python"),
        Path("/cache"),
        command.config.model.device,
        command.prompt_path,
        command.prompt_sha256,
    )

    # When
    report_sha256 = sha256_bytes((report.model_dump_json(indent=2) + "\n").encode())
    manifest = build_run_manifest(
        command,
        report_sha256=report_sha256,
        worker_ready=ready,
        worker_argv=worker_args,
    )

    # Then
    assert manifest.report_sha256 == report_sha256
    assert manifest.prompt_path == "prompts/openexr-poc-revision.md"
    assert manifest.prompt_sha256 == command.prompt_sha256
    assert (
        manifest.target.repository_url == "https://github.com/AcademySoftwareFoundation/openexr.git"
    )
    assert manifest.target.tree_oid == ZERO_TREE
    assert manifest.target.source_archive_sha256 == "5" * 64
    assert manifest.model_id == "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
    assert manifest.model_revision == "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
    assert manifest.image_id == IMAGE_ID
    assert manifest.fuzzer_seeds == (20_260_821, 20_260_822)
    assert command.config.run.particle_seed_base == PARTICLE_SEED_BASE
    assert command.config.run.proof_seed == PROOF_SEED
    assert manifest.particle_seed_base == PARTICLE_SEED_BASE
    assert manifest.particle_seeds == (
        PARTICLE_SEED_BASE,
        PARTICLE_SEED_BASE + 1,
        PARTICLE_SEED_BASE + 2,
        PARTICLE_SEED_BASE + 3,
    )
    assert manifest.proof_seed == PROOF_SEED
    assert manifest.initial_steps == 8
    assert manifest.revision_steps == 2
    assert manifest.probe_seconds == 1
    assert manifest.proof_seconds == 10
    assert manifest.worker_argv == worker_args
    assert manifest.worker_argv_sha256 == sha256_bytes(
        json.dumps(worker_args, separators=(",", ":")).encode()
    )
    assert manifest.worker_ready is not None
    assert manifest.worker_ready.load_count == 1


def test_poc_revision_load_returns_verified_prompt_path(tmp_path: Path) -> None:
    # Given
    runtime = _executable(tmp_path)
    source = Path("/home/seory0/.cache/dllm-agent/targets/openexr-v3.4.14-pinned")
    config = tmp_path / "config.json"
    _ = config.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")

    # When
    loaded = load_poc_revision_command(config, source, runtime)

    # Then
    assert loaded.prompt_path == ROOT / "prompts/openexr-poc-revision.md"
    assert loaded.prompt_sha256 == sha256_bytes(loaded.prompt_path.read_bytes())
    assert loaded.config.run.particle_seed_base == PARTICLE_SEED_BASE
    assert loaded.config.run.proof_seed == PROOF_SEED


def test_poc_revision_checked_in_config_exposes_explicit_seed_schedule() -> None:
    # Given
    config_path = ROOT / "configs/openexr-v3.4.14-poc-revision.json"

    # When
    config = PocRevisionCommandConfig.model_validate_json(config_path.read_bytes())
    command = VerifiedPocRevisionCommand(
        config=config,
        config_sha256=sha256_bytes(config_path.read_bytes()),
        prompt_path=ROOT / config.prompt.path,
        prompt_sha256=config.prompt.sha256,
        target=_target_evidence(),
    )
    manifest = build_run_manifest(
        command,
        report_sha256="6" * 64,
        worker_ready=_ready(),
        worker_argv=revision_worker_command(
            Path("/runtime/python"),
            Path("/cache"),
            command.config.model.device,
            command.prompt_path,
            command.prompt_sha256,
        ),
    )

    # Then
    assert isinstance(config.run, RevisionRunRequest)
    assert config.run.particle_seed_base == PARTICLE_SEED_BASE
    assert config.run.proof_seed == PROOF_SEED
    assert manifest.particle_seed_base == config.run.particle_seed_base
    assert manifest.particle_seeds == (
        PARTICLE_SEED_BASE,
        PARTICLE_SEED_BASE + 1,
        PARTICLE_SEED_BASE + 2,
        PARTICLE_SEED_BASE + 3,
    )
    assert manifest.proof_seed == config.run.proof_seed


def test_poc_revision_in_context_setup_error_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    runtime = _executable(tmp_path)
    config = tmp_path / "config.json"
    source = tmp_path / "source"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "report.json"
    _ = config.write_text(json.dumps(_config(tmp_path)), encoding="utf-8")
    _ = source.mkdir()

    class ReadyWorker:
        ready: WorkerReadyProvenance = _ready()

        def __init__(  # noqa: PLR0913, PLR0917
            self,
            runtime_python: Path,
            cache_dir: Path,
            prompt_path: Path,
            prompt_sha256: str,
            device: str,
            timeout_seconds: float,
        ) -> None:
            del runtime_python, cache_dir, prompt_path, prompt_sha256, device, timeout_seconds

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc_value, traceback

    @contextmanager
    def fake_prepare(probe_config: ProbeConfig) -> Generator[_FakeProbe, None, None]:
        del probe_config
        yield _FakeProbe()

    def fake_run_poc_revision(  # noqa: PLR0913, PLR0917
        revision_config: RevisionConfig,
        run_request: RevisionRunRequest,
        sampler: RevisionSampler,
        probe: ProbeRunner,
        proof_runner: ProofRunner,
        artifact_dir: Path,
    ) -> PocRevisionReport:
        del revision_config, run_request, sampler, probe, proof_runner, artifact_dir
        detail = "revision worker returned mismatched request hash"
        raise HuntSetupError(detail)

    def fake_load_poc_revision_command(
        config_path: Path, source_path: Path, runtime_path: Path
    ) -> VerifiedPocRevisionCommand:
        del config_path, source_path, runtime_path
        return _verified_command()

    monkeypatch.setattr(
        cli,
        "load_poc_revision_command",
        fake_load_poc_revision_command,
    )
    worker_module = ModuleType("dllm_agent.runtime.revision_client")
    worker_module.__dict__["RevisionWorkerClient"] = ReadyWorker
    pipeline_module = ModuleType("dllm_agent.revision.pipeline")
    pipeline_module.__dict__["run_poc_revision"] = fake_run_poc_revision
    probe_module = ModuleType("dllm_agent.targets.openexr.probe")
    probe_module.__dict__["prepare_openexr_probe"] = fake_prepare
    proof_module = ModuleType("dllm_agent.targets.openexr.proof")
    proof_module.__dict__["OpenExrProofRunner"] = _FakeProofRunner
    proof_module.__dict__["ProofBoundaryError"] = _FakeProofBoundaryError
    monkeypatch.setitem(sys.modules, "dllm_agent.runtime.revision_client", worker_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.revision.pipeline", pipeline_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.targets.openexr.probe", probe_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.targets.openexr.proof", proof_module)

    # When
    result = RUNNER.invoke(
        cli.app,
        [
            "poc-revision",
            "--config",
            str(config),
            "--source",
            str(source),
            "--runtime-python",
            str(runtime),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--artifact-dir",
            str(artifacts),
            "--output",
            str(output),
        ],
    )

    # Then
    assert result.exit_code == 2
    assert "revision worker returned mismatched request hash" in result.stderr
    assert "TypeError" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
    assert not (artifacts / "run-manifest.json").exists()


def test_libfuzzer_only_command_does_not_construct_revision_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    runtime = _executable(tmp_path)
    source = Path("/home/seory0/.cache/dllm-agent/targets/openexr-v3.4.14-pinned")
    config = tmp_path / "config.json"
    _ = config.write_text(json.dumps(_config(tmp_path, arm="libfuzzer-only")), encoding="utf-8")
    worker_calls: list[tuple[str, ...]] = []

    class FailingWorker:
        def __init__(  # noqa: PLR0913, PLR0917
            self,
            runtime_python: Path,
            cache_dir: Path,
            prompt_path: Path,
            prompt_sha256: str,
            device: str,
            timeout_seconds: float,
        ) -> None:
            del prompt_sha256, timeout_seconds
            worker_calls.append((str(runtime_python), str(cache_dir), str(prompt_path), device))
            detail = "libfuzzer-only must not start worker"
            raise AssertionError(detail)

    @contextmanager
    def fake_prepare(probe_config: ProbeConfig) -> Generator[_FakeProbe, None, None]:
        del probe_config
        yield _FakeProbe()

    def fake_run_poc_revision(  # noqa: PLR0913, PLR0917
        revision_config: RevisionConfig,
        run_request: RevisionRunRequest,
        sampler: RevisionSampler,
        probe: ProbeRunner,
        proof_runner: ProofRunner,
        artifact_dir: Path,
    ) -> PocRevisionReport:
        del revision_config, run_request, sampler, probe, proof_runner, artifact_dir
        return _report()

    def fake_revision_worker_command(
        runtime_python: Path,
        cache_dir: Path,
        device: str,
        prompt_path: Path,
        prompt_sha256: str,
    ) -> tuple[str, ...]:
        del runtime_python, cache_dir, device, prompt_path, prompt_sha256
        detail = "libfuzzer-only must not build worker argv"
        raise AssertionError(detail)

    worker_module = ModuleType("dllm_agent.runtime.revision_client")
    worker_module.__dict__["RevisionWorkerClient"] = FailingWorker
    process_module = ModuleType("dllm_agent.runtime.revision_process")
    process_module.__dict__["revision_worker_command"] = fake_revision_worker_command
    pipeline_module = ModuleType("dllm_agent.revision.pipeline")
    pipeline_module.__dict__["run_poc_revision"] = fake_run_poc_revision
    probe_module = ModuleType("dllm_agent.targets.openexr.probe")
    probe_module.__dict__["prepare_openexr_probe"] = fake_prepare
    proof_module = ModuleType("dllm_agent.targets.openexr.proof")
    proof_module.__dict__["OpenExrProofRunner"] = _FakeProofRunner
    proof_module.__dict__["ProofBoundaryError"] = _FakeProofBoundaryError
    monkeypatch.setitem(sys.modules, "dllm_agent.runtime.revision_client", worker_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.runtime.revision_process", process_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.revision.pipeline", pipeline_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.targets.openexr.probe", probe_module)
    monkeypatch.setitem(sys.modules, "dllm_agent.targets.openexr.proof", proof_module)

    # When
    report, command, ready, argv = cli.run_poc_revision_command(
        config,
        source,
        runtime,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    manifest = build_run_manifest(
        command,
        report_sha256=sha256_bytes((report.model_dump_json(indent=2) + "\n").encode()),
        worker_ready=ready,
        worker_argv=argv,
    )

    # Then
    assert worker_calls == []
    assert argv == ()
    assert ready is None
    assert manifest.model_worker_started is False
    assert manifest.model_timeout_seconds == 0.0
    assert manifest.initial_steps == 0
    assert manifest.revision_steps == 0
    assert manifest.particle_seed_base == PARTICLE_SEED_BASE
    assert manifest.particle_seeds == (
        PARTICLE_SEED_BASE,
        PARTICLE_SEED_BASE + 1,
        PARTICLE_SEED_BASE + 2,
        PARTICLE_SEED_BASE + 3,
    )
    assert manifest.proof_seed == PROOF_SEED


def _config(
    tmp_path: Path,
    mutation: ConfigMutation | None = None,
    *,
    arm: str = "selective-revision",
) -> dict[str, JsonValue]:
    del tmp_path
    prompt_sha256 = hashlib.sha256(
        (ROOT / "prompts/openexr-poc-revision.md").read_bytes()
    ).hexdigest()
    config: dict[str, JsonValue] = {
        "schema_version": "dllm-poc-revision-command/v1",
        "revision": {
            "schema_version": "dllm-poc-revision-config/v1",
            "particle_count": 4,
            "epochs": 1,
            "initial_steps": 8,
            "revision_steps": 2,
            "probe_seconds": 1,
            "fuzzer_seeds": [20_260_821, 20_260_822],
            "proof_seconds": 10,
        },
        "run": {
            "schema_version": "dllm-poc-revision-run/v1",
            "arm": arm,
            "pseudo_feedback_rotation_seed": 0,
            "particle_seed_base": PARTICLE_SEED_BASE,
            "proof_seed": PROOF_SEED,
        },
        "target": {
            "repository_url": "https://github.com/AcademySoftwareFoundation/openexr.git",
            "commit_sha": TARGET_COMMIT,
            "security_policy_url": "https://github.com/AcademySoftwareFoundation/openexr/security/policy",
            "security_policy_sha256": SECURITY_SHA256,
        },
        "probe": {
            "image_id": IMAGE_ID,
            "harness_sha256": HARNESS_SHA256,
            "rss_limit_mb": 256,
            "timeout_seconds": 1,
        },
        "model": {
            "model_id": "ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
            "revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
            "device": "cuda:0",
            "timeout_seconds": 60,
        },
        "prompt": {"path": "prompts/openexr-poc-revision.md", "sha256": prompt_sha256},
    }
    match mutation:
        case None:
            pass
        case "extra":
            config["extra"] = True
        case "image":
            probe = _nested(config, "probe")
            probe["image_id"] = "sha256:" + ("0" * 64)
        case "commit":
            target = _nested(config, "target")
            target["commit_sha"] = "0" * 40
        case "security":
            target = _nested(config, "target")
            target["security_policy_sha256"] = "1" * 64
        case "escape":
            prompt = _nested(config, "prompt")
            prompt["path"] = "../escape.md"
    return config


def _nested(config: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = config[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return value


def _executable(tmp_path: Path) -> Path:
    runtime = tmp_path / "python"
    _ = runtime.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime.chmod(0o755)
    return runtime


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _ = (source / "SECURITY.md").write_text("report privately\n", encoding="utf-8")
    return source


def _report() -> PocRevisionReport:
    return PocRevisionReport(
        schema_version="dllm-poc-revision-report/v1",
        config_hash_sha256="0" * 64,
        snapshots=(),
        probe_feedback=(),
        field_attributions=(),
        remask_decisions=(),
        revision_events_sha256="1" * 64,
        accounting_rejected=0,
        accounting_dedup=0,
        proof=None,
    )


def _verified_command() -> VerifiedPocRevisionCommand:
    config = PocRevisionCommandConfig.model_validate_json(json.dumps(_config(Path())))
    return VerifiedPocRevisionCommand(
        config=config,
        config_sha256="2" * 64,
        prompt_path=ROOT / "prompts/openexr-poc-revision.md",
        prompt_sha256=config.prompt.sha256,
        target=_target_evidence(),
    )


def _target_evidence() -> TargetEvidence:
    return TargetEvidence(
        repository_url="https://github.com/AcademySoftwareFoundation/openexr.git",
        commit_sha=TARGET_COMMIT,
        tree_oid=ZERO_TREE,
        source_archive_sha256="5" * 64,
        security_policy_url="https://github.com/AcademySoftwareFoundation/openexr/security/policy",
        security_policy_sha256=SECURITY_SHA256,
    )


def _ready() -> WorkerReadyProvenance:
    return WorkerReadyProvenance(
        model_id="ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
        revision="10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        remote_code_sha256="298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104",
        device="cuda:0",
        load_count=1,
    )


class _FakeProbe:
    pass


class _FakeProofRunner:
    pass


class _FakeProofBoundaryError(Exception):
    pass
