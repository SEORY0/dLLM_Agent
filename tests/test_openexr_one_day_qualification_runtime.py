"""Qualification runtime collector tests."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

import pytest
from pydantic import TypeAdapter
from tools.stable_diffcoder_payloads import MODEL_ID

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day import qualification_runtime as runtime_module
from dllm_agent.experiments.openexr_one_day.coordinator import CampaignConfig, CoordinatorError
from dllm_agent.experiments.openexr_one_day.qualification_collectors import (
    CoverageQualificationRequest,
    collect_external_coverage_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_runtime import (
    CoverageQualificationArtifact,
    ProductionCoverageQualificationRunner,
    QualificationRuntimePaths,
    build_production_collector_inputs,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import WRAPPER_NAME
from tests.test_openexr_one_day_qualification_collectors import (
    ArtifactFiles,
    _artifact_files,
    _config,
    _hash_inputs,
)


@dataclass(slots=True)
class FakeProbeCommand:
    calls: list[tuple[str, ...]] = field(default_factory=list)
    report_total: int = 20

    def run(
        self, argv: tuple[str, ...], _environment: Mapping[str, str], _timeout: float
    ) -> CompletedProcess[bytes]:
        self.calls.append(argv)
        root = _mounted_root(argv)
        if "/usr/local/bin/llvm-profdata" in argv:
            _ = _host_path(root, argv[-1]).write_bytes(f"profile-{len(self.calls)}".encode())
            return CompletedProcess(argv, 0, b"", b"")
        if "/usr/local/bin/llvm-cov" in argv:
            return CompletedProcess(argv, 0, self._report(), b"")
        profile = next(item for item in argv if item.startswith("LLVM_PROFILE_FILE="))
        _ = _host_path(root, profile.split("=", 1)[1]).write_bytes(
            f"raw-{len(self.calls)}".encode()
        )
        return CompletedProcess(argv, 0, b"run\n", b"")

    def _report(self) -> bytes:
        payload = {
            "data": [
                {
                    "files": [
                        {
                            "filename": "/opt/openexr-src/src/lib/OpenEXR/ImfInputFile.cpp",
                            "summary": {
                                "branches": {"covered": 5, "count": self.report_total},
                                "regions": {"covered": 6, "count": 21},
                                "lines": {"covered": 7, "count": 22},
                            },
                        },
                    ]
                }
            ]
        }
        return json.dumps(payload, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class FakeDockerInspectRunner:
    digests: dict[str, str]

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        if argv[:4] != ("docker", "image", "inspect", "--format"):
            raise AssertionError(argv)
        return CompletedProcess(argv, 0, f"{self.digests[argv[5]]}\n".encode(), b"")


@dataclass(frozen=True, slots=True)
class FakeSystemRunnerFactory:
    runner: FakeDockerInspectRunner

    def __call__(self) -> FakeDockerInspectRunner:
        return self.runner


def test_runtime_runner_replays_exact_c0_twice_with_real_docker_replayer(
    data_root: Path, tmp_path: Path
) -> None:
    """Given C0 config, when coverage qualifies twice, then hash-bound artifacts are persisted."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    root = config.artifact_root / "qualification-runtime" / "coverage"
    command = FakeProbeCommand()
    runner = ProductionCoverageQualificationRunner(
        config=config,
        c0_source_path=files.c0,
        artifact_root=root,
        cpu_index=18,
        command_runner=command.run,
    )
    measurements = collect_external_coverage_measurements(config, runner)
    first_artifact = Path(str(measurements["coverage_artifact_path"]))
    repeated_artifact = Path(str(measurements["repeated_coverage_artifact_path"]))

    artifact = CoverageQualificationArtifact.model_validate_json(first_artifact.read_bytes())

    assert measurements["target_only_branch_count"] == 5
    assert measurements["repeated_target_only_branch_count"] == 5
    assert artifact.branches_covered == 5
    assert artifact.branches_total == 20
    assert first_artifact.name == "coverage-qualification-first.json"
    assert repeated_artifact.name == "coverage-qualification-repeated.json"
    assert measurements["coverage_artifact_sha256"] == sha256_bytes(first_artifact.read_bytes())
    assert first_artifact.read_bytes() == canonical_json_bytes(
        _JSON.validate_json(first_artifact.read_bytes())
    )
    assert (root / "first" / "corpus" / "c0").read_bytes() == b"\x00\xff"
    assert (root / "repeated" / "corpus" / "c0").read_bytes() == b"\x00\xff"
    assert len(command.calls) == 6
    assert {config.coverage_image_digest in call for call in command.calls} == {True}


def test_runtime_runner_rejects_unsafe_artifact_roots_before_replay(
    data_root: Path, tmp_path: Path
) -> None:
    """Given symlink or escaped roots, when coverage qualifies, then no replay writes start."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    command = FakeProbeCommand()
    outside = config.artifact_root.parent / f"{config.artifact_root.name}-coverage-outside"
    link_target = config.artifact_root / "actual-parent"
    link_target.mkdir(parents=True)
    symlink_parent = config.artifact_root / "link-parent"
    symlink_parent.symlink_to(link_target, target_is_directory=True)

    for root in (outside, symlink_parent / "coverage"):
        runner = ProductionCoverageQualificationRunner(
            config=config,
            c0_source_path=files.c0,
            artifact_root=root,
            command_runner=command.run,
        )
        with pytest.raises(CoordinatorError, match="qualification coverage artifact root"):
            _ = runner.run(_coverage_request(config))

    assert command.calls == []
    assert not outside.exists()
    assert not (symlink_parent / "coverage").exists()


def test_runtime_runner_rejects_repeated_total_mismatch(data_root: Path, tmp_path: Path) -> None:
    """Given a changed target branch total, when replay repeats, then qualification fails."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    root = config.artifact_root / "qualification-runtime" / "coverage"
    command = FakeProbeCommand()
    runner = ProductionCoverageQualificationRunner(
        config=config,
        c0_source_path=files.c0,
        artifact_root=root,
        command_runner=command.run,
    )
    request = _coverage_request(config)
    first = runner.run(request)
    command.report_total = 21

    with pytest.raises(CoordinatorError, match="target branch total mismatch"):
        _ = runner.run(request.model_copy(update={"repeated": True}))

    artifact = CoverageQualificationArtifact.model_validate_json(first.artifact_path.read_bytes())
    assert artifact.branches_total == 20


def test_runtime_factory_builds_collector_inputs_with_system_runner(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given explicit artifacts, when factory builds inputs, then provenance is bound exactly."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    paths = _runtime_paths(config.artifact_root, files)
    fake_runner = FakeDockerInspectRunner(
        {
            "search:qualified": config.search_image_digest,
            "coverage:qualified": config.coverage_image_digest,
        }
    )
    monkeypatch.setattr(
        runtime_module,
        "SystemQualificationCommandRunner",
        FakeSystemRunnerFactory(fake_runner),
    )

    inputs = build_production_collector_inputs(config=config, paths=paths, model_id=MODEL_ID)

    assert inputs.config == config
    assert inputs.measurements_root == config.artifact_root / "measurements"
    assert inputs.evidence_root == config.artifact_root / "qualification"
    assert inputs.hash_artifacts == _hash_inputs(files)
    assert isinstance(inputs.coverage_runner, ProductionCoverageQualificationRunner)


def test_runtime_factory_rejects_unverified_or_mismatched_inputs(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given stale config/provenance, when factory builds inputs, then it fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    paths = _runtime_paths(config.artifact_root, files)
    fake_runner = FakeDockerInspectRunner(
        {
            "search:qualified": config.search_image_digest,
            "coverage:qualified": config.coverage_image_digest,
            "coverage:stale": "sha256:" + ("9" * 64),
        }
    )
    monkeypatch.setattr(
        runtime_module,
        "SystemQualificationCommandRunner",
        FakeSystemRunnerFactory(fake_runner),
    )

    with pytest.raises(CoordinatorError, match="config digest"):
        _ = build_production_collector_inputs(
            config=config.model_copy(update={"config_sha256": "1" * 64}),
            paths=paths,
            model_id=MODEL_ID,
        )

    bad_paths = replace(paths, coverage_image_ref="coverage:stale")
    with pytest.raises(CoordinatorError, match="coverage_image_digest"):
        _ = build_production_collector_inputs(config=config, paths=bad_paths, model_id=MODEL_ID)


def _runtime_paths(root: Path, files: ArtifactFiles) -> QualificationRuntimePaths:
    return QualificationRuntimePaths(
        target_harness_path=files.harness,
        remote_code_path=files.remote,
        search_binary_path=files.search,
        coverage_binary_path=files.coverage,
        coverage_build_manifest_path=files.coverage_manifest,
        c0_source_path=files.c0,
        dictionary_path=files.dictionary,
        search_image_ref="search:qualified",
        coverage_image_ref="coverage:qualified",
        smoke_artifact_path=root / "smoke" / WRAPPER_NAME,
    )


def _coverage_request(config: CampaignConfig) -> CoverageQualificationRequest:
    return CoverageQualificationRequest(
        target_commit=config.target_commit,
        image_digest=config.coverage_image_digest,
        binary_sha256=config.coverage_binary_sha256,
        build_manifest_sha256=config.coverage_build_manifest_sha256,
    )


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("qualification-runtime")
    digest = sha256(str(root).encode()).hexdigest()[:12]
    return Path("/data/dllm-agent") / f"qualification-runtime-{digest}"


def _host_path(root: Path, container_path: str) -> Path:
    assert container_path.startswith("/work/")
    return root / container_path.removeprefix("/work/")


def _mounted_root(argv: tuple[str, ...]) -> Path:
    volume = argv[argv.index("-v") + 1]
    host, container, mode = volume.split(":", maxsplit=2)
    assert container == "/work"
    assert mode == "rw"
    return Path(host)


_JSON = TypeAdapter(dict[str, JsonValue])

if TYPE_CHECKING:
    from collections.abc import Mapping
