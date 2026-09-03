"""Production runtime adapters for OpenEXR qualification collectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field, ValidationError
from tools.stable_diffcoder_payloads import MODEL_ID

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import CampaignConfig, CoordinatorError
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRequest,
    CoverageSeriesRequest,
    CoverageStepRequest,
    CoverageStepResult,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.qualification import SystemQualificationCommandRunner
from dllm_agent.experiments.openexr_one_day.qualification_collectors import (
    CollectorInputs,
    CoverageQualificationRequest,
    CoverageQualificationResult,
    HashArtifactInputs,
    collect_hash_measurements,
)
from dllm_agent.targets.openexr.probe_coverage import (
    CoverageBoundaryError,
    DockerCoverageReplayer,
    ProbeCommand,
)

_C0_NAME: Final = "c0"

if TYPE_CHECKING:
    import pathlib


@dataclass(frozen=True, slots=True)
class QualificationRuntimePaths:
    """Explicit file and image inputs required by production qualification collection."""

    target_harness_path: pathlib.Path
    remote_code_path: pathlib.Path
    search_binary_path: pathlib.Path
    coverage_binary_path: pathlib.Path
    coverage_build_manifest_path: pathlib.Path
    c0_source_path: pathlib.Path
    dictionary_path: pathlib.Path
    search_image_ref: str
    coverage_image_ref: str
    smoke_artifact_path: pathlib.Path
    measurements_root: pathlib.Path | None = None
    evidence_root: pathlib.Path | None = None
    coverage_artifact_root: pathlib.Path | None = None
    disk_path: pathlib.Path | None = None
    coverage_cpu_index: int = 18


class CoverageQualificationArtifact(FrozenModel):
    """Canonical artifact binding one qualification replay to Docker coverage evidence."""

    schema_version: Literal["dllm-openexr-one-day-coverage-qualification/v1"] = (
        "dllm-openexr-one-day-coverage-qualification/v1"
    )
    request_sha256: Sha256
    series_request_sha256: Sha256
    step_result_sha256: Sha256
    branches_covered: int = Field(ge=0)
    branches_total: int = Field(gt=0)
    target_only_branch_count: int = Field(ge=0)
    request: CoverageQualificationRequest
    series_request: CoverageSeriesRequest
    step_result: CoverageStepResult


@dataclass(frozen=True, slots=True)
class ProductionCoverageQualificationRunner:
    """Replay exact C0 through the real Docker coverage replayer."""

    config: CampaignConfig
    c0_source_path: pathlib.Path
    artifact_root: pathlib.Path
    cpu_index: int = 18
    command_runner: ProbeCommand | None = None

    def run(self, request: CoverageQualificationRequest) -> CoverageQualificationResult:
        """Run one isolated target-only replay and persist its canonical artifact."""
        self._validate_request(request)
        _validate_coverage_artifact_root(self.config.artifact_root, self.artifact_root)
        replay_root = self.artifact_root / _run_name(request)
        corpus = _corpus_request(replay_root)
        c0_path = _stage_c0(corpus.corpus_root, self.c0_source_path, self.config.c0_sha256)
        record = snapshot_corpus(corpus, 0, 0.0)
        series = CoverageSeriesRequest(
            execution_profile="qualification-smoke",
            schedule_seconds=(0, 300),
            corpus_index=corpus,
            target_commit=self.config.target_commit,
            image_digest=request.image_digest,
            binary_sha256=request.binary_sha256,
            build_manifest_sha256=request.build_manifest_sha256,
            cpu_index=self.cpu_index,
        )
        replayer = DockerCoverageReplayer(series, runner=self.command_runner)
        step = CoverageStepRequest(
            scheduled_elapsed_seconds=0,
            corpus_set_sha256=record.cumulative_set_sha256,
            new_input_paths=(c0_path,),
            new_input_sha256=(self.config.c0_sha256,),
            prior_profdata_sha256=None,
        )
        try:
            result = replayer.replay(step)
        except CoverageBoundaryError as error:
            reason = f"qualification coverage replay failed: {error.reason}"
            raise CoordinatorError(reason=reason) from error
        _validate_repeated_total(self.artifact_root, request, result)
        artifact_path = self.artifact_root / f"coverage-qualification-{_run_name(request)}.json"
        branch_count = result.counts.branches_covered
        artifact = CoverageQualificationArtifact(
            request_sha256=_model_sha256(request),
            series_request_sha256=_model_sha256(series),
            step_result_sha256=_model_sha256(result),
            branches_covered=branch_count,
            branches_total=result.counts.branches_total,
            target_only_branch_count=branch_count,
            request=request,
            series_request=series,
            step_result=result,
        )
        content = canonical_json_bytes(artifact.model_dump(mode="json"))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists() or artifact_path.is_symlink():
            raise CoordinatorError(reason="qualification coverage artifact already exists")
        _ = artifact_path.write_bytes(content)
        return CoverageQualificationResult(
            target_only_branch_count=branch_count,
            artifact_path=artifact_path,
            artifact_sha256=sha256_bytes(content),
        )

    def _validate_request(self, request: CoverageQualificationRequest) -> None:
        if request.target_commit != self.config.target_commit:
            raise CoordinatorError(reason="qualification target_commit mismatch")
        if request.image_digest != self.config.coverage_image_digest:
            raise CoordinatorError(reason="qualification coverage_image_digest mismatch")
        if request.binary_sha256 != self.config.coverage_binary_sha256:
            raise CoordinatorError(reason="qualification coverage_binary_sha256 mismatch")
        if request.build_manifest_sha256 != self.config.coverage_build_manifest_sha256:
            raise CoordinatorError(reason="qualification coverage_build_manifest_sha256 mismatch")


def build_production_collector_inputs(
    *, config: CampaignConfig, paths: QualificationRuntimePaths, model_id: str = MODEL_ID
) -> CollectorInputs:
    """Build production collector inputs from a digest-verified config and explicit paths."""
    _require_verified_config(config)
    command_runner = SystemQualificationCommandRunner()
    hash_artifacts = _hash_inputs(paths)
    _ = collect_hash_measurements(config, hash_artifacts, command_runner)
    coverage_root = (
        config.artifact_root / "qualification-runtime" / "coverage"
        if paths.coverage_artifact_root is None
        else paths.coverage_artifact_root
    )
    return CollectorInputs(
        config=config,
        measurements_root=(
            config.artifact_root / "measurements"
            if paths.measurements_root is None
            else paths.measurements_root
        ),
        evidence_root=(
            config.artifact_root / "qualification"
            if paths.evidence_root is None
            else paths.evidence_root
        ),
        hash_artifacts=hash_artifacts,
        model_id=model_id,
        coverage_runner=ProductionCoverageQualificationRunner(
            config=config,
            c0_source_path=paths.c0_source_path,
            artifact_root=coverage_root,
            cpu_index=paths.coverage_cpu_index,
        ),
        smoke_artifact_path=paths.smoke_artifact_path,
        disk_path=config.artifact_root if paths.disk_path is None else paths.disk_path,
        command_runner=command_runner,
    )


def _hash_inputs(paths: QualificationRuntimePaths) -> HashArtifactInputs:
    return HashArtifactInputs(
        target_harness_path=paths.target_harness_path,
        remote_code_path=paths.remote_code_path,
        search_binary_path=paths.search_binary_path,
        coverage_binary_path=paths.coverage_binary_path,
        coverage_build_manifest_path=paths.coverage_build_manifest_path,
        c0_source_path=paths.c0_source_path,
        dictionary_path=paths.dictionary_path,
        search_image_ref=paths.search_image_ref,
        coverage_image_ref=paths.coverage_image_ref,
    )


def _require_verified_config(config: CampaignConfig) -> None:
    payload = config.model_dump(mode="json")
    if config.config_sha256 != config_payload_sha256(payload):
        raise CoordinatorError(reason="config digest mismatch")
    if config.qualification_state != "qualified":
        raise CoordinatorError(reason="qualification config must be qualified")


def _validate_coverage_artifact_root(
    config_root: pathlib.Path, artifact_root: pathlib.Path
) -> None:
    try:
        _ = artifact_root.resolve(strict=False).relative_to(config_root.resolve(strict=False))
    except ValueError as error:
        reason = "qualification coverage artifact root escapes config artifact root"
        raise CoordinatorError(reason=reason) from error
    for path in (artifact_root, *artifact_root.parents):
        if path.is_symlink():
            raise CoordinatorError(reason="qualification coverage artifact root contains symlink")


def _validate_repeated_total(
    artifact_root: pathlib.Path,
    request: CoverageQualificationRequest,
    result: CoverageStepResult,
) -> None:
    if not request.repeated:
        return
    try:
        first = CoverageQualificationArtifact.model_validate_json(
            (artifact_root / "coverage-qualification-first.json").read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="qualification target branch total mismatch") from error
    if first.branches_total != result.counts.branches_total:
        raise CoordinatorError(reason="qualification target branch total mismatch")


def _corpus_request(root: pathlib.Path) -> CorpusIndexRequest:
    return CorpusIndexRequest(
        execution_profile="qualification-smoke",
        schedule_seconds=(0, 300),
        artifact_root=root,
        corpus_root=root / "corpus",
        store_root=root / "corpus-inputs",
        index_path=root / "corpus-index.jsonl",
    )


def _stage_c0(
    corpus_root: pathlib.Path, c0_source_path: pathlib.Path, expected_sha256: str
) -> pathlib.Path:
    try:
        if c0_source_path.is_symlink() or not c0_source_path.is_file():
            raise CoordinatorError(reason="qualification C0 source unavailable")
        content = bytes.fromhex(c0_source_path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CoordinatorError(reason="invalid C0 source fixture") from error
    if sha256_bytes(content) != expected_sha256:
        raise CoordinatorError(reason="qualification c0_sha256 mismatch")
    corpus_root.mkdir(parents=True, exist_ok=True)
    path = corpus_root / _C0_NAME
    if path.exists() or path.is_symlink():
        raise CoordinatorError(reason="qualification C0 corpus already exists")
    _ = path.write_bytes(content)
    return path


def _run_name(request: CoverageQualificationRequest) -> str:
    return "repeated" if request.repeated else "first"


def _model_sha256(value: FrozenModel) -> str:
    return sha256_bytes(canonical_json_bytes(value.model_dump(mode="json")))
