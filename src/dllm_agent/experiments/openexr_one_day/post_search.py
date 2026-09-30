"""Production post-search coverage and proof composition for Task 5.

# noqa: SIZE_OK
"""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Self, assert_never, override

from pydantic import ConfigDict, Field, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellResult,
    CellRunRequest,
    SearchAccounting,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CorpusIndexRequest,
    CoverageReplayer,
    CoverageSeries,
    CoverageSeriesError,
    CoverageSeriesRequest,
    build_coverage_series,
    validate_coverage_series_backings,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    ExactProofRunner,
    ProofBatchRequest,
    ProofBatchResult,
    ProofClock,
    run_deferred_proof_batch,
)
from dllm_agent.targets.openexr.probe_coverage import DockerCoverageReplayer
from dllm_agent.targets.openexr.proof import OpenExrProofRunner


class PostSearchEvidenceError(ValueError):
    """Raised when persisted post-search artifacts are not canonical."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the evidence failure reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CampaignConfigView(Protocol):
    """Coverage/proof fields required from the verified campaign config."""

    coverage_image_digest: str
    coverage_binary_sha256: str
    coverage_build_manifest_sha256: str
    proof_duration_seconds: int
    proof_seed_base: int
    search_image_digest: str


class CoverageBuilder(Protocol):
    """Task4 coverage-series builder seam."""

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        """Build target-only coverage from a complete corpus index."""
        ...


class CoverageReplayerFactory(Protocol):
    """Factory for request-bound Task4 coverage replayers."""

    def build(self, request: CoverageSeriesRequest) -> CoverageReplayer:
        """Build a replayer for one immutable coverage request."""
        ...


class ProofBuilder(Protocol):
    """Task4 deferred proof-batch builder seam."""

    def run(
        self, request: ProofBatchRequest, runner: ExactProofRunner, clock: ProofClock
    ) -> ProofBatchResult:
        """Run deferred exact proof over the cell crash queue."""
        ...


class DeepParseReplayer(Protocol):
    """Final corpus OpenEXR deep-parse replay boundary."""

    @property
    def checker_binary_sha256(self) -> Sha256:
        """Return the exact checker binary configured for replay."""
        ...

    def replay(self, request: DeepParseRequest) -> DeepParseResult:
        """Replay checkOpenEXRFile-equivalent parsing over the exact final corpus."""
        ...


@dataclass(frozen=True, slots=True)
class DefaultCoverageBuilder:
    """Default adapter around Task4 `build_coverage_series`."""

    def build(self, request: CoverageSeriesRequest, replayer: CoverageReplayer) -> CoverageSeries:
        """Call the Task4 coverage builder."""
        return build_coverage_series(request, replayer)


@dataclass(frozen=True, slots=True)
class DefaultCoverageReplayerFactory:
    """Default adapter around Task4 Docker coverage replay."""

    def build(self, request: CoverageSeriesRequest) -> CoverageReplayer:
        """Bind the Docker replayer to one coverage request."""
        return DockerCoverageReplayer(request)


@dataclass(frozen=True, slots=True)
class DefaultProofBuilder:
    """Default adapter around Task4 `run_deferred_proof_batch`."""

    def run(
        self, request: ProofBatchRequest, runner: ExactProofRunner, clock: ProofClock
    ) -> ProofBatchResult:
        """Call the Task4 deferred proof batch."""
        return run_deferred_proof_batch(request, runner, clock)


class PostSearchOutcome(FrozenModel):
    """Typed coverage/proof accounting consumed by Task6 reports."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    coverage_replay_count: int = Field(ge=0)
    proof_replay_count: int = Field(ge=0)
    coverage_series_sha256: Sha256
    proof_result_sha256: Sha256
    coverage_envelope_sha256: Sha256 | None = None
    proof_envelope_sha256: Sha256 | None = None
    coverage_series_path: Path | None = None
    proof_result_path: Path | None = None
    coverage_status: Literal["available", "unavailable"] = "unavailable"
    proof_status: Literal["available", "unavailable"] = "unavailable"
    proof_wall_seconds: float = Field(ge=0.0)
    proof_cpu_seconds: float = Field(ge=0.0)
    deep_parse_status: Literal["available", "unavailable"] = "unavailable"
    deep_parse_replay_count: int = Field(default=0, ge=0)
    deep_parse_success_count: int | None = Field(default=None, ge=0)
    deep_parse_total_count: int | None = Field(default=None, ge=0)
    deep_parse_result_sha256: Sha256 | None = None
    deep_parse_envelope_sha256: Sha256 | None = None
    deep_parse_result_path: Path | None = None
    deep_parse_envelope_path: Path | None = None


class CoverageSeriesEnvelope(FrozenModel):
    """Strict post-search coverage envelope bound to one completed cell result."""

    schema_version: Literal["dllm-openexr-coverage-envelope/v1"] = (
        "dllm-openexr-coverage-envelope/v1"
    )
    cell_request: CellRunRequest
    cell_request_sha256: Sha256
    cell_result: CellResult
    cell_result_sha256: Sha256
    coverage_request: CoverageSeriesRequest
    coverage_request_sha256: Sha256
    coverage_series: CoverageSeries
    coverage_series_sha256: Sha256


class ProofBatchEnvelope(FrozenModel):
    """Strict post-search proof envelope bound to one completed cell result."""

    schema_version: Literal["dllm-openexr-proof-envelope/v1"] = "dllm-openexr-proof-envelope/v1"
    cell_request: CellRunRequest
    cell_request_sha256: Sha256
    cell_result: CellResult
    cell_result_sha256: Sha256
    proof_request: ProofBatchRequest
    proof_request_sha256: Sha256
    proof_result: ProofBatchResult
    proof_result_sha256: Sha256


class DeepParseRequest(FrozenModel):
    """Exact final-corpus parse replay request for one completed cell."""

    schema_version: Literal["dllm-openexr-deep-parse-request/v1"] = (
        "dllm-openexr-deep-parse-request/v1"
    )
    cell_id: str
    artifact_root: Path
    cell_request_sha256: Sha256
    cell_result_sha256: Sha256
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    checker_binary_sha256: Sha256
    build_manifest_sha256: Sha256
    final_corpus_set_sha256: Sha256
    input_paths: tuple[Path, ...] = Field(min_length=1)
    input_sha256: tuple[Sha256, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _paired_inputs(self) -> Self:
        if len(self.input_paths) != len(self.input_sha256):
            message = "deep-parse input paths and hashes must have equal length"
            raise ValueError(message)
        return self


class DeepParseEvidenceArtifact(FrozenModel):
    """One byte-exact deep-parse artifact below a declared evidence root."""

    relative_path: Path
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @model_validator(mode="after")
    def _safe_relative_path(self) -> Self:
        if (
            self.relative_path.is_absolute()
            or self.relative_path == Path()
            or ".." in self.relative_path.parts
        ):
            message = "deep-parse evidence path must be a safe relative path"
            raise ValueError(message)
        return self


class DeepParseRunEvidence(FrozenModel):
    """Canonical process evidence for one checker invocation."""

    input_evidence: DeepParseEvidenceArtifact
    argv_evidence: DeepParseEvidenceArtifact
    stdout_evidence: DeepParseEvidenceArtifact
    stderr_evidence: DeepParseEvidenceArtifact
    exit_evidence: DeepParseEvidenceArtifact


class DeepParseVerdict(DeepParseRunEvidence):
    """Per-input final-corpus parse verdict."""

    input_path: Path
    input_sha256: Sha256
    size_bytes: int = Field(gt=0)
    parse_success: bool


class DeepParseResult(FrozenModel):
    """Aggregated final-corpus parse replay result."""

    schema_version: Literal["dllm-openexr-deep-parse-result/v1"] = (
        "dllm-openexr-deep-parse-result/v1"
    )
    replay_kind: Literal["docker", "test-only-hash"]
    evidence_root: Path
    build_manifest_evidence: DeepParseEvidenceArtifact | None = None
    verdicts: tuple[DeepParseVerdict, ...] = Field(min_length=1)
    success_count: int = Field(ge=0)
    total_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _counts_match(self) -> Self:
        if self.total_count != len(self.verdicts) or self.success_count != sum(
            1 for verdict in self.verdicts if verdict.parse_success
        ):
            message = "deep-parse counts do not match verdicts"
            raise ValueError(message)
        match self.replay_kind:
            case "docker":
                if self.build_manifest_evidence is None:
                    message = "Docker deep-parse result requires build manifest evidence"
                    raise ValueError(message)
            case "test-only-hash":
                pass
            case _ as unreachable:
                assert_never(unreachable)
        return self


class DeepParseEnvelope(FrozenModel):
    """Strict deep-parse envelope bound to coverage final corpus and Task4 result."""

    schema_version: Literal["dllm-openexr-deep-parse-envelope/v1"] = (
        "dllm-openexr-deep-parse-envelope/v1"
    )
    cell_request: CellRunRequest
    cell_request_sha256: Sha256
    cell_result: CellResult
    cell_result_sha256: Sha256
    deep_parse_request: DeepParseRequest
    deep_parse_request_sha256: Sha256
    deep_parse_request_evidence: DeepParseEvidenceArtifact
    deep_parse_result: DeepParseResult
    deep_parse_result_sha256: Sha256
    deep_parse_result_evidence: DeepParseEvidenceArtifact


@dataclass(frozen=True, slots=True)
class HashOnlyDeepParseReplayer:
    """Explicitly test-only deterministic compatibility replayer."""

    checker_binary_sha256: Sha256 = "0" * 64

    def replay(self, request: DeepParseRequest) -> DeepParseResult:
        """Create dereferenceable deterministic evidence without executing Docker."""
        evidence_root = request.artifact_root / "post-search" / "deep-parse-test-evidence"
        manifest_evidence = persist_deep_parse_evidence(
            evidence_root,
            Path("provenance/build-manifest.json"),
            canonical_json_bytes({"kind": "test-only-hash-replayer"}),
        )
        verdicts: list[DeepParseVerdict] = []
        for index, (path, digest) in enumerate(
            zip(request.input_paths, request.input_sha256, strict=True)
        ):
            content = path.read_bytes()
            if sha256_bytes(content) != digest:
                reason = "deep-parse input hash mismatch"
                raise PostSearchEvidenceError(reason)
            prefix = Path("runs") / f"{index:06d}-{digest}"
            run_evidence = DeepParseRunEvidence(
                input_evidence=persist_deep_parse_evidence(
                    evidence_root, prefix / "input.exr", content
                ),
                argv_evidence=persist_deep_parse_evidence(
                    evidence_root,
                    prefix / "argv.json",
                    canonical_json_bytes(
                        {
                            "checker_binary_sha256": request.checker_binary_sha256,
                            "input_sha256": digest,
                        }
                    ),
                ),
                stdout_evidence=persist_deep_parse_evidence(
                    evidence_root, prefix / "stdout.bin", b"hash-only-deep-parse:success"
                ),
                stderr_evidence=persist_deep_parse_evidence(
                    evidence_root, prefix / "stderr.bin", b""
                ),
                exit_evidence=persist_deep_parse_evidence(
                    evidence_root,
                    prefix / "exit.json",
                    canonical_json_bytes({"return_code": 0}),
                ),
            )
            verdicts.append(
                DeepParseVerdict(
                    input_evidence=run_evidence.input_evidence,
                    argv_evidence=run_evidence.argv_evidence,
                    stdout_evidence=run_evidence.stdout_evidence,
                    stderr_evidence=run_evidence.stderr_evidence,
                    exit_evidence=run_evidence.exit_evidence,
                    input_path=path,
                    input_sha256=digest,
                    size_bytes=len(content),
                    parse_success=True,
                )
            )
        return DeepParseResult(
            replay_kind="test-only-hash",
            evidence_root=evidence_root,
            build_manifest_evidence=manifest_evidence,
            verdicts=tuple(verdicts),
            success_count=len(verdicts),
            total_count=len(verdicts),
        )


@dataclass(frozen=True, slots=True)
class ProductionPostSearch:
    """Run Task4 post-search coverage and proof after all search cells finish."""

    config: CampaignConfigView | None = None
    coverage_replayer: CoverageReplayer | None = None
    coverage_replayer_factory: CoverageReplayerFactory = field(
        default_factory=DefaultCoverageReplayerFactory
    )
    coverage_builder: CoverageBuilder = field(default_factory=DefaultCoverageBuilder)
    proof_builder: ProofBuilder = field(default_factory=DefaultProofBuilder)
    proof_runner: ExactProofRunner = field(default_factory=OpenExrProofRunner)
    clock: ProofClock = field(default_factory=lambda: _ProcessClock())  # noqa: PLW0108
    deep_parse_replayer: DeepParseReplayer | None = None

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        """Build coverage and proof requests from the completed search result."""
        corpus_index = CorpusIndexRequest(
            execution_profile=request.execution_profile,
            schedule_seconds=request.schedule_seconds,
            artifact_root=request.artifact_root,
            corpus_root=request.corpus_root,
            store_root=request.artifact_root / "corpus-inputs",
            index_path=request.artifact_root / "corpus-index.jsonl",
        )
        image_digest = request.build.image_digest
        binary_sha256 = request.build.binary_sha256
        build_manifest_sha256 = request.build.binary_sha256
        duration_seconds = 30
        proof_seed = request.fuzzer_seed
        search_image_digest = request.build.image_digest
        if self.config is not None:
            image_digest = self.config.coverage_image_digest
            binary_sha256 = self.config.coverage_binary_sha256
            build_manifest_sha256 = self.config.coverage_build_manifest_sha256
            duration_seconds = self.config.proof_duration_seconds
            proof_seed = _proof_seed(self.config.proof_seed_base, request.fuzzer_seed)
            search_image_digest = self.config.search_image_digest
        coverage_request = CoverageSeriesRequest(
            execution_profile=request.execution_profile,
            schedule_seconds=request.schedule_seconds,
            corpus_index=corpus_index,
            target_commit=request.target_commit,
            image_digest=image_digest,
            binary_sha256=binary_sha256,
            build_manifest_sha256=build_manifest_sha256,
            cpu_index=request.cpu_index,
        )
        coverage_path = _coverage_path(request)
        coverage = _load_or_build_coverage(
            path=coverage_path,
            cell_request=request,
            cell_result=result,
            coverage_request=coverage_request,
            replayer=self.coverage_replayer
            or self.coverage_replayer_factory.build(coverage_request),
            builder=self.coverage_builder,
        )
        proof_request = ProofBatchRequest(
            cell_id=request.cell_id,
            cell_root=request.artifact_root,
            crash_queue_path=request.crash_root / "occurrences.jsonl",
            proof_root=request.artifact_root / "proof",
            target_commit=request.target_commit,
            proof_seed=proof_seed,
            duration_seconds=duration_seconds,
            search_finished_monotonic_seconds=result.finished_monotonic_seconds,
            search_cpu_seconds=result.search_accounting.cpu_seconds,
            search_image_digest=search_image_digest,
        )
        proof_path = _proof_path(request)
        proof = _load_or_build_proof(
            path=proof_path,
            cell_request=request,
            cell_result=result,
            proof_request=proof_request,
            builder=self.proof_builder,
            runner=self.proof_runner,
            clock=self.clock,
        )
        deep_parse_path = _deep_parse_path(request)
        if self.deep_parse_replayer is None:
            reason = "deep-parse replayer unavailable"
            raise PostSearchEvidenceError(reason)
        deep_parse = _load_or_build_deep_parse(
            path=deep_parse_path,
            cell_request=request,
            cell_result=result,
            coverage=coverage,
            image_digest=image_digest,
            build_manifest_sha256=build_manifest_sha256,
            replayer=self.deep_parse_replayer,
        )
        return _outcome(coverage_path, coverage, proof_path, proof, deep_parse_path, deep_parse)

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        """Build coverage-only frozen evidence for a treatment-failed cell."""
        corpus_index = CorpusIndexRequest(
            execution_profile=request.execution_profile,
            schedule_seconds=request.schedule_seconds,
            artifact_root=request.artifact_root,
            corpus_root=request.corpus_root,
            store_root=request.artifact_root / "corpus-inputs",
            index_path=request.artifact_root / "corpus-index.jsonl",
        )
        image_digest = request.build.image_digest
        binary_sha256 = request.build.binary_sha256
        build_manifest_sha256 = request.build.binary_sha256
        if self.config is not None:
            image_digest = self.config.coverage_image_digest
            binary_sha256 = self.config.coverage_binary_sha256
            build_manifest_sha256 = self.config.coverage_build_manifest_sha256
        coverage_request = CoverageSeriesRequest(
            execution_profile=request.execution_profile,
            schedule_seconds=request.schedule_seconds,
            corpus_index=corpus_index,
            target_commit=request.target_commit,
            image_digest=image_digest,
            binary_sha256=binary_sha256,
            build_manifest_sha256=build_manifest_sha256,
            cpu_index=request.cpu_index,
        )
        coverage_path = _coverage_path(request)
        coverage = _load_or_build_treatment_coverage(
            path=coverage_path,
            cell_request=request,
            coverage_request=coverage_request,
            replayer=self.coverage_replayer
            or self.coverage_replayer_factory.build(coverage_request),
            builder=self.coverage_builder,
        )
        if self.deep_parse_replayer is None:
            reason = "deep-parse replayer unavailable"
            raise PostSearchEvidenceError(reason)
        deep_parse_path = _deep_parse_path(request)
        placeholder = _treatment_placeholder_result(request, coverage)
        deep_parse = _load_or_build_deep_parse(
            path=deep_parse_path,
            cell_request=request,
            cell_result=placeholder,
            coverage=coverage,
            image_digest=image_digest,
            build_manifest_sha256=build_manifest_sha256,
            replayer=self.deep_parse_replayer,
        )
        return _coverage_only_outcome(coverage_path, coverage, deep_parse_path, deep_parse)


def load_post_search_outcome(request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
    """Load and validate persisted post-search artifacts for one completed cell."""
    coverage_path = _coverage_path(request)
    proof_path = _proof_path(request)
    coverage = _load_coverage_envelope(coverage_path, request, result).coverage_series
    proof = _load_proof_envelope(proof_path, request, result).proof_result
    deep_parse_path = _deep_parse_path(request)
    deep_parse = _load_deep_parse_envelope(deep_parse_path, request, result).deep_parse_result
    return _outcome(coverage_path, coverage, proof_path, proof, deep_parse_path, deep_parse)


def _load_or_build_coverage(  # noqa: PLR0913
    *,
    path: Path,
    cell_request: CellRunRequest,
    cell_result: CellResult,
    coverage_request: CoverageSeriesRequest,
    replayer: CoverageReplayer,
    builder: CoverageBuilder,
) -> CoverageSeries:
    if path.exists():
        return _load_coverage_envelope(
            path, cell_request, cell_result, coverage_request
        ).coverage_series
    series = builder.build(coverage_request, replayer)
    envelope = CoverageSeriesEnvelope(
        cell_request=cell_request,
        cell_request_sha256=_model_sha256(cell_request),
        cell_result=cell_result,
        cell_result_sha256=_model_sha256(cell_result),
        coverage_request=coverage_request,
        coverage_request_sha256=_model_sha256(coverage_request),
        coverage_series=series,
        coverage_series_sha256=_model_sha256(series),
    )
    _persist_canonical_model(path, envelope)
    return _load_coverage_envelope(
        path, cell_request, cell_result, coverage_request
    ).coverage_series


def _load_or_build_proof(  # noqa: PLR0913
    *,
    path: Path,
    cell_request: CellRunRequest,
    cell_result: CellResult,
    proof_request: ProofBatchRequest,
    builder: ProofBuilder,
    runner: ExactProofRunner,
    clock: ProofClock,
) -> ProofBatchResult:
    if path.exists():
        return _load_proof_envelope(path, cell_request, cell_result).proof_result
    result = builder.run(proof_request, runner, clock)
    envelope = ProofBatchEnvelope(
        cell_request=cell_request,
        cell_request_sha256=_model_sha256(cell_request),
        cell_result=cell_result,
        cell_result_sha256=_model_sha256(cell_result),
        proof_request=proof_request,
        proof_request_sha256=_model_sha256(proof_request),
        proof_result=result,
        proof_result_sha256=_model_sha256(result),
    )
    _persist_canonical_model(path, envelope)
    return _load_proof_envelope(path, cell_request, cell_result).proof_result


def _load_or_build_treatment_coverage(
    *,
    path: Path,
    cell_request: CellRunRequest,
    coverage_request: CoverageSeriesRequest,
    replayer: CoverageReplayer,
    builder: CoverageBuilder,
) -> CoverageSeries:
    if path.exists():
        return _load_treatment_coverage_envelope(
            path, cell_request, coverage_request
        ).coverage_series
    series = builder.build(coverage_request, replayer)
    envelope = CoverageSeriesEnvelope(
        cell_request=cell_request,
        cell_request_sha256=_model_sha256(cell_request),
        cell_result=_treatment_placeholder_result(cell_request, series),
        cell_result_sha256=sha256_bytes(b"treatment-failure"),
        coverage_request=coverage_request,
        coverage_request_sha256=_model_sha256(coverage_request),
        coverage_series=series,
        coverage_series_sha256=_model_sha256(series),
    )
    _persist_canonical_model(path, envelope)
    return _load_treatment_coverage_envelope(path, cell_request, coverage_request).coverage_series


def _load_or_build_deep_parse(  # noqa: PLR0913
    *,
    path: Path,
    cell_request: CellRunRequest,
    cell_result: CellResult,
    coverage: CoverageSeries,
    image_digest: str,
    build_manifest_sha256: str,
    replayer: DeepParseReplayer,
) -> DeepParseResult:
    request = _deep_parse_request(
        cell_request,
        cell_result,
        coverage,
        image_digest,
        replayer.checker_binary_sha256,
        build_manifest_sha256,
    )
    if path.exists():
        envelope = _load_deep_parse_envelope(path, cell_request, cell_result)
        _require_current_deep_parse_request(envelope, request)
        _require_production_deep_parse_result(envelope.deep_parse_result, replayer)
        return envelope.deep_parse_result
    result = replayer.replay(request)
    _require_production_deep_parse_result(result, replayer)
    request_payload = canonical_json_bytes(request.model_dump(mode="json"))
    result_payload = canonical_json_bytes(result.model_dump(mode="json"))
    request_evidence = persist_deep_parse_evidence(
        cell_request.artifact_root,
        Path("post-search/deep-parse-request.json"),
        request_payload,
    )
    result_evidence = persist_deep_parse_evidence(
        cell_request.artifact_root,
        Path("post-search/deep-parse-result.json"),
        result_payload,
    )
    envelope = DeepParseEnvelope(
        cell_request=cell_request,
        cell_request_sha256=_model_sha256(cell_request),
        cell_result=cell_result,
        cell_result_sha256=_model_sha256(cell_result),
        deep_parse_request=request,
        deep_parse_request_sha256=request_evidence.sha256,
        deep_parse_request_evidence=request_evidence,
        deep_parse_result=result,
        deep_parse_result_sha256=result_evidence.sha256,
        deep_parse_result_evidence=result_evidence,
    )
    _persist_canonical_model(path, envelope)
    loaded = _load_deep_parse_envelope(path, cell_request, cell_result)
    _require_current_deep_parse_request(loaded, request)
    _require_production_deep_parse_result(loaded.deep_parse_result, replayer)
    return loaded.deep_parse_result


def _persist_canonical_model[
    ModelT: (CoverageSeriesEnvelope, ProofBatchEnvelope, DeepParseEnvelope)
](path: Path, model: ModelT) -> None:
    payload = canonical_json_bytes(model.model_dump(mode="json"))
    _write_once_bytes(path, payload)


def _load_canonical_model[ModelT: (CoverageSeriesEnvelope, ProofBatchEnvelope, DeepParseEnvelope)](
    path: Path, model_type: type[ModelT]
) -> ModelT:
    raw = path.read_bytes()
    try:
        model = model_type.model_validate_json(raw)
    except ValueError as error:
        reason = "invalid post-search artifact"
        raise PostSearchEvidenceError(reason) from error
    if raw != canonical_json_bytes(model.model_dump(mode="json")):
        reason = "noncanonical post-search artifact"
        raise PostSearchEvidenceError(reason)
    return model


def _load_coverage_envelope(
    path: Path,
    request: CellRunRequest,
    result: CellResult,
    expected_coverage_request: CoverageSeriesRequest | None = None,
) -> CoverageSeriesEnvelope:
    envelope = _load_canonical_model(path, CoverageSeriesEnvelope)
    _require_common_envelope("coverage envelope", envelope, request, result)
    if envelope.coverage_series_sha256 != _model_sha256(envelope.coverage_series):
        reason = "coverage envelope result hash mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.coverage_request_sha256 != _model_sha256(envelope.coverage_request):
        reason = "coverage envelope request hash mismatch"
        raise PostSearchEvidenceError(reason)
    _require_coverage_contract(
        "coverage envelope",
        envelope,
        request,
        expected_coverage_request,
    )
    _validate_coverage_backings(envelope.coverage_series, request.artifact_root)
    return envelope


def _load_treatment_coverage_envelope(
    path: Path,
    request: CellRunRequest,
    expected_coverage_request: CoverageSeriesRequest | None = None,
) -> CoverageSeriesEnvelope:
    envelope = _load_canonical_model(path, CoverageSeriesEnvelope)
    if envelope.cell_request != request or envelope.cell_request_sha256 != _model_sha256(request):
        reason = "coverage envelope request mismatch"
        raise PostSearchEvidenceError(reason)
    expected_placeholder = _treatment_placeholder_result(request, envelope.coverage_series)
    if envelope.cell_result != expected_placeholder:
        reason = "coverage envelope result mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.cell_result_sha256 != sha256_bytes(b"treatment-failure"):
        reason = "coverage envelope result mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.coverage_series_sha256 != _model_sha256(envelope.coverage_series):
        reason = "coverage envelope result hash mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.coverage_request_sha256 != _model_sha256(envelope.coverage_request):
        reason = "coverage envelope request hash mismatch"
        raise PostSearchEvidenceError(reason)
    _require_coverage_contract(
        "coverage envelope",
        envelope,
        request,
        expected_coverage_request,
    )
    _validate_coverage_backings(envelope.coverage_series, request.artifact_root)
    return envelope


def _require_coverage_contract(
    label: str,
    envelope: CoverageSeriesEnvelope,
    request: CellRunRequest,
    expected_coverage_request: CoverageSeriesRequest | None,
) -> None:
    if (
        expected_coverage_request is not None
        and envelope.coverage_request != expected_coverage_request
    ):
        reason = f"{label} current request mismatch"
        raise PostSearchEvidenceError(reason)
    if (
        envelope.coverage_request.execution_profile != request.execution_profile
        or envelope.coverage_request.schedule_seconds != request.schedule_seconds
        or envelope.coverage_request.corpus_index.execution_profile != request.execution_profile
        or envelope.coverage_request.corpus_index.schedule_seconds != request.schedule_seconds
    ):
        reason = f"{label} profile schedule mismatch"
        raise PostSearchEvidenceError(reason)
    corpus_schedule = tuple(
        row.scheduled_elapsed_seconds for row in envelope.coverage_series.corpus_index
    )
    point_schedule = tuple(
        point.scheduled_elapsed_seconds for point in envelope.coverage_series.points
    )
    if corpus_schedule != request.schedule_seconds or point_schedule != request.schedule_seconds:
        reason = f"{label} coverage schedule mismatch"
        raise PostSearchEvidenceError(reason)


def _validate_coverage_backings(series: CoverageSeries, artifact_root: Path) -> None:
    try:
        validate_coverage_series_backings(artifact_root, series)
    except CoverageSeriesError as error:
        reason = f"coverage backing validation failed: {error}"
        raise PostSearchEvidenceError(reason) from error


def _load_proof_envelope(
    path: Path, request: CellRunRequest, result: CellResult
) -> ProofBatchEnvelope:
    envelope = _load_canonical_model(path, ProofBatchEnvelope)
    _require_common_envelope("proof envelope", envelope, request, result)
    if envelope.proof_result_sha256 != _model_sha256(envelope.proof_result):
        reason = "proof envelope result hash mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.proof_request_sha256 != _model_sha256(envelope.proof_request):
        reason = "proof envelope request hash mismatch"
        raise PostSearchEvidenceError(reason)
    return envelope


def _load_deep_parse_envelope(
    path: Path, request: CellRunRequest, result: CellResult
) -> DeepParseEnvelope:
    envelope = _load_canonical_model(path, DeepParseEnvelope)
    _require_common_envelope("deep-parse envelope", envelope, request, result)
    if envelope.deep_parse_request_sha256 != _model_sha256(envelope.deep_parse_request):
        reason = "deep-parse envelope request hash mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.deep_parse_result_sha256 != _model_sha256(envelope.deep_parse_result):
        reason = "deep-parse envelope result hash mismatch"
        raise PostSearchEvidenceError(reason)
    _validate_deep_parse_bindings(envelope, request, result)
    _validate_deep_parse_backings(envelope, request.artifact_root)
    return envelope


def _validate_deep_parse_bindings(
    envelope: DeepParseEnvelope, request: CellRunRequest, result: CellResult
) -> None:
    if envelope.deep_parse_request.cell_request_sha256 != _model_sha256(
        request
    ) or envelope.deep_parse_request.cell_result_sha256 != _model_sha256(result):
        reason = "deep-parse envelope binding mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.deep_parse_request.artifact_root != request.artifact_root:
        reason = "deep-parse envelope artifact root mismatch"
        raise PostSearchEvidenceError(reason)
    expected_inputs = tuple(
        (path, digest)
        for path, digest in zip(
            envelope.deep_parse_request.input_paths,
            envelope.deep_parse_request.input_sha256,
            strict=True,
        )
    )
    result_inputs = tuple(
        (verdict.input_path, verdict.input_sha256)
        for verdict in envelope.deep_parse_result.verdicts
    )
    if result_inputs != expected_inputs:
        reason = "deep-parse result input set mismatch"
        raise PostSearchEvidenceError(reason)


def _validate_deep_parse_backings(envelope: DeepParseEnvelope, artifact_root: Path) -> None:
    try:
        _ = envelope.deep_parse_result.evidence_root.resolve(strict=True).relative_to(
            artifact_root.resolve(strict=True)
        )
    except (FileNotFoundError, ValueError) as error:
        reason = "deep-parse result evidence root escapes artifact root"
        raise PostSearchEvidenceError(reason) from error
    match envelope.deep_parse_result.replay_kind:
        case "docker":
            manifest = envelope.deep_parse_result.build_manifest_evidence
            if (
                manifest is None
                or manifest.sha256 != envelope.deep_parse_request.build_manifest_sha256
            ):
                reason = "deep-parse build manifest backing mismatch"
                raise PostSearchEvidenceError(reason)
        case "test-only-hash":
            pass
        case _ as unreachable:
            assert_never(unreachable)
    request_bytes = _read_deep_parse_evidence(artifact_root, envelope.deep_parse_request_evidence)
    result_bytes = _read_deep_parse_evidence(artifact_root, envelope.deep_parse_result_evidence)
    if request_bytes != canonical_json_bytes(envelope.deep_parse_request.model_dump(mode="json")):
        reason = "deep-parse request backing mismatch"
        raise PostSearchEvidenceError(reason)
    if result_bytes != canonical_json_bytes(envelope.deep_parse_result.model_dump(mode="json")):
        reason = "deep-parse result backing mismatch"
        raise PostSearchEvidenceError(reason)
    validate_deep_parse_result_evidence(envelope.deep_parse_result)


def _require_current_deep_parse_request(
    envelope: DeepParseEnvelope, expected: DeepParseRequest
) -> None:
    if envelope.deep_parse_request != expected:
        reason = "deep-parse existing envelope current request mismatch"
        raise PostSearchEvidenceError(reason)


def _require_production_deep_parse_result(
    result: DeepParseResult, replayer: DeepParseReplayer
) -> None:
    """Reject synthetic results unless the caller selected the explicit test-only seam."""
    if result.replay_kind == "test-only-hash" and not isinstance(
        replayer, HashOnlyDeepParseReplayer
    ):
        reason = "production deep parse rejects test-only replay evidence"
        raise PostSearchEvidenceError(reason)


def _require_common_envelope(
    label: str,
    envelope: CoverageSeriesEnvelope | ProofBatchEnvelope | DeepParseEnvelope,
    request: CellRunRequest,
    result: CellResult,
) -> None:
    if envelope.cell_request != request or envelope.cell_request_sha256 != _model_sha256(request):
        reason = f"{label} request mismatch"
        raise PostSearchEvidenceError(reason)
    if envelope.cell_result != result or envelope.cell_result_sha256 != _model_sha256(result):
        reason = f"{label} result mismatch"
        raise PostSearchEvidenceError(reason)


def _outcome(  # noqa: PLR0913, PLR0917
    coverage_path: Path,
    coverage: CoverageSeries,
    proof_path: Path,
    proof: ProofBatchResult,
    deep_parse_path: Path,
    deep_parse: DeepParseResult,
) -> PostSearchOutcome:
    return PostSearchOutcome(
        coverage_replay_count=len(coverage.points),
        proof_replay_count=proof.proof_accounting.replay_count,
        coverage_series_sha256=_model_sha256(coverage),
        proof_result_sha256=_model_sha256(proof),
        coverage_envelope_sha256=sha256_bytes(coverage_path.read_bytes()),
        proof_envelope_sha256=sha256_bytes(proof_path.read_bytes()),
        coverage_series_path=coverage_path,
        proof_result_path=proof_path,
        coverage_status="available",
        proof_status="available",
        proof_wall_seconds=proof.proof_accounting.wall_seconds,
        proof_cpu_seconds=proof.proof_accounting.cpu_seconds,
        deep_parse_status="available",
        deep_parse_replay_count=deep_parse.total_count,
        deep_parse_success_count=deep_parse.success_count,
        deep_parse_total_count=deep_parse.total_count,
        deep_parse_result_sha256=_model_sha256(deep_parse),
        deep_parse_envelope_sha256=sha256_bytes(deep_parse_path.read_bytes()),
        deep_parse_result_path=deep_parse_path.with_name("deep-parse-result.json"),
        deep_parse_envelope_path=deep_parse_path,
    )


def _coverage_only_outcome(
    coverage_path: Path,
    coverage: CoverageSeries,
    deep_parse_path: Path,
    deep_parse: DeepParseResult,
) -> PostSearchOutcome:
    return PostSearchOutcome(
        coverage_replay_count=len(coverage.points),
        proof_replay_count=0,
        coverage_series_sha256=_model_sha256(coverage),
        proof_result_sha256="0" * 64,
        coverage_envelope_sha256=sha256_bytes(coverage_path.read_bytes()),
        coverage_series_path=coverage_path,
        coverage_status="available",
        proof_status="unavailable",
        proof_wall_seconds=0.0,
        proof_cpu_seconds=0.0,
        deep_parse_status="available",
        deep_parse_replay_count=deep_parse.total_count,
        deep_parse_success_count=deep_parse.success_count,
        deep_parse_total_count=deep_parse.total_count,
        deep_parse_result_sha256=_model_sha256(deep_parse),
        deep_parse_envelope_sha256=sha256_bytes(deep_parse_path.read_bytes()),
        deep_parse_result_path=deep_parse_path.with_name("deep-parse-result.json"),
        deep_parse_envelope_path=deep_parse_path,
    )


def _deep_parse_request(  # noqa: PLR0913, PLR0917
    request: CellRunRequest,
    result: CellResult,
    coverage: CoverageSeries,
    image_digest: str,
    checker_binary_sha256: str,
    build_manifest_sha256: str,
) -> DeepParseRequest:
    final = coverage.corpus_index[-1]
    return DeepParseRequest(
        cell_id=request.cell_id,
        artifact_root=request.artifact_root,
        cell_request_sha256=_model_sha256(request),
        cell_result_sha256=_model_sha256(result),
        target_commit=request.target_commit,
        image_digest=image_digest,
        checker_binary_sha256=checker_binary_sha256,
        build_manifest_sha256=build_manifest_sha256,
        final_corpus_set_sha256=final.cumulative_set_sha256,
        input_paths=tuple(request.artifact_root / item for item in _stored_paths(coverage)),
        input_sha256=final.cumulative_hashes,
    )


def _stored_paths(coverage: CoverageSeries) -> tuple[Path, ...]:
    by_hash = {
        item.sha256: Path(item.stored_path)
        for row in coverage.corpus_index
        for item in row.new_inputs
    }
    return tuple(by_hash[digest] for digest in coverage.corpus_index[-1].cumulative_hashes)


def _treatment_placeholder_result(request: CellRunRequest, coverage: CoverageSeries) -> CellResult:
    return CellResult(
        cell_id=request.cell_id,
        status="failed",
        started_monotonic_seconds=request.start_monotonic_seconds,
        finished_monotonic_seconds=request.start_monotonic_seconds,
        coverage_freeze_elapsed_seconds=0.0,
        attempt_count=1,
        manifest_sha256="0" * 64,
        event_stream_sha256="0" * 64,
        corpus_index_stream_sha256="0" * 64,
        crash_stream_sha256="0" * 64,
        crash_occurrence_count=0,
        corpus_index_scheduled_seconds=tuple(
            row.scheduled_elapsed_seconds for row in coverage.corpus_index
        ),
        search_accounting=SearchAccounting(cpu_seconds=0.0, wall_seconds=0.0),
    )


def _coverage_path(request: CellRunRequest) -> Path:
    return request.artifact_root / "post-search" / "coverage-envelope.json"


def _proof_path(request: CellRunRequest) -> Path:
    return request.artifact_root / "post-search" / "proof-envelope.json"


def _deep_parse_path(request: CellRunRequest) -> Path:
    return request.artifact_root / "post-search" / "deep-parse-envelope.json"


def _model_sha256(model: FrozenModel) -> Sha256:
    return sha256_bytes(canonical_json_bytes(model.model_dump(mode="json")))


def persist_deep_parse_evidence(
    root: Path, relative_path: Path, payload: bytes
) -> DeepParseEvidenceArtifact:
    """Persist one immutable deep-parse backing and return its exact descriptor."""
    descriptor = DeepParseEvidenceArtifact(
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=sha256_bytes(payload),
    )
    path = _deep_parse_evidence_path(root, descriptor.relative_path, create_parent=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            reason = "deep-parse evidence content mismatch"
            raise PostSearchEvidenceError(reason)
        return descriptor
    try:
        with path.open("xb") as handle:
            _ = handle.write(payload)
    except FileExistsError as error:
        reason = "deep-parse evidence content mismatch"
        raise PostSearchEvidenceError(reason) from error
    return descriptor


def validate_deep_parse_result_evidence(result: DeepParseResult) -> None:
    """Require every result hash to resolve to the declared exact backing bytes."""
    artifacts = tuple(
        artifact
        for verdict in result.verdicts
        for artifact in (
            verdict.input_evidence,
            verdict.argv_evidence,
            verdict.stdout_evidence,
            verdict.stderr_evidence,
            verdict.exit_evidence,
        )
    )
    if result.build_manifest_evidence is not None:
        artifacts = (result.build_manifest_evidence, *artifacts)
    for artifact in artifacts:
        _ = _read_deep_parse_evidence(result.evidence_root, artifact)
    for verdict in result.verdicts:
        if (
            verdict.input_evidence.sha256 != verdict.input_sha256
            or verdict.input_evidence.size_bytes != verdict.size_bytes
        ):
            reason = "deep-parse input backing mismatch"
            raise PostSearchEvidenceError(reason)


def _read_deep_parse_evidence(root: Path, artifact: DeepParseEvidenceArtifact) -> bytes:
    path = _deep_parse_evidence_path(root, artifact.relative_path, create_parent=False)
    if path.is_symlink() or not path.is_file():
        reason = "deep-parse evidence is not a regular non-symlink file"
        raise PostSearchEvidenceError(reason)
    content = path.read_bytes()
    if len(content) != artifact.size_bytes or sha256_bytes(content) != artifact.sha256:
        reason = "deep-parse evidence hash or size mismatch"
        raise PostSearchEvidenceError(reason)
    return content


def _deep_parse_evidence_path(root: Path, relative_path: Path, *, create_parent: bool) -> Path:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        reason = "unsafe deep-parse evidence root"
        raise PostSearchEvidenceError(reason)
    if create_parent:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        reason = "deep-parse evidence root is missing"
        raise PostSearchEvidenceError(reason)
    current = root
    for part in relative_path.parts[:-1]:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            reason = "unsafe deep-parse evidence directory"
            raise PostSearchEvidenceError(reason)
        if create_parent:
            current.mkdir(exist_ok=True)
    path = root / relative_path
    try:
        _ = path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        reason = "deep-parse evidence path escapes evidence root"
        raise PostSearchEvidenceError(reason) from error
    return path


def _write_once_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            reason = "post-search artifact mismatch"
            raise PostSearchEvidenceError(reason)
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                reason = "post-search artifact mismatch"
                raise PostSearchEvidenceError(reason) from None
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _proof_seed(proof_seed_base: int, fuzzer_seed: int) -> int:
    return proof_seed_base + fuzzer_seed


@dataclass(frozen=True, slots=True)
class _ProcessClock:
    """Production clock for proof accounting."""

    def monotonic(self) -> float:
        """Return current monotonic time."""
        return time.monotonic()

    def process_time(self) -> float:
        """Return current process CPU time."""
        return time.process_time()
