"""Hash-bound corpus indexing and incremental external coverage series.

Module-size opt-out ``# noqa: SIZE_OK``: Task 4 owns one coverage-series module.
"""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, Protocol, Self, override

from pydantic import Field, TypeAdapter, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.ledger import JsonValue, append_event, validate_ledger
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    persist_backing_bytes,
    validate_backing_file,
)

type ExecutionProfile = Literal["main", "qualification-smoke"]
SCHEDULE_SECONDS: Final = tuple(range(0, 10_801, 900))
QUALIFICATION_SMOKE_SCHEDULE_SECONDS: Final = (0, 300)
_PROFILE_SCHEDULES: Final[dict[ExecutionProfile, tuple[int, ...]]] = {
    "main": SCHEDULE_SECONDS,
    "qualification-smoke": QUALIFICATION_SMOKE_SCHEDULE_SECONDS,
}
_PROFILE_DURATIONS: Final[dict[ExecutionProfile, int]] = {
    "main": 10_800,
    "qualification-smoke": 300,
}
_MAX_CORPUS_INPUT_BYTES: Final = 1_048_576
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class CoverageSeriesError(Exception):
    reason: str

    def __init__(self, reason: str) -> None:  # noqa: D107
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CorpusIndexRequest(FrozenModel):
    schema_version: Literal["dllm-openexr-corpus-index-request/v1"] = (
        "dllm-openexr-corpus-index-request/v1"
    )
    execution_profile: ExecutionProfile = "main"
    schedule_seconds: tuple[int, ...] = SCHEDULE_SECONDS
    artifact_root: Path
    corpus_root: Path
    store_root: Path
    index_path: Path

    @model_validator(mode="after")
    def _request_contract(self) -> Self:
        _validate_profile_schedule(self.execution_profile, self.schedule_seconds)
        root = _data_root(self.artifact_root)
        for path in (self.corpus_root, self.store_root, self.index_path):
            _require_below(root, path)
        return self


class CorpusInput(FrozenModel):
    sha256: Sha256
    size_bytes: int = Field(gt=0, le=_MAX_CORPUS_INPUT_BYTES)
    stored_path: str = Field(min_length=1)


class CorpusIndexRecord(FrozenModel):
    schema_version: Literal["dllm-openexr-corpus-index/v1"] = "dllm-openexr-corpus-index/v1"
    scheduled_elapsed_seconds: int
    observation_kind: Literal["observed", "frozen"]
    source_elapsed_seconds: float = Field(ge=0.0)
    observed_lateness_seconds: float = Field(ge=0.0)
    new_inputs: tuple[CorpusInput, ...]
    cumulative_hashes: tuple[Sha256, ...]
    cumulative_count: int = Field(ge=1)
    cumulative_bytes: int = Field(gt=0)
    cumulative_set_sha256: Sha256


class TargetCoverageCounts(FrozenModel):
    target_file_count: int = Field(gt=0)
    branches_covered: int = Field(ge=0)
    branches_total: int = Field(gt=0)
    regions_covered: int = Field(ge=0)
    regions_total: int = Field(gt=0)
    lines_covered: int = Field(ge=0)
    lines_total: int = Field(gt=0)

    @model_validator(mode="after")
    def _covered_within_totals(self) -> Self:
        if (
            self.branches_covered > self.branches_total
            or self.regions_covered > self.regions_total
            or self.lines_covered > self.lines_total
        ):
            message = "covered counts exceed totals"
            raise ValueError(message)
        return self


class CoverageStepRequest(FrozenModel):
    scheduled_elapsed_seconds: int
    corpus_set_sha256: Sha256
    new_input_paths: tuple[Path, ...] = Field(min_length=1)
    new_input_sha256: tuple[Sha256, ...] = Field(min_length=1)
    prior_profdata_sha256: Sha256 | None

    @model_validator(mode="after")
    def _paired_inputs(self) -> Self:
        if len(self.new_input_paths) != len(self.new_input_sha256):
            message = "coverage input paths and hashes must have equal length"
            raise ValueError(message)
        return self


class CoverageStepResult(FrozenModel):
    counts: TargetCoverageCounts
    run_argv_sha256: tuple[Sha256, ...] = Field(min_length=1)
    profraw_sha256: tuple[Sha256, ...] = Field(min_length=1)
    merge_argv_sha256: Sha256
    profdata_sha256: Sha256
    report_argv_sha256: Sha256
    report_sha256: Sha256
    run_argv_evidence: tuple[ArtifactBackingEvidence, ...] = ()
    profraw_evidence: tuple[ArtifactBackingEvidence, ...] = ()
    merge_argv_evidence: ArtifactBackingEvidence | None = None
    profdata_evidence: ArtifactBackingEvidence | None = None
    report_argv_evidence: ArtifactBackingEvidence | None = None
    report_evidence: ArtifactBackingEvidence | None = None


class CoverageReplayer(Protocol):
    def replay(self, request: CoverageStepRequest) -> CoverageStepResult: ...


class CoverageSeriesRequest(FrozenModel):
    schema_version: Literal["dllm-openexr-coverage-request/v1"] = "dllm-openexr-coverage-request/v1"
    execution_profile: ExecutionProfile = "main"
    schedule_seconds: tuple[int, ...] = SCHEDULE_SECONDS
    corpus_index: CorpusIndexRequest
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    binary_sha256: Sha256
    build_manifest_sha256: Sha256
    cpu_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _request_contract(self) -> Self:
        _validate_profile_schedule(self.execution_profile, self.schedule_seconds)
        if (
            self.corpus_index.execution_profile != self.execution_profile
            or self.corpus_index.schedule_seconds != self.schedule_seconds
        ):
            message = "coverage request profile schedule conflicts with corpus index"
            raise ValueError(message)
        return self


class CoveragePoint(FrozenModel):
    schema_version: Literal["dllm-openexr-coverage-point/v1"] = "dllm-openexr-coverage-point/v1"
    scheduled_elapsed_seconds: int
    observation_kind: Literal["observed", "frozen"]
    source_elapsed_seconds: float
    corpus_set_sha256: Sha256
    replayed_input_sha256: tuple[Sha256, ...]
    branches_covered: int
    branches_total: int
    regions_covered: int
    regions_total: int
    lines_covered: int
    lines_total: int
    run_argv_sha256: tuple[Sha256, ...] = ()
    run_argv_evidence: tuple[ArtifactBackingEvidence, ...] = ()
    profraw_sha256: tuple[Sha256, ...] = ()
    profraw_evidence: tuple[ArtifactBackingEvidence, ...] = ()
    merge_argv_sha256: Sha256 | None = None
    merge_argv_evidence: ArtifactBackingEvidence | None = None
    profdata_sha256: Sha256
    profdata_evidence: ArtifactBackingEvidence | None = None
    report_argv_sha256: Sha256 | None = None
    report_argv_evidence: ArtifactBackingEvidence | None = None
    report_sha256: Sha256
    report_evidence: ArtifactBackingEvidence | None = None


class CoverageSeries(FrozenModel):
    schema_version: Literal["dllm-openexr-coverage-series/v1"] = "dllm-openexr-coverage-series/v1"
    corpus_index: tuple[CorpusIndexRecord, ...]
    points: tuple[CoveragePoint, ...]

    @model_validator(mode="after")
    def _points_have_backing_evidence(self) -> Self:
        validate_coverage_series_backings(Path(), self, reopen=False)
        return self


def snapshot_corpus(
    request: CorpusIndexRequest, scheduled_elapsed_seconds: int, observed_elapsed_seconds: float
) -> CorpusIndexRecord:
    prior = load_corpus_index(request)
    schedule = request.schedule_seconds
    if (
        len(prior) >= len(schedule)
        or scheduled_elapsed_seconds != schedule[len(prior)]
        or (prior and prior[-1].observation_kind == "frozen")
    ):
        raise CoverageSeriesError(reason="corpus index schedule gap")
    current = _scan_corpus(request)
    previous_hashes = prior[-1].cumulative_hashes if prior else ()
    cumulative = tuple(sorted(set(previous_hashes) | set(current)))
    new_hashes = tuple(item for item in cumulative if item not in previous_hashes)
    inputs = tuple(current[item] for item in new_hashes)
    # fmt: off
    record = CorpusIndexRecord(scheduled_elapsed_seconds=scheduled_elapsed_seconds, observation_kind="observed", source_elapsed_seconds=observed_elapsed_seconds, observed_lateness_seconds=max(0.0, observed_elapsed_seconds - scheduled_elapsed_seconds), new_inputs=inputs, cumulative_hashes=cumulative, cumulative_count=len(cumulative), cumulative_bytes=sum(_stored_size(request, item) for item in cumulative), cumulative_set_sha256=_set_hash(cumulative))  # noqa: E501
    _ = append_event(request.index_path, timestamp_utc=f"monotonic+{scheduled_elapsed_seconds:05d}", event_type="corpus_index", payload=_JSON_OBJECT.validate_python(record.model_dump(mode="json")))  # noqa: E501
    # fmt: on
    return record


def freeze_corpus_index(
    request: CorpusIndexRequest, *, freeze_elapsed_seconds: float
) -> tuple[CorpusIndexRecord, ...]:
    rows = list(load_corpus_index(request))
    if not rows:
        raise CoverageSeriesError(reason="cannot freeze an empty corpus index")
    while len(rows) < len(request.schedule_seconds):
        prior = rows[-1]
        scheduled = request.schedule_seconds[len(rows)]
        # fmt: off
        record = CorpusIndexRecord(scheduled_elapsed_seconds=scheduled, observation_kind="frozen", source_elapsed_seconds=freeze_elapsed_seconds, observed_lateness_seconds=0.0, new_inputs=(), cumulative_hashes=prior.cumulative_hashes, cumulative_count=prior.cumulative_count, cumulative_bytes=prior.cumulative_bytes, cumulative_set_sha256=prior.cumulative_set_sha256)  # noqa: E501
        _validate_record(request, tuple(rows), record)
        _ = append_event(request.index_path, timestamp_utc=f"frozen+{freeze_elapsed_seconds:011.3f}+{scheduled:05d}", event_type="corpus_index", payload=_JSON_OBJECT.validate_python(record.model_dump(mode="json")))  # noqa: E501
        # fmt: on
        rows.append(record)
    return tuple(rows)


def load_corpus_index(request: CorpusIndexRequest) -> tuple[CorpusIndexRecord, ...]:
    if not request.index_path.exists():
        return ()
    _ = validate_ledger(request.index_path)
    rows: list[CorpusIndexRecord] = []
    for line in request.index_path.read_bytes().splitlines():
        outer = _JSON_OBJECT.validate_json(line)
        if outer.get("event_type") != "corpus_index":
            raise CoverageSeriesError(reason="unexpected corpus index event")
        payload = json.dumps(outer.get("payload"), separators=(",", ":"), sort_keys=True)
        record = CorpusIndexRecord.model_validate_json(payload)
        _validate_record(request, tuple(rows), record)
        rows.append(record)
    return tuple(rows)


def build_coverage_series(
    request: CoverageSeriesRequest, replayer: CoverageReplayer
) -> CoverageSeries:
    rows = load_corpus_index(request.corpus_index)
    if tuple(row.scheduled_elapsed_seconds for row in rows) != request.schedule_seconds:
        raise CoverageSeriesError(reason="coverage requires complete request schedule corpus index")
    points: list[CoveragePoint] = []
    result: CoverageStepResult | None = None
    for row in rows:
        if row.new_inputs:
            # fmt: off
            step = CoverageStepRequest(scheduled_elapsed_seconds=row.scheduled_elapsed_seconds, corpus_set_sha256=row.cumulative_set_sha256, new_input_paths=tuple(request.corpus_index.artifact_root / item.stored_path for item in row.new_inputs), new_input_sha256=tuple(item.sha256 for item in row.new_inputs), prior_profdata_sha256=None if result is None else result.profdata_sha256)  # noqa: E501
            # fmt: on
            result = replayer.replay(step)
            _validate_coverage_result_evidence(request.corpus_index.artifact_root, result)
        if result is None:
            raise CoverageSeriesError(reason="initial corpus snapshot has no replayable input")
        points.append(_point(row, result))
    return CoverageSeries(corpus_index=rows, points=tuple(points))


def _scan_corpus(request: CorpusIndexRequest) -> dict[str, CorpusInput]:
    if not request.corpus_root.is_dir():
        raise CoverageSeriesError(reason="corpus root is not a directory")
    result: dict[str, CorpusInput] = {}
    for path in sorted(request.corpus_root.rglob("*")):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise CoverageSeriesError(reason=f"unsafe corpus path: {path}")
        if not path.is_file():
            continue
        content = path.read_bytes()
        if not content or len(content) > _MAX_CORPUS_INPUT_BYTES:
            raise CoverageSeriesError(reason=f"invalid corpus input size: {path}")
        digest = sha256_bytes(content)
        destination = request.store_root / digest
        stored_path = destination.relative_to(request.artifact_root).as_posix()
        _ = persist_backing_bytes(request.artifact_root, stored_path, content)
        result[digest] = CorpusInput(
            sha256=digest,
            size_bytes=len(content),
            stored_path=stored_path,
        )
    return result


def _validate_record(
    request: CorpusIndexRequest, prior: tuple[CorpusIndexRecord, ...], record: CorpusIndexRecord
) -> None:
    _validate_schedule(request.schedule_seconds, prior, record)
    previous = prior[-1].cumulative_hashes if prior else ()
    _validate_cumulative_record(request, previous, record)
    if record.observation_kind == "observed":
        _validate_observed_record(prior, record)
        return
    _validate_frozen_record(prior, previous, record)


def _validate_schedule(
    schedule: tuple[int, ...], prior: tuple[CorpusIndexRecord, ...], record: CorpusIndexRecord
) -> None:
    if len(prior) >= len(schedule) or record.scheduled_elapsed_seconds != schedule[len(prior)]:
        raise CoverageSeriesError(reason="corpus index schedule gap")


def _validate_cumulative_record(
    request: CorpusIndexRequest,
    previous: tuple[str, ...],
    record: CorpusIndexRecord,
) -> None:
    for item in record.new_inputs:
        _validate_corpus_input(request, item)
    expected_new = tuple(item for item in record.cumulative_hashes if item not in previous)
    if (
        record.cumulative_hashes != tuple(sorted(record.cumulative_hashes))
        or tuple(item.sha256 for item in record.new_inputs) != expected_new
    ):
        raise CoverageSeriesError(reason="corpus index cumulative-set mismatch")
    if set(previous) - set(record.cumulative_hashes):
        raise CoverageSeriesError(reason="corpus index cumulative set decreased")
    if record.cumulative_count != len(record.cumulative_hashes):
        raise CoverageSeriesError(reason="corpus index count mismatch")
    if record.cumulative_set_sha256 != _set_hash(record.cumulative_hashes):
        raise CoverageSeriesError(reason="corpus index set hash mismatch")
    if record.cumulative_bytes != sum(
        _stored_size(request, item) for item in record.cumulative_hashes
    ):
        raise CoverageSeriesError(reason="corpus index byte count mismatch")


def _validate_observed_record(
    prior: tuple[CorpusIndexRecord, ...], record: CorpusIndexRecord
) -> None:
    if prior and prior[-1].observation_kind == "frozen":
        raise CoverageSeriesError(reason="observed corpus row follows frozen row")
    if record.source_elapsed_seconds < record.scheduled_elapsed_seconds or (
        record.observed_lateness_seconds
        != record.source_elapsed_seconds - record.scheduled_elapsed_seconds
    ):
        raise CoverageSeriesError(reason="observed corpus timing mismatch")


def _validate_frozen_record(
    prior: tuple[CorpusIndexRecord, ...],
    previous: tuple[str, ...],
    record: CorpusIndexRecord,
) -> None:
    if (
        not prior
        or record.new_inputs
        or record.source_elapsed_seconds > record.scheduled_elapsed_seconds
    ):
        raise CoverageSeriesError(reason="invalid frozen corpus row")
    if (
        record.cumulative_hashes != previous
        or record.cumulative_bytes != prior[-1].cumulative_bytes
        or (
            prior[-1].observation_kind == "frozen"
            and record.source_elapsed_seconds != prior[-1].source_elapsed_seconds
        )
    ):
        raise CoverageSeriesError(reason="frozen corpus row changed last value")


def _validate_corpus_input(request: CorpusIndexRequest, item: CorpusInput) -> None:
    try:
        content = validate_backing_file(
            request.artifact_root,
            ArtifactBackingEvidence(
                relative_path=item.stored_path,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
            ),
        )
    except EvidenceBackingError as error:
        raise CoverageSeriesError(reason=str(error)) from error
    try:
        _ = (
            (request.artifact_root / item.stored_path)
            .resolve()
            .relative_to(request.store_root.resolve())
        )
    except ValueError as error:
        raise CoverageSeriesError(reason="corpus input backing path escapes store root") from error
    if not content:
        raise CoverageSeriesError(reason="corpus input backing evidence is empty")


def _stored_size(request: CorpusIndexRequest, digest: str) -> int:
    path = request.store_root / digest
    if path.is_symlink() or not path.is_file():
        raise CoverageSeriesError(reason=f"missing corpus content: {digest}")
    content = path.read_bytes()
    if sha256_bytes(content) != digest:
        raise CoverageSeriesError(reason=f"corpus content hash mismatch: {digest}")
    return len(content)


def _validate_coverage_result_evidence(root: Path, result: CoverageStepResult) -> None:
    if (
        len(result.run_argv_evidence) != len(result.run_argv_sha256)
        or len(result.profraw_evidence) != len(result.profraw_sha256)
        or result.merge_argv_evidence is None
        or result.profdata_evidence is None
        or result.report_argv_evidence is None
        or result.report_evidence is None
    ):
        raise CoverageSeriesError(reason="coverage result lacks dereferenceable evidence")
    pairs = (
        *zip(result.run_argv_sha256, result.run_argv_evidence, strict=True),
        *zip(result.profraw_sha256, result.profraw_evidence, strict=True),
        (result.merge_argv_sha256, result.merge_argv_evidence),
        (result.profdata_sha256, result.profdata_evidence),
        (result.report_argv_sha256, result.report_argv_evidence),
        (result.report_sha256, result.report_evidence),
    )
    for expected, evidence in pairs:
        if evidence.sha256 != expected:
            raise CoverageSeriesError(reason="coverage evidence hash mismatch")
        try:
            _ = validate_backing_file(root, evidence)
        except EvidenceBackingError as error:
            raise CoverageSeriesError(reason=str(error)) from error


def validate_coverage_series_backings(
    artifact_root: Path, series: CoverageSeries, *, reopen: bool = True
) -> None:
    for point in series.points:
        _validate_coverage_point_backings(artifact_root, point, reopen=reopen)


def _validate_coverage_point_backings(
    artifact_root: Path, point: CoveragePoint, *, reopen: bool
) -> None:
    if (
        len(point.run_argv_evidence) != len(point.run_argv_sha256)
        or len(point.profraw_evidence) != len(point.profraw_sha256)
        or point.merge_argv_sha256 is None
        or point.merge_argv_evidence is None
        or point.profdata_evidence is None
        or point.report_argv_sha256 is None
        or point.report_argv_evidence is None
        or point.report_evidence is None
    ):
        raise CoverageSeriesError(reason="coverage backing evidence is incomplete")
    pairs = (
        *zip(point.run_argv_sha256, point.run_argv_evidence, strict=True),
        *zip(point.profraw_sha256, point.profraw_evidence, strict=True),
        (point.merge_argv_sha256, point.merge_argv_evidence),
        (point.profdata_sha256, point.profdata_evidence),
        (point.report_argv_sha256, point.report_argv_evidence),
        (point.report_sha256, point.report_evidence),
    )
    for expected, evidence in pairs:
        if evidence.sha256 != expected:
            raise CoverageSeriesError(reason="coverage backing evidence hash mismatch")
        if reopen:
            try:
                _ = validate_backing_file(artifact_root, evidence)
            except EvidenceBackingError as error:
                raise CoverageSeriesError(reason=str(error)) from error


def _point(row: CorpusIndexRecord, result: CoverageStepResult) -> CoveragePoint:
    counts = result.counts
    if result.profdata_evidence is None or result.report_evidence is None:
        raise CoverageSeriesError(reason="coverage result lacks dereferenceable evidence")
    # fmt: off
    return CoveragePoint(scheduled_elapsed_seconds=row.scheduled_elapsed_seconds, observation_kind=row.observation_kind, source_elapsed_seconds=row.source_elapsed_seconds, corpus_set_sha256=row.cumulative_set_sha256, replayed_input_sha256=tuple(item.sha256 for item in row.new_inputs), branches_covered=counts.branches_covered, branches_total=counts.branches_total, regions_covered=counts.regions_covered, regions_total=counts.regions_total, lines_covered=counts.lines_covered, lines_total=counts.lines_total, run_argv_sha256=result.run_argv_sha256, run_argv_evidence=result.run_argv_evidence, profraw_sha256=result.profraw_sha256, profraw_evidence=result.profraw_evidence, merge_argv_sha256=result.merge_argv_sha256, merge_argv_evidence=result.merge_argv_evidence, profdata_sha256=result.profdata_sha256, profdata_evidence=result.profdata_evidence, report_argv_sha256=result.report_argv_sha256, report_argv_evidence=result.report_argv_evidence, report_sha256=result.report_sha256, report_evidence=result.report_evidence)  # noqa: E501
    # fmt: on


def _set_hash(hashes: tuple[str, ...]) -> str:
    return sha256_bytes("\n".join(hashes).encode())


def execution_profile_duration_seconds(profile: ExecutionProfile) -> int:
    return _PROFILE_DURATIONS[profile]


def _validate_profile_schedule(
    profile: ExecutionProfile, schedule_seconds: tuple[int, ...]
) -> None:
    if schedule_seconds != _PROFILE_SCHEDULES[profile]:
        message = "execution profile requires its preregistered schedule"
        raise ValueError(message)


def _data_root(path: Path) -> Path:
    root = path.resolve()
    if not root.is_absolute() or root == Path("/") or Path("/data") not in root.parents:
        message = "artifact root must be a private absolute /data path"
        raise ValueError(message)
    return root


def _require_below(root: Path, path: Path) -> None:
    try:
        _ = path.resolve().relative_to(root)
    except ValueError as error:
        message = "cell evidence path escapes artifact root"
        raise ValueError(message) from error
