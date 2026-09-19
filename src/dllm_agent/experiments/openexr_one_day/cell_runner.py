"""Absolute-deadline OpenEXR search cell and crash evidence stream.

Module-size opt-out ``# noqa: SIZE_OK``: the task forbids another production helper module.
"""

# ruff: noqa: D101, D102, D103, D107

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, Protocol, Self, final, override

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import (
    EMPTY_STREAM_SHA256,
    JsonValue,
    LedgerError,
    append_event,
    validate_ledger,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRecord,
    CorpusIndexRequest,
    CoverageSeriesError,
    ExecutionProfile,
    execution_profile_duration_seconds,
    freeze_corpus_index,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    CrashOccurrenceRecord,
    ProofBatchError,
    append_crash_occurrence,
    load_crash_stream,
)

EXPECTED_SANITIZERS: Final = ("address", "undefined")
_MAX_RESTART_ATTEMPTS: Final = 1_024
_MAX_ARTIFACT_BYTES: Final = 1_048_576
_MAX_LOG_BYTES: Final = 8_388_608
_CELL_DURATION_SECONDS: Final = 10_800.0
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_UBSAN_OPTIONS: Final = "halt_on_error=1:print_stacktrace=1"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MANIFEST_NAME: Final = "run-manifest.json"
_EVENTS_NAME: Final = "events.jsonl"
_RESULT_NAME: Final = "final-result.json"
type SearchArtifactKind = Literal["sanitizer", "timeout", "oom", "slow_unit", "crash"]
type ArtifactCategory = Literal["inputs", "timeout", "oom", "slow_unit", "crash"]
type _ArtifactBacking = tuple[str, ArtifactCategory, str, int]
_ARTIFACT_CATEGORIES: Final[tuple[ArtifactCategory, ...]] = (
    "inputs",
    "timeout",
    "oom",
    "slow_unit",
    "crash",
)


class CellEvidenceError(Exception):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class SearchBuildProvenance(FrozenModel):
    schema_version: Literal["dllm-openexr-search-build/v1"] = "dllm-openexr-search-build/v1"
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    binary_sha256: Sha256
    sanitizers: tuple[Literal["address", "undefined"], ...]

    @model_validator(mode="after")
    def _exact_sanitizers(self) -> Self:
        if self.sanitizers != EXPECTED_SANITIZERS:
            message = "search build requires canonical address and undefined sanitizers"
            raise ValueError(message)
        return self


class CellRunRequest(FrozenModel):
    schema_version: Literal["dllm-openexr-cell-request/v1"] = "dllm-openexr-cell-request/v1"
    execution_profile: ExecutionProfile = "main"
    schedule_seconds: tuple[int, ...] = SCHEDULE_SECONDS
    cell_id: str = Field(min_length=1)
    artifact_root: Path
    corpus_root: Path
    crash_root: Path
    log_root: Path
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    harness_sha256: Sha256
    config_sha256: Sha256
    bank_sha256: Sha256
    start_monotonic_seconds: float
    deadline_monotonic_seconds: float
    fuzzer_seed: int = Field(ge=0)
    max_input_length: int = Field(gt=0, le=_MAX_ARTIFACT_BYTES)
    target_timeout_seconds: int = Field(gt=0)
    rss_limit_mb: int = Field(ge=64)
    cpu_index: int = Field(ge=0)
    dictionary_path: Path
    dictionary_sha256: Sha256
    build: SearchBuildProvenance

    @model_validator(mode="after")
    def _fixed_private_cell(self) -> Self:
        root = _private_data_root(self.artifact_root)
        paths = (self.corpus_root, self.crash_root, self.log_root)
        for path in paths:
            _require_below(root, path)
        if len({path.resolve() for path in paths}) != len(paths):
            message = "cell roots must be distinct"
            raise ValueError(message)
        duration_seconds = execution_profile_duration_seconds(self.execution_profile)
        if self.deadline_monotonic_seconds - self.start_monotonic_seconds != duration_seconds:
            message = "cell deadline must match execution profile duration"
            raise ValueError(message)
        if self.schedule_seconds[-1] != duration_seconds:
            message = "cell schedule must end at execution profile duration"
            raise ValueError(message)
        _ = CorpusIndexRequest(
            execution_profile=self.execution_profile,
            schedule_seconds=self.schedule_seconds,
            artifact_root=self.artifact_root,
            corpus_root=self.corpus_root,
            store_root=self.artifact_root / "corpus-inputs",
            index_path=self.artifact_root / "corpus-index.jsonl",
        )
        _validate_dictionary_file(self.dictionary_path, self.dictionary_sha256)
        return self


class SearchArtifact(FrozenModel):
    name: str = Field(min_length=1)
    kind: SearchArtifactKind
    content: bytes = Field(min_length=1, max_length=_MAX_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _safe_name(self) -> Self:
        if Path(self.name).name != self.name:
            message = "artifact name must be one path component"
            raise ValueError(message)
        return self


class SearchAttemptResult(FrozenModel):
    outcome: Literal["completed", "target_crash", "failed", "deadline"]
    return_code: int
    log: bytes = Field(max_length=8_388_608)
    cpu_seconds: float = Field(ge=0.0)
    artifacts: tuple[SearchArtifact, ...] = Field(max_length=64)


class SearchLaunchRequest(FrozenModel):
    execution_profile: ExecutionProfile = "main"
    schedule_seconds: tuple[int, ...] = SCHEDULE_SECONDS
    cell_id: str
    attempt_index: int = Field(ge=0)
    attempt_root: Path
    artifact_root: Path
    corpus_root: Path
    crash_root: Path
    log_root: Path
    absolute_deadline_monotonic_seconds: float
    remaining_seconds: float = Field(gt=0.0, le=10_800.0)
    fuzzer_seed: int = Field(ge=0)
    max_input_length: int
    target_timeout_seconds: int
    rss_limit_mb: int
    cpu_index: int
    dictionary_path: Path
    dictionary_sha256: Sha256
    build: SearchBuildProvenance

    @model_validator(mode="after")
    def _private_paths(self) -> Self:
        duration_seconds = execution_profile_duration_seconds(self.execution_profile)
        if (
            self.schedule_seconds[-1] != duration_seconds
            or self.remaining_seconds > duration_seconds
        ):
            message = "search launch timing exceeds execution profile"
            raise ValueError(message)
        _ = CorpusIndexRequest(
            execution_profile=self.execution_profile,
            schedule_seconds=self.schedule_seconds,
            artifact_root=self.artifact_root,
            corpus_root=self.corpus_root,
            store_root=self.artifact_root / "corpus-inputs",
            index_path=self.artifact_root / "corpus-index.jsonl",
        )
        root = _private_data_root(self.artifact_root)
        for path in (self.attempt_root, self.corpus_root, self.crash_root, self.log_root):
            _require_below(root, path)
        _validate_dictionary_file(self.dictionary_path, self.dictionary_sha256)
        return self


class SearchProcess(Protocol):
    def wait_until(self, deadline_monotonic_seconds: float) -> SearchAttemptResult | None: ...
    def terminate(self) -> SearchAttemptResult: ...


class SearchLauncher(Protocol):
    def launch(self, request: SearchLaunchRequest) -> SearchProcess: ...


class CellClock(Protocol):
    def monotonic(self) -> float: ...


class SearchChildProcess(Protocol):
    returncode: int | None

    def communicate(
        self,
        input: bytes | None = None,  # noqa: A002
        timeout: float | None = None,
    ) -> tuple[bytes, bytes | None]: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


type SearchPopenFactory = Callable[[tuple[str, ...]], SearchChildProcess]


@dataclass(frozen=True, slots=True)
class _DockerProcessRuntime:
    monotonic: Callable[[], float]
    cpu_clock: Callable[[], float]
    on_finish: Callable[[DockerSearchProcess], None]


@final
class DockerSearchLauncher:
    def __init__(
        self,
        popen_factory: SearchPopenFactory | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] | None = None,
    ) -> None:
        self._popen = _start_search if popen_factory is None else popen_factory
        self._monotonic = monotonic
        self._cpu_clock = _children_cpu_seconds if cpu_clock is None else cpu_clock
        self._active: set[DockerSearchProcess] = set()
        self._active_lock = threading.Lock()

    def launch(self, request: SearchLaunchRequest) -> DockerSearchProcess:
        try:
            request.attempt_root.mkdir(parents=True, exist_ok=False)
            request.log_root.mkdir(parents=True, exist_ok=True)
            started_cpu_seconds = self._cpu_clock()
            child = self._popen(build_long_probe_argv(request))
        except OSError as error:
            raise CellEvidenceError(reason=f"docker search launch failed: {error}") from error
        process = DockerSearchProcess(
            request,
            child,
            _DockerProcessRuntime(self._monotonic, self._cpu_clock, self._discard),
            started_cpu_seconds,
        )
        with self._active_lock:
            self._active.add(process)
        return process

    def cancel_active(self) -> None:
        """Terminate, bounded-kill, and reap every registered Docker search."""
        with self._active_lock:
            active = tuple(self._active)
        for process in active:
            process.request_terminate()
        for process in active:
            _ = process.reap_terminated()

    def _discard(self, process: DockerSearchProcess) -> None:
        with self._active_lock:
            self._active.discard(process)


@final
class DockerSearchProcess:
    def __init__(
        self,
        request: SearchLaunchRequest,
        child: SearchChildProcess,
        runtime: _DockerProcessRuntime,
        started_cpu_seconds: float,
    ) -> None:
        self._request = request
        self._child = child
        self._monotonic = runtime.monotonic
        self._cpu_clock = runtime.cpu_clock
        self._started_cpu_seconds = started_cpu_seconds
        self._on_finish = runtime.on_finish
        self._result: SearchAttemptResult | None = None
        self._condition = threading.Condition()
        self._communicating = False
        self._termination_requested = False
        self._kill_requested = False

    def wait_until(self, deadline_monotonic_seconds: float) -> SearchAttemptResult | None:
        try:
            return self._communicate_once(
                timeout=max(0.0, deadline_monotonic_seconds - self._monotonic()),
                forced_deadline=False,
            )
        except subprocess.TimeoutExpired:
            return None

    def terminate(self) -> SearchAttemptResult:
        self.request_terminate()
        return self.reap_terminated()

    def request_terminate(self) -> None:
        """Send TERM once without competing with an existing communicator."""
        with self._condition:
            if self._result is not None or self._termination_requested:
                return
            self._termination_requested = True
        self._child.terminate()

    def reap_terminated(self) -> SearchAttemptResult:
        """Wait five seconds for TERM, then KILL and wait five more seconds."""
        result, ready = self._await_result_or_idle(5.0)
        if result is not None:
            return result
        if not ready:
            self._request_kill()
            result, ready = self._await_result_or_idle(5.0)
            if result is not None:
                return result
            if not ready:
                raise CellEvidenceError(reason="docker search could not be reaped")
        try:
            return self._communicate_once(timeout=5.0, forced_deadline=True)
        except subprocess.TimeoutExpired:
            self._request_kill()
            try:
                return self._communicate_once(timeout=5.0, forced_deadline=True)
            except subprocess.TimeoutExpired as error:
                raise CellEvidenceError(reason="docker search could not be reaped") from error

    def _request_kill(self) -> None:
        with self._condition:
            if self._result is not None or self._kill_requested:
                return
            self._kill_requested = True
        self._child.kill()

    def _await_result_or_idle(
        self, timeout_seconds: float
    ) -> tuple[SearchAttemptResult | None, bool]:
        with self._condition:
            if self._result is not None or not self._communicating:
                return self._result, True
            ready = self._condition.wait_for(
                lambda: self._result is not None or not self._communicating,
                timeout=timeout_seconds,
            )
            return self._result, ready

    def _communicate_once(self, *, timeout: float, forced_deadline: bool) -> SearchAttemptResult:
        with self._condition:
            if self._result is not None:
                return self._result
            if self._communicating:
                raise CellEvidenceError(reason="docker search already has an active waiter")
            self._communicating = True
        try:
            streams = self._child.communicate(timeout=timeout)
            return self._finish(
                streams,
                forced_deadline=forced_deadline or self._termination_requested,
            )
        finally:
            with self._condition:
                self._communicating = False
                self._condition.notify_all()

    def _finish(
        self, streams: tuple[bytes | None, bytes | None], *, forced_deadline: bool
    ) -> SearchAttemptResult:
        return_code = self._child.returncode
        if return_code is None:
            raise CellEvidenceError(reason="docker search ended without a return code")
        # _start_search merges stderr into stdout (stderr=STDOUT), so a real
        # Popen.communicate() returns (stdout, None); fakes return both streams.
        log = (streams[0] or b"") + (streams[1] or b"")
        if len(log) > _MAX_LOG_BYTES:
            raise CellEvidenceError(reason="docker search log exceeds evidence limit")
        artifacts = _harvest_attempt(self._request, log)
        _persist_bytes(
            self._request.log_root / f"attempt-{self._request.attempt_index:06d}.log", log
        )
        outcome = _process_outcome(return_code, artifacts, forced_deadline)
        self._result = SearchAttemptResult(
            outcome=outcome,
            return_code=return_code,
            log=log,
            cpu_seconds=max(0.0, self._cpu_clock() - self._started_cpu_seconds),
            artifacts=artifacts,
        )
        self._on_finish(self)
        return self._result


class SearchAccounting(FrozenModel):
    schema_version: Literal["dllm-openexr-search-accounting/v1"] = (
        "dllm-openexr-search-accounting/v1"
    )
    cpu_seconds: float = Field(ge=0.0)
    wall_seconds: float = Field(ge=0.0)


class CellManifest(FrozenModel):
    schema_version: Literal["dllm-openexr-cell-manifest/v1"] = "dllm-openexr-cell-manifest/v1"
    request: CellRunRequest


class CellArtifactEvidence(FrozenModel):
    schema_version: Literal["dllm-openexr-cell-artifact/v1"] = "dllm-openexr-cell-artifact/v1"
    sha256: Sha256
    kind: SearchArtifactKind
    category: ArtifactCategory
    stored_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0, le=_MAX_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _kind_matches_category(self) -> Self:
        expected: ArtifactCategory = "inputs" if self.kind == "sanitizer" else self.kind
        if self.category != expected:
            message = "cell artifact kind and category conflict"
            raise ValueError(message)
        return self


class CellEvent(FrozenModel):
    schema_version: Literal["dllm-openexr-cell-event/v2"] = "dllm-openexr-cell-event/v2"
    sequence: int = Field(ge=0)
    attempt_index: int = Field(ge=0)
    outcome: Literal["completed", "target_crash", "failed", "deadline"]
    return_code: int
    observed_monotonic_seconds: float
    remaining_seconds_at_launch: float = Field(gt=0.0)
    cpu_seconds: float = Field(ge=0.0)
    log_sha256: Sha256
    artifact_sha256: tuple[Sha256, ...]
    artifacts: tuple[CellArtifactEvidence, ...]
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def _canonical_artifacts(self) -> Self:
        evidence_hashes = tuple(artifact.sha256 for artifact in self.artifacts)
        if self.artifact_sha256 != evidence_hashes or len(evidence_hashes) != len(
            set(evidence_hashes)
        ):
            message = "cell event artifact digests must be unique"
            raise ValueError(message)
        sanitizer_present = any(artifact.kind == "sanitizer" for artifact in self.artifacts)
        if sanitizer_present and self.outcome != "target_crash":
            message = "sanitizer artifact requires target_crash outcome"
            raise ValueError(message)
        if self.outcome == "target_crash" and not sanitizer_present:
            message = "target_crash outcome requires sanitizer artifact"
            raise ValueError(message)
        return self


class CellResult(FrozenModel):
    schema_version: Literal["dllm-openexr-cell-result/v1"] = "dllm-openexr-cell-result/v1"
    execution_profile: ExecutionProfile = "main"
    cell_id: str
    status: Literal["completed", "failed", "deadline", "restart_limit"]
    started_monotonic_seconds: float
    finished_monotonic_seconds: float
    coverage_freeze_elapsed_seconds: float | None
    attempt_count: int = Field(ge=1)
    manifest_sha256: Sha256
    event_stream_sha256: Sha256
    corpus_index_stream_sha256: Sha256
    crash_stream_sha256: Sha256
    crash_occurrence_count: int = Field(ge=0)
    corpus_index_scheduled_seconds: tuple[int, ...]
    proof_replay_count: Literal[0] = 0
    search_accounting: SearchAccounting


def run_openexr_cell(
    request: CellRunRequest,
    launcher: SearchLauncher,
    clock: CellClock,
    *,
    preinitialized_corpus_index: tuple[CorpusIndexRecord, ...] = (),
) -> CellResult:
    _require_fresh_cell(request, preinitialized_corpus_index)
    _prepare_roots(request)
    manifest = CellManifest(request=request)
    manifest_payload = canonical_json_bytes(manifest.model_dump(mode="json"))
    _persist_new_evidence(request.artifact_root / _MANIFEST_NAME, manifest_payload)
    manifest_hash = sha256_bytes(manifest_payload)
    index_request = _index_request(request)
    rows = list(preinitialized_corpus_index)
    if not rows:
        rows.append(
            snapshot_corpus(index_request, 0, clock.monotonic() - request.start_monotonic_seconds)
        )
    cpu_seconds = 0.0
    attempt = 0
    last_observed: float | None = None
    status: Literal["completed", "failed", "deadline", "restart_limit"] = "failed"
    while (
        clock.monotonic() < request.deadline_monotonic_seconds and attempt < _MAX_RESTART_ATTEMPTS
    ):
        launch = _launch_request(request, attempt, clock.monotonic())
        process = launcher.launch(launch)
        outcome: SearchAttemptResult | None = None
        while outcome is None:
            next_deadline = _next_wait_deadline(request, len(rows))
            outcome = process.wait_until(next_deadline)
            while len(rows) < len(request.schedule_seconds) and clock.monotonic() >= (
                request.start_monotonic_seconds + request.schedule_seconds[len(rows)]
            ):
                scheduled = request.schedule_seconds[len(rows)]
                rows.append(
                    snapshot_corpus(
                        index_request,
                        scheduled,
                        clock.monotonic() - request.start_monotonic_seconds,
                    )
                )
            if outcome is None and clock.monotonic() >= request.deadline_monotonic_seconds:
                outcome = process.terminate()
        cpu_seconds += outcome.cpu_seconds
        artifact_evidence = _record_artifacts(request, outcome, attempt, manifest_hash)
        last_observed = min(clock.monotonic(), request.deadline_monotonic_seconds)
        _append_cell_event(
            request,
            CellEvent(
                sequence=attempt,
                attempt_index=attempt,
                outcome=outcome.outcome,
                return_code=outcome.return_code,
                observed_monotonic_seconds=last_observed,
                remaining_seconds_at_launch=launch.remaining_seconds,
                cpu_seconds=outcome.cpu_seconds,
                log_sha256=sha256_bytes(outcome.log),
                artifact_sha256=tuple(item.sha256 for item in artifact_evidence),
                artifacts=artifact_evidence,
                manifest_sha256=manifest_hash,
            ),
        )
        attempt += 1
        if (
            outcome.outcome == "target_crash"
            and clock.monotonic() < request.deadline_monotonic_seconds
        ):
            continue
        status = _terminal_status(outcome, clock.monotonic(), request.deadline_monotonic_seconds)
        break
    if attempt >= _MAX_RESTART_ATTEMPTS and clock.monotonic() < request.deadline_monotonic_seconds:
        status = "restart_limit"
    # validate_cell_evidence requires finished == the last event's observed time
    # exactly; a second clock read here would drift from it by real elapsed time.
    finished = (
        last_observed
        if last_observed is not None
        else min(clock.monotonic(), request.deadline_monotonic_seconds)
    )
    freeze_elapsed = (
        finished - request.start_monotonic_seconds
        if finished < request.deadline_monotonic_seconds
        else None
    )
    if freeze_elapsed is not None:
        rows = list(freeze_corpus_index(index_request, freeze_elapsed_seconds=freeze_elapsed))
    occurrences = load_crash_occurrences(request)
    event_stream = _ledger_hash(request.artifact_root / _EVENTS_NAME, allow_missing=False)
    corpus_stream = _ledger_hash(index_request.index_path, allow_missing=False)
    crash_stream = _ledger_hash(request.crash_root / "occurrences.jsonl", allow_missing=True)
    # fmt: off
    result = CellResult(execution_profile=request.execution_profile, cell_id=request.cell_id, status=status, started_monotonic_seconds=request.start_monotonic_seconds, finished_monotonic_seconds=finished, coverage_freeze_elapsed_seconds=freeze_elapsed, attempt_count=attempt, manifest_sha256=manifest_hash, event_stream_sha256=event_stream, corpus_index_stream_sha256=corpus_stream, crash_stream_sha256=crash_stream, crash_occurrence_count=len(occurrences), corpus_index_scheduled_seconds=tuple(row.scheduled_elapsed_seconds for row in rows), search_accounting=SearchAccounting(cpu_seconds=cpu_seconds, wall_seconds=finished - request.start_monotonic_seconds))  # noqa: E501
    # fmt: on
    _persist_new_evidence(
        request.artifact_root / _RESULT_NAME,
        canonical_json_bytes(result.model_dump(mode="json")),
    )
    return validate_cell_evidence(request)


def load_cell_manifest(request: CellRunRequest) -> CellManifest:
    path = request.artifact_root / _MANIFEST_NAME
    raw = _read_evidence(path)
    try:
        manifest = CellManifest.model_validate_json(raw)
    except ValidationError as error:
        raise CellEvidenceError(reason="invalid cell manifest") from error
    canonical = canonical_json_bytes(manifest.model_dump(mode="json"))
    if raw != canonical or manifest.request != request:
        raise CellEvidenceError(reason="noncanonical or mismatched cell manifest")
    return manifest


def load_cell_events(request: CellRunRequest) -> tuple[CellEvent, ...]:
    manifest_hash = sha256_bytes(
        canonical_json_bytes(load_cell_manifest(request).model_dump(mode="json"))
    )
    path = request.artifact_root / _EVENTS_NAME
    if not path.exists() and not path.is_symlink():
        return ()
    try:
        _ = validate_ledger(path)
        records: list[CellEvent] = []
        for line in path.read_bytes().splitlines():
            outer = _JSON_OBJECT.validate_json(line)
            if outer.get("event_type") != "search_attempt":
                raise CellEvidenceError(reason="unexpected cell event type")
            event = CellEvent.model_validate_json(canonical_json_bytes(outer.get("payload")))
            if (
                outer.get("sequence") != len(records)
                or event.sequence != len(records)
                or event.attempt_index != event.sequence
                or event.manifest_sha256 != manifest_hash
            ):
                raise CellEvidenceError(reason="noncanonical cell event sequence or binding")
            records.append(event)
    except (LedgerError, ValidationError) as error:
        raise CellEvidenceError(reason=f"invalid cell event stream: {error}") from error
    return tuple(records)


def load_cell_result(request: CellRunRequest) -> CellResult:
    raw = _read_evidence(request.artifact_root / _RESULT_NAME)
    try:
        result = CellResult.model_validate_json(raw)
    except ValidationError as error:
        raise CellEvidenceError(reason="invalid final cell result") from error
    if raw != canonical_json_bytes(result.model_dump(mode="json")):
        raise CellEvidenceError(reason="noncanonical final cell result")
    if (
        result.cell_id != request.cell_id
        or result.execution_profile != request.execution_profile
        or result.started_monotonic_seconds != request.start_monotonic_seconds
        or result.finished_monotonic_seconds > request.deadline_monotonic_seconds
    ):
        raise CellEvidenceError(reason="final cell result conflicts with request")
    return result


def validate_cell_evidence(request: CellRunRequest) -> CellResult:
    manifest = load_cell_manifest(request)
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest.model_dump(mode="json")))
    events = load_cell_events(request)
    result = load_cell_result(request)
    try:
        rows = load_corpus_index(_index_request(request))
    except (CoverageSeriesError, ValidationError) as error:
        raise CellEvidenceError(reason=f"invalid cell corpus index: {error}") from error
    occurrences = load_crash_occurrences(request)
    if not events:
        raise CellEvidenceError(reason="cell evidence artifacts conflict")
    _validate_attempt_logs(request, events)
    _validate_attempt_artifacts(request, events, occurrences)
    event_hash = _ledger_hash(request.artifact_root / _EVENTS_NAME, allow_missing=False)
    corpus_hash = _ledger_hash(_index_request(request).index_path, allow_missing=False)
    crash_hash = _ledger_hash(request.crash_root / "occurrences.jsonl", allow_missing=True)
    frozen = tuple(row for row in rows if row.observation_kind == "frozen")
    freeze = result.coverage_freeze_elapsed_seconds
    event_times = tuple(event.observed_monotonic_seconds for event in events)
    final_event = events[-1]
    expected_finished = final_event.observed_monotonic_seconds
    expected_freeze = (
        expected_finished - request.start_monotonic_seconds
        if expected_finished < request.deadline_monotonic_seconds
        else None
    )
    if freeze is None:
        freeze_valid = not frozen
    else:
        freeze_valid = bool(frozen) and all(row.source_elapsed_seconds == freeze for row in frozen)
    if (
        result.manifest_sha256 != manifest_hash
        or result.event_stream_sha256 != event_hash
        or result.corpus_index_stream_sha256 != corpus_hash
        or result.crash_stream_sha256 != crash_hash
        or result.attempt_count != len(events)
        or result.status != _expected_cell_status(request, final_event, len(events))
        or result.finished_monotonic_seconds != expected_finished
        or result.coverage_freeze_elapsed_seconds != expected_freeze
        or result.search_accounting.cpu_seconds != sum(event.cpu_seconds for event in events)
        or result.crash_occurrence_count != len(occurrences)
        or result.corpus_index_scheduled_seconds != request.schedule_seconds
        or tuple(row.scheduled_elapsed_seconds for row in rows) != request.schedule_seconds
        or result.search_accounting.wall_seconds
        != result.finished_monotonic_seconds - result.started_monotonic_seconds
        or any(
            observed < request.start_monotonic_seconds
            or observed > request.deadline_monotonic_seconds
            for observed in event_times
        )
        or any(first > second for first, second in pairwise(event_times))
        or not freeze_valid
    ):
        raise CellEvidenceError(reason="cell evidence artifacts conflict")
    return result


def _expected_cell_status(
    request: CellRunRequest, final_event: CellEvent, attempt_count: int
) -> Literal["completed", "failed", "deadline", "restart_limit"]:
    if final_event.outcome == "completed":
        status: Literal["completed", "failed", "deadline", "restart_limit"] = "completed"
    elif (
        final_event.outcome == "deadline"
        or final_event.observed_monotonic_seconds >= request.deadline_monotonic_seconds
    ):
        status = "deadline"
    else:
        status = "failed"
    if attempt_count >= _MAX_RESTART_ATTEMPTS and (
        final_event.observed_monotonic_seconds < request.deadline_monotonic_seconds
    ):
        return "restart_limit"
    return status


def load_crash_occurrences(request: CellRunRequest) -> tuple[CrashOccurrenceRecord, ...]:
    try:
        records = load_crash_stream(request.artifact_root, request.crash_root / "occurrences.jsonl")
    except ProofBatchError as error:
        raise CellEvidenceError(reason=str(error)) from error
    manifest_hash = sha256_bytes(
        canonical_json_bytes(load_cell_manifest(request).model_dump(mode="json"))
    )
    if any(record.cell_manifest_sha256 != manifest_hash for record in records):
        raise CellEvidenceError(reason="crash occurrence lacks persisted manifest binding")
    return records


def build_long_probe_argv(request: SearchLaunchRequest) -> tuple[str, ...]:
    attempt = request.attempt_root.relative_to(request.artifact_root).as_posix()
    # fmt: off
    return ("docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true", "--pids-limit", "128", "--user", f"{os.getuid()}:{os.getgid()}", "--cpuset-cpus", str(request.cpu_index), "--memory", f"{request.rss_limit_mb}m", "--tmpfs", _TMPFS, "--env", f"ASAN_OPTIONS={_ASAN_OPTIONS}", "--env", f"UBSAN_OPTIONS={_UBSAN_OPTIONS}", "-v", f"{request.artifact_root}:/work:rw", "-v", f"{request.dictionary_path.resolve()}:/work/openexr.dict:ro", "-w", "/work", "--pull", "never", request.build.image_digest, "/opt/dllm/bin/poc_probe-asan", "/work/corpus", f"-artifact_prefix=/work/{attempt}/", "-dict=/work/openexr.dict", "-use_value_profile=1", f"-max_total_time={int(request.remaining_seconds)}", f"-timeout={request.target_timeout_seconds}", f"-rss_limit_mb={request.rss_limit_mb}", f"-max_len={request.max_input_length}", f"-seed={request.fuzzer_seed}", "-print_final_stats=1")  # noqa: E501
    # fmt: on


def _record_artifacts(
    request: CellRunRequest, result: SearchAttemptResult, attempt: int, manifest_hash: str
) -> tuple[CellArtifactEvidence, ...]:
    sanitizer_present = any(artifact.kind == "sanitizer" for artifact in result.artifacts)
    if sanitizer_present and result.outcome != "target_crash":
        raise CellEvidenceError(reason="sanitizer artifact requires target_crash outcome")
    if result.outcome == "target_crash" and not sanitizer_present:
        raise CellEvidenceError(reason="target_crash outcome requires sanitizer artifact")
    log_hash = sha256_bytes(result.log)
    _persist_bytes(request.log_root / f"attempt-{attempt:06d}.log", result.log)
    evidence: list[CellArtifactEvidence] = []
    for artifact in result.artifacts:
        digest = sha256_bytes(artifact.content)
        category: ArtifactCategory = "inputs" if artifact.kind == "sanitizer" else artifact.kind
        destination = request.crash_root / category / digest
        _persist_bytes(destination, artifact.content)
        evidence.append(
            CellArtifactEvidence(
                sha256=digest,
                kind=artifact.kind,
                category=category,
                stored_path=destination.relative_to(request.artifact_root).as_posix(),
                size_bytes=len(artifact.content),
            )
        )
        if artifact.kind == "sanitizer":
            queue = request.crash_root / "occurrences.jsonl"
            # fmt: off
            record = CrashOccurrenceRecord(sequence=len(load_crash_stream(request.artifact_root, queue)), cell_id=request.cell_id, attempt_index=attempt, artifact_name=artifact.name, input_path=destination.relative_to(request.artifact_root).as_posix(), input_sha256=digest, size_bytes=len(artifact.content), search_log_sha256=log_hash, cell_manifest_sha256=manifest_hash)  # noqa: E501
            # fmt: on
            append_crash_occurrence(request.artifact_root, queue, record)
    return tuple(evidence)


def _validate_attempt_logs(request: CellRunRequest, events: tuple[CellEvent, ...]) -> None:
    entries = _directory_entries(request.log_root, "attempt log evidence")
    expected_names = tuple(f"attempt-{event.attempt_index:06d}.log" for event in events)
    if tuple(path.name for path in entries) != expected_names:
        raise CellEvidenceError(reason="attempt log evidence set mismatch")
    for event in events:
        path = request.log_root / f"attempt-{event.attempt_index:06d}.log"
        content = _read_backing_bytes(path, "attempt log evidence")
        if sha256_bytes(content) != event.log_sha256:
            raise CellEvidenceError(reason="attempt log evidence hash mismatch")


def _validate_attempt_artifacts(
    request: CellRunRequest,
    events: tuple[CellEvent, ...],
    occurrences: tuple[CrashOccurrenceRecord, ...],
) -> None:
    backing_items = _artifact_backings(request)
    event_artifacts = tuple(artifact for event in events for artifact in event.artifacts)
    expected_backing = {
        (artifact.sha256, artifact.category, artifact.stored_path, artifact.size_bytes)
        for artifact in event_artifacts
    }
    if set(backing_items) != expected_backing:
        raise CellEvidenceError(reason="attempt artifact evidence set mismatch")
    queue = request.crash_root / "occurrences.jsonl"
    if (queue.exists() or queue.is_symlink()) != bool(occurrences):
        raise CellEvidenceError(reason="sanitizer occurrence evidence set mismatch")
    expected_occurrences = tuple(
        sorted(
            (
                event.attempt_index,
                request.cell_id,
                event.log_sha256,
                artifact.sha256,
                artifact.stored_path,
                artifact.size_bytes,
                event.manifest_sha256,
            )
            for event in events
            for artifact in event.artifacts
            if artifact.kind == "sanitizer"
        )
    )
    actual_occurrences = tuple(
        sorted(
            (
                occurrence.attempt_index,
                occurrence.cell_id,
                occurrence.search_log_sha256,
                occurrence.input_sha256,
                occurrence.input_path,
                occurrence.size_bytes,
                occurrence.cell_manifest_sha256,
            )
            for occurrence in occurrences
        )
    )
    if expected_occurrences != actual_occurrences:
        raise CellEvidenceError(reason="sanitizer occurrence evidence binding mismatch")


def _artifact_backings(request: CellRunRequest) -> tuple[_ArtifactBacking, ...]:
    root_entries = _directory_entries(request.crash_root, "attempt artifact evidence")
    allowed_names = {*_ARTIFACT_CATEGORIES, "occurrences.jsonl"}
    if any(path.name not in allowed_names for path in root_entries):
        raise CellEvidenceError(reason="attempt artifact evidence contains an unexpected path")
    result: list[_ArtifactBacking] = []
    seen: set[str] = set()
    for category in _ARTIFACT_CATEGORIES:
        root = request.crash_root / category
        if not root.exists() and not root.is_symlink():
            continue
        for path in _directory_entries(root, "attempt artifact evidence"):
            content = _read_backing_bytes(path, "attempt artifact evidence")
            digest = sha256_bytes(content)
            if path.name != digest or digest in seen:
                raise CellEvidenceError(
                    reason="attempt artifact evidence is mismatched or ambiguous"
                )
            seen.add(digest)
            result.append(
                (
                    digest,
                    category,
                    path.relative_to(request.artifact_root).as_posix(),
                    len(content),
                )
            )
    return tuple(result)


def _directory_entries(path: Path, evidence_kind: str) -> tuple[Path, ...]:
    if path.is_symlink() or not path.is_dir():
        raise CellEvidenceError(reason=f"{evidence_kind} directory is missing or unsafe")
    try:
        return tuple(sorted(path.iterdir()))
    except OSError as error:
        raise CellEvidenceError(reason=f"cannot read {evidence_kind} directory") from error


def _read_backing_bytes(path: Path, evidence_kind: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CellEvidenceError(reason=f"{evidence_kind} file is missing or unsafe")
    try:
        return path.read_bytes()
    except OSError as error:
        raise CellEvidenceError(reason=f"cannot read {evidence_kind} file") from error


def _require_fresh_cell(
    request: CellRunRequest, preinitialized_corpus_index: tuple[CorpusIndexRecord, ...]
) -> None:
    evidence = (
        request.artifact_root / _MANIFEST_NAME,
        request.artifact_root / _EVENTS_NAME,
        request.artifact_root / _RESULT_NAME,
        request.crash_root / "occurrences.jsonl",
        request.artifact_root / "attempts",
    )
    if any(path.exists() or path.is_symlink() for path in evidence):
        raise CellEvidenceError(reason="existing cell evidence requires Task 5 resume validation")
    for directory in (request.crash_root, request.log_root):
        if directory.is_symlink() or (
            directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
        ):
            raise CellEvidenceError(
                reason="existing cell evidence requires Task 5 resume validation"
            )
    index_request = _index_request(request)
    if preinitialized_corpus_index:
        try:
            loaded = load_corpus_index(index_request)
        except CoverageSeriesError as error:
            raise CellEvidenceError(reason="invalid preinitialized corpus index") from error
        scheduled = tuple(row.scheduled_elapsed_seconds for row in preinitialized_corpus_index)
        if (
            loaded != preinitialized_corpus_index
            or scheduled != request.schedule_seconds[: len(scheduled)]
            or any(row.observation_kind == "frozen" for row in preinitialized_corpus_index)
        ):
            raise CellEvidenceError(reason="invalid preinitialized corpus index")
        return
    if index_request.index_path.exists() or index_request.store_root.exists():
        raise CellEvidenceError(reason="existing cell evidence requires Task 5 resume validation")


def _persist_new_evidence(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CellEvidenceError(reason=f"existing cell evidence: {path.name}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            _ = handle.write(payload)
    except FileExistsError as error:
        raise CellEvidenceError(reason=f"existing cell evidence: {path.name}") from error
    except OSError as error:
        raise CellEvidenceError(reason=f"cannot persist cell evidence: {path.name}") from error


def _read_evidence(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CellEvidenceError(reason=f"missing or unsafe cell evidence: {path.name}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise CellEvidenceError(reason=f"cannot read cell evidence: {path.name}") from error


def _append_cell_event(request: CellRunRequest, event: CellEvent) -> None:
    path = request.artifact_root / _EVENTS_NAME
    prior = load_cell_events(request)
    if event.sequence != len(prior):
        raise CellEvidenceError(reason="cell event sequence gap")
    try:
        _ = append_event(
            path,
            timestamp_utc=f"monotonic+{event.observed_monotonic_seconds:016.6f}",
            event_type="search_attempt",
            payload=_JSON_OBJECT.validate_python(event.model_dump(mode="json")),
        )
    except (LedgerError, ValidationError) as error:
        raise CellEvidenceError(reason=f"cannot append cell event: {error}") from error
    if load_cell_events(request)[-1] != event:
        raise CellEvidenceError(reason="persisted cell event is noncanonical")


def _ledger_hash(path: Path, *, allow_missing: bool) -> str:
    if not path.exists() and not path.is_symlink():
        if allow_missing:
            return EMPTY_STREAM_SHA256
        raise CellEvidenceError(reason=f"missing cell ledger: {path.name}")
    if path.is_symlink() or not path.is_file():
        raise CellEvidenceError(reason=f"unsafe cell ledger: {path.name}")
    try:
        return validate_ledger(path).final_stream_sha256
    except LedgerError as error:
        raise CellEvidenceError(reason=f"invalid cell ledger: {error}") from error


def _prepare_roots(request: CellRunRequest) -> None:
    for path in (request.artifact_root, request.corpus_root, request.crash_root, request.log_root):
        path.mkdir(parents=True, exist_ok=True)


def _index_request(request: CellRunRequest) -> CorpusIndexRequest:
    # fmt: off
    return CorpusIndexRequest(execution_profile=request.execution_profile, schedule_seconds=request.schedule_seconds, artifact_root=request.artifact_root, corpus_root=request.corpus_root, store_root=request.artifact_root / "corpus-inputs", index_path=request.artifact_root / "corpus-index.jsonl")  # noqa: E501
    # fmt: on


def _launch_request(request: CellRunRequest, attempt: int, now: float) -> SearchLaunchRequest:
    # fmt: off
    return SearchLaunchRequest(execution_profile=request.execution_profile, schedule_seconds=request.schedule_seconds, cell_id=request.cell_id, attempt_index=attempt, attempt_root=request.artifact_root / "attempts" / f"{attempt:06d}", artifact_root=request.artifact_root, corpus_root=request.corpus_root, crash_root=request.crash_root, log_root=request.log_root, absolute_deadline_monotonic_seconds=request.deadline_monotonic_seconds, remaining_seconds=request.deadline_monotonic_seconds - now, fuzzer_seed=request.fuzzer_seed, max_input_length=request.max_input_length, target_timeout_seconds=request.target_timeout_seconds, rss_limit_mb=request.rss_limit_mb, cpu_index=request.cpu_index, dictionary_path=request.dictionary_path, dictionary_sha256=request.dictionary_sha256, build=request.build)  # noqa: E501
    # fmt: on


def _next_wait_deadline(request: CellRunRequest, row_count: int) -> float:
    if row_count >= len(request.schedule_seconds):
        return request.deadline_monotonic_seconds
    return min(
        request.start_monotonic_seconds + request.schedule_seconds[row_count],
        request.deadline_monotonic_seconds,
    )


def _terminal_status(
    result: SearchAttemptResult, now: float, deadline: float
) -> Literal["completed", "failed", "deadline"]:
    if result.outcome == "completed":
        return "completed"
    if result.outcome == "deadline" or now >= deadline:
        return "deadline"
    return "failed"


def _persist_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise CellEvidenceError(reason="content-addressed crash mismatch")
        return
    with path.open("xb") as handle:
        _ = handle.write(content)


def _start_search(argv: tuple[str, ...]) -> SearchChildProcess:
    return subprocess.Popen(  # noqa: S603 - fixed shell-free Docker argv.
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _harvest_attempt(request: SearchLaunchRequest, log: bytes) -> tuple[SearchArtifact, ...]:
    artifacts: list[SearchArtifact] = []
    for path in sorted(request.attempt_root.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise CellEvidenceError(reason=f"unsafe search artifact: {path.name}")
        content = path.read_bytes()
        if not content or len(content) > _MAX_ARTIFACT_BYTES:
            raise CellEvidenceError(reason=f"invalid search artifact size: {path.name}")
        artifacts.append(
            SearchArtifact(name=path.name, kind=_artifact_kind(path.name, log), content=content)
        )
    return tuple(artifacts)


def _artifact_kind(
    name: str, log: bytes
) -> Literal["sanitizer", "timeout", "oom", "slow_unit", "crash"]:
    if name.startswith("timeout-"):
        return "timeout"
    if name.startswith("oom-"):
        return "oom"
    if name.startswith("slow-unit-"):
        return "slow_unit"
    markers = (b"AddressSanitizer", b"UndefinedBehaviorSanitizer", b"runtime error:")
    return "sanitizer" if any(marker in log for marker in markers) else "crash"


def _process_outcome(
    return_code: int, artifacts: tuple[SearchArtifact, ...], forced_deadline: bool
) -> Literal["completed", "target_crash", "failed", "deadline"]:
    if forced_deadline:
        return "deadline"
    if return_code == 0:
        return "completed"
    if any(item.kind == "sanitizer" for item in artifacts):
        return "target_crash"
    return "failed"


def _private_data_root(path: Path) -> Path:
    root = path.resolve()
    if root == Path("/") or Path("/data") not in root.parents:
        message = "artifact root must be a private absolute /data path"
        raise ValueError(message)
    return root


def _require_below(root: Path, path: Path) -> None:
    try:
        _ = path.resolve().relative_to(root)
    except ValueError as error:
        message = "evidence path escapes cell root"
        raise ValueError(message) from error


def _validate_dictionary_file(path: Path, expected_sha256: Sha256) -> None:
    if ".." in path.parts:
        message = "dictionary path must not escape its base directory"
        raise ValueError(message)
    if path.is_symlink() or not path.is_file():
        message = "dictionary file is missing or unsafe"
        raise ValueError(message)
    if sha256_bytes(path.read_bytes()) != expected_sha256:
        message = "dictionary hash mismatch"
        raise ValueError(message)
