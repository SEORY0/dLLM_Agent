"""Validated crash stream and resumable deferred OpenEXR proof batch.

Module-size opt-out ``# noqa: SIZE_OK`` keeps Task 4 proof evidence in one owned module.
"""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.ledger import (
    JsonValue,
    LedgerError,
    append_event,
    validate_ledger,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    persist_backing_bytes,
    validate_backing_file,
)
from dllm_agent.targets.openexr.proof_execution import OPENEXR_PROOF_IMAGE_DIGEST

type ProofStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer"]
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MAX_INPUT_BYTES: Final = 1_048_576
_REPLAY_COUNT: Final = 2


class ProofBatchError(Exception):
    reason: str

    def __init__(self, reason: str) -> None:  # noqa: D107
        self.reason = reason
        super().__init__(reason)


class CrashOccurrenceRecord(FrozenModel):
    schema_version: Literal["dllm-openexr-crash-occurrence/v1"] = "dllm-openexr-crash-occurrence/v1"
    sequence: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    attempt_index: int = Field(ge=0)
    artifact_name: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    input_sha256: Sha256
    size_bytes: int = Field(gt=0, le=_MAX_INPUT_BYTES)
    search_log_sha256: Sha256
    cell_manifest_sha256: Sha256


class ProofBatchRequest(FrozenModel):
    schema_version: Literal["dllm-openexr-proof-request/v1"] = "dllm-openexr-proof-request/v1"
    cell_id: str = Field(min_length=1)
    cell_root: Path
    crash_queue_path: Path
    proof_root: Path
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    proof_seed: int = Field(ge=0)
    duration_seconds: int = Field(gt=0)
    search_finished_monotonic_seconds: float
    search_cpu_seconds: float = Field(ge=0.0)
    search_image_digest: str = Field(
        default=OPENEXR_PROOF_IMAGE_DIGEST, pattern=r"^sha256:[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def _private_paths(self) -> Self:
        root = self.cell_root.resolve()
        if root == Path("/") or Path("/data") not in root.parents:
            message = "cell root must be a private absolute /data path"
            raise ValueError(message)
        _require_below(root, self.crash_queue_path)
        proof = self.proof_root.resolve()
        if proof == Path("/") or Path("/data") not in proof.parents:
            message = "proof root must be a private absolute /data path"
            raise ValueError(message)
        return self


class ExactProofRequestModel(FrozenModel):
    snapshot: bytes = Field(min_length=1, max_length=_MAX_INPUT_BYTES)
    snapshot_sha256: Sha256
    duration_seconds: int = Field(gt=0)
    proof_seed: int = Field(ge=0)
    search_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ProofOutcome(Protocol):
    @property
    def oracle_status(self) -> ProofStatus: ...
    @property
    def input_sha256(self) -> str: ...
    @property
    def proof_log_sha256(self) -> str: ...
    @property
    def sanitizer_class(self) -> str | None: ...
    @property
    def root_signature(self) -> str | None: ...
    @property
    def run_argv_sha256(self) -> tuple[str, ...]: ...


@runtime_checkable
class ByteBackedProofOutcome(ProofOutcome, Protocol):
    @property
    def proof_log(self) -> bytes: ...
    @property
    def run_argv_bytes(self) -> tuple[bytes, ...]: ...
    @property
    def result_bytes(self) -> bytes: ...


class ExactProofRunner(Protocol):
    def run(self, request: ExactProofRequestModel) -> ProofOutcome: ...


class ProofClock(Protocol):
    def monotonic(self) -> float: ...
    def process_time(self) -> float: ...


class ProofVerdict(FrozenModel):
    schema_version: Literal["dllm-openexr-proof-verdict/v1"] = "dllm-openexr-proof-verdict/v1"
    input_sha256: Sha256
    first_owner: CrashOccurrenceRecord
    target_commit: str
    proof_seed: int
    oracle_status: ProofStatus
    sanitizer_class: str | None
    root_signature: str | None
    cause_id: Sha256 | None
    proof_log_sha256: Sha256
    proof_log_evidence: ArtifactBackingEvidence
    run_argv_sha256: tuple[Sha256, ...] = Field(min_length=2)
    run_argv_evidence: tuple[ArtifactBackingEvidence, ...] = Field(min_length=2)
    result_sha256: Sha256
    result_evidence: ArtifactBackingEvidence
    replay_count: Literal[2] = 2
    wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _canonical_cause(self) -> Self:
        if self.oracle_status != "proved":
            if self.cause_id is not None:
                message = "non-proved verdict cannot bind a proved cause"
                raise ValueError(message)
            return self
        sanitizer = None if self.sanitizer_class is None else self.sanitizer_class.strip()
        if not sanitizer:
            message = "proved verdict requires nonblank sanitizer class"
            raise ValueError(message)
        root = None if self.root_signature is None else " ".join(self.root_signature.split())
        if not root:
            message = "proved verdict requires nonblank root signature"
            raise ValueError(message)
        if sanitizer != self.sanitizer_class or root != self.root_signature:
            message = "proved verdict signature must be normalized"
            raise ValueError(message)
        expected = proved_cause_id(self.target_commit, sanitizer, root)
        if self.cause_id != expected:
            message = "proved verdict cause does not match canonical signature"
            raise ValueError(message)
        return self


class CauseRecord(FrozenModel):
    cause_id: Sha256
    sanitizer_class: str
    root_signature: str
    input_sha256: tuple[Sha256, ...] = Field(min_length=1)


class CauseIndex(FrozenModel):
    schema_version: Literal["dllm-openexr-cause-index/v1"] = "dllm-openexr-cause-index/v1"
    causes: tuple[CauseRecord, ...]


class SearchAccountingRecord(FrozenModel):
    schema_version: Literal["dllm-openexr-search-accounting/v1"] = (
        "dllm-openexr-search-accounting/v1"
    )
    cpu_seconds: float = Field(ge=0.0)


class ProofAccounting(FrozenModel):
    schema_version: Literal["dllm-openexr-proof-accounting/v1"] = "dllm-openexr-proof-accounting/v1"
    replay_count: int = Field(ge=0)
    wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float = Field(ge=0.0)


class ProofBatchResult(FrozenModel):
    schema_version: Literal["dllm-openexr-proof-batch/v1"] = "dllm-openexr-proof-batch/v1"
    verdicts: tuple[ProofVerdict, ...]
    cause_index: CauseIndex
    search_accounting: SearchAccountingRecord
    proof_accounting: ProofAccounting


def proved_cause_id(target_commit: str, sanitizer_class: str, root_signature: str) -> str:
    sanitizer = sanitizer_class.strip()
    if not sanitizer:
        raise ProofBatchError(reason="proved cause requires nonblank sanitizer class")
    root = " ".join(root_signature.split())
    if not root:
        raise ProofBatchError(reason="proved cause requires nonblank root signature")
    return sha256_bytes(f"{target_commit}\n{sanitizer}\n{root}".encode())


def append_crash_occurrence(
    cell_root: Path, queue_path: Path, record: CrashOccurrenceRecord
) -> None:
    prior = load_crash_stream(cell_root, queue_path)
    if record.sequence != len(prior):
        raise ProofBatchError(reason="crash occurrence sequence gap")
    _validate_occurrence(cell_root, record)
    # fmt: off
    _ = append_event(queue_path, timestamp_utc=f"search-occurrence+{record.sequence:06d}", event_type="sanitizer_occurrence", payload=_JSON_OBJECT.validate_python(record.model_dump(mode="json")))  # noqa: E501
    # fmt: on


def load_crash_stream(cell_root: Path, queue_path: Path) -> tuple[CrashOccurrenceRecord, ...]:
    if not queue_path.exists():
        return ()
    try:
        _ = validate_ledger(queue_path)
    except LedgerError as error:
        raise ProofBatchError(reason=str(error)) from error
    records: list[CrashOccurrenceRecord] = []
    for line in queue_path.read_bytes().splitlines():
        outer = _JSON_OBJECT.validate_json(line)
        if outer.get("event_type") != "sanitizer_occurrence":
            raise ProofBatchError(reason="unexpected crash stream event")
        record = CrashOccurrenceRecord.model_validate(outer.get("payload"))
        if record.sequence != len(records):
            raise ProofBatchError(reason="crash occurrence sequence gap")
        _validate_occurrence(cell_root, record)
        records.append(record)
    return tuple(records)


def run_deferred_proof_batch(
    request: ProofBatchRequest, runner: ExactProofRunner, clock: ProofClock
) -> ProofBatchResult:
    if clock.monotonic() < request.search_finished_monotonic_seconds:
        raise ProofBatchError(reason="proof cannot run before search completion")
    occurrences = load_crash_stream(request.cell_root, request.crash_queue_path)
    owners: dict[str, CrashOccurrenceRecord] = {}
    for occurrence in occurrences:
        if occurrence.cell_id != request.cell_id:
            raise ProofBatchError(reason="crash occurrence belongs to a different cell")
        _ = owners.setdefault(occurrence.input_sha256, occurrence)
    verdict_path = request.proof_root / "verdicts.jsonl"
    existing = {
        item.input_sha256: item for item in _load_verdicts(request.proof_root, verdict_path)
    }
    current_inputs = set(owners)
    persisted_inputs = set(existing)
    if persisted_inputs - current_inputs:
        raise ProofBatchError(reason="proof verdict outside current deduplicated crash set")
    cause_path = request.proof_root / "cause-index.json"
    completed = cause_path.is_symlink() or cause_path.exists()
    if completed and persisted_inputs != current_inputs:
        raise ProofBatchError(reason="completed proof input set differs from current crash set")
    verdicts: list[ProofVerdict] = []
    for input_sha, owner in owners.items():
        expected = existing.get(input_sha)
        if expected is not None:
            _validate_resumed(request, owner, expected)
            verdicts.append(expected)
            continue
        exact = _exact_request(request, owner)
        wall_start, cpu_start = clock.monotonic(), clock.process_time()
        outcome = runner.run(exact)
        # fmt: off
        verdict = _verdict(request, owner, outcome, clock.monotonic() - wall_start, clock.process_time() - cpu_start)  # noqa: E501
        # fmt: on
        _append_verdict(request.proof_root, verdict_path, verdict)
        verdicts.append(verdict)
    persisted = {
        item.input_sha256: item for item in _load_verdicts(request.proof_root, verdict_path)
    }
    if set(persisted) != current_inputs:
        raise ProofBatchError(reason="completed proof verdict set is not exact")
    verdicts = []
    for input_sha, owner in owners.items():
        verdict = persisted[input_sha]
        _validate_resumed(request, owner, verdict)
        verdicts.append(verdict)
    cause_index = _cause_index(tuple(verdicts))
    _persist_cause_index(cause_path, cause_index)
    # fmt: off
    return ProofBatchResult(verdicts=tuple(verdicts), cause_index=cause_index, search_accounting=SearchAccountingRecord(cpu_seconds=request.search_cpu_seconds), proof_accounting=ProofAccounting(replay_count=sum(item.replay_count for item in verdicts), wall_seconds=sum(item.wall_seconds for item in verdicts), cpu_seconds=sum(item.cpu_seconds for item in verdicts)))  # noqa: E501
    # fmt: on


def _verdict(
    request: ProofBatchRequest,
    owner: CrashOccurrenceRecord,
    outcome: ProofOutcome,
    wall_seconds: float,
    cpu_seconds: float,
) -> ProofVerdict:
    if outcome.input_sha256 != owner.input_sha256 or len(outcome.run_argv_sha256) < _REPLAY_COUNT:
        raise ProofBatchError(reason="proof result does not bind exact input and two replays")
    if not isinstance(outcome, ByteBackedProofOutcome):
        raise ProofBatchError(reason="proof outcome lacks dereferenceable evidence")
    try:
        proof_log = outcome.proof_log
        run_argv_bytes = outcome.run_argv_bytes
        result_bytes = outcome.result_bytes
    except AttributeError as error:
        raise ProofBatchError(reason="proof outcome lacks dereferenceable evidence") from error
    if len(run_argv_bytes) != len(outcome.run_argv_sha256):
        raise ProofBatchError(reason="proof outcome lacks dereferenceable evidence")
    cause: str | None = None
    root = None if outcome.root_signature is None else " ".join(outcome.root_signature.split())
    root = root or None
    sanitizer = None if outcome.sanitizer_class is None else outcome.sanitizer_class.strip()
    sanitizer = sanitizer or None
    if outcome.oracle_status == "proved":
        if sanitizer is None:
            raise ProofBatchError(reason="proved verdict requires nonblank sanitizer class")
        if root is None:
            raise ProofBatchError(reason="proved verdict requires nonblank root signature")
        cause = proved_cause_id(request.target_commit, sanitizer, root)
    proof_log_evidence = _persist_outcome_evidence(
        request.proof_root,
        f"evidence/{owner.input_sha256}/proof.log",
        proof_log,
        outcome.proof_log_sha256,
    )
    run_argv_evidence = tuple(
        _persist_outcome_evidence(
            request.proof_root,
            f"evidence/{owner.input_sha256}/run-argv-{index:04d}.bin",
            content,
            digest,
        )
        for index, (content, digest) in enumerate(
            zip(run_argv_bytes, outcome.run_argv_sha256, strict=True)
        )
    )
    result_sha256 = sha256_bytes(result_bytes)
    result_evidence = _persist_outcome_evidence(
        request.proof_root,
        f"evidence/{owner.input_sha256}/result.bin",
        result_bytes,
        result_sha256,
    )
    # fmt: off
    return ProofVerdict(input_sha256=owner.input_sha256, first_owner=owner, target_commit=request.target_commit, proof_seed=request.proof_seed, oracle_status=outcome.oracle_status, sanitizer_class=sanitizer, root_signature=root, cause_id=cause, proof_log_sha256=outcome.proof_log_sha256, proof_log_evidence=proof_log_evidence, run_argv_sha256=outcome.run_argv_sha256, run_argv_evidence=run_argv_evidence, result_sha256=result_sha256, result_evidence=result_evidence, wall_seconds=wall_seconds, cpu_seconds=cpu_seconds)  # noqa: E501
    # fmt: on


def _exact_request(
    request: ProofBatchRequest, owner: CrashOccurrenceRecord
) -> ExactProofRequestModel:
    content = (request.cell_root / owner.input_path).read_bytes()
    # fmt: off
    return ExactProofRequestModel(snapshot=content, snapshot_sha256=owner.input_sha256, duration_seconds=request.duration_seconds, proof_seed=request.proof_seed, search_image_digest=request.search_image_digest)  # noqa: E501
    # fmt: on


def _cause_index(verdicts: tuple[ProofVerdict, ...]) -> CauseIndex:
    groups: dict[str, list[str]] = {}
    signatures: dict[str, tuple[str, str]] = {}
    for verdict in verdicts:
        if verdict.cause_id is not None and verdict.sanitizer_class and verdict.root_signature:
            groups.setdefault(verdict.cause_id, []).append(verdict.input_sha256)
            signatures[verdict.cause_id] = (verdict.sanitizer_class, verdict.root_signature)
    # fmt: off
    return CauseIndex(causes=tuple(CauseRecord(cause_id=cause, sanitizer_class=signatures[cause][0], root_signature=signatures[cause][1], input_sha256=tuple(sorted(groups[cause]))) for cause in sorted(groups)))  # noqa: E501
    # fmt: on


def _load_verdicts(root: Path, path: Path) -> tuple[ProofVerdict, ...]:
    if not path.exists():
        return ()
    try:
        _ = validate_ledger(path)
    except LedgerError as error:
        raise ProofBatchError(reason=str(error)) from error
    verdicts: list[ProofVerdict] = []
    seen_inputs: set[str] = set()
    try:
        for line in path.read_bytes().splitlines():
            outer = _JSON_OBJECT.validate_json(line)
            if outer.get("event_type") != "proof_verdict":
                raise ProofBatchError(reason="unexpected proof verdict event")
            payload = json.dumps(
                outer.get("payload"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            verdict = ProofVerdict.model_validate_json(payload)
            if verdict.input_sha256 in seen_inputs:
                raise ProofBatchError(reason="duplicate proof verdict for input")
            _validate_verdict_evidence(root, verdict)
            seen_inputs.add(verdict.input_sha256)
            verdicts.append(verdict)
    except ValidationError as error:
        raise ProofBatchError(reason="invalid proof verdict evidence") from error
    return tuple(verdicts)


def _append_verdict(root: Path, path: Path, verdict: ProofVerdict) -> None:
    _ = _load_verdicts(root, path)
    # fmt: off
    _ = append_event(path, timestamp_utc=f"proof-input+{verdict.input_sha256}", event_type="proof_verdict", payload=_JSON_OBJECT.validate_python(verdict.model_dump(mode="json")))  # noqa: E501
    # fmt: on


def _validate_resumed(
    request: ProofBatchRequest, owner: CrashOccurrenceRecord, verdict: ProofVerdict
) -> None:
    # fmt: off
    if verdict.first_owner != owner or verdict.target_commit != request.target_commit or verdict.proof_seed != request.proof_seed:  # noqa: E501
    # fmt: on
        raise ProofBatchError(reason="resumed proof verdict conflicts with canonical request")
    _validate_verdict_evidence(request.proof_root, verdict)


def _persist_outcome_evidence(
    root: Path, relative_path: str, content: bytes, expected_sha256: str
) -> ArtifactBackingEvidence:
    actual = sha256_bytes(content)
    if actual != expected_sha256:
        raise ProofBatchError(reason="proof evidence hash mismatch")
    try:
        return persist_backing_bytes(root, relative_path, content)
    except EvidenceBackingError as error:
        raise ProofBatchError(reason=str(error)) from error


def _validate_verdict_evidence(root: Path, verdict: ProofVerdict) -> None:
    if (
        verdict.proof_log_evidence.sha256 != verdict.proof_log_sha256
        or len(verdict.run_argv_evidence) != len(verdict.run_argv_sha256)
        or verdict.result_evidence.sha256 != verdict.result_sha256
    ):
        raise ProofBatchError(reason="proof verdict evidence hash mismatch")
    evidence = (
        (verdict.proof_log_sha256, verdict.proof_log_evidence),
        *zip(verdict.run_argv_sha256, verdict.run_argv_evidence, strict=True),
        (verdict.result_sha256, verdict.result_evidence),
    )
    for expected, item in evidence:
        if item.sha256 != expected:
            raise ProofBatchError(reason="proof verdict evidence hash mismatch")
        try:
            _ = validate_backing_file(root, item)
        except EvidenceBackingError as error:
            raise ProofBatchError(reason=str(error)) from error


def _persist_cause_index(path: Path, index: CauseIndex) -> None:
    # fmt: off
    payload = json.dumps(index.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()  # noqa: E501
    # fmt: on
    if path.is_symlink():
        raise ProofBatchError(reason="cause index conflicts with canonical proof verdicts")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ProofBatchError(reason="cause index conflicts with canonical proof verdicts")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        _ = handle.write(payload)


def _validate_occurrence(cell_root: Path, record: CrashOccurrenceRecord) -> None:
    raw = Path(record.input_path)
    if raw.is_absolute():
        raise ProofBatchError(reason="crash input path must be relative")
    path = (cell_root / raw).resolve()
    _require_below(cell_root.resolve(), path)
    if path.is_symlink() or not path.is_file():
        raise ProofBatchError(reason="crash input is not a regular file")
    content = path.read_bytes()
    if len(content) != record.size_bytes or sha256_bytes(content) != record.input_sha256:
        raise ProofBatchError(reason="crash input content mismatch")


def _require_below(root: Path, path: Path) -> None:
    try:
        _ = path.resolve().relative_to(root)
    except ValueError as error:
        raise ProofBatchError(reason="proof evidence path escapes root") from error
