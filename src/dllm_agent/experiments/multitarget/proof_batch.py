"""Resumable batch proof replay for completed multitarget campaigns."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, override

from pydantic import TypeAdapter, ValidationError

from dllm_agent.experiments.multitarget.artifacts import (
    CanonicalValue,
    canonical_json_bytes,
    write_json_artifact,
)
from dllm_agent.experiments.multitarget.docker_runtime import RuntimeProfile, runtime_profile
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
from dllm_agent.experiments.multitarget.proof import (
    AsanSignature,
    ProofReplayRequest,
    ProofReplayResult,
    ProofRunEvidence,
    ProofStatus,
    proof_result_validation_error,
    prove_input,
)

# allow: SIZE_OK - scoped batch engine must stay in the requested ownership file.

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dllm_agent.experiments.multitarget.budget import Target
    from dllm_agent.experiments.multitarget.campaign_execution import FinishedCell

type JsonObject = dict[str, object]

_DEFAULT_TIMEOUT_SECONDS = 60
_TEMP_PREFIX = ".proof-input-"
_TEMP_SUFFIX = ".tmp"
_JSON_OBJECT: TypeAdapter[JsonObject] = TypeAdapter(dict[str, object])


class ProveInput(Protocol):
    """Proof replay seam used by tests and the live batch runner."""

    def __call__(self, request: ProofReplayRequest) -> ProofReplayResult:
        """Replay one exact input twice."""
        ...


class WriteJsonArtifact(Protocol):
    """Canonical JSON write seam used by tests and the live batch runner."""

    def __call__(
        self,
        *,
        experiment_root: Path,
        relative_path: Path,
        value: CanonicalValue,
    ) -> ArtifactEvidence:
        """Persist one canonical JSON artifact."""
        ...


@dataclass(slots=True)
class ProofBatchError(Exception):
    """Raised when proof batch inputs or resume artifacts are unsafe."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ProofCpuPair:
    """Two distinct CPUs assigned to one two-replay proof worker slot."""

    first: int
    second: int


@dataclass(frozen=True, slots=True)
class ProofReplayRequestRecord:
    """Relative-path manifest for a proof replay request artifact."""

    profile: RuntimeProfile
    image_id: str
    input_path: str
    input_sha256: str
    first_seed: int
    second_seed: int
    first_container_name: str
    second_container_name: str
    first_cpu: int
    second_cpu: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class ProofBatchRecord:
    """Canonical per-target/input proof result artifact."""

    key: str
    target: Target
    input_sha256: str
    owner_cell: Mapping[str, object]
    source_finding_path: str
    proof_input: ArtifactEvidence
    request: ProofReplayRequestRecord
    result: ProofReplayResult


@dataclass(frozen=True, slots=True)
class ProofBatchItem:
    """One proof record plus its persistence evidence."""

    key: str
    record: ProofBatchRecord
    artifact: ArtifactEvidence
    skipped: bool


@dataclass(frozen=True, slots=True)
class ProofBatchSummary:
    """Compact deterministic proof-batch evidence for one ledger entry."""

    scheduled_count: int
    executed_count: int
    skipped_count: int
    replay_count: int
    wall_seconds: float
    results: tuple[ProofBatchItem, ...]
    result_artifact_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Task:
    target: Target
    input_sha256: str
    owner: FinishedCell
    source_path: Path
    source_relative_path: str
    content: bytes

    @property
    def key(self) -> str:
        return f"{self.target}/{self.input_sha256}"


def prove_finished_crashes_batch(  # noqa: PLR0913 - explicit batch seams are the API.
    *,
    experiment_root: Path,
    finished_cells: tuple[FinishedCell, ...],
    proof_images: Mapping[Target, str],
    cpu_pairs: tuple[ProofCpuPair, ...],
    prove: ProveInput = prove_input,
    write_artifact: WriteJsonArtifact = write_json_artifact,
) -> ProofBatchSummary:
    """Prove deduplicated `crash-*` findings and resume valid existing results."""
    started = time.monotonic()
    root = experiment_root.resolve()
    _validate_cpu_pairs(cpu_pairs)
    tasks = _proof_tasks(root, finished_cells)
    scheduled = tuple(
        _scheduled_task(root, proof_images, cpu_pairs, task, index)
        for index, task in enumerate(tasks)
    )
    items_by_key: dict[str, ProofBatchItem] = {}
    pending: list[_ScheduledTask] = []
    for task in scheduled:
        item = _resume_item(root, task)
        if item is not None:
            items_by_key[task.task.key] = item
        else:
            pending.append(task)

    executed_items = _execute_pending(
        root,
        pending,
        prove,
        write_artifact,
        worker_count=len(cpu_pairs),
    )
    for item in executed_items:
        items_by_key[item.key] = item
    results = tuple(items_by_key[task.task.key] for task in scheduled)

    return ProofBatchSummary(
        scheduled_count=len(scheduled),
        executed_count=sum(1 for item in results if not item.skipped),
        skipped_count=sum(1 for item in results if item.skipped),
        replay_count=sum(len(item.record.result.runs) for item in results if not item.skipped),
        wall_seconds=round(time.monotonic() - started, 6),
        results=results,
        result_artifact_paths=tuple(item.artifact.path for item in results),
    )


@dataclass(frozen=True, slots=True)
class _ScheduledTask:
    task: _Task
    proof_input: ArtifactEvidence
    request: ProofReplayRequest
    request_record: ProofReplayRequestRecord
    shard_index: int


def _proof_tasks(root: Path, finished_cells: tuple[FinishedCell, ...]) -> tuple[_Task, ...]:
    owners: dict[tuple[Target, str], _Task] = {}
    for cell in finished_cells:
        for finding in cell.result.findings:
            name = Path(finding.path).name
            if not name.startswith("crash-") or name.startswith("slow-unit-"):
                continue
            source = _evidence_path(root, finding)
            content = source.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            if actual != finding.sha256:
                raise ProofBatchError(reason=f"finding sha256 mismatch: {finding.path}")
            if len(content) != finding.size_bytes:
                raise ProofBatchError(reason=f"finding size mismatch: {finding.path}")
            key = (cell.cell.target, actual)
            if key not in owners:
                owners[key] = _Task(
                    target=cell.cell.target,
                    input_sha256=actual,
                    owner=cell,
                    source_path=source,
                    source_relative_path=_relative(root, source),
                    content=content,
                )
    return tuple(owners[key] for key in sorted(owners))


def _scheduled_task(
    root: Path,
    proof_images: Mapping[Target, str],
    cpu_pairs: tuple[ProofCpuPair, ...],
    task: _Task,
    index: int,
) -> _ScheduledTask:
    shard_index = index % len(cpu_pairs)
    pair = cpu_pairs[shard_index]
    input_path = root / "proof-inputs" / task.target / task.input_sha256
    proof_input = _ensure_proof_input(root, input_path, task.content)
    request = ProofReplayRequest(
        profile=runtime_profile(task.target),
        image_id=proof_images[task.target],
        input_path=input_path,
        input_sha256=task.input_sha256,
        first_seed=_seed(task.owner.cell.seeds.proof_replay_a, index),
        second_seed=_seed(task.owner.cell.seeds.proof_replay_b, index),
        first_container_name=_container_name(task, index, "a"),
        second_container_name=_container_name(task, index, "b"),
        first_cpu=pair.first,
        second_cpu=pair.second,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return _ScheduledTask(
        task=task,
        proof_input=proof_input,
        request=request,
        request_record=_request_record(root, request),
        shard_index=shard_index,
    )


def _resume_item(root: Path, scheduled: _ScheduledTask) -> ProofBatchItem | None:
    relative_path = _result_relative_path(scheduled.task)
    path = root / relative_path
    if not path.exists():
        return None
    payload = path.read_bytes()
    try:
        raw = _JSON_OBJECT.validate_json(payload)
    except ValidationError as error:
        raise ProofBatchError(reason=f"malformed proof result: {path}") from error
    if canonical_json_bytes(raw) != payload:
        raise ProofBatchError(reason=f"malformed proof result: {path}")
    record = _record_from_json(raw, scheduled)
    if _executable_request(root, record.request) != scheduled.request:
        raise ProofBatchError(reason=f"proof result conflicts: {path}")
    validation_error = proof_result_validation_error(scheduled.request, record.result)
    if validation_error is not None:
        raise ProofBatchError(reason=f"proof result conflicts: {path}")
    expected = _record(scheduled, record.result)
    if canonical_json_bytes(record) != canonical_json_bytes(expected):
        raise ProofBatchError(reason=f"proof result conflicts: {path}")
    artifact = ArtifactEvidence(
        path=relative_path.as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    return ProofBatchItem(scheduled.task.key, record, artifact, skipped=True)


def _execute_pending(
    root: Path,
    pending: list[_ScheduledTask],
    prove: ProveInput,
    write_artifact: WriteJsonArtifact,
    *,
    worker_count: int,
) -> tuple[ProofBatchItem, ...]:
    shards: tuple[list[_ScheduledTask], ...] = tuple([] for _ in range(worker_count))
    for task in pending:
        shards[task.shard_index].append(task)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = tuple(
            executor.submit(_execute_shard, root, tuple(shard), prove, write_artifact)
            for shard in shards
            if shard
        )
        shard_results = tuple(future.result() for future in futures)
    return tuple(item for result in shard_results for item in result)


def _execute_shard(
    root: Path,
    shard: tuple[_ScheduledTask, ...],
    prove: ProveInput,
    write_artifact: WriteJsonArtifact,
) -> tuple[ProofBatchItem, ...]:
    items: list[ProofBatchItem] = []
    for task in shard:
        record = _record(task, prove(task.request))
        artifact = write_artifact(
            experiment_root=root,
            relative_path=_result_relative_path(task.task),
            value=record,
        )
        items.append(ProofBatchItem(task.task.key, record, artifact, skipped=False))
    return tuple(items)


def _record(scheduled: _ScheduledTask, result: ProofReplayResult) -> ProofBatchRecord:
    return ProofBatchRecord(
        key=scheduled.task.key,
        target=scheduled.task.target,
        input_sha256=scheduled.task.input_sha256,
        owner_cell=scheduled.task.owner.cell.to_manifest(),
        source_finding_path=scheduled.task.source_relative_path,
        proof_input=scheduled.proof_input,
        request=scheduled.request_record,
        result=result,
    )


def _request_record(root: Path, request: ProofReplayRequest) -> ProofReplayRequestRecord:
    return ProofReplayRequestRecord(
        profile=request.profile,
        image_id=request.image_id,
        input_path=_relative(root, request.input_path),
        input_sha256=request.input_sha256,
        first_seed=request.first_seed,
        second_seed=request.second_seed,
        first_container_name=request.first_container_name,
        second_container_name=request.second_container_name,
        first_cpu=request.first_cpu,
        second_cpu=request.second_cpu,
        timeout_seconds=request.timeout_seconds,
    )


def _executable_request(root: Path, record: ProofReplayRequestRecord) -> ProofReplayRequest:
    input_path = root / record.input_path
    _ = _relative(root, input_path)
    return ProofReplayRequest(
        profile=record.profile,
        image_id=record.image_id,
        input_path=input_path,
        input_sha256=record.input_sha256,
        first_seed=record.first_seed,
        second_seed=record.second_seed,
        first_container_name=record.first_container_name,
        second_container_name=record.second_container_name,
        first_cpu=record.first_cpu,
        second_cpu=record.second_cpu,
        timeout_seconds=record.timeout_seconds,
    )


def _ensure_proof_input(root: Path, destination: Path, content: bytes) -> ArtifactEvidence:
    relative_path = _relative_path(root, destination)
    sha256 = hashlib.sha256(content).hexdigest()
    if destination.exists():
        existing = destination.read_bytes()
        if hashlib.sha256(existing).hexdigest() != sha256:
            raise ProofBatchError(reason=f"proof input sha256 mismatch: {relative_path}")
        if len(existing) != len(content):
            raise ProofBatchError(reason=f"proof input size mismatch: {relative_path}")
        return ArtifactEvidence(relative_path.as_posix(), sha256, len(content))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=_TEMP_PREFIX,
            suffix=_TEMP_SUFFIX,
        )
        with os.fdopen(descriptor, "wb") as handle:
            _ = handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _ = Path(temp_name).replace(destination)
    except OSError as error:
        detail = error.strerror or "cannot write proof input"
        raise ProofBatchError(reason=f"{relative_path}: {detail}") from error
    finally:
        if temp_name:
            with suppress(OSError):
                Path(temp_name).unlink(missing_ok=True)
    return ArtifactEvidence(relative_path.as_posix(), sha256, len(content))


def _record_from_json(raw: JsonObject, scheduled: _ScheduledTask) -> ProofBatchRecord:
    try:
        target = _string(raw, "target")
        return ProofBatchRecord(
            key=_string(raw, "key"),
            target=_target(target),
            input_sha256=_string(raw, "input_sha256"),
            owner_cell=_object(raw, "owner_cell"),
            source_finding_path=_string(raw, "source_finding_path"),
            proof_input=_artifact(_object(raw, "proof_input")),
            request=_request_record_from_json(_object(raw, "request"), scheduled.request.profile),
            result=_result(_object(raw, "result"), scheduled.request.profile),
        )
    except (KeyError, TypeError, ProofBatchError) as error:
        raise ProofBatchError(reason="malformed proof result") from error


def _request_record_from_json(raw: JsonObject, profile: RuntimeProfile) -> ProofReplayRequestRecord:
    input_path = _string(raw, "input_path")
    if Path(input_path).is_absolute():
        raise ProofBatchError(reason="request input_path must be relative")
    return ProofReplayRequestRecord(
        profile=profile,
        image_id=_string(raw, "image_id"),
        input_path=input_path,
        input_sha256=_string(raw, "input_sha256"),
        first_seed=_integer(raw, "first_seed"),
        second_seed=_integer(raw, "second_seed"),
        first_container_name=_string(raw, "first_container_name"),
        second_container_name=_string(raw, "second_container_name"),
        first_cpu=_integer(raw, "first_cpu"),
        second_cpu=_integer(raw, "second_cpu"),
        timeout_seconds=_integer(raw, "timeout_seconds"),
    )


def _result(raw: JsonObject, profile: RuntimeProfile) -> ProofReplayResult:
    signature = _optional_signature(raw.get("signature"))
    return ProofReplayResult(
        status=_status(_string(raw, "status")),
        profile=profile,
        image_id=_string(raw, "image_id"),
        input_sha256=_string(raw, "input_sha256"),
        runs=tuple(_run(item) for item in _array(raw, "runs")),
        signature=signature,
        error=_optional_string(raw.get("error")),
    )


def _run(raw: object) -> ProofRunEvidence:
    item = _as_json_object(raw, "run must be an object")
    return ProofRunEvidence(
        container_name=_string(item, "container_name"),
        proof_seed=_integer(item, "proof_seed"),
        return_code=_integer(item, "return_code"),
        log=_bytes(_object(item, "log")),
        log_sha256=_string(item, "log_sha256"),
        argv_sha256=_string(item, "argv_sha256"),
        signature=_optional_signature(item.get("signature")),
        error=_optional_string(item.get("error")),
    )


def _optional_signature(raw: object) -> AsanSignature | None:
    if raw is None:
        return None
    item = _as_json_object(raw, "signature must be an object")
    return AsanSignature(
        sanitizer_class=_string(item, "sanitizer_class"),
        first_target_frame=_string(item, "first_target_frame"),
    )


def _artifact(raw: JsonObject) -> ArtifactEvidence:
    return ArtifactEvidence(
        path=_string(raw, "path"),
        sha256=_string(raw, "sha256"),
        size_bytes=_integer(raw, "size_bytes"),
    )


def _bytes(raw: JsonObject) -> bytes:
    data = base64.b64decode(_string(raw, "data_base64"), validate=True)
    if hashlib.sha256(data).hexdigest() != _string(raw, "sha256"):
        raise ProofBatchError(reason="bytes sha256 mismatch")
    if len(data) != _integer(raw, "size_bytes"):
        raise ProofBatchError(reason="bytes size mismatch")
    return data


def _validate_cpu_pairs(cpu_pairs: tuple[ProofCpuPair, ...]) -> None:
    if not cpu_pairs:
        raise ProofBatchError(reason="at least one proof CPU pair is required")
    seen: set[int] = set()
    for pair in cpu_pairs:
        if pair.first == pair.second:
            raise ProofBatchError(reason="proof CPU pair must contain distinct CPUs")
        if pair.first < 0 or pair.second < 0:
            raise ProofBatchError(reason="proof CPUs must be nonnegative")
        if pair.first in seen or pair.second in seen:
            raise ProofBatchError(reason="proof CPU pairs must be fixed distinct CPUs")
        seen.add(pair.first)
        seen.add(pair.second)


def _evidence_path(root: Path, evidence: ArtifactEvidence) -> Path:
    raw = Path(evidence.path)
    path = raw if raw.is_absolute() else root / raw
    resolved = path.resolve()
    _ = _relative(root, resolved)
    if not resolved.is_file():
        raise ProofBatchError(reason=f"finding is not a regular file: {evidence.path}")
    return resolved


def _result_relative_path(task: _Task) -> Path:
    return Path("proof-results") / task.target / f"{task.input_sha256}.json"


def _container_name(task: _Task, index: int, suffix: str) -> str:
    return f"dllm-proof-{task.target}-{task.input_sha256}-{index:06d}-{suffix}"


def _seed(namespace: int, global_index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{global_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _relative_path(root: Path, path: Path) -> Path:
    return Path(_relative(root, path))


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ProofBatchError(reason=f"path is outside experiment root: {path}") from error


def _target(value: str) -> Target:
    match value:
        case "libpng" | "libtiff" | "libxml2":
            return value
        case _:
            raise ProofBatchError(reason=f"unsupported target: {value}")


def _status(value: str) -> ProofStatus:
    match value:
        case "proved" | "not_reproduced" | "crash_without_sanitizer" | "failed":
            return value
        case _:
            raise ProofBatchError(reason=f"unsupported proof status: {value}")


def _string(raw: JsonObject, field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ProofBatchError(reason=f"{field} must be a string")
    return value


def _integer(raw: JsonObject, field: str) -> int:
    value = raw[field]
    if not isinstance(value, int):
        raise ProofBatchError(reason=f"{field} must be an integer")
    return value


def _object(raw: JsonObject, field: str) -> JsonObject:
    value = raw[field]
    return _as_json_object(value, f"{field} must be an object")


def _array(raw: JsonObject, field: str) -> tuple[object, ...]:
    value = raw[field]
    if not _is_json_array(value):
        raise ProofBatchError(reason=f"{field} must be an array")
    return tuple(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ProofBatchError(reason="optional string has invalid type")


def _as_json_object(value: object, reason: str) -> JsonObject:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError as error:
        raise ProofBatchError(reason=reason) from error


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
