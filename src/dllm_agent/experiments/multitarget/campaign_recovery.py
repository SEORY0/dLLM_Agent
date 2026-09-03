"""Strict recovery of launched multitarget campaign cells from manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, override

from pydantic import TypeAdapter, ValidationError

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.campaign_evidence import launched_cell_manifest
from dllm_agent.experiments.multitarget.campaign_execution import FinishedCell, LaunchedCell
from dllm_agent.experiments.multitarget.container_monitor import (
    CanarySeries,
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.corpus import (
    CorpusFileEvidence,
    CorpusSourceKind,
    StagedCorpus,
)
from dllm_agent.experiments.multitarget.docker_runtime import (
    ContainerRequest,
    RuntimeProfile,
    container_argv,
    runtime_profile,
)
from dllm_agent.experiments.multitarget.ledger import (
    ArtifactEvidence,
    artifact_evidence,
    command_evidence,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.budget import Arm, Target
    from dllm_agent.experiments.multitarget.schedule import CampaignSchedule

type JsonObject = dict[str, object]

_MANIFEST_ADAPTER: TypeAdapter[list[JsonObject]] = TypeAdapter(list[JsonObject])
_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
_CANARY_POINT_COUNT = 2


@dataclass(slots=True)
class CampaignRecoveryError(Exception):
    """Raised when a launch manifest cannot be trusted for recovery."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


def load_launched_cells_manifest(
    *,
    experiment_root: Path,
    manifest_path: Path,
    schedule: CampaignSchedule,
) -> tuple[LaunchedCell, ...]:
    """Rehydrate launched cells from a generated main-launches.json artifact."""
    root = experiment_root.resolve()
    manifest = _read_manifest(root, manifest_path)
    if len(manifest) != len(schedule.cells):
        raise CampaignRecoveryError(reason="launch manifest cell count does not match schedule")

    seen_cells: set[tuple[str, str, int]] = set()
    seen_containers: set[str] = set()
    launched: list[LaunchedCell] = []
    for index, item in enumerate(manifest):
        cell = schedule.cells[index]
        manifest_cell = _require_object(item.get("cell"), "cell")
        cell_key = _cell_key(manifest_cell)
        if cell_key in seen_cells:
            raise CampaignRecoveryError(reason="duplicate cell in launch manifest")
        seen_cells.add(cell_key)
        if canonical_json_bytes(manifest_cell) != canonical_json_bytes(cell.to_manifest()):
            raise CampaignRecoveryError(reason="cell manifest does not match schedule")

        container_id = _require_str(item.get("container_id"), "container_id")
        if container_id in seen_containers:
            raise CampaignRecoveryError(reason="duplicate container id in launch manifest")
        seen_containers.add(container_id)

        corpus = _staged_corpus(root, _require_object(item.get("corpus"), "corpus"))
        request_manifest = _require_object(item.get("request"), "request")
        request = _container_request(
            root,
            runtime_profile(cell.target),
            cell.image_id,
            cell.cpu_index,
            request_manifest,
        )
        argv = tuple(
            _require_str(value, "request.argv[]")
            for value in _require_list(
                request_manifest.get("argv"),
                "request.argv",
            )
        )
        if container_argv(request) != argv:
            raise CampaignRecoveryError(reason="request argv does not match reconstructed request")
        expected_command = command_evidence(argv, {}).to_json()
        if canonical_json_bytes(
            _require_object(request_manifest.get("command_evidence"), "request.command_evidence")
        ) != canonical_json_bytes(expected_command):
            raise CampaignRecoveryError(reason="request command evidence does not match argv")
        if request.corpus_dir != corpus.corpus_dir:
            raise CampaignRecoveryError(reason="request corpus_dir does not match corpus evidence")

        launched.append(
            LaunchedCell(
                cell=cell,
                work_dir=request.shared_dir.parent,
                shared_dir=request.shared_dir,
                staged_corpus=corpus,
                request=request,
                launch=ContainerLaunch(container_id=container_id),
            )
        )
    return tuple(launched)


def load_finished_cells_manifest(
    *,
    experiment_root: Path,
    launch_manifest_path: Path,
    schedule: CampaignSchedule,
) -> tuple[FinishedCell, ...]:
    """Rehydrate finished cells from canonical per-cell result artifacts."""
    root = experiment_root.resolve()
    launched = load_launched_cells_manifest(
        experiment_root=root,
        manifest_path=launch_manifest_path,
        schedule=schedule,
    )
    _validate_finished_result_set(root, schedule)
    finished: list[FinishedCell] = []
    for item in launched:
        result_manifest = _read_canonical_object(
            root,
            _finished_result_path(item.cell.target, item.cell.arm, item.cell.repetition),
        )
        nested_launch = _require_object(result_manifest.get("launch"), "launch")
        if canonical_json_bytes(nested_launch) != canonical_json_bytes(
            launched_cell_manifest(root, item)
        ):
            raise CampaignRecoveryError(
                reason="nested launch does not match trusted launch manifest"
            )

        log_evidence = _artifact(
            root,
            _require_object(result_manifest.get("log"), "log"),
            "log evidence",
        )
        expected_log_path = item.work_dir / "main.log"
        if Path(log_evidence.path) != expected_log_path:
            raise CampaignRecoveryError(reason="log evidence path does not match main.log")
        logs_text = expected_log_path.read_text(encoding="utf-8")
        if hashlib.sha256(logs_text.encode()).hexdigest() != log_evidence.sha256:
            raise CampaignRecoveryError(reason="log evidence does not match main.log")

        run_result = _run_result(
            root,
            _require_object(result_manifest.get("result"), "result"),
            logs_text,
            log_evidence.sha256,
        )
        if run_result.container_id != item.launch.container_id:
            raise CampaignRecoveryError(reason="result container id does not match launch")

        finished.append(
            FinishedCell(
                cell=item.cell,
                work_dir=item.work_dir,
                staged_corpus=item.staged_corpus,
                request=item.request,
                launch=item.launch,
                result=run_result,
                log_path=expected_log_path,
            )
        )
    return tuple(finished)


def _read_manifest(root: Path, manifest_path: Path) -> list[JsonObject]:
    path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    resolved = path.resolve()
    try:
        _ = resolved.relative_to(root)
    except ValueError as error:
        raise CampaignRecoveryError(reason="manifest path is outside experiment root") from error
    try:
        return _MANIFEST_ADAPTER.validate_json(resolved.read_bytes())
    except ValidationError as error:
        raise CampaignRecoveryError(reason="launch manifest must be a list of objects") from error


def _read_canonical_object(root: Path, relative_path: Path) -> JsonObject:
    path = _bounded_child(root, relative_path, "finished result path")
    try:
        content = path.read_bytes()
        parsed = _OBJECT_ADAPTER.validate_json(content)
    except OSError as error:
        detail = error.strerror or "cannot read finished result"
        raise CampaignRecoveryError(
            reason=f"finished result set is incomplete: {detail}"
        ) from error
    except ValidationError as error:
        raise CampaignRecoveryError(reason="finished result manifest must be an object") from error
    if content != canonical_json_bytes(parsed):
        raise CampaignRecoveryError(reason="finished result manifest is not canonical JSON")
    return parsed


def _validate_finished_result_set(root: Path, schedule: CampaignSchedule) -> None:
    expected = {
        _finished_result_path(cell.target, cell.arm, cell.repetition) for cell in schedule.cells
    }
    results_root = root / "results" / "cells"
    actual: set[Path] = set()
    if results_root.exists():
        actual = {item.relative_to(root) for item in results_root.rglob("*.json") if item.is_file()}
    if actual != expected:
        raise CampaignRecoveryError(reason="finished result set does not match schedule")


def _finished_result_path(target: Target, arm: Arm, repetition: int) -> Path:
    return Path("results/cells") / target / arm / f"rep-{repetition}.json"


def _run_result(
    root: Path,
    result: JsonObject,
    logs_text: str,
    log_sha256: str,
) -> ContainerRunResult:
    result_log_sha256 = _require_str(result.get("log_sha256"), "result.log_sha256")
    if result_log_sha256 != log_sha256:
        raise CampaignRecoveryError(reason="result log hash does not match log evidence")
    if result_log_sha256 != hashlib.sha256(logs_text.encode()).hexdigest():
        raise CampaignRecoveryError(reason="result log hash does not match main.log")
    return ContainerRunResult(
        container_id=_require_str(result.get("container_id"), "result.container_id"),
        exit_code=_require_optional_int(result.get("exit_code"), "result.exit_code"),
        docker_status=_require_optional_str(result.get("docker_status"), "result.docker_status"),
        oom_killed=_require_optional_bool(result.get("oom_killed"), "result.oom_killed"),
        timed_out=_require_bool(result.get("timed_out"), "result.timed_out"),
        process_error=_require_optional_str(result.get("process_error"), "result.process_error"),
        logs_text=logs_text,
        log_sha256=result_log_sha256,
        metrics=_metrics(_require_object(result.get("metrics"), "result.metrics")),
        canaries=_canaries(_require_object(result.get("canaries"), "result.canaries")),
        findings=tuple(
            _artifact(root, _require_object(item, "result.findings[]"), "finding evidence")
            for item in _require_list(result.get("findings"), "result.findings")
        ),
    )


def _artifact(root: Path, item: JsonObject, label: str) -> ArtifactEvidence:
    relative_path = _require_str(item.get("path"), f"{label}.path")
    path = _bounded_child(root, Path(relative_path), label)
    if not path.is_file():
        raise CampaignRecoveryError(reason=f"{label} path must be a regular file")
    expected = ArtifactEvidence(
        path=relative_path,
        sha256=_require_str(item.get("sha256"), f"{label}.sha256"),
        size_bytes=_require_int(item.get("size_bytes"), f"{label}.size_bytes"),
    )
    actual = artifact_evidence(path)
    if actual.sha256 != expected.sha256 or actual.size_bytes != expected.size_bytes:
        raise CampaignRecoveryError(reason=f"{label} does not match file content")
    return actual


def _metrics(value: JsonObject) -> FuzzerMetrics:
    return FuzzerMetrics(
        coverage_edges=_require_int(value.get("coverage_edges"), "result.metrics.coverage_edges"),
        value_profile_features=_require_int(
            value.get("value_profile_features"),
            "result.metrics.value_profile_features",
        ),
        executions_per_second=_require_int(
            value.get("executions_per_second"),
            "result.metrics.executions_per_second",
        ),
        executed_units=_require_int(value.get("executed_units"), "result.metrics.executed_units"),
    )


def _canaries(value: JsonObject) -> dict[str, CanarySeries]:
    return {
        bug: CanarySeries(
            reached=_series_points(_require_list(series.get("reached"), "result.canaries.reached")),
            triggered=_series_points(
                _require_list(series.get("triggered"), "result.canaries.triggered")
            ),
        )
        for bug, raw_series in value.items()
        for series in (_require_object(raw_series, "result.canaries[]"),)
    }


def _series_points(values: list[object]) -> tuple[tuple[int, int], ...]:
    return tuple(_series_point(_require_list(item, "result.canaries.point")) for item in values)


def _series_point(value: list[object]) -> tuple[int, int]:
    if len(value) != _CANARY_POINT_COUNT:
        raise CampaignRecoveryError(reason="result.canaries.point must have two values")
    return (
        _require_int(value[0], "result.canaries.point.time"),
        _require_int(value[1], "result.canaries.point.count"),
    )


def _container_request(
    root: Path,
    profile: RuntimeProfile,
    image_id: str,
    cpu_index: int,
    request: JsonObject,
) -> ContainerRequest:
    return ContainerRequest(
        name=_require_str(request.get("name"), "request.name"),
        profile=profile,
        image_id=image_id,
        shared_dir=_relative_path(
            root,
            _require_str(request.get("shared_dir"), "request.shared_dir"),
        ),
        corpus_dir=_relative_path(
            root,
            _require_str(request.get("corpus_dir"), "request.corpus_dir"),
        ),
        duration_seconds=_require_int(request.get("duration_seconds"), "request.duration_seconds"),
        fuzzer_seed=_require_int(request.get("fuzzer_seed"), "request.fuzzer_seed"),
        cpu_index=cpu_index,
        poll_seconds=_require_int(request.get("poll_seconds"), "request.poll_seconds"),
    )


def _cell_key(cell: JsonObject) -> tuple[str, str, int]:
    return (
        _require_str(cell.get("target"), "cell.target"),
        _require_str(cell.get("arm"), "cell.arm"),
        _require_int(cell.get("repetition"), "cell.repetition"),
    )


def _staged_corpus(root: Path, corpus: JsonObject) -> StagedCorpus:
    corpus_dir = _relative_path(root, _require_str(corpus.get("corpus_dir"), "corpus.corpus_dir"))
    files = tuple(
        _corpus_file(corpus_dir, _require_object(item, "corpus.files[]"))
        for item in _require_list(corpus.get("files"), "corpus.files")
    )
    base_file_count = _require_int(corpus.get("base_file_count"), "corpus.base_file_count")
    candidate_file_count = _require_int(
        corpus.get("candidate_file_count"), "corpus.candidate_file_count"
    )
    if sum(1 for item in files if item.source_kind == "base") != base_file_count:
        raise CampaignRecoveryError(reason="corpus count does not match base evidence")
    if sum(1 for item in files if item.source_kind == "particle") != candidate_file_count:
        raise CampaignRecoveryError(reason="corpus count does not match candidate evidence")
    tree_sha256 = _require_str(corpus.get("tree_sha256"), "corpus.tree_sha256")
    if _tree_hash(files) != tree_sha256:
        raise CampaignRecoveryError(reason="corpus tree hash does not match file evidence")
    return StagedCorpus(
        corpus_dir=corpus_dir,
        base_file_count=base_file_count,
        candidate_file_count=candidate_file_count,
        files=files,
        tree_sha256=tree_sha256,
    )


def _corpus_file(corpus_dir: Path, item: JsonObject) -> CorpusFileEvidence:
    relative_path = _require_str(item.get("relative_path"), "corpus.files[].relative_path")
    path = _bounded_corpus_file(corpus_dir, Path(relative_path))
    source_kind = _source_kind(_require_str(item.get("source_kind"), "corpus.files[].source_kind"))
    evidence = CorpusFileEvidence(
        relative_path=relative_path,
        size_bytes=_require_int(item.get("size_bytes"), "corpus.files[].size_bytes"),
        sha256=_require_str(item.get("sha256"), "corpus.files[].sha256"),
        source_kind=source_kind,
    )
    actual = artifact_evidence(path)
    if actual.sha256 != evidence.sha256 or actual.size_bytes != evidence.size_bytes:
        raise CampaignRecoveryError(reason="corpus file evidence does not match file content")
    return evidence


def _bounded_corpus_file(corpus_dir: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute():
        raise CampaignRecoveryError(reason="corpus file evidence path must be relative")
    path = corpus_dir / relative_path
    try:
        _ = path.parent.resolve().relative_to(corpus_dir)
    except ValueError as error:
        raise CampaignRecoveryError(
            reason="corpus file evidence path is outside corpus directory"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise CampaignRecoveryError(
            reason="corpus file evidence path must be a regular non-symlink file"
        )
    return path


def _relative_path(root: Path, value: str) -> Path:
    return _bounded_child(root, Path(value), "relative path")


def _bounded_child(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute():
        raise CampaignRecoveryError(reason=f"{label} must be relative")
    resolved = (root / relative).resolve()
    try:
        _ = resolved.relative_to(root)
    except ValueError as error:
        raise CampaignRecoveryError(reason=f"{label} is outside experiment root") from error
    return resolved


def _source_kind(value: str) -> CorpusSourceKind:
    match value:
        case "base":
            return "base"
        case "particle":
            return "particle"
        case _:
            raise CampaignRecoveryError(reason="corpus source_kind is unsupported")


def _tree_hash(files: tuple[CorpusFileEvidence, ...]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode())
        digest.update(b"\0")
        digest.update(item.source_kind.encode())
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode())
        digest.update(b"\0")
        digest.update(item.sha256.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_object(value: object, label: str) -> JsonObject:
    if not _is_json_object(value):
        raise CampaignRecoveryError(reason=f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not _is_json_list(value):
        raise CampaignRecoveryError(reason=f"{label} must be a list")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CampaignRecoveryError(reason=f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CampaignRecoveryError(reason=f"{label} must be an integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CampaignRecoveryError(reason=f"{label} must be a boolean")
    return value


def _require_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, label)


def _require_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, label)


def _require_optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _require_bool(value, label)


def _is_json_object(value: object) -> TypeGuard[JsonObject]:
    return isinstance(value, dict)


def _is_json_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
