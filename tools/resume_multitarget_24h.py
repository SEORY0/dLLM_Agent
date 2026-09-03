"""Resume post-processing for the completed interrupted multitarget campaign."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

from pydantic import TypeAdapter, ValidationError

if __package__ in {None, ""}:
    _REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "src"))

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.budget import ExperimentSpec, Target
from dllm_agent.experiments.multitarget.campaign_recording import ExperimentRecorder
from dllm_agent.experiments.multitarget.campaign_recovery import load_finished_cells_manifest
from dllm_agent.experiments.multitarget.container_monitor import supervisor_exited_cleanly
from dllm_agent.experiments.multitarget.ledger import artifact_evidence, validate_ledger
from dllm_agent.experiments.multitarget.proof_batch import (
    ProofBatchSummary,
    ProofCpuPair,
    prove_finished_crashes_batch,
)
from dllm_agent.experiments.multitarget.schedule import CampaignSchedule, build_campaign_schedule
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.run_multitarget_24h import PROOF_IMAGES, SEARCH_IMAGES, write_evidence_tree_sidecar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dllm_agent.experiments.multitarget.campaign_execution import FinishedCell

type JsonObject = dict[str, object]

_INDEX_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(dict[str, object])
_LEDGER_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(dict[str, object])
_RESULTS_ADAPTER: Final[TypeAdapter[list[JsonObject]]] = TypeAdapter(list[dict[str, object]])
_OBJECT_LIST_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])
_OWNER_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(dict[str, object])
_TARGET_HASHES_ADAPTER: Final[TypeAdapter[list[JsonObject]]] = TypeAdapter(list[dict[str, object]])
_REPORT_PATH: Final = Path("results/report.md")
_INDEX_PATH: Final = Path("proof/batch-index.json")
_DOCS_REPORT: Final = Path("docs/multitarget-24h-results-20260822.md")
_EXPECTED_FINISHED_CELLS: Final = 15
_SHA256_HEX_LENGTH: Final = 64
_PROOF_STATUSES: Final = ("proved", "not_reproduced", "crash_without_sanitizer", "failed")


class ProofBatchRunner(Protocol):
    """Injected proof-batch seam for tests and production."""

    def __call__(
        self,
        *,
        experiment_root: Path,
        finished_cells: tuple[FinishedCell, ...],
        proof_images: Mapping[Target, str],
        cpu_pairs: tuple[ProofCpuPair, ...],
    ) -> ProofBatchSummary:
        """Run or resume exact-input proof replay."""
        ...


@dataclass(frozen=True, slots=True)
class ResumeArguments:
    """Typed recovery CLI arguments."""

    experiment_root: Path
    proof_cpu_pairs: tuple[ProofCpuPair, ...]


@dataclass(slots=True)
class ResumeRecoveryError(Exception):
    """Raised when interrupted campaign post-processing cannot be trusted."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ProofRecoveryContext:
    """Inputs that define one recovery proof phase."""

    root: Path
    recorder: ExperimentRecorder
    schedule: CampaignSchedule
    finished_cells: tuple[FinishedCell, ...]
    cpu_pairs: tuple[ProofCpuPair, ...]
    source_hashes: Mapping[str, str]


def main(arguments: Sequence[str] | None = None) -> int:
    """CLI entrypoint for recovery-only post-processing."""
    args = parse_arguments(arguments)
    return run_recovery(args, repo_root=Path(__file__).resolve().parents[1])


def parse_arguments(arguments: Sequence[str] | None) -> ResumeArguments:
    """Parse recovery CLI arguments."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--experiment-root", type=Path, required=True)
    _ = parser.add_argument(
        "--proof-cpu-pairs",
        default=",".join(f"{cpu}-{cpu + 1}" for cpu in range(15, 39, 2)),
        help="Comma-separated exclusive CPU pairs, for example 15-16,17-18.",
    )
    parsed = parser.parse_args(arguments)
    try:
        proof_cpu_pairs = _parse_cpu_pairs(cast("str", parsed.proof_cpu_pairs))
    except ResumeRecoveryError as error:
        parser.error(str(error))
    return ResumeArguments(
        experiment_root=cast("Path", parsed.experiment_root),
        proof_cpu_pairs=proof_cpu_pairs,
    )


def run_recovery(
    args: ResumeArguments,
    *,
    repo_root: Path,
    schedule: CampaignSchedule | None = None,
    finished_cells: tuple[FinishedCell, ...] | None = None,
    proof_batch: ProofBatchRunner = prove_finished_crashes_batch,
) -> int:
    """Resume proof, report, ledger validation, and evidence-tree sidecar."""
    root = args.experiment_root.resolve()
    resolved_schedule = schedule or build_campaign_schedule(
        ExperimentSpec(),
        available_cpus=40,
        image_ids=SEARCH_IMAGES,
    )
    resolved_finished = finished_cells or load_finished_cells_manifest(
        experiment_root=root,
        launch_manifest_path=root / "manifests/main-launches.json",
        schedule=resolved_schedule,
    )
    _verify_finished_cells(resolved_schedule, resolved_finished)
    recorder = ExperimentRecorder(root)
    source_hashes = _source_hashes(repo_root)
    index = _load_or_run_proof(
        ProofRecoveryContext(
            root=root,
            recorder=recorder,
            schedule=resolved_schedule,
            finished_cells=resolved_finished,
            cpu_pairs=args.proof_cpu_pairs,
            source_hashes=source_hashes,
        ),
        proof_batch,
    )
    report = render_report(
        schedule=resolved_schedule,
        finished_cells=resolved_finished,
        batch_index=index,
    )
    _write_report_once(root, repo_root, recorder, report)
    _ = validate_ledger(root / "events.jsonl")
    _write_or_validate_sidecar(root)
    return 0


def build_batch_index(
    *,
    experiment_root: Path,
    schedule: CampaignSchedule,
    summary: ProofBatchSummary,
    cpu_pairs: tuple[ProofCpuPair, ...],
    source_hashes: Mapping[str, str],
) -> JsonObject:
    """Build the deterministic proof batch index bound to persisted result artifacts."""
    _ = experiment_root
    status_counts: Counter[str] = Counter(item.record.result.status for item in summary.results)
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    for item in summary.results:
        by_target[item.record.target][item.record.result.status] += 1
    proofed_pairs = sorted(
        (item.record.target, item.record.input_sha256) for item in summary.results
    )
    proved_pairs = sorted(
        (item.record.target, item.record.input_sha256)
        for item in summary.results
        if item.record.result.status == "proved"
    )
    return {
        "schema_version": 1,
        "schedule_manifest_sha256": schedule.manifest_hash,
        "source_hashes": dict(sorted(source_hashes.items())),
        "result_artifacts": [item.artifact.to_json() for item in summary.results],
        "status_counts": dict(sorted(status_counts.items())),
        "by_target": {
            target: dict(sorted(counts.items())) for target, counts in sorted(by_target.items())
        },
        "proofed_inputs": [_target_hash(target, digest) for target, digest in proofed_pairs],
        "proved_inputs": [_target_hash(target, digest) for target, digest in proved_pairs],
        "proof_images": dict(sorted(PROOF_IMAGES.items())),
        "cpu_pairs": [{"first": pair.first, "second": pair.second} for pair in cpu_pairs],
        "operational": {
            "executed_count": summary.executed_count,
            "recorded_replay_count": sum(len(item.record.result.runs) for item in summary.results),
            "replay_count": summary.replay_count,
            "scheduled_count": summary.scheduled_count,
            "skipped_count": summary.skipped_count,
            "wall_seconds": summary.wall_seconds,
        },
        "results": [
            {
                "key": item.key,
                "target": item.record.target,
                "input_sha256": item.record.input_sha256,
                "owner_cell": dict(item.record.owner_cell),
                "status": item.record.result.status,
                "artifact": item.artifact.to_json(),
            }
            for item in summary.results
        ],
    }


def render_report(
    *,
    schedule: CampaignSchedule,
    finished_cells: tuple[FinishedCell, ...],
    batch_index: JsonObject,
) -> str:
    """Render a Markdown-only research report from recovered artifacts."""
    proved_by_cell = _proved_by_cell(finished_cells, batch_index)
    rows = [
        (
            f"| {item.cell.target} | {item.cell.arm} | {item.result.metrics.coverage_edges} | "
            f"{item.result.metrics.value_profile_features} | "
            f"{item.result.metrics.executed_units} | "
            f"{_effective_units_per_second(item):.3f} | {len(_crash_keys(item))} | "
            f"{_triggered_count(item)} | {_first_trigger_time(item)} | "
            f"{item.result.oom_killed} | {proved_by_cell[_cell_key(item)]} |"
        )
        for item in finished_cells
    ]
    unique_crashes = len({key for item in finished_cells for key in _crash_keys(item)})
    canary_ids = sorted(
        canary_id
        for item in finished_cells
        for canary_id, series in item.result.canaries.items()
        if series.triggered and series.triggered[-1][1] > 0
    )
    effects = _effect_lines(finished_cells)
    return "\n".join(
        (
            "# dLLM-native multi-target 24-hour recovery report",
            "",
            (
                "Status: 15/15 finished cells verified against the frozen schedule "
                f"`{schedule.manifest_hash}`."
            ),
            "This is an n=1 descriptive comparison; percentages summarize this run only.",
            "A fatal canary trigger is benchmark evidence, not a vulnerability.",
            (
                "No novel vulnerability claim is made; independent proof is limited to "
                "the hashes below."
            ),
            "",
            "## Recovered Outcomes",
            "",
            "First trigger time (logical search seconds) includes feedback-arm probe time.",
            "",
            (
                "| Target | Arm | Final coverage | Features | Executed units | "
                "Effective units/s | Unique crash candidates | Distinct triggered canaries | "
                "First trigger time | OOM state | Independently proved in this cell |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
            *rows,
            "",
            f"Unique crash candidates: {unique_crashes}.",
            (
                f"Distinct triggered canary IDs: {len(set(canary_ids))} "
                f"({', '.join(sorted(set(canary_ids)))})"
            ),
            "",
            "## Selective vs Pseudo",
            "",
            "Primary selective vs pseudo percent effects:",
            *effects,
            f"geometric mean direction: {_geomean_direction(finished_cells)}.",
            (
                "geometric mean coverage-per-million-executions direction: "
                f"{_geomean_cpm_direction(finished_cells)}."
            ),
            "",
            "## Independent Proof",
            "",
            f"Status counts: {_status_counts_text(batch_index)}.",
            f"Independently ASan-proved target/hash pairs: {_proved_pairs_text(batch_index)}.",
            "",
        )
    )


def _load_or_run_proof(context: ProofRecoveryContext, proof_batch: ProofBatchRunner) -> JsonObject:
    index_path = context.root / _INDEX_PATH
    if index_path.exists():
        return _read_existing_index(context, index_path)
    if not _ledger_has_event(context.root, "recovery-proof-started"):
        _ = context.recorder.event(
            "recovery-proof-started",
            {"cell_count": len(context.finished_cells)},
        )
    started = time.monotonic()
    summary = proof_batch(
        experiment_root=context.root,
        finished_cells=context.finished_cells,
        proof_images=PROOF_IMAGES,
        cpu_pairs=context.cpu_pairs,
    )
    operational_summary = ProofBatchSummary(
        scheduled_count=summary.scheduled_count,
        executed_count=summary.executed_count,
        skipped_count=summary.skipped_count,
        replay_count=summary.replay_count,
        wall_seconds=summary.wall_seconds or round(time.monotonic() - started, 6),
        results=summary.results,
        result_artifact_paths=summary.result_artifact_paths,
    )
    index = build_batch_index(
        experiment_root=context.root,
        schedule=context.schedule,
        summary=operational_summary,
        cpu_pairs=context.cpu_pairs,
        source_hashes=context.source_hashes,
    )
    _ = context.recorder.record_json(
        event_type="recovery-proof-batch-indexed",
        relative_path=_INDEX_PATH,
        value=index,
        payload={
            "executed_count": operational_summary.executed_count,
            "skipped_count": operational_summary.skipped_count,
            "source_hashes": dict(sorted(context.source_hashes.items())),
        },
    )
    return _read_canonical_index(context.root / _INDEX_PATH)


def _read_existing_index(context: ProofRecoveryContext, index_path: Path) -> JsonObject:
    index = _read_canonical_index(index_path)
    _validate_index_context(index, context)
    return index


def _validate_index_context(index: JsonObject, context: ProofRecoveryContext) -> None:
    expected_source_hashes = dict(sorted(context.source_hashes.items()))
    expected_proof_images = dict(sorted(PROOF_IMAGES.items()))
    expected_cpu_pairs = [
        {"first": pair.first, "second": pair.second} for pair in context.cpu_pairs
    ]
    expected_proofed = _expected_proofed_inputs(context.finished_cells)
    results = _index_results(index)
    result_artifacts = _index_artifacts(index)
    result_records = [
        _validate_result_artifact(context.root, artifact) for artifact in result_artifacts
    ]

    _require_index_equal(
        index.get("schedule_manifest_sha256"),
        context.schedule.manifest_hash,
        "schedule manifest hash",
    )
    _require_index_equal(index.get("source_hashes"), expected_source_hashes, "source hashes")
    _require_index_equal(index.get("proof_images"), expected_proof_images, "proof images")
    _require_index_equal(index.get("cpu_pairs"), expected_cpu_pairs, "CPU pairs")
    _require_index_equal(index.get("proofed_inputs"), expected_proofed, "proofed inputs")
    _require_index_equal(len(results), len(expected_proofed), "result count")
    _require_index_equal(len(result_artifacts), len(results), "result artifact count")

    expected_artifacts = [
        _json_object(result.get("artifact"), "result artifact") for result in results
    ]
    _require_index_equal(result_artifacts, expected_artifacts, "result artifacts")
    _require_index_equal(index.get("status_counts"), _status_counts(results), "status counts")
    _require_index_equal(index.get("by_target"), _by_target(results), "by-target status counts")
    _require_index_equal(index.get("proofed_inputs"), _proofed_inputs(results), "proofed inputs")
    _require_index_equal(index.get("proved_inputs"), _proved_inputs(results), "proved inputs")
    _validate_result_records(results, result_records)
    _validate_operational(index, results, result_records)


def _index_results(index: JsonObject) -> list[JsonObject]:
    try:
        return _RESULTS_ADAPTER.validate_python(index.get("results"))
    except ValidationError as error:
        raise ResumeRecoveryError(reason="proof batch index results are invalid") from error


def _index_artifacts(index: JsonObject) -> list[JsonObject]:
    try:
        return _RESULTS_ADAPTER.validate_python(index.get("result_artifacts"))
    except ValidationError as error:
        raise ResumeRecoveryError(
            reason="proof batch index result artifacts are invalid"
        ) from error


def _expected_proofed_inputs(finished_cells: tuple[FinishedCell, ...]) -> list[JsonObject]:
    return [
        _target_hash(target, digest)
        for target, digest in sorted({key for item in finished_cells for key in _crash_keys(item)})
    ]


def _validate_result_artifact(root: Path, artifact: JsonObject) -> JsonObject:
    relative_path = _artifact_path(artifact)
    path = (root / relative_path).resolve()
    try:
        _ = path.relative_to(root.resolve())
    except ValueError as error:
        raise ResumeRecoveryError(
            reason="proof batch index result artifact is outside experiment root"
        ) from error
    try:
        content = path.read_bytes()
    except OSError as error:
        detail = error.strerror or "cannot read artifact"
        raise ResumeRecoveryError(
            reason=f"proof batch index result artifact cannot be read: {detail}"
        ) from error
    _require_index_equal(_sha256(content), _artifact_sha256(artifact), "result artifact hash")
    _require_index_equal(len(content), _artifact_size(artifact), "result artifact size")
    try:
        result_record = _INDEX_ADAPTER.validate_json(content)
    except ValidationError as error:
        raise ResumeRecoveryError(
            reason="proof batch index result artifact is invalid JSON"
        ) from error
    if content != canonical_json_bytes(result_record):
        raise ResumeRecoveryError(reason="proof batch index result artifact is not canonical JSON")
    return result_record


def _validate_result_records(results: list[JsonObject], records: list[JsonObject]) -> None:
    for result, record in zip(results, records, strict=True):
        _require_index_equal(result.get("key"), record.get("key"), "result key")
        _require_index_equal(result.get("target"), record.get("target"), "result target")
        _require_index_equal(
            result.get("input_sha256"),
            record.get("input_sha256"),
            "result input hash",
        )
        _require_index_equal(
            result.get("owner_cell"),
            record.get("owner_cell"),
            "result owner cell",
        )
        record_result = _json_object(record.get("result"), "result record payload")
        _require_index_equal(result.get("status"), record_result.get("status"), "result status")


def _validate_operational(
    index: JsonObject,
    results: list[JsonObject],
    records: list[JsonObject],
) -> None:
    operational = _json_object(index.get("operational"), "operational summary")
    recorded_replay_count = sum(_recorded_replay_count(record) for record in records)
    _require_index_equal(operational.get("scheduled_count"), len(results), "scheduled count")
    _require_index_equal(
        operational.get("recorded_replay_count"),
        recorded_replay_count,
        "recorded replay count",
    )


def _recorded_replay_count(record: JsonObject) -> int:
    result = _json_object(record.get("result"), "result record payload")
    try:
        runs = _OBJECT_LIST_ADAPTER.validate_python(result.get("runs"))
    except ValidationError as error:
        raise ResumeRecoveryError(reason="proof batch index result runs are invalid") from error
    return len(runs)


def _status_counts(results: list[JsonObject]) -> JsonObject:
    return dict(sorted(Counter(_string_field(result, "status") for result in results).items()))


def _by_target(results: list[JsonObject]) -> JsonObject:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        counts[_string_field(result, "target")][_string_field(result, "status")] += 1
    return {target: dict(sorted(values.items())) for target, values in sorted(counts.items())}


def _proofed_inputs(results: list[JsonObject]) -> list[JsonObject]:
    return [
        _target_hash(_string_field(result, "target"), _string_field(result, "input_sha256"))
        for result in sorted(results, key=_result_sort_key)
    ]


def _proved_inputs(results: list[JsonObject]) -> list[JsonObject]:
    return [
        _target_hash(_string_field(result, "target"), _string_field(result, "input_sha256"))
        for result in sorted(results, key=_result_sort_key)
        if result.get("status") == "proved"
    ]


def _result_sort_key(result: JsonObject) -> tuple[str, str]:
    return (_string_field(result, "target"), _string_field(result, "input_sha256"))


def _json_object(value: object, label: str) -> JsonObject:
    try:
        return _OWNER_ADAPTER.validate_python(value)
    except ValidationError as error:
        raise ResumeRecoveryError(reason=f"proof batch index {label} is invalid") from error


def _string_field(value: JsonObject, field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str):
        raise ResumeRecoveryError(reason=f"proof batch index {field} is invalid")
    return item


def _artifact_path(artifact: JsonObject) -> Path:
    path = _string_field(artifact, "path")
    relative = Path(path)
    if relative.is_absolute():
        raise ResumeRecoveryError(reason="proof batch index result artifact path is absolute")
    return relative


def _artifact_sha256(artifact: JsonObject) -> str:
    return _string_field(artifact, "sha256")


def _artifact_size(artifact: JsonObject) -> int:
    size = artifact.get("size_bytes")
    if not isinstance(size, int):
        raise ResumeRecoveryError(reason="proof batch index result artifact size is invalid")
    return size


def _require_index_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ResumeRecoveryError(
            reason=f"proof batch index {label} does not match recovery context"
        )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_report_once(
    root: Path,
    repo_root: Path,
    recorder: ExperimentRecorder,
    report: str,
) -> None:
    report_path = root / _REPORT_PATH
    docs_path = repo_root / _DOCS_REPORT
    if report_path.exists():
        _require_same(report_path, report.encode(), "report")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _ = report_path.write_text(report, encoding="utf-8")
        _ = recorder.record_existing(
            event_type="markdown-report-written",
            artifact_path=report_path,
            payload={"format": "markdown"},
        )
    if docs_path.exists():
        _require_same(docs_path, report.encode(), "repository report")
    else:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_bytes = report.encode()
        _ = docs_path.write_bytes(docs_bytes)
        _ = recorder.record_existing(
            event_type="repository-report-copied",
            artifact_path=report_path,
            payload={
                "repository_path": _DOCS_REPORT.as_posix(),
                "repository_copy_evidence": {
                    "path": _DOCS_REPORT.as_posix(),
                    "sha256": hashlib.sha256(docs_bytes).hexdigest(),
                    "size_bytes": len(docs_bytes),
                },
            },
        )


def _read_canonical_index(path: Path) -> JsonObject:
    payload = path.read_bytes()
    try:
        index = _INDEX_ADAPTER.validate_json(payload)
    except ValidationError as error:
        raise ResumeRecoveryError(reason="proof batch index is not valid JSON") from error
    if payload != canonical_json_bytes(index):
        raise ResumeRecoveryError(reason="proof batch index is not canonical JSON")
    if index.get("schema_version") != 1:
        raise ResumeRecoveryError(reason="proof batch index schema version is unsupported")
    return index


def _write_or_validate_sidecar(root: Path) -> None:
    sidecar = root.with_name(f"{root.name}.evidence-tree-sha256.txt")
    if sidecar.exists():
        content = sidecar.read_text(encoding="utf-8")
        digest = content.split()[0] if content.split() else ""
        if not content.endswith(f"  {root.name}\n") or len(digest) != _SHA256_HEX_LENGTH:
            raise ResumeRecoveryError(reason="evidence tree sidecar is malformed")
        if digest != evidence_tree_sha256(root):
            raise ResumeRecoveryError(reason="evidence tree sidecar is malformed")
        return
    _ = write_evidence_tree_sidecar(root)


def _verify_finished_cells(
    schedule: CampaignSchedule,
    finished_cells: tuple[FinishedCell, ...],
) -> None:
    if len(finished_cells) != _EXPECTED_FINISHED_CELLS or len(finished_cells) != len(
        schedule.cells
    ):
        raise ResumeRecoveryError(reason="expected exactly 15 finished cells")
    if tuple(item.cell for item in finished_cells) != schedule.cells:
        raise ResumeRecoveryError(reason="finished cells do not match frozen schedule order")
    failed = [item for item in finished_cells if not supervisor_exited_cleanly(item.result)]
    if failed:
        names = ", ".join(f"{item.cell.target}/{item.cell.arm}" for item in failed)
        raise ResumeRecoveryError(reason=f"unclean campaign supervisors: {names}")


def _parse_cpu_pairs(raw: str) -> tuple[ProofCpuPair, ...]:
    pairs: list[ProofCpuPair] = []
    for chunk in raw.split(","):
        left, separator, right = chunk.partition("-")
        if separator == "":
            raise ResumeRecoveryError(reason="proof CPU pairs must use N-M integer syntax")
        try:
            pairs.append(ProofCpuPair(first=int(left), second=int(right)))
        except ValueError as error:
            raise ResumeRecoveryError(
                reason="proof CPU pairs must use N-M integer syntax"
            ) from error
    return tuple(pairs)


def _source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        Path("tools/resume_multitarget_24h.py"),
        Path("src/dllm_agent/experiments/multitarget/campaign_recovery.py"),
        Path("src/dllm_agent/experiments/multitarget/proof.py"),
        Path("src/dllm_agent/experiments/multitarget/proof_batch.py"),
    )
    return {
        path.as_posix(): artifact_evidence(repo_root / path).sha256
        for path in paths
        if (repo_root / path).is_file()
    }


def _ledger_has_event(root: Path, event_type: str) -> bool:
    ledger = root / "events.jsonl"
    if not ledger.exists():
        return False
    _ = validate_ledger(ledger)
    for line in ledger.read_text(encoding="utf-8").splitlines():
        record = _LEDGER_ADAPTER.validate_json(line)
        if record.get("event_type") == event_type:
            return True
    return False


def _require_same(path: Path, expected: bytes, label: str) -> None:
    if path.read_bytes() != expected:
        raise ResumeRecoveryError(reason=f"existing {label} differs from recovered report")


def _target_hash(target: str, digest: str) -> JsonObject:
    return {"sha256": digest, "target": target}


def _cell_key(item: FinishedCell) -> tuple[str, str, int]:
    return (item.cell.target, item.cell.arm, item.cell.repetition)


def _proved_by_cell(
    finished_cells: tuple[FinishedCell, ...],
    batch_index: JsonObject,
) -> Counter[tuple[str, str, int]]:
    counts: Counter[tuple[str, str, int]] = Counter()
    proved = _target_hash_set(batch_index, "proved_inputs")
    for item in finished_cells:
        counts[_cell_key(item)] = len(set(_crash_keys(item)) & proved)
    return counts


def _target_hash_set(batch_index: JsonObject, field: str) -> set[tuple[str, str]]:
    try:
        items = _TARGET_HASHES_ADAPTER.validate_python(batch_index.get(field))
    except ValidationError:
        return set()
    pairs: set[tuple[str, str]] = set()
    for item in items:
        target = item.get("target")
        digest = item.get("sha256")
        if isinstance(target, str) and isinstance(digest, str):
            pairs.add((target, digest))
    return pairs


def _crash_keys(item: FinishedCell) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item.cell.target, finding.sha256)
        for finding in item.result.findings
        if Path(finding.path).name.startswith("crash-")
    )


def _effective_units_per_second(item: FinishedCell) -> float:
    seconds = max(item.cell.allocation.search_worker_seconds, 1)
    return item.result.metrics.executed_units / seconds


def _triggered_count(item: FinishedCell) -> int:
    return sum(
        1
        for series in item.result.canaries.values()
        if series.triggered and series.triggered[-1][1] > 0
    )


def _first_trigger_time(item: FinishedCell) -> str:
    times = [
        point[0]
        for series in item.result.canaries.values()
        for point in series.triggered
        if point[1] > 0
    ]
    if not times:
        return "none"
    offset = (
        item.cell.allocation.initial_probe_worker_seconds
        + item.cell.allocation.revision_probe_worker_seconds
    )
    return str(min(times) + offset)


def _effect_lines(finished_cells: tuple[FinishedCell, ...]) -> list[str]:
    lines: list[str] = []
    for target in ("libpng", "libtiff", "libxml2"):
        selective = _cell_for(finished_cells, target, "selective-revision")
        pseudo = _cell_for(finished_cells, target, "pseudo-feedback")
        coverage = _percent(
            selective.result.metrics.coverage_edges,
            pseudo.result.metrics.coverage_edges,
        )
        features = _percent(
            selective.result.metrics.value_profile_features,
            pseudo.result.metrics.value_profile_features,
        )
        units = _percent(
            selective.result.metrics.executed_units,
            pseudo.result.metrics.executed_units,
        )
        cpm = _percent(_coverage_per_million(selective), _coverage_per_million(pseudo))
        lines.append(
            "".join(
                (
                    f"- {target}: coverage {coverage:+.2f}%, ",
                    f"features {features:+.2f}%, executed units {units:+.2f}%, ",
                    f"coverage-per-million-executions {cpm:+.2f}%.",
                )
            )
        )
    return lines


def _cell_for(finished_cells: tuple[FinishedCell, ...], target: str, arm: str) -> FinishedCell:
    for item in finished_cells:
        if item.cell.target == target and item.cell.arm == arm:
            return item
    raise ResumeRecoveryError(reason=f"missing cell for {target}/{arm}")


def _percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return ((numerator / denominator) - 1.0) * 100.0


def _geomean_direction(finished_cells: tuple[FinishedCell, ...]) -> str:
    ratios: list[float] = [
        _ratio(
            _cell_for(
                finished_cells,
                target,
                "selective-revision",
            ).result.metrics.coverage_edges,
            _cell_for(
                finished_cells,
                target,
                "pseudo-feedback",
            ).result.metrics.coverage_edges,
        )
        for target in ("libpng", "libtiff", "libxml2")
    ]
    product: float = 1.0
    for ratio in ratios:
        product *= ratio
    geomean: float = math.pow(product, 1.0 / len(ratios))
    direction = "higher" if geomean >= 1.0 else "lower"
    delta: float = (geomean - 1.0) * 100.0
    return f"selective-revision {direction} than pseudo-feedback ({delta:+.2f}%)"


def _geomean_cpm_direction(finished_cells: tuple[FinishedCell, ...]) -> str:
    ratios: list[float] = [
        _ratio(
            _coverage_per_million(_cell_for(finished_cells, target, "selective-revision")),
            _coverage_per_million(_cell_for(finished_cells, target, "pseudo-feedback")),
        )
        for target in ("libpng", "libtiff", "libxml2")
    ]
    product: float = 1.0
    for ratio in ratios:
        product *= ratio
    geomean: float = math.pow(product, 1.0 / len(ratios))
    direction = "higher" if geomean >= 1.0 else "lower"
    delta: float = (geomean - 1.0) * 100.0
    return f"selective-revision {direction} than pseudo-feedback ({delta:+.2f}%)"


def _coverage_per_million(item: FinishedCell) -> float:
    executed = item.result.metrics.executed_units
    if executed == 0:
        return 0.0
    return item.result.metrics.coverage_edges * 1_000_000.0 / executed


def _status_counts_text(batch_index: JsonObject) -> str:
    try:
        counts = _OWNER_ADAPTER.validate_python(batch_index.get("status_counts"))
    except ValidationError:
        counts = {}
    return ", ".join(f"{status}: {_int_count(counts.get(status))}" for status in _PROOF_STATUSES)


def _proved_pairs_text(batch_index: JsonObject) -> str:
    pairs = sorted(_target_hash_set(batch_index, "proved_inputs"))
    if not pairs:
        return "none"
    return ", ".join(f"{target}/{digest}" for target, digest in pairs)


def _int_count(value: object) -> int:
    if isinstance(value, int):
        return value
    return 0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
