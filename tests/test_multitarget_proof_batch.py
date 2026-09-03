"""Resumable exact-input proof batch contracts."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from subprocess import CompletedProcess
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.budget import BudgetAllocation
from dllm_agent.experiments.multitarget.campaign_execution import FinishedCell
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.corpus import StagedCorpus
from dllm_agent.experiments.multitarget.docker_runtime import ContainerRequest, runtime_profile
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
from dllm_agent.experiments.multitarget.proof import (
    ProofReplayRequest,
    ProofReplayResult,
    prove_input,
)
from dllm_agent.experiments.multitarget.proof_batch import (
    ProofBatchError,
    ProofBatchRecord,
    ProofCpuPair,
    ProofReplayRequestRecord,
    ProveInput,
    prove_finished_crashes_batch,
)
from dllm_agent.experiments.multitarget.schedule import (
    CellSchedule,
    SeedNamespace,
    WorkerLedgerItem,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from dllm_agent.experiments.multitarget.artifacts import CanonicalValue
    from dllm_agent.experiments.multitarget.budget import Arm, Target

type JsonObject = dict[str, object]

_IMAGE_IDS: Mapping[Target, str] = {
    "libpng": "sha256:" + ("a" * 64),
    "libtiff": "sha256:" + ("b" * 64),
    "libxml2": "sha256:" + ("c" * 64),
}


def test_deduplicates_crash_findings_by_target_sha_and_excludes_slow_units(
    tmp_path: Path,
) -> None:
    """Given repeated crash artifacts, when proven, then only first owners run."""
    # Given
    root = tmp_path / "experiment"
    content = b"same crash"
    first = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", content),))
    second = _finished_cell(
        root,
        "libpng",
        "selective-revision",
        0,
        (("slow-unit-timeout", b"slow"), ("crash-b", content), ("oom-1", b"oom")),
    )
    calls: list[ProofReplayRequest] = []

    # When
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(first, second),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1),),
        prove=_prove_success(calls),
        write_artifact=_write_json,
    )

    # Then
    crash_sha = _sha256(content)
    assert summary.scheduled_count == 1
    assert summary.executed_count == 1
    assert summary.skipped_count == 0
    assert tuple(item.key for item in summary.results) == (f"libpng/{crash_sha}",)
    assert calls[0].input_sha256 == crash_sha
    assert calls[0].input_path == root / "proof-inputs/libpng" / crash_sha


def test_builds_exact_requests_with_deterministic_seeds_names_and_cpu_pairs(
    tmp_path: Path,
) -> None:
    """Given several unique crashes, when proven, then requests are stable."""
    # Given
    root = tmp_path / "experiment"
    first = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    second = _finished_cell(root, "libtiff", "pseudo-feedback", 0, (("crash-b", b"b"),))
    third = _finished_cell(root, "libpng", "selective-revision", 1, (("crash-c", b"c"),))
    calls: list[ProofReplayRequest] = []

    # When
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(first, second, third),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=2, second=3), ProofCpuPair(first=4, second=5)),
        prove=_prove_success(calls),
        write_artifact=_write_json,
    )

    # Then
    assert summary.replay_count == 6
    assert tuple(request.input_sha256 for request in calls) == (
        _sha256(b"c"),
        _sha256(b"a"),
        _sha256(b"b"),
    )
    assert tuple((request.first_cpu, request.second_cpu) for request in calls) == (
        (2, 3),
        (4, 5),
        (2, 3),
    )
    assert all(request.first_cpu != request.second_cpu for request in calls)
    assert (
        len(
            {
                name
                for request in calls
                for name in (request.first_container_name, request.second_container_name)
            }
        )
        == 6
    )
    assert calls[0].first_container_name.endswith("-a")
    assert calls[0].second_container_name.endswith("-b")
    assert calls[0].first_seed != calls[0].second_seed
    assert tuple(summary.result_artifact_paths) == tuple(
        f"proof-results/{request.profile.target_id}/{request.input_sha256}.json"
        for request in calls
    )


def test_runs_parallel_but_returns_and_writes_in_deterministic_order(tmp_path: Path) -> None:
    """Given uneven proof durations, when run, then output order follows schedule."""
    # Given
    root = tmp_path / "experiment"
    first = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    second = _finished_cell(root, "libpng", "pseudo-feedback", 0, (("crash-b", b"b"),))
    started: list[str] = []
    written: list[str] = []

    def prove(request: ProofReplayRequest) -> ProofReplayResult:
        started.append(request.input_sha256)
        if request.input_sha256 == _sha256(b"a"):
            time.sleep(0.05)
        return _proof_result(request)

    def write_artifact(
        *,
        experiment_root: Path,
        relative_path: Path,
        value: CanonicalValue,
    ) -> ArtifactEvidence:
        written.append(relative_path.as_posix())
        return _write_json(
            experiment_root=experiment_root,
            relative_path=relative_path,
            value=value,
        )

    # When
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(first, second),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1), ProofCpuPair(first=2, second=3)),
        prove=prove,
        write_artifact=write_artifact,
    )

    # Then
    assert started == sorted((_sha256(b"a"), _sha256(b"b")))
    assert tuple(item.key for item in summary.results) == (
        f"libpng/{_sha256(b'b')}",
        f"libpng/{_sha256(b'a')}",
    )
    assert written == list(summary.result_artifact_paths)


def test_persists_each_completed_proof_before_later_interruption_and_resumes(
    tmp_path: Path,
) -> None:
    """Given a later proof interruption, when retried, then completed work is skipped."""
    # Given
    root = tmp_path / "experiment"
    first = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    second = _finished_cell(root, "libpng", "pseudo-feedback", 0, (("crash-b", b"b"),))
    first_sha = _sha256(b"b")
    second_sha = _sha256(b"a")
    calls: list[str] = []

    def prove_then_interrupt(request: ProofReplayRequest) -> ProofReplayResult:
        calls.append(request.input_sha256)
        if request.input_sha256 == second_sha:
            message = "interrupted"
            raise RuntimeError(message)
        return _proof_result(request)

    # When / Then
    with pytest.raises(RuntimeError, match="interrupted"):
        _ = prove_finished_crashes_batch(
            experiment_root=root,
            finished_cells=(first, second),
            proof_images=_IMAGE_IDS,
            cpu_pairs=(ProofCpuPair(first=0, second=1),),
            prove=prove_then_interrupt,
            write_artifact=_write_json,
        )
    persisted = root / "proof-results/libpng" / f"{first_sha}.json"
    assert persisted.is_file()

    retry_calls: list[ProofReplayRequest] = []
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(first, second),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1),),
        prove=_prove_success(retry_calls),
        write_artifact=_write_json,
    )

    assert calls == [first_sha, second_sha]
    assert tuple(request.input_sha256 for request in retry_calls) == (second_sha,)
    assert summary.executed_count == 1
    assert summary.skipped_count == 1


def test_one_worker_shard_owns_each_cpu_pair_without_overlap(tmp_path: Path) -> None:
    """Given uneven proof durations, when sharded, then CPUs never overlap."""
    # Given
    root = tmp_path / "experiment"
    cells = (
        _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),)),
        _finished_cell(root, "libpng", "pseudo-feedback", 0, (("crash-b", b"b"),)),
        _finished_cell(root, "libtiff", "libfuzzer-only", 0, (("crash-c", b"c"),)),
    )
    lock = threading.Lock()
    active_cpus: set[int] = set()
    overlaps: list[int] = []
    pair_zero_starts = 0

    def prove(request: ProofReplayRequest) -> ProofReplayResult:
        nonlocal pair_zero_starts
        cpus = (request.first_cpu, request.second_cpu)
        with lock:
            overlaps.extend(cpu for cpu in cpus if cpu in active_cpus)
            active_cpus.update(cpus)
            if request.first_cpu == 0:
                pair_zero_starts += 1
                is_first_pair_zero = pair_zero_starts == 1
            else:
                is_first_pair_zero = False
        time.sleep(0.12 if is_first_pair_zero else 0.02)
        with lock:
            for cpu in cpus:
                active_cpus.discard(cpu)
        return _proof_result(request)

    # When
    _ = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=cells,
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1), ProofCpuPair(first=2, second=3)),
        prove=prove,
        write_artifact=_write_json,
    )

    # Then
    assert overlaps == []


def test_persisted_record_uses_relative_request_input_path(tmp_path: Path) -> None:
    """Given a proof result, when persisted, then root paths do not leak to JSON."""
    # Given
    root = tmp_path / "experiment"
    cell = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))

    # When
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(cell,),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1),),
        prove=_prove_success([]),
        write_artifact=_write_json,
    )

    # Then
    payload = (root / summary.result_artifact_paths[0]).read_text(encoding="utf-8")
    assert str(root) not in payload
    assert f"proof-inputs/libpng/{_sha256(b'a')}" in payload


def test_resume_uses_valid_existing_matching_result_without_replay(tmp_path: Path) -> None:
    """Given a canonical existing proof artifact, when resumed, then it is skipped."""
    # Given
    root = tmp_path / "experiment"
    cell = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    request = _expected_request(root, cell, b"a", ProofCpuPair(first=0, second=1), 0)
    proof_input = root / "proof-inputs/libpng" / _sha256(b"a")
    proof_input.parent.mkdir(parents=True)
    _ = proof_input.write_bytes(b"a")
    record = ProofBatchRecord(
        key=f"libpng/{_sha256(b'a')}",
        target="libpng",
        input_sha256=_sha256(b"a"),
        owner_cell=cell.cell.to_manifest(),
        source_finding_path="cells/libpng/libfuzzer-only/rep-0/shared/findings/crash-a",
        proof_input=ArtifactEvidence(
            path=f"proof-inputs/libpng/{_sha256(b'a')}",
            sha256=_sha256(b"a"),
            size_bytes=1,
        ),
        request=_request_record(root, request),
        result=_proof_result(request),
    )
    result_path = root / "proof-results/libpng" / f"{_sha256(b'a')}.json"
    result_path.parent.mkdir(parents=True)
    _ = result_path.write_bytes(canonical_json_bytes(record))
    calls: list[ProofReplayRequest] = []

    # When
    summary = prove_finished_crashes_batch(
        experiment_root=root,
        finished_cells=(cell,),
        proof_images=_IMAGE_IDS,
        cpu_pairs=(ProofCpuPair(first=0, second=1),),
        prove=_prove_success(calls),
        write_artifact=_write_json,
    )

    # Then
    assert calls == []
    assert summary.executed_count == 0
    assert summary.skipped_count == 1
    assert summary.replay_count == 0
    assert summary.results[0].record.result.status == "proved"


@pytest.mark.parametrize(
    "case_name",
    [
        "proved_empty_runs",
        "mismatched_seed_name",
        "bad_log_hash",
        "bad_argv_hash",
        "wrong_status_classification",
    ],
)
def test_resume_rejects_forged_canonical_proof_results(
    tmp_path: Path,
    case_name: str,
) -> None:
    """Given canonical but forged proof JSON, when resumed, then it fails closed."""
    # Given
    root = tmp_path / "experiment"
    cell = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    request = _expected_request(root, cell, b"a", ProofCpuPair(first=0, second=1), 0)
    proof_input = root / "proof-inputs/libpng" / _sha256(b"a")
    proof_input.parent.mkdir(parents=True)
    _ = proof_input.write_bytes(b"a")
    record = _record(root, cell, request, _forged_result(case_name, request))
    result_path = root / "proof-results/libpng" / f"{_sha256(b'a')}.json"
    result_path.parent.mkdir(parents=True)
    _ = result_path.write_bytes(canonical_json_bytes(record))

    # When / Then
    with pytest.raises(ProofBatchError, match="proof result conflicts"):
        _ = prove_finished_crashes_batch(
            experiment_root=root,
            finished_cells=(cell,),
            proof_images=_IMAGE_IDS,
            cpu_pairs=(ProofCpuPair(first=0, second=1),),
            prove=_prove_success([]),
            write_artifact=_write_json,
        )


@pytest.mark.parametrize(
    ("case_name", "message"),
    [
        ("bad_finding_hash", "finding sha256 mismatch"),
        ("tamper_proof_input", "proof input sha256 mismatch"),
        ("malformed_result", "malformed proof result"),
        ("conflicting_result", "proof result conflicts"),
    ],
)
def test_rejects_tampered_inputs_and_resume_artifacts(
    tmp_path: Path,
    case_name: str,
    message: str,
) -> None:
    """Given tampered artifacts, when batching, then execution fails closed."""
    # Given
    root = tmp_path / "experiment"
    cell = _finished_cell(root, "libpng", "libfuzzer-only", 0, (("crash-a", b"a"),))
    if case_name == "bad_finding_hash":
        finding = root / cell.result.findings[0].path
        bad = replace(cell.result.findings[0], sha256="0" * 64)
        assert finding.read_bytes() == b"a"
        cell = replace(cell, result=replace(cell.result, findings=(bad,)))
    if case_name == "tamper_proof_input":
        proof_input = root / "proof-inputs/libpng" / _sha256(b"a")
        proof_input.parent.mkdir(parents=True)
        _ = proof_input.write_bytes(b"tampered")
    if case_name in {"malformed_result", "conflicting_result"}:
        result_path = root / "proof-results/libpng" / f"{_sha256(b'a')}.json"
        result_path.parent.mkdir(parents=True)
        if case_name == "malformed_result":
            _ = result_path.write_text("{", encoding="utf-8")
        else:
            request = _expected_request(root, cell, b"a", ProofCpuPair(first=0, second=1), 0)
            _ = (root / "proof-inputs/libpng").mkdir(parents=True, exist_ok=True)
            _ = (root / "proof-inputs/libpng" / _sha256(b"a")).write_bytes(b"a")
            record = ProofBatchRecord(
                key=f"libxml2/{_sha256(b'a')}",
                target="libxml2",
                input_sha256=_sha256(b"a"),
                owner_cell=cell.cell.to_manifest(),
                source_finding_path="cells/libpng/libfuzzer-only/rep-0/shared/findings/crash-a",
                proof_input=ArtifactEvidence(
                    path=f"proof-inputs/libpng/{_sha256(b'a')}",
                    sha256=_sha256(b"a"),
                    size_bytes=1,
                ),
                request=_request_record(root, request),
                result=_proof_result(request),
            )
            _ = result_path.write_bytes(canonical_json_bytes(record))

    # When / Then
    with pytest.raises(ProofBatchError, match=message):
        _ = prove_finished_crashes_batch(
            experiment_root=root,
            finished_cells=(cell,),
            proof_images=_IMAGE_IDS,
            cpu_pairs=(ProofCpuPair(first=0, second=1),),
            prove=_prove_success([]),
            write_artifact=_write_json,
        )


def _finished_cell(
    root: Path,
    target: Target,
    arm: Arm,
    repetition: int,
    findings: tuple[tuple[str, bytes], ...],
) -> FinishedCell:
    cell = _cell(target, arm, repetition)
    work_dir = root / "cells" / target / arm / f"rep-{repetition}"
    shared_dir = work_dir / "shared"
    findings_dir = shared_dir / "findings"
    findings_dir.mkdir(parents=True)
    evidences: list[ArtifactEvidence] = []
    for name, content in findings:
        path = findings_dir / name
        _ = path.write_bytes(content)
        evidences.append(
            ArtifactEvidence(
                path=path.relative_to(root).as_posix(),
                sha256=_sha256(content),
                size_bytes=len(content),
            )
        )
    log_path = work_dir / "main.log"
    _ = log_path.write_text("log", encoding="utf-8")
    return FinishedCell(
        cell=cell,
        work_dir=work_dir,
        staged_corpus=StagedCorpus(
            corpus_dir=work_dir / "corpus",
            files=(),
            tree_sha256="1" * 64,
            base_file_count=0,
            candidate_file_count=0,
        ),
        request=ContainerRequest(
            name=f"main-{target}-{arm}-{repetition}",
            profile=runtime_profile(target),
            image_id=_IMAGE_IDS[target],
            shared_dir=shared_dir,
            corpus_dir=work_dir / "corpus",
            duration_seconds=7,
            fuzzer_seed=cell.seeds.main_fuzzer,
            cpu_index=cell.cpu_index,
            poll_seconds=5,
        ),
        launch=ContainerLaunch(container_id=f"container-{target}-{arm}-{repetition}"),
        result=ContainerRunResult(
            container_id=f"container-{target}-{arm}-{repetition}",
            exit_code=0,
            docker_status="exited",
            oom_killed=False,
            timed_out=False,
            process_error=None,
            logs_text="log",
            log_sha256=_sha256(b"log"),
            metrics=FuzzerMetrics(0, 0, 0, 0),
            canaries={},
            findings=tuple(evidences),
        ),
        log_path=log_path,
    )


def _cell(target: Target, arm: Arm, repetition: int) -> CellSchedule:
    return CellSchedule(
        target=target,
        arm=arm,
        repetition=repetition,
        cpu_index=repetition,
        image_id=_IMAGE_IDS[target],
        allocation=BudgetAllocation(
            search_worker_seconds=1,
            initial_probe_worker_seconds=1,
            revision_probe_worker_seconds=1,
            campaign_worker_seconds=1,
            proof_worker_seconds=10,
        ),
        worker_ledger=(WorkerLedgerItem("main-campaign", 1),),
        seeds=SeedNamespace(
            model_particles=1 + repetition,
            model_revision=2 + repetition,
            pseudo_permutation=3 + repetition,
            initial_probe=4 + repetition,
            revision_probe=5 + repetition,
            main_fuzzer=6 + repetition,
            proof_replay_a=100 + repetition,
            proof_replay_b=200 + repetition,
        ),
    )


def _prove_success(calls: list[ProofReplayRequest]) -> ProveInput:
    def prove(request: ProofReplayRequest) -> ProofReplayResult:
        calls.append(request)
        return _proof_result(request)

    return prove


def _record(
    root: Path,
    cell: FinishedCell,
    request: ProofReplayRequest,
    result: ProofReplayResult,
) -> ProofBatchRecord:
    return ProofBatchRecord(
        key=f"{cell.cell.target}/{request.input_sha256}",
        target=cell.cell.target,
        input_sha256=request.input_sha256,
        owner_cell=cell.cell.to_manifest(),
        source_finding_path="cells/libpng/libfuzzer-only/rep-0/shared/findings/crash-a",
        proof_input=ArtifactEvidence(
            path=f"proof-inputs/libpng/{request.input_sha256}",
            sha256=request.input_sha256,
            size_bytes=1,
        ),
        request=_request_record(root, request),
        result=result,
    )


def _forged_result(case_name: str, request: ProofReplayRequest) -> ProofReplayResult:
    valid = _proof_result(request)
    match case_name:
        case "proved_empty_runs":
            return replace(valid, runs=())
        case "mismatched_seed_name":
            first, second = valid.runs
            forged = replace(first, proof_seed=first.proof_seed + 1)
            return replace(valid, runs=(forged, second))
        case "bad_log_hash":
            first, second = valid.runs
            forged = replace(first, log_sha256="0" * 64)
            return replace(valid, runs=(forged, second))
        case "bad_argv_hash":
            first, second = valid.runs
            forged = replace(first, argv_sha256="0" * 64)
            return replace(valid, runs=(forged, second))
        case "wrong_status_classification":
            return replace(valid, status="not_reproduced", signature=None)
        case _:
            raise AssertionError(case_name)


def _proof_result(request: ProofReplayRequest) -> ProofReplayResult:
    first_log = _asan_log(b"0x111")
    second_log = _asan_log(b"0x999")
    logs = [first_log, second_log]

    def runner(
        argv: Sequence[str],
        *,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> CompletedProcess[bytes]:
        assert capture_output is True
        assert check is False
        assert timeout > 0
        return CompletedProcess(tuple(argv), 1, b"", logs.pop(0))

    return prove_input(request, runner=runner)


def _asan_log(address: bytes) -> bytes:
    return (
        b"==1==ERROR: AddressSanitizer: heap-buffer-overflow on address "
        + address
        + b"\n    #0 "
        + address
        + b" in target_frame /src/target.c:1\n"
    )


def _expected_request(
    root: Path,
    cell: FinishedCell,
    content: bytes,
    pair: ProofCpuPair,
    global_index: int,
) -> ProofReplayRequest:
    input_sha256 = _sha256(content)
    first_seed = _seed(cell.cell.seeds.proof_replay_a, global_index)
    second_seed = _seed(cell.cell.seeds.proof_replay_b, global_index)
    return ProofReplayRequest(
        profile=runtime_profile(cell.cell.target),
        image_id=_IMAGE_IDS[cell.cell.target],
        input_path=root / "proof-inputs" / cell.cell.target / input_sha256,
        input_sha256=input_sha256,
        first_seed=first_seed,
        second_seed=second_seed,
        first_container_name=f"dllm-proof-{cell.cell.target}-{input_sha256}-{global_index:06d}-a",
        second_container_name=f"dllm-proof-{cell.cell.target}-{input_sha256}-{global_index:06d}-b",
        first_cpu=pair.first,
        second_cpu=pair.second,
        timeout_seconds=60,
    )


def _request_record(root: Path, request: ProofReplayRequest) -> ProofReplayRequestRecord:
    return ProofReplayRequestRecord(
        profile=request.profile,
        image_id=request.image_id,
        input_path=request.input_path.relative_to(root).as_posix(),
        input_sha256=request.input_sha256,
        first_seed=request.first_seed,
        second_seed=request.second_seed,
        first_container_name=request.first_container_name,
        second_container_name=request.second_container_name,
        first_cpu=request.first_cpu,
        second_cpu=request.second_cpu,
        timeout_seconds=request.timeout_seconds,
    )


def _seed(namespace: int, global_index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{global_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _write_json(
    *,
    experiment_root: Path,
    relative_path: Path,
    value: CanonicalValue,
) -> ArtifactEvidence:
    destination = experiment_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    if destination.exists():
        if destination.read_bytes() != payload:
            msg = f"artifact already exists: {destination}"
            raise AssertionError(msg)
    else:
        _ = destination.write_bytes(payload)
    return ArtifactEvidence(
        path=relative_path.as_posix(),
        sha256=_sha256(payload),
        size_bytes=len(payload),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
