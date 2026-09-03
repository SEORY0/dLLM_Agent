"""Deferred OpenEXR exact-proof batch behavior.

Module-size opt-out ``# noqa: SIZE_OK`` keeps Task 4 proof regressions in one owned test.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Generator, Mapping  # noqa: TC003
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess
from typing import Literal, final

import pytest
from pydantic import TypeAdapter

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue, append_event
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    CrashOccurrenceRecord,
    ExactProofRequestModel,
    ProofBatchError,
    ProofBatchRequest,
    ProofVerdict,
    append_crash_occurrence,
    run_deferred_proof_batch,
)
from dllm_agent.targets.openexr.proof import prove_snapshot, proved_cause_id
from tests.openexr_one_day_task4_fakes import (
    FakeProofOutcome,
    ManualClock,
    ScriptedProofRunner,
)

SHA_A = "7" * 64
SHA_B = "8" * 64
TARGET = "777d231a179de1711d2a942810d9559216ab3f4c"
_BLANK_ROOT_ASAN = b"==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class ProofInterruptedError(RuntimeError):
    """Test-only interruption after one persisted proof verdict."""


@final
class InterruptAfterFirstRunner:
    """Return one proof outcome, then simulate process interruption."""

    def __init__(self, clock: ManualClock, outcome: FakeProofOutcome) -> None:
        self._clock = clock
        self._outcome = outcome
        self.requests: list[ExactProofRequestModel] = []

    def run(self, request: ExactProofRequestModel) -> FakeProofOutcome:
        """Persist the first request outcome and interrupt the second request."""
        if self.requests:
            detail = "interrupted after first verdict"
            raise ProofInterruptedError(detail)
        self.requests.append(request)
        self._clock.now += 1.0
        self._clock.cpu += 2.0
        return self._outcome


@final
class HashOnlyProofOutcome:
    """Legacy fake that exposes hashes without dereferenceable bytes."""

    oracle_status: Literal["proved"] = "proved"
    sanitizer_class = "heap-buffer-overflow"
    root_signature = "Imf::read|LLVMFuzzerTestOneInput"

    def __init__(self, input_sha: str) -> None:
        self.input_sha256 = input_sha
        self.proof_log_sha256 = SHA_A
        self.run_argv_sha256 = (SHA_A, SHA_B)

    @property
    def proof_log(self) -> bytes:
        raise AttributeError

    @property
    def run_argv_bytes(self) -> tuple[bytes, ...]:
        raise AttributeError

    @property
    def result_bytes(self) -> bytes:
        raise AttributeError


def test_proved_cause_id_normalizes_signature_and_binds_target() -> None:
    # Given
    # When
    first = proved_cause_id(
        TARGET, "heap-buffer-overflow", " Imf::read   | LLVMFuzzerTestOneInput "
    )
    second = proved_cause_id(TARGET, "heap-buffer-overflow", "Imf::read | LLVMFuzzerTestOneInput")

    # Then
    assert first == second
    assert (
        first
        == hashlib.sha256(
            b"\n".join(
                (
                    b"777d231a179de1711d2a942810d9559216ab3f4c",
                    b"heap-buffer-overflow",
                    b"Imf::read | LLVMFuzzerTestOneInput",
                )
            )
        ).hexdigest()
    )


def test_batch_deduplicates_exact_inputs_resumes_and_keeps_accounting_separate() -> None:
    # Given
    input_bytes = b"same crash bytes"
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    outcome = _fake_outcome(input_sha)

    # When
    with _data_root() as root:
        cell_root = root / "cell"
        crash_root = cell_root / "crashes"
        input_path = crash_root / "inputs" / input_sha
        input_path.parent.mkdir(parents=True)
        _ = input_path.write_bytes(input_bytes)
        queue = crash_root / "occurrences.jsonl"
        first = CrashOccurrenceRecord(
            sequence=0,
            cell_id="cell-a",
            attempt_index=0,
            artifact_name="crash-a",
            input_path=input_path.relative_to(cell_root).as_posix(),
            input_sha256=input_sha,
            size_bytes=len(input_bytes),
            search_log_sha256=SHA_A,
            cell_manifest_sha256=SHA_B,
        )
        second = first.model_copy(update={"sequence": 1, "attempt_index": 1})
        append_crash_occurrence(cell_root, queue, first)
        append_crash_occurrence(cell_root, queue, second)
        request = ProofBatchRequest(
            cell_id="cell-a",
            cell_root=cell_root,
            crash_queue_path=queue,
            proof_root=root / "proof",
            target_commit=TARGET,
            proof_seed=20_260_820,
            duration_seconds=30,
            search_finished_monotonic_seconds=20.0,
            search_cpu_seconds=10.0,
        )
        clock = ManualClock(20.0, cpu=30.0)
        runner = ScriptedProofRunner(clock, (outcome,), wall_seconds=2.0, cpu_seconds=4.0)
        first_result = run_deferred_proof_batch(request, runner, clock)
        forbidden = ScriptedProofRunner(clock, (), wall_seconds=9.0, cpu_seconds=9.0)
        resumed = run_deferred_proof_batch(request, forbidden, clock)

        # Then
        assert len(runner.requests) == 1
        assert runner.requests[0].snapshot == input_bytes
        assert runner.requests[0].proof_seed == 20_260_820
        assert forbidden.requests == []
        assert len(first_result.verdicts) == 1
        assert resumed.verdicts == first_result.verdicts
        assert first_result.search_accounting.cpu_seconds == 10.0
        assert first_result.proof_accounting.cpu_seconds == 4.0
        assert first_result.proof_accounting.replay_count == 2
        assert len(first_result.cause_index.causes) == 1
        assert first_result.cause_index.causes[0].input_sha256 == (input_sha,)


def test_distinct_inputs_with_same_signature_share_one_cause() -> None:
    # Given
    contents = (b"first", b"second")
    hashes = tuple(hashlib.sha256(item).hexdigest() for item in contents)
    outcomes = tuple(_fake_outcome(input_sha) for input_sha in hashes)

    # When
    with _data_root() as root:
        cell_root = root / "cell"
        queue = cell_root / "crashes/occurrences.jsonl"
        for sequence, (content, input_sha) in enumerate(zip(contents, hashes, strict=True)):
            path = cell_root / "crashes/inputs" / input_sha
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(content)
            append_crash_occurrence(
                cell_root,
                queue,
                CrashOccurrenceRecord(
                    sequence=sequence,
                    cell_id="cell-a",
                    attempt_index=sequence,
                    artifact_name=f"crash-{sequence}",
                    input_path=path.relative_to(cell_root).as_posix(),
                    input_sha256=input_sha,
                    size_bytes=len(content),
                    search_log_sha256=SHA_A,
                    cell_manifest_sha256=SHA_B,
                ),
            )
        request = ProofBatchRequest(
            cell_id="cell-a",
            cell_root=cell_root,
            crash_queue_path=queue,
            proof_root=root / "proof",
            target_commit=TARGET,
            proof_seed=20_260_820,
            duration_seconds=30,
            search_finished_monotonic_seconds=20.0,
            search_cpu_seconds=10.0,
        )
        clock = ManualClock(20.0)
        result = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(clock, outcomes, 1.0, 2.0),
            clock,
        )

        # Then
        assert len(result.verdicts) == 2
        assert len(result.cause_index.causes) == 1
        assert result.cause_index.causes[0].input_sha256 == tuple(sorted(hashes))


def test_proof_batch_refuses_to_run_before_search_completion() -> None:
    # Given
    # When / Then
    with _data_root() as root:
        cell_root = root / "cell"
        queue = cell_root / "crashes/occurrences.jsonl"
        queue.parent.mkdir(parents=True)
        _ = queue.write_bytes(b"")
        request = ProofBatchRequest(
            cell_id="cell-a",
            cell_root=cell_root,
            crash_queue_path=queue,
            proof_root=root / "proof",
            target_commit=TARGET,
            proof_seed=20_260_820,
            duration_seconds=30,
            search_finished_monotonic_seconds=20.0,
            search_cpu_seconds=10.0,
        )
        clock = ManualClock(19.0)
        runner = ScriptedProofRunner(clock, (), 1.0, 1.0)
        with pytest.raises(ProofBatchError, match="search completion"):
            _ = run_deferred_proof_batch(request, runner, clock)
        assert runner.requests == []


def test_proof_resume_rejects_non_verdict_event_type() -> None:
    # Given
    with _data_root() as root:
        request, (input_sha,) = _proof_fixture(root, (b"wrong event type",))
        clock = ManualClock(20.0)
        result = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(clock, (_fake_outcome(input_sha),), 1.0, 2.0),
            clock,
        )
        verdict_path = request.proof_root / "verdicts.jsonl"
        _ = verdict_path.write_bytes(b"")
        _append_proof_event(verdict_path, result.verdicts[0], "not_proof_verdict")
        forbidden = ScriptedProofRunner(clock, (), 1.0, 1.0)

        # When / Then
        with pytest.raises(ProofBatchError, match="unexpected proof verdict event"):
            _ = run_deferred_proof_batch(request, forbidden, clock)
        assert forbidden.requests == []


def test_proof_resume_rejects_duplicate_input_verdicts() -> None:
    # Given
    with _data_root() as root:
        request, (input_sha,) = _proof_fixture(root, (b"duplicate verdict",))
        clock = ManualClock(20.0)
        result = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(clock, (_fake_outcome(input_sha),), 1.0, 2.0),
            clock,
        )
        _append_proof_event(
            request.proof_root / "verdicts.jsonl",
            result.verdicts[0],
            "proof_verdict",
        )
        forbidden = ScriptedProofRunner(clock, (), 1.0, 1.0)

        # When / Then
        with pytest.raises(ProofBatchError, match="duplicate proof verdict"):
            _ = run_deferred_proof_batch(request, forbidden, clock)
        assert forbidden.requests == []


def test_proof_resume_rejects_verdict_outside_current_crash_set() -> None:
    # Given
    with _data_root() as root:
        request, hashes = _proof_fixture(root, (b"retained crash", b"stale crash"))
        clock = ManualClock(20.0)
        result = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(
                clock,
                tuple(_fake_outcome(item) for item in hashes),
                1.0,
                2.0,
            ),
            clock,
        )
        _ = request.crash_queue_path.write_bytes(b"")
        append_crash_occurrence(
            request.cell_root, request.crash_queue_path, result.verdicts[0].first_owner
        )
        forbidden = ScriptedProofRunner(clock, (), 1.0, 1.0)

        # When / Then
        with pytest.raises(ProofBatchError, match="outside current deduplicated crash set"):
            _ = run_deferred_proof_batch(request, forbidden, clock)
        assert forbidden.requests == []


def test_interrupted_proof_batch_resumes_unique_canonical_subset() -> None:
    # Given
    with _data_root() as root:
        request, hashes = _proof_fixture(root, (b"first interrupted", b"second resumed"))
        clock = ManualClock(20.0)
        interrupted = InterruptAfterFirstRunner(clock, _fake_outcome(hashes[0]))

        # When
        with pytest.raises(ProofInterruptedError, match="interrupted after first verdict"):
            _ = run_deferred_proof_batch(request, interrupted, clock)

        # Then
        verdict_path = request.proof_root / "verdicts.jsonl"
        assert len(verdict_path.read_bytes().splitlines()) == 1
        assert not (request.proof_root / "cause-index.json").exists()

        # When
        resumed_runner = ScriptedProofRunner(
            clock,
            (_fake_outcome(hashes[1]),),
            1.0,
            2.0,
        )
        resumed = run_deferred_proof_batch(request, resumed_runner, clock)

        # Then
        assert tuple(item.input_sha256 for item in resumed.verdicts) == hashes
        assert [item.snapshot_sha256 for item in resumed_runner.requests] == [hashes[1]]
        assert (request.proof_root / "cause-index.json").is_file()


def test_completed_proof_batch_requires_exact_current_input_set() -> None:
    # Given
    with _data_root() as root:
        request, (first_sha,) = _proof_fixture(root, (b"completed input",))
        clock = ManualClock(20.0)
        _ = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(clock, (_fake_outcome(first_sha),), 1.0, 2.0),
            clock,
        )
        second_sha = _append_crash_input(request, b"late crash", sequence=1)
        forbidden = ScriptedProofRunner(
            clock,
            (_fake_outcome(second_sha),),
            1.0,
            2.0,
        )

        # When / Then
        with pytest.raises(ProofBatchError, match="completed proof input set"):
            _ = run_deferred_proof_batch(request, forbidden, clock)
        assert forbidden.requests == []


def test_proof_batch_rejects_hash_only_outcome_without_backing_bytes() -> None:
    # Given
    with _data_root() as root:
        request, (input_sha,) = _proof_fixture(root, (b"hash-only outcome",))
        clock = ManualClock(20.0)

        # When / Then
        with pytest.raises(ProofBatchError, match="dereferenceable evidence"):
            _ = run_deferred_proof_batch(
                request,
                ScriptedProofRunner(
                    clock,
                    (HashOnlyProofOutcome(input_sha),),
                    1.0,
                    2.0,
                ),
                clock,
            )


def test_proof_resume_rejects_tampered_backing_evidence() -> None:
    # Given
    with _data_root() as root:
        request, (input_sha,) = _proof_fixture(root, (b"tampered proof evidence",))
        clock = ManualClock(20.0)
        result = run_deferred_proof_batch(
            request,
            ScriptedProofRunner(clock, (_fake_outcome(input_sha),), 1.0, 2.0),
            clock,
        )
        proof_log = request.proof_root / result.verdicts[0].proof_log_evidence.relative_path
        _ = proof_log.write_bytes(b"tampered")

        # When / Then
        with pytest.raises(ProofBatchError, match="content mismatch"):
            _ = run_deferred_proof_batch(
                request,
                ScriptedProofRunner(clock, (), 1.0, 1.0),
                clock,
            )


def test_proved_verdict_rejects_blank_normalized_root_signature() -> None:
    # Given
    with _data_root() as root:
        request, (input_sha,) = _proof_fixture(root, (b"blank root",))
        outcome = _fake_outcome(input_sha, root_signature=" \t  ")
        clock = ManualClock(20.0)

        # When / Then
        with pytest.raises(ProofBatchError, match="nonblank root signature"):
            _ = run_deferred_proof_batch(
                request,
                ScriptedProofRunner(clock, (outcome,), 1.0, 2.0),
                clock,
            )
        with pytest.raises(ProofBatchError, match="nonblank root signature"):
            _ = proved_cause_id(TARGET, "heap-buffer-overflow", " \t  ")


def test_exact_proof_downgrades_asan_without_root_frames() -> None:
    # Given
    snapshot = b"blank-root-snapshot"
    request = ExactProofRequestModel(
        snapshot=snapshot,
        snapshot_sha256=sha256_bytes(snapshot),
        duration_seconds=1,
        proof_seed=7,
    )
    calls: list[tuple[str, ...]] = []

    def runner(
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        calls.append(argv)
        return CompletedProcess(argv, 1, _BLANK_ROOT_ASAN, b"")

    # When
    result = prove_snapshot(request, runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.root_signature is None
    assert len(calls) == 2
    assert result.run_argv_bytes == tuple(
        json.dumps(call, separators=(",", ":")).encode() for call in calls
    )
    assert result.run_argv_sha256 == tuple(sha256_bytes(item) for item in result.run_argv_bytes)
    result_payload = _JSON_OBJECT.validate_json(result.result_bytes)
    assert result_payload["input_sha256"] == request.snapshot_sha256
    assert result_payload["proof_log_sha256"] == result.proof_log_sha256
    assert "proof_log" not in result_payload


def _proof_fixture(
    root: Path, contents: tuple[bytes, ...]
) -> tuple[ProofBatchRequest, tuple[str, ...]]:
    cell_root = root / "cell"
    queue = cell_root / "crashes/occurrences.jsonl"
    hashes = tuple(
        _append_input(cell_root, queue, content, sequence)
        for sequence, content in enumerate(contents)
    )
    return (
        ProofBatchRequest(
            cell_id="cell-a",
            cell_root=cell_root,
            crash_queue_path=queue,
            proof_root=root / "proof",
            target_commit=TARGET,
            proof_seed=20_260_820,
            duration_seconds=30,
            search_finished_monotonic_seconds=20.0,
            search_cpu_seconds=10.0,
        ),
        hashes,
    )


def _append_crash_input(request: ProofBatchRequest, content: bytes, sequence: int) -> str:
    return _append_input(request.cell_root, request.crash_queue_path, content, sequence)


def _append_input(cell_root: Path, queue: Path, content: bytes, sequence: int) -> str:
    input_sha = sha256_bytes(content)
    path = cell_root / "crashes/inputs" / input_sha
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    append_crash_occurrence(
        cell_root,
        queue,
        CrashOccurrenceRecord(
            sequence=sequence,
            cell_id="cell-a",
            attempt_index=sequence,
            artifact_name=f"crash-{sequence}",
            input_path=path.relative_to(cell_root).as_posix(),
            input_sha256=input_sha,
            size_bytes=len(content),
            search_log_sha256=SHA_A,
            cell_manifest_sha256=SHA_B,
        ),
    )
    return input_sha


def _fake_outcome(
    input_sha: str,
    *,
    root_signature: str = "Imf::read|LLVMFuzzerTestOneInput",
) -> FakeProofOutcome:
    proof_log = f"proof-log:{input_sha}".encode()
    run_argv_bytes = (
        json.dumps(("proof", input_sha, 0), separators=(",", ":")).encode(),
        json.dumps(("proof", input_sha, 1), separators=(",", ":")).encode(),
    )
    result_bytes = json.dumps(
        {"input_sha256": input_sha, "status": "proved"},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return FakeProofOutcome(
        oracle_status="proved",
        input_sha256=input_sha,
        proof_log_sha256=sha256_bytes(proof_log),
        proof_log=proof_log,
        sanitizer_class="heap-buffer-overflow",
        root_signature=root_signature,
        run_argv_sha256=tuple(sha256_bytes(item) for item in run_argv_bytes),
        run_argv_bytes=run_argv_bytes,
        result_bytes=result_bytes,
    )


def _append_proof_event(path: Path, verdict: ProofVerdict, event_type: str) -> None:
    _ = append_event(
        path,
        timestamp_utc=f"proof-input+{verdict.input_sha256}",
        event_type=event_type,
        payload=_JSON_OBJECT.validate_python(verdict.model_dump(mode="json")),
    )


@contextmanager
def _data_root() -> Generator[Path]:
    with tempfile.TemporaryDirectory(prefix="dllm-task4-proof-", dir="/data") as directory:
        yield Path(directory)
