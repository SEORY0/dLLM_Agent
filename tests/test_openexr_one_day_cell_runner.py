"""Long-running OpenEXR search-cell behavior.

Module-size opt-out ``# noqa: SIZE_OK`` keeps Task 4 cell regressions in one owned test.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Generator  # noqa: TC003
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest
from pydantic import TypeAdapter, ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue, append_event, validate_ledger
from dllm_agent.experiments.openexr_one_day import cell_runner
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    CellEvent,
    CellEvidenceError,
    CellManifest,
    CellRunRequest,
    DockerSearchLauncher,
    SearchArtifact,
    SearchAttemptResult,
    SearchBuildProvenance,
    SearchLaunchRequest,
    load_crash_occurrences,
    run_openexr_cell,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRequest,
    ExecutionProfile,
    load_corpus_index,
)
from dllm_agent.experiments.openexr_one_day.proof_batch import (
    CrashOccurrenceRecord,
    append_crash_occurrence,
)
from dllm_agent.targets.openexr.probe import ProbeRequest, build_long_probe_argv
from tests.openexr_one_day_task4_fakes import (
    FakeSearchPopen,
    ManualClock,
    ProcessScript,
    ScriptedSearchLauncher,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_F = "f" * 64
IMAGE = f"sha256:{'e' * 64}"
START = 1_000.0
DEADLINE = 11_800.0
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
type _LogMutation = Literal["missing", "extra", "symlink", "wrong_attempt", "byte_mismatch"]
type _ArtifactMutation = Literal["missing", "extra", "symlink", "byte_mismatch", "ambiguous"]
type _OccurrenceMutation = Literal["wrong_attempt", "duplicate"]
type _InvalidSanitizerOutcome = Literal["failed", "completed", "deadline"]
type _NonSanitizerKind = Literal["timeout", "oom", "slow_unit", "crash"]


def test_micro_probe_duration_cap_remains_thirty_seconds() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = ProbeRequest(snapshot=b"x", fuzzer_seed=1, duration_seconds=31, max_len=8)


def test_cell_request_rejects_asan_only_build_provenance() -> None:
    # When / Then
    with _data_root() as root, pytest.raises(ValidationError, match=r"address.*undefined"):
        _ = _request(root, sanitizers=("address",))


def test_target_crash_restarts_against_one_absolute_deadline_without_proof() -> None:
    # Given
    crash_bytes = b"exact crashing bytes"
    crash = SearchAttemptResult(
        outcome="target_crash",
        return_code=1,
        log=b"ERROR: AddressSanitizer: heap-buffer-overflow\n",
        cpu_seconds=6.0,
        artifacts=(SearchArtifact(name="crash-a", kind="sanitizer", content=crash_bytes),),
    )
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=4.0, artifacts=()
    )
    terminated = SearchAttemptResult(
        outcome="deadline", return_code=124, log=b"deadline\n", cpu_seconds=0.0, artifacts=()
    )

    # When
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        launcher = ScriptedSearchLauncher(
            clock,
            (
                ProcessScript(1_004.0, crash, terminated),
                ProcessScript(DEADLINE, completed, terminated),
            ),
        )
        result = run_openexr_cell(request, launcher, clock)
        occurrences = load_crash_occurrences(request)
        persisted = Path(request.crash_root) / "inputs" / hashlib.sha256(crash_bytes).hexdigest()

        # Then
        assert tuple(item.remaining_seconds for item in launcher.requests) == (10_800.0, 10_796.0)
        assert {item.absolute_deadline_monotonic_seconds for item in launcher.requests} == {
            DEADLINE
        }
        assert {item.fuzzer_seed for item in launcher.requests} == {71}
        assert {item.corpus_root for item in launcher.requests} == {request.corpus_root}
        assert result.attempt_count == 2
        assert result.search_accounting.cpu_seconds == 10.0
        assert result.proof_replay_count == 0
        assert persisted.read_bytes() == crash_bytes
        assert len(occurrences) == 1
        assert occurrences[0].input_sha256 == hashlib.sha256(crash_bytes).hexdigest()
        assert (
            occurrences[0].cell_manifest_sha256
            == hashlib.sha256(
                (request.artifact_root / "run-manifest.json").read_bytes()
            ).hexdigest()
        )


def test_fixed_deadline_terminates_resident_process_and_keeps_exact_indexes() -> None:
    # Given
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"late\n", cpu_seconds=99.0, artifacts=()
    )
    terminated = SearchAttemptResult(
        outcome="deadline", return_code=124, log=b"killed\n", cpu_seconds=10.0, artifacts=()
    )

    # When
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        launcher = ScriptedSearchLauncher(
            clock, (ProcessScript(DEADLINE + 50.0, completed, terminated),)
        )
        result = run_openexr_cell(request, launcher, clock)

        # Then
        assert launcher.processes[0].terminate_count == 1
        assert result.finished_monotonic_seconds == DEADLINE
        assert result.corpus_index_scheduled_seconds == tuple(range(0, 10_801, 900))
        assert result.search_accounting.cpu_seconds == 10.0


def test_qualification_smoke_cell_uses_300_second_profile_schedule() -> None:
    # Given
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"smoke\n", cpu_seconds=2.0, artifacts=()
    )

    # When
    with _data_root() as root:
        request = _request(
            root,
            execution_profile="qualification-smoke",
            deadline_monotonic_seconds=START + 300.0,
            schedule_seconds=(0, 300),
        )
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        launcher = ScriptedSearchLauncher(
            clock,
            (ProcessScript(START + 300.0, completed, completed),),
        )
        result = run_openexr_cell(request, launcher, clock)
        rows = load_corpus_index(
            CorpusIndexRequest(
                execution_profile="qualification-smoke",
                schedule_seconds=(0, 300),
                artifact_root=request.artifact_root,
                corpus_root=request.corpus_root,
                store_root=request.artifact_root / "corpus-inputs",
                index_path=request.artifact_root / "corpus-index.jsonl",
            )
        )

        # Then
        assert tuple(item.remaining_seconds for item in launcher.requests) == (300.0,)
        assert result.execution_profile == "qualification-smoke"
        assert result.corpus_index_scheduled_seconds == (0, 300)
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == (0, 300)
        assert cell_runner.validate_cell_evidence(request) == result


def test_early_failed_cell_freezes_explicit_curve_through_deadline() -> None:
    # Given
    failed = SearchAttemptResult(
        outcome="failed", return_code=2, log=b"controller failed\n", cpu_seconds=3.0, artifacts=()
    )

    # When
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        result = run_openexr_cell(
            request,
            ScriptedSearchLauncher(clock, (ProcessScript(START + 10.0, failed, failed),)),
            clock,
        )
        rows = load_corpus_index(
            CorpusIndexRequest(
                artifact_root=request.artifact_root,
                corpus_root=request.corpus_root,
                store_root=request.artifact_root / "corpus-inputs",
                index_path=request.artifact_root / "corpus-index.jsonl",
            )
        )

        # Then
        assert result.status == "failed"
        assert result.finished_monotonic_seconds == START + 10.0
        assert result.coverage_freeze_elapsed_seconds == 10.0
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == tuple(range(0, 10_801, 900))
        assert tuple(row.observation_kind for row in rows) == ("observed", *("frozen",) * 12)
        assert tuple(row.source_elapsed_seconds for row in rows) == (0.0, *(10.0,) * 12)
        assert len({row.cumulative_set_sha256 for row in rows}) == 1


def test_cell_persists_canonical_manifest_events_and_result_before_launch() -> None:
    # Given
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=2.0, artifacts=()
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        seen_manifest: list[bytes] = []

        def popen(_argv: tuple[str, ...]) -> FakeSearchPopen:
            seen_manifest.append((request.artifact_root / "run-manifest.json").read_bytes())
            return FakeSearchPopen(completed.log, completed.return_code)

        cpu_ticks = iter((1.0, 3.0))

        # When
        result = run_openexr_cell(
            request,
            DockerSearchLauncher(
                popen_factory=popen,
                monotonic=clock.monotonic,
                cpu_clock=lambda: next(cpu_ticks),
            ),
            clock,
        )

        # Then
        expected_manifest = canonical_json_bytes(
            CellManifest(request=request).model_dump(mode="json")
        )
        assert seen_manifest == [expected_manifest]
        assert cell_runner.load_cell_manifest(request) == CellManifest(request=request)
        assert cell_runner.load_cell_result(request) == result
        assert cell_runner.validate_cell_evidence(request) == result
        events = cell_runner.load_cell_events(request)
        assert tuple(event.sequence for event in events) == (0,)
        assert events[0].outcome == "completed"
        assert events[0].model_dump(mode="json").get("artifacts") == []
        assert events[0].manifest_sha256 == result.manifest_sha256
        assert validate_ledger(request.artifact_root / "events.jsonl").final_stream_sha256 == (
            result.event_stream_sha256
        )
        assert (request.artifact_root / "final-result.json").read_bytes() == canonical_json_bytes(
            result.model_dump(mode="json")
        )


@pytest.mark.parametrize(
    "artifact_name", ["run-manifest.json", "events.jsonl", "final-result.json"]
)
def test_fresh_cell_rejects_existing_evidence_before_launch(artifact_name: str) -> None:
    # Given
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        evidence = request.artifact_root / artifact_name
        evidence.parent.mkdir(parents=True, exist_ok=True)
        _ = evidence.write_bytes(b"existing")
        clock = ManualClock(START)
        launcher = ScriptedSearchLauncher(clock, ())

        # When / Then
        with pytest.raises(CellEvidenceError, match="existing cell evidence"):
            _ = run_openexr_cell(request, launcher, clock)
        assert launcher.requests == []


@pytest.mark.parametrize(
    "artifact_name", ["run-manifest.json", "events.jsonl", "final-result.json"]
)
def test_cell_evidence_validation_rejects_noncanonical_tampering(artifact_name: str) -> None:
    # Given
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=2.0, artifacts=()
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        _ = run_openexr_cell(
            request,
            ScriptedSearchLauncher(clock, (ProcessScript(START + 1.0, completed, completed),)),
            clock,
        )
        evidence = request.artifact_root / artifact_name
        _ = evidence.write_bytes(evidence.read_bytes() + b"\n")

        # When / Then
        with pytest.raises(CellEvidenceError):
            _ = cell_runner.validate_cell_evidence(request)


def test_cell_evidence_validation_rejects_canonical_result_tampering() -> None:
    # Given
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=2.0, artifacts=()
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        result = run_openexr_cell(
            request,
            ScriptedSearchLauncher(clock, (ProcessScript(START + 1.0, completed, completed),)),
            clock,
        )
        tampered = result.model_copy(update={"status": "failed"})
        _ = (request.artifact_root / "final-result.json").write_bytes(
            canonical_json_bytes(tampered.model_dump(mode="json"))
        )

        # When / Then
        with pytest.raises(CellEvidenceError, match="cell evidence artifacts conflict"):
            _ = cell_runner.validate_cell_evidence(request)


def test_cell_evidence_rejects_canonically_rehashed_wrong_log_digest() -> None:
    # Given
    with _data_root() as root:
        request = _run_completed_docker_cell(root)
        events = cell_runner.load_cell_events(request)
        tampered = (events[0].model_copy(update={"log_sha256": SHA_F}),)
        _rewrite_cell_events(request, tampered)

        # When / Then
        with pytest.raises(CellEvidenceError, match="attempt log evidence"):
            _ = cell_runner.validate_cell_evidence(request)


def test_cell_evidence_rejects_canonically_rehashed_wrong_artifact_digest() -> None:
    # Given
    with _data_root() as root:
        request, _digest = _run_sanitizer_cell(root)
        events = cell_runner.load_cell_events(request)
        tampered = (
            events[0].model_copy(update={"artifact_sha256": (SHA_F,)}),
            events[1],
        )
        _rewrite_cell_events(request, tampered)

        # When / Then
        with pytest.raises(
            CellEvidenceError, match=r"invalid cell event stream|attempt artifact evidence"
        ):
            _ = cell_runner.validate_cell_evidence(request)


def test_cell_evidence_rejects_canonical_non_sanitizer_promotion_to_inputs() -> None:
    # Given
    with _data_root() as root:
        request, digest = _run_plain_artifact_cell(root)
        event = cell_runner.load_cell_events(request)[0]
        source = request.crash_root / "crash" / digest
        promoted = request.crash_root / "inputs" / digest
        promoted.parent.mkdir()
        _ = source.rename(promoted)
        append_crash_occurrence(
            request.artifact_root,
            request.crash_root / "occurrences.jsonl",
            CrashOccurrenceRecord(
                sequence=0,
                cell_id=request.cell_id,
                attempt_index=0,
                artifact_name="promoted-crash",
                input_path=promoted.relative_to(request.artifact_root).as_posix(),
                input_sha256=digest,
                size_bytes=promoted.stat().st_size,
                search_log_sha256=event.log_sha256,
                cell_manifest_sha256=event.manifest_sha256,
            ),
        )
        _rewrite_crash_result(request, 1)

        # When / Then
        with pytest.raises(CellEvidenceError, match="attempt artifact evidence"):
            _ = cell_runner.validate_cell_evidence(request)


@pytest.mark.parametrize("outcome", ["failed", "completed", "deadline"])
def test_cell_rejects_sanitizer_artifact_on_non_target_crash_outcome(
    outcome: _InvalidSanitizerOutcome,
) -> None:
    # Given
    invalid = SearchAttemptResult(
        outcome=outcome,
        return_code=2,
        log=b"ERROR: AddressSanitizer: invalid-outcome\n",
        cpu_seconds=1.0,
        artifacts=(SearchArtifact(name="invalid-sanitizer", kind="sanitizer", content=b"invalid"),),
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)

        # When / Then
        with pytest.raises(CellEvidenceError, match=r"sanitizer artifact.*target_crash"):
            _ = run_openexr_cell(
                request,
                ScriptedSearchLauncher(
                    clock,
                    (ProcessScript(START + 1.0, invalid, invalid),),
                ),
                clock,
            )


def test_cell_rejects_target_crash_outcome_without_sanitizer_artifact() -> None:
    # Given
    invalid = SearchAttemptResult(
        outcome="target_crash",
        return_code=1,
        log=b"crash without sanitizer\n",
        cpu_seconds=1.0,
        artifacts=(SearchArtifact(name="plain-crash", kind="crash", content=b"plain"),),
    )
    completed = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=1.0, artifacts=()
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)

        # When / Then
        with pytest.raises(CellEvidenceError, match=r"target_crash.*sanitizer artifact"):
            _ = run_openexr_cell(
                request,
                ScriptedSearchLauncher(
                    clock,
                    (
                        ProcessScript(START + 1.0, invalid, invalid),
                        ProcessScript(START + 2.0, completed, completed),
                    ),
                ),
                clock,
            )


@pytest.mark.parametrize("kind", ["timeout", "oom", "slow_unit", "crash"])
def test_cell_event_persists_non_sanitizer_artifact_kind_and_path(
    kind: _NonSanitizerKind,
) -> None:
    # Given
    content = f"{kind}-artifact".encode()
    digest = hashlib.sha256(content).hexdigest()
    failed = SearchAttemptResult(
        outcome="failed",
        return_code=2,
        log=b"failed\n",
        cpu_seconds=1.0,
        artifacts=(SearchArtifact(name=f"{kind}-artifact", kind=kind, content=content),),
    )
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        _ = run_openexr_cell(
            request,
            ScriptedSearchLauncher(clock, (ProcessScript(START + 1.0, failed, failed),)),
            clock,
        )

        # When
        artifacts = (
            cell_runner.load_cell_events(request)[0].model_dump(mode="json").get("artifacts")
        )

        # Then
        assert artifacts == [
            {
                "schema_version": "dllm-openexr-cell-artifact/v1",
                "sha256": digest,
                "kind": kind,
                "category": kind,
                "stored_path": f"crashes/{kind}/{digest}",
                "size_bytes": len(content),
            }
        ]


def test_cell_evidence_rejects_occurrence_for_non_sanitizer_artifact() -> None:
    # Given
    with _data_root() as root:
        request, digest = _run_plain_artifact_cell(root)
        event = cell_runner.load_cell_events(request)[0]
        artifact = request.crash_root / "crash" / digest
        append_crash_occurrence(
            request.artifact_root,
            request.crash_root / "occurrences.jsonl",
            CrashOccurrenceRecord(
                sequence=0,
                cell_id=request.cell_id,
                attempt_index=0,
                artifact_name="plain-crash",
                input_path=artifact.relative_to(request.artifact_root).as_posix(),
                input_sha256=digest,
                size_bytes=artifact.stat().st_size,
                search_log_sha256=event.log_sha256,
                cell_manifest_sha256=event.manifest_sha256,
            ),
        )
        _rewrite_crash_result(request, 1)

        # When / Then
        with pytest.raises(CellEvidenceError, match="sanitizer occurrence evidence"):
            _ = cell_runner.validate_cell_evidence(request)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "symlink", "wrong_attempt", "byte_mismatch"]
)
def test_cell_evidence_rejects_invalid_attempt_log_backing(mutation: _LogMutation) -> None:
    # Given
    with _data_root() as root:
        request = _run_completed_docker_cell(root)
        _mutate_log_backing(request, mutation)

        # When / Then
        with pytest.raises(CellEvidenceError, match="attempt log evidence"):
            _ = cell_runner.validate_cell_evidence(request)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink", "byte_mismatch", "ambiguous"])
def test_cell_evidence_rejects_invalid_attempt_artifact_backing(
    mutation: _ArtifactMutation,
) -> None:
    # Given
    with _data_root() as root:
        request, digest = _run_plain_artifact_cell(root)
        _mutate_artifact_backing(request, digest, mutation)

        # When / Then
        with pytest.raises(CellEvidenceError, match="attempt artifact evidence"):
            _ = cell_runner.validate_cell_evidence(request)


def test_cell_evidence_rejects_duplicate_artifact_digest_in_canonical_event() -> None:
    # Given
    with _data_root() as root:
        request, digest = _run_plain_artifact_cell(root)
        events = cell_runner.load_cell_events(request)
        tampered = (events[0].model_copy(update={"artifact_sha256": (digest, digest)}),)
        _rewrite_cell_events(request, tampered)

        # When / Then
        with pytest.raises(CellEvidenceError):
            _ = cell_runner.validate_cell_evidence(request)


@pytest.mark.parametrize("mutation", ["wrong_attempt", "duplicate"])
def test_cell_evidence_rejects_noncanonical_sanitizer_occurrence_binding(
    mutation: _OccurrenceMutation,
) -> None:
    # Given
    with _data_root() as root:
        request, _digest = _run_sanitizer_cell(root)
        occurrences = load_crash_occurrences(request)
        queue = request.crash_root / "occurrences.jsonl"
        if mutation == "wrong_attempt":
            _ = queue.write_bytes(b"")
            append_crash_occurrence(
                request.artifact_root,
                queue,
                occurrences[0].model_copy(update={"attempt_index": 1}),
            )
            occurrence_count = 1
        else:
            append_crash_occurrence(
                request.artifact_root,
                queue,
                occurrences[0].model_copy(update={"sequence": 1}),
            )
            occurrence_count = 2
        _rewrite_crash_result(request, occurrence_count)

        # When / Then
        with pytest.raises(CellEvidenceError, match="sanitizer occurrence evidence"):
            _ = cell_runner.validate_cell_evidence(request)


def test_long_probe_argv_uses_cpuset_and_persistent_private_roots() -> None:
    # Given
    with _data_root() as root:
        request = _launch_request(root)

        # When
        argv = build_long_probe_argv(request)

        # Then
        assert argv[argv.index("--cpuset-cpus") + 1] == "7"
        assert "--cpus" not in argv
        assert f"{request.artifact_root}:/work:rw" in argv
        assert f"{request.dictionary_path.resolve()}:/work/openexr.dict:ro" in argv
        # Docker reads a relative -v source as a *named volume* and rejects it
        # (exit 125), so every bind-mount source must be an absolute path.
        mount_sources = [argv[i + 1].split(":")[0] for i, arg in enumerate(argv) if arg == "-v"]
        assert mount_sources
        assert all(source.startswith("/") for source in mount_sources)
        assert "-dict=/work/openexr.dict" in argv
        assert "-max_total_time=7200" in argv
        assert "-seed=71" in argv


def test_cell_request_rejects_dictionary_hash_mismatch() -> None:
    # Given
    with _data_root() as root:
        dictionary = root / "openexr.dict"
        _ = dictionary.write_bytes(b'"EXR"')

        # When / Then
        with pytest.raises(ValidationError, match="dictionary hash mismatch"):
            _ = _request(root, dictionary_path=dictionary, dictionary_sha256=SHA_F)


def test_cell_request_rejects_symlink_dictionary() -> None:
    # Given
    with _data_root() as root:
        target = root / "openexr.dict"
        link = root / "linked.dict"
        _ = target.write_bytes(b'"EXR"')
        link.symlink_to(target)

        # When / Then
        with pytest.raises(ValidationError, match="dictionary file is missing or unsafe"):
            _ = _request(root, dictionary_path=link, dictionary_sha256=sha256_bytes(b'"EXR"'))


def test_launch_request_rejects_dictionary_relative_parent_escape() -> None:
    # Given
    with _data_root() as root:
        request = _launch_request(root)
        escaped = request.model_copy(update={"dictionary_path": Path("../openexr.dict")})

        # When / Then
        with pytest.raises(ValidationError, match="dictionary path must not escape"):
            _ = SearchLaunchRequest.model_validate(escaped.model_dump())


def test_production_launcher_harvests_sanitizer_artifact_and_persists_log() -> None:
    # Given
    with _data_root() as root:
        request = _launch_request(root)
        seen: list[tuple[str, ...]] = []

        def popen(argv: tuple[str, ...]) -> FakeSearchPopen:
            seen.append(argv)
            _ = (request.attempt_root / "crash-deadbeef").write_bytes(b"exact-input")
            return FakeSearchPopen(b"ERROR: AddressSanitizer: heap-buffer-overflow\n", 1)

        cpu_ticks = iter((4.0, 6.5))
        launcher = DockerSearchLauncher(
            popen_factory=popen,
            monotonic=lambda: START,
            cpu_clock=lambda: next(cpu_ticks),
        )

        # When
        result = launcher.launch(request).wait_until(START + 5.0)

        # Then
        assert result is not None
        assert result.outcome == "target_crash"
        assert result.cpu_seconds == 2.5
        assert result.artifacts[0].content == b"exact-input"
        assert (request.log_root / "attempt-000002.log").read_bytes() == result.log
        assert seen == [build_long_probe_argv(request)]


def test_production_launcher_terminates_at_supervisor_deadline() -> None:
    # Given
    with _data_root() as root:
        request = _launch_request(root)
        child = FakeSearchPopen(b"terminated\n", 143, timeouts=1)
        launcher = DockerSearchLauncher(
            popen_factory=lambda _argv: child,
            monotonic=lambda: START,
            cpu_clock=lambda: 0.0,
        )
        process = launcher.launch(request)

        # When
        waiting = process.wait_until(START + 5.0)
        result = process.terminate()

        # Then
        assert waiting is None
        assert child.terminate_count == 1
        assert result.outcome == "deadline"


def test_truncated_crash_queue_is_rejected_before_resume() -> None:
    # Given
    crash = SearchAttemptResult(
        outcome="target_crash",
        return_code=1,
        log=b"ERROR: AddressSanitizer: x\n",
        cpu_seconds=1.0,
        artifacts=(SearchArtifact(name="crash-x", kind="sanitizer", content=b"x"),),
    )
    done = SearchAttemptResult(
        outcome="completed", return_code=0, log=b"done\n", cpu_seconds=1.0, artifacts=()
    )

    # When / Then
    with _data_root() as root:
        request = _request(root)
        _seed_corpus(request.corpus_root)
        clock = ManualClock(START)
        launcher = ScriptedSearchLauncher(
            clock,
            (
                ProcessScript(1_001.0, crash, done),
                ProcessScript(DEADLINE, done, done),
            ),
        )
        _ = run_openexr_cell(request, launcher, clock)
        queue = Path(request.crash_root) / "occurrences.jsonl"
        _ = queue.write_bytes(queue.read_bytes().removesuffix(b"\n"))
        with pytest.raises(CellEvidenceError, match="truncated"):
            _ = load_crash_occurrences(request)


def _request(  # noqa: PLR0913
    root: Path,
    *,
    execution_profile: ExecutionProfile = "main",
    deadline_monotonic_seconds: float = DEADLINE,
    schedule_seconds: tuple[int, ...] = SCHEDULE_SECONDS,
    sanitizers: tuple[Literal["address", "undefined"], ...] = ("address", "undefined"),
    dictionary_path: Path | None = None,
    dictionary_sha256: str = SHA_B,
) -> CellRunRequest:
    artifact_root = root / "cell"
    dictionary = artifact_root / "openexr.dict" if dictionary_path is None else dictionary_path
    if dictionary_path is None:
        dictionary.parent.mkdir(parents=True, exist_ok=True)
        _ = dictionary.write_bytes(b'"EXR"')
        dictionary_sha256 = sha256_bytes(b'"EXR"')
    return CellRunRequest(
        execution_profile=execution_profile,
        schedule_seconds=schedule_seconds,
        cell_id="rep-0/selective-revision",
        artifact_root=artifact_root,
        corpus_root=artifact_root / "corpus",
        crash_root=artifact_root / "crashes",
        log_root=artifact_root / "logs",
        target_commit="777d231a179de1711d2a942810d9559216ab3f4c",
        harness_sha256=SHA_A,
        config_sha256=SHA_B,
        bank_sha256=SHA_C,
        start_monotonic_seconds=START,
        deadline_monotonic_seconds=deadline_monotonic_seconds,
        fuzzer_seed=71,
        max_input_length=1_024,
        target_timeout_seconds=5,
        rss_limit_mb=2_048,
        cpu_index=7,
        dictionary_path=dictionary,
        dictionary_sha256=dictionary_sha256,
        build=SearchBuildProvenance(
            image_digest=IMAGE,
            binary_sha256=SHA_D,
            sanitizers=sanitizers,
        ),
    )


def _launch_request(root: Path) -> SearchLaunchRequest:
    artifact_root = root / "cell"
    dictionary = artifact_root / "openexr.dict"
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    _ = dictionary.write_bytes(b'"EXR"')
    return SearchLaunchRequest(
        cell_id="cell-a",
        attempt_index=2,
        attempt_root=artifact_root / "attempts/000002",
        artifact_root=artifact_root,
        corpus_root=artifact_root / "corpus",
        crash_root=artifact_root / "crashes",
        log_root=artifact_root / "logs",
        absolute_deadline_monotonic_seconds=DEADLINE,
        remaining_seconds=7_200.0,
        fuzzer_seed=71,
        max_input_length=1_024,
        target_timeout_seconds=5,
        rss_limit_mb=2_048,
        cpu_index=7,
        dictionary_path=dictionary,
        dictionary_sha256=sha256_bytes(b'"EXR"'),
        build=SearchBuildProvenance(
            image_digest=IMAGE,
            binary_sha256=SHA_D,
            sanitizers=("address", "undefined"),
        ),
    )


def _run_completed_docker_cell(root: Path) -> CellRunRequest:
    request = _request(root)
    _seed_corpus(request.corpus_root)
    clock = ManualClock(START)
    cpu_ticks = iter((1.0, 3.0))
    _ = run_openexr_cell(
        request,
        DockerSearchLauncher(
            popen_factory=lambda _argv: FakeSearchPopen(b"done\n", 0),
            monotonic=clock.monotonic,
            cpu_clock=lambda: next(cpu_ticks),
        ),
        clock,
    )
    return request


def _run_plain_artifact_cell(root: Path) -> tuple[CellRunRequest, str]:
    request = _request(root)
    _seed_corpus(request.corpus_root)
    clock = ManualClock(START)
    content = b"plain non-sanitizer artifact"

    def popen(_argv: tuple[str, ...]) -> FakeSearchPopen:
        _ = (request.artifact_root / "attempts/000000/plain-crash").write_bytes(content)
        return FakeSearchPopen(b"failed\n", 2)

    cpu_ticks = iter((1.0, 3.0))
    _ = run_openexr_cell(
        request,
        DockerSearchLauncher(
            popen_factory=popen,
            monotonic=clock.monotonic,
            cpu_clock=lambda: next(cpu_ticks),
        ),
        clock,
    )
    return request, hashlib.sha256(content).hexdigest()


def _run_sanitizer_cell(root: Path) -> tuple[CellRunRequest, str]:
    request = _request(root)
    _seed_corpus(request.corpus_root)
    clock = ManualClock(START)
    content = b"sanitizer artifact"
    launches = 0

    def popen(_argv: tuple[str, ...]) -> FakeSearchPopen:
        nonlocal launches
        attempt = launches
        launches += 1
        if attempt == 0:
            _ = (request.artifact_root / "attempts/000000/crash-a").write_bytes(content)
            return FakeSearchPopen(b"ERROR: AddressSanitizer: heap-buffer-overflow\n", 1)
        return FakeSearchPopen(b"done\n", 0)

    cpu_ticks = iter((1.0, 2.0, 2.0, 3.0))
    _ = run_openexr_cell(
        request,
        DockerSearchLauncher(
            popen_factory=popen,
            monotonic=clock.monotonic,
            cpu_clock=lambda: next(cpu_ticks),
        ),
        clock,
    )
    return request, hashlib.sha256(content).hexdigest()


def _rewrite_cell_events(request: CellRunRequest, events: tuple[CellEvent, ...]) -> None:
    result = cell_runner.load_cell_result(request)
    path = request.artifact_root / "events.jsonl"
    _ = path.write_bytes(b"")
    for event in events:
        _ = append_event(
            path,
            timestamp_utc=f"monotonic+{event.observed_monotonic_seconds:016.6f}",
            event_type="search_attempt",
            payload=_JSON_OBJECT.validate_python(event.model_dump(mode="json")),
        )
    tampered = result.model_copy(
        update={"event_stream_sha256": validate_ledger(path).final_stream_sha256}
    )
    _ = (request.artifact_root / "final-result.json").write_bytes(
        canonical_json_bytes(tampered.model_dump(mode="json"))
    )


def _rewrite_crash_result(request: CellRunRequest, occurrence_count: int) -> None:
    result = cell_runner.load_cell_result(request)
    stream = validate_ledger(request.crash_root / "occurrences.jsonl")
    tampered = result.model_copy(
        update={
            "crash_stream_sha256": stream.final_stream_sha256,
            "crash_occurrence_count": occurrence_count,
        }
    )
    _ = (request.artifact_root / "final-result.json").write_bytes(
        canonical_json_bytes(tampered.model_dump(mode="json"))
    )


def _mutate_log_backing(request: CellRunRequest, mutation: _LogMutation) -> None:
    path = request.log_root / "attempt-000000.log"
    if mutation == "missing":
        path.unlink()
        return
    if mutation == "extra":
        _ = (request.log_root / "attempt-999999.log").write_bytes(b"extra\n")
        return
    if mutation == "symlink":
        target = request.artifact_root / "outside-log"
        _ = target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        return
    if mutation == "wrong_attempt":
        _ = path.rename(request.log_root / "attempt-000001.log")
        return
    _ = path.write_bytes(b"tampered\n")


def _mutate_artifact_backing(
    request: CellRunRequest, digest: str, mutation: _ArtifactMutation
) -> None:
    path = request.crash_root / "crash" / digest
    if mutation == "missing":
        path.unlink()
        return
    if mutation == "extra":
        content = b"extra artifact"
        extra = hashlib.sha256(content).hexdigest()
        destination = request.crash_root / "timeout" / extra
        destination.parent.mkdir()
        _ = destination.write_bytes(content)
        return
    if mutation == "symlink":
        target = request.artifact_root / "outside-artifact"
        _ = target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
        return
    if mutation == "byte_mismatch":
        _ = path.write_bytes(b"tampered artifact")
        return
    duplicate = request.crash_root / "timeout" / digest
    duplicate.parent.mkdir()
    _ = duplicate.write_bytes(path.read_bytes())


@contextmanager
def _data_root() -> Generator[Path]:
    with tempfile.TemporaryDirectory(prefix="dllm-task4-cell-", dir="/data") as directory:
        yield Path(directory)


def _seed_corpus(path: Path) -> None:
    path.mkdir(parents=True)
    _ = (path / "seed").write_bytes(b"seed")
