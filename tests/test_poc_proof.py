"""Exact-input OpenEXR ASan proof tests."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.targets.openexr.particle import decode_particle, materialize_particle
from dllm_agent.targets.openexr.proof import (
    OPENEXR_PROOF_IMAGE_DIGEST,
    OPENEXR_PROOF_SEED,
    OPENEXR_TARGET_COMMIT,
    OpenExrProofRunner,
    ProofBoundaryError,
    ProofCommand,
    prove_snapshot,
)

_INPUT = b"exact-exr-snapshot"
_ASAN_LOG = (
    b"==7==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7f\n"
    b"    #0 0x111 in Imf_3_4::InputFile::readPixels src/lib/OpenEXR/ImfInputFile.cpp:17\n"
    b"    #1 0x222 in LLVMFuzzerTestOneInput src/test/oss-fuzz/openexr_exrcheck_fuzzer.cc:8\n"
)
_MIN_ASAN_LOG = (
    b"==9==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x8f\n"
    b"    #0 0x333 in Imf_3_4::InputFile::readPixels src/lib/OpenEXR/ImfInputFile.cpp:17\n"
    b"    #1 0x444 in LLVMFuzzerTestOneInput src/test/oss-fuzz/openexr_exrcheck_fuzzer.cc:8\n"
)


@dataclass(frozen=True, slots=True)
class SeededProofRequest:
    """Target-side exact proof request fixture with explicit seed provenance."""

    snapshot: bytes
    snapshot_sha256: str
    duration_seconds: int
    proof_seed: int
    search_image_digest: str


class ScriptedProofRunner:
    """Proof command double that records exact replay inputs."""

    def __init__(self, results: tuple[CompletedProcess[bytes], ...]) -> None:
        self._results: list[CompletedProcess[bytes]] = list(results)
        self.calls: list[tuple[str, ...]] = []
        self.inputs: list[bytes] = []

    def __call__(
        self, command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        """Record proof execution and return the next scripted process."""
        assert environment["ASAN_OPTIONS"] == "detect_leaks=0:abort_on_error=1:symbolize=1"
        assert timeout_seconds >= 30.0
        self.calls.append(command)
        root = _host_work_root(command)
        self.inputs.append((root / "input.exr").read_bytes())
        result = self._results.pop(0)
        if "-minimize_crash=1" in command:
            _ = (root / "artifacts/minimized").write_bytes(b"minimized-exr")
        return result


def test_probe_sanitizer_flag_cannot_prove_clean_fresh_replay() -> None:
    # Given
    runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))

    # When
    result = prove_snapshot(_request(probe_sanitizer_observed=True), runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.sanitizer is None
    assert result.input_sha256 == sha256_bytes(_INPUT)
    assert result.crash_artifact_sha256 == ()
    assert runner.inputs == [_INPUT, _INPUT]


def test_proved_requires_two_consistent_fresh_process_asan_reproductions() -> None:
    # Given
    different_log = _ASAN_LOG.replace(b"heap-buffer-overflow", b"stack-buffer-overflow")
    runner = ScriptedProofRunner((_process(1, _ASAN_LOG), _process(1, different_log)))

    # When
    result = prove_snapshot(_request(), runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.sanitizer == "AddressSanitizer"
    assert result.root_signature is None
    assert runner.inputs == [_INPUT, _INPUT]


def test_request_seed_is_used_and_clean_result_waits_for_two_replays_before_classification() -> (
    None
):
    # Given
    runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))

    # When
    result = prove_snapshot(_request(seed=314159), runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.proof_seed == 314159
    assert len(runner.calls) == 2
    assert _seed_args(runner.calls) == ("-seed=314159", "-seed=314160")
    assert result.run_argv_sha256 == tuple(sha256_bytes(_argv_bytes(call)) for call in runner.calls)


def test_different_request_seeds_produce_different_argv_hash_evidence() -> None:
    # Given
    first_runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))
    second_runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))

    # When
    first = prove_snapshot(_request(seed=7), first_runner)
    second = prove_snapshot(_request(seed=11), second_runner)

    # Then
    assert first.proof_seed == 7
    assert second.proof_seed == 11
    assert _seed_args(first_runner.calls) == ("-seed=7", "-seed=8")
    assert _seed_args(second_runner.calls) == ("-seed=11", "-seed=12")
    assert first.run_argv_sha256 != second.run_argv_sha256


def test_negative_request_seed_fails_closed_before_execution() -> None:
    # Given
    runner = ScriptedProofRunner((_process(0, b"unused\n"),))

    # When / Then
    with pytest.raises(ProofBoundaryError, match="proof seed must be nonnegative"):
        _ = prove_snapshot(_request(seed=-1), runner)
    assert runner.calls == []


def test_timeout_and_docker_setup_fail_closed_without_oracle_status() -> None:
    # Given / When / Then
    for code in (124, 125):
        runner = ScriptedProofRunner((_process(code, b"boundary failure\n"),))
        with pytest.raises(ProofBoundaryError, match="proof boundary process failed"):
            _ = prove_snapshot(_request(), runner)


def test_ordinary_nonzero_without_crash_signal_is_not_reproduced_after_two_replays() -> None:
    # Given
    runner = ScriptedProofRunner((_process(2, b"usage error\n"), _process(2, b"usage error\n")))

    # When
    result = prove_snapshot(_request(), runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.sanitizer is None
    assert runner.inputs == [_INPUT, _INPUT]


def test_segfault_without_asan_is_crash_without_sanitizer_after_two_replays() -> None:
    # Given
    runner = ScriptedProofRunner(
        (_process(139, b"Segmentation fault\n"), _process(139, b"Segmentation fault\n"))
    )

    # When
    result = prove_snapshot(_request(), runner)

    # Then
    assert result.oracle_status == "crash_without_sanitizer"
    assert result.sanitizer is None
    assert result.proof_log == b"Segmentation fault\nSegmentation fault\n"
    assert runner.inputs == [_INPUT, _INPUT]


def test_consistent_asan_replay_is_proved_and_hash_binds_minimization_evidence() -> None:
    # Given
    original = materialize_particle(
        decode_particle((1, 1, 1, 1, 0, 0, 0, 0, 1, 1), "validity-preserving")
    )
    structured = materialize_particle(
        decode_particle((0, 1, 1, 1, 0, 0, 0, 0, 1, 1), "validity-preserving")
    )
    runner = ScriptedProofRunner(
        (
            _process(1, _ASAN_LOG),
            _process(1, _ASAN_LOG),
            _process(1, _ASAN_LOG),
            _process(1, _ASAN_LOG),
            _process(1, b"minimize\n"),
            _process(1, _MIN_ASAN_LOG),
        )
    )

    # When
    result = prove_snapshot(_request(original.content, seed=4242), runner)

    # Then
    assert result.oracle_status == "proved"
    assert result.proof_seed == 4242
    assert result.input_bytes == original.content
    assert result.input_sha256 == original.sha256
    assert result.structured_minimization_status == "reduced"
    assert result.structured_minimized_input_sha256 == structured.sha256
    assert result.structured_minimization_log_sha256 == sha256_bytes(
        f"reduced:{original.sha256}->{structured.sha256}:origin_x\n".encode()
    )
    assert result.minimized_input_bytes == b"minimized-exr"
    assert result.minimized_input_sha256 == sha256_bytes(b"minimized-exr")
    assert result.sanitizer == "AddressSanitizer"
    assert result.root_signature is not None
    assert result.proof_log_sha256 == sha256_bytes(result.proof_log)
    assert result.minimization_log_sha256 == sha256_bytes(b"minimize\n")
    assert result.image_digest == OPENEXR_PROOF_IMAGE_DIGEST
    assert result.target_commit == OPENEXR_TARGET_COMMIT
    assert _seed_args(runner.calls) == (
        "-seed=4242",
        "-seed=4243",
        "-seed=4242",
        "-seed=4242",
        "-seed=4242",
        "-seed=4242",
    )
    assert result.run_argv_sha256 == tuple(sha256_bytes(_argv_bytes(call)) for call in runner.calls)
    assert runner.inputs == [
        original.content,
        original.content,
        structured.content,
        structured.content,
        structured.content,
        b"minimized-exr",
    ]


def test_libfuzzer_minimization_timeout_or_setup_failure_fails_closed() -> None:
    # Given / When / Then
    for code in (124, 125):
        runner = ScriptedProofRunner(
            (_process(1, _ASAN_LOG), _process(1, _ASAN_LOG), _process(code, b"minimize\n"))
        )
        with pytest.raises(ProofBoundaryError, match=f"proof boundary process failed: exit {code}"):
            _ = prove_snapshot(_request(), runner)


def test_minimized_rerun_timeout_or_setup_failure_fails_closed() -> None:
    # Given / When / Then
    for code in (124, 125):
        runner = ScriptedProofRunner(
            (
                _process(1, _ASAN_LOG),
                _process(1, _ASAN_LOG),
                _process(1, b"minimize\n"),
                _process(code, b"rerun\n"),
            )
        )
        with pytest.raises(ProofBoundaryError, match=f"proof boundary process failed: exit {code}"):
            _ = prove_snapshot(_request(), runner)


def test_nonzero_crash_without_asan_is_distinct_from_proof() -> None:
    # Given
    runner = ScriptedProofRunner(
        (_process(139, b"Segmentation fault\n"), _process(139, b"Segmentation fault\n"))
    )

    # When
    result = prove_snapshot(_request(), runner)

    # Then
    assert result.oracle_status == "crash_without_sanitizer"
    assert result.sanitizer is None
    assert result.proof_log == b"Segmentation fault\nSegmentation fault\n"


def test_exact_input_sha_mismatch_fails_before_execution() -> None:
    # Given
    runner = ScriptedProofRunner((_process(0, b"unused"),))
    request = SeededProofRequest(
        snapshot=_INPUT,
        snapshot_sha256=sha256_bytes(b"different"),
        duration_seconds=1,
        proof_seed=OPENEXR_PROOF_SEED,
        search_image_digest=OPENEXR_PROOF_IMAGE_DIGEST,
    )

    # When / Then
    with pytest.raises(ProofBoundaryError, match="snapshot sha256 mismatch"):
        _ = prove_snapshot(request, runner)
    assert runner.calls == []


def test_proof_runner_implements_task4_summary_interface() -> None:
    # Given
    runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))
    proof_runner = OpenExrProofRunner(command=runner)

    # When
    result = proof_runner.run(_request())

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.input_sha256 == sha256_bytes(_INPUT)
    assert result.proof_log_sha256 == sha256_bytes(b"clean\nclean again\n")
    assert result.proof_log == b"clean\nclean again\n"
    assert result.summary().proof_log_sha256 == result.proof_log_sha256


def test_unsafe_symlink_and_oversized_minimization_artifacts_fail_closed(tmp_path: Path) -> None:
    # Given
    safe = tmp_path / "safe"
    _ = safe.write_bytes(b"x")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "link").symlink_to(safe)

    # When / Then
    with pytest.raises(ProofBoundaryError, match="unsafe artifact"):
        _ = prove_snapshot(_request(), _artifact_runner(artifact_root))

    huge_root = tmp_path / "huge"
    huge_root.mkdir()
    _ = (huge_root / "minimized").write_bytes(b"x" * 1_048_577)
    with pytest.raises(ProofBoundaryError, match="artifact size limit"):
        _ = prove_snapshot(_request(), _artifact_runner(huge_root))


def test_clean_result_does_not_fabricate_crash_or_minimized_artifact() -> None:
    # Given
    runner = ScriptedProofRunner((_process(0, b"clean\n"), _process(0, b"clean again\n")))

    # When
    result = prove_snapshot(_request(), runner)

    # Then
    assert result.oracle_status == "not_reproduced"
    assert result.minimized_input_sha256 is None
    assert result.minimization_log_sha256 is None
    assert result.structured_minimization_status == "unavailable"
    assert result.structured_minimization_log_sha256 == sha256_bytes(b"structured-unavailable\n")
    assert result.crash_artifact_sha256 == ()
    assert len(runner.calls) == 2


def _request(
    snapshot: bytes = _INPUT,
    *,
    probe_sanitizer_observed: bool = False,
    seed: int = OPENEXR_PROOF_SEED,
    image_digest: str = OPENEXR_PROOF_IMAGE_DIGEST,
) -> SeededProofRequest:
    return SeededProofRequest(
        snapshot=snapshot,
        snapshot_sha256=sha256_bytes(snapshot),
        duration_seconds=1 if not probe_sanitizer_observed else 2,
        proof_seed=seed,
        search_image_digest=image_digest,
    )


def _process(returncode: int, log: bytes) -> CompletedProcess[bytes]:
    return CompletedProcess(("proof",), returncode, b"", log)


def _host_work_root(command: tuple[str, ...]) -> Path:
    volume = command[command.index("-v") + 1]
    return Path(volume.split(":", maxsplit=1)[0])


def _argv_bytes(command: tuple[str, ...]) -> bytes:
    return ("[" + ",".join(f'"{item}"' for item in command) + "]").encode()


def _seed_args(calls: list[tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(item for call in calls for item in call if item.startswith("-seed="))


def _artifact_runner(artifact_root: Path) -> ProofCommand:
    def run(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment
        _ = timeout_seconds
        root = _host_work_root(command)
        if "-minimize_crash=1" in command:
            target = root / "artifacts"
            for child in artifact_root.iterdir():
                if child.is_symlink():
                    (target / child.name).symlink_to(child.readlink())
                else:
                    _ = (target / child.name).write_bytes(child.read_bytes())
            return _process(1, b"minimize\n")
        return _process(1, _ASAN_LOG)

    return run
