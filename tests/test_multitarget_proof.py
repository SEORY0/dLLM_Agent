"""Target-generic Magma ASan proof replay tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from subprocess import CompletedProcess, TimeoutExpired
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile
from dllm_agent.experiments.multitarget.proof import (
    ProofReplayError,
    ProofReplayRequest,
    prove_input,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_IMAGE = "sha256:" + ("1" * 64)
_ASAN_LOG = (
    b"==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7f\n"
    b"    #0 0x111 in png_read_info /src/libpng/pngread.c:101\n"
    b"    #1 0x222 in LLVMFuzzerTestOneInput /src/libpng/contrib/oss-fuzz/libpng_read_fuzzer.cc:9\n"
)
_SAME_ASAN_LOG = _ASAN_LOG.replace(b"0x111", b"0x999")


class ScriptedDockerRunner:
    """subprocess.run double that records shell-free Docker proof argv."""

    def __init__(
        self,
        results: tuple[CompletedProcess[bytes] | TimeoutExpired | OSError, ...],
    ) -> None:
        self._results: list[CompletedProcess[bytes] | TimeoutExpired | OSError] = list(results)
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        capture_output: bool,
        check: bool,
        timeout: float,
    ) -> CompletedProcess[bytes]:
        # Given/Then: production must use subprocess directly without a shell.
        assert capture_output is True
        assert check is False
        assert timeout > 0
        self.calls.append(tuple(argv))
        result = self._results.pop(0)
        if isinstance(result, TimeoutExpired | OSError):
            raise result
        return result


def test_proved_requires_two_fresh_asan_replays_with_same_target_signature(
    tmp_path: Path,
) -> None:
    # Given
    input_path = _input_file(tmp_path)
    runner = ScriptedDockerRunner((_process(1, _ASAN_LOG), _process(1, _SAME_ASAN_LOG)))

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "proved"
    assert result.input_sha256 == _sha256(input_path)
    assert result.signature is not None
    assert result.signature.sanitizer_class == "heap-buffer-overflow"
    assert result.signature.first_target_frame == "png_read_info /src/libpng/pngread.c:101"
    assert tuple(run.return_code for run in result.runs) == (1, 1)
    assert tuple(run.signature for run in result.runs) == (result.signature, result.signature)
    assert len(runner.calls) == 2
    assert runner.calls[0] != runner.calls[1]


def test_mismatched_asan_signature_is_not_a_proof(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    different = _ASAN_LOG.replace(b"heap-buffer-overflow", b"stack-buffer-overflow")
    runner = ScriptedDockerRunner((_process(1, _ASAN_LOG), _process(1, different)))

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "not_reproduced"
    assert result.signature is None
    assert len(result.runs) == 2


def test_fatal_signal_without_asan_is_not_a_proof(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    runner = ScriptedDockerRunner(
        (_process(139, b"Segmentation fault\n"), _process(139, b"Segmentation fault\n"))
    )

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "crash_without_sanitizer"
    assert result.signature is None


def test_timeout_fails_closed(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    timeout = TimeoutExpired(("docker", "run"), timeout=5, output=b"partial", stderr=b"late")
    runner = ScriptedDockerRunner((timeout, _process(0, b"clean second replay")))

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "failed"
    assert result.error == "docker run timed out after 5s"
    assert len(runner.calls) == 2
    assert len(result.runs) == 2
    assert result.runs[0].return_code == 124
    assert result.runs[0].log == b"partiallate"
    assert result.runs[1].return_code == 0


def test_os_error_still_records_both_replay_attempts(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    runner = ScriptedDockerRunner((OSError("docker unavailable"), _process(1, _ASAN_LOG)))

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "failed"
    assert result.error == "docker unavailable"
    assert len(runner.calls) == 2
    assert tuple(run.container_name for run in result.runs) == (
        "dllm-proof-libpng-0",
        "dllm-proof-libpng-1",
    )
    assert result.runs[0].return_code == 125
    assert result.runs[1].signature is not None


def test_docker_argv_uses_read_only_exact_input_and_pinned_runtime(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    runner = ScriptedDockerRunner((_process(0, b"clean"), _process(0, b"clean")))

    # When
    result = prove_input(_request(input_path), runner=runner)

    # Then
    assert result.status == "not_reproduced"
    for index, call in enumerate(runner.calls):
        assert call[:2] == ("docker", "run")
        assert "--rm" in call
        assert "--network" in call
        assert "none" in call
        assert "--cap-drop" in call
        assert "ALL" in call
        assert "--security-opt" in call
        assert "no-new-privileges" in call
        assert "--pids-limit" in call
        assert "--memory" in call
        assert f"{input_path}:/proof/input:ro" in call
        assert "--entrypoint" in call
        assert "/magma_out/libpng_read_fuzzer" in call
        assert _IMAGE in call
        assert call[-2:] == ("-runs=1", "/proof/input")
        assert f"dllm-proof-libpng-{index}" in call
        assert f"-seed={100 + index}" in call
    assert tuple(run.argv_sha256 for run in result.runs) == tuple(
        hashlib.sha256(b"\0".join(item.encode() for item in call)).hexdigest()
        for call in runner.calls
    )


def test_input_hash_mismatch_fails_before_docker(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    runner = ScriptedDockerRunner((_process(0, b"unused"),))
    request = _request(input_path, input_sha256="0" * 64)

    # When / Then
    with pytest.raises(ProofReplayError, match="input sha256 mismatch"):
        _ = prove_input(request, runner=runner)
    assert runner.calls == []


def test_distinct_seed_name_and_cpu_are_required(tmp_path: Path) -> None:
    # Given
    input_path = _input_file(tmp_path)
    base = _request(input_path)

    # When / Then
    with pytest.raises(ProofReplayError, match="proof seeds must be distinct"):
        _ = prove_input(
            replace(base, first_seed=7, second_seed=7),
            runner=ScriptedDockerRunner(()),
        )
    with pytest.raises(ProofReplayError, match="container names must be distinct"):
        _ = prove_input(
            replace(base, first_container_name="same", second_container_name="same"),
            runner=ScriptedDockerRunner(()),
        )
    with pytest.raises(ProofReplayError, match="proof CPUs must be distinct"):
        _ = prove_input(
            replace(base, first_cpu=2, second_cpu=2),
            runner=ScriptedDockerRunner(()),
        )


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "poc.png"
    _ = path.write_bytes(b"png poc")
    return path


def _request(input_path: Path, *, input_sha256: str | None = None) -> ProofReplayRequest:
    return ProofReplayRequest(
        profile=runtime_profile("libpng"),
        image_id=_IMAGE,
        input_path=input_path,
        input_sha256=input_sha256 or _sha256(input_path),
        first_seed=100,
        second_seed=101,
        first_container_name="dllm-proof-libpng-0",
        second_container_name="dllm-proof-libpng-1",
        first_cpu=0,
        second_cpu=1,
        timeout_seconds=5,
    )


def _process(return_code: int, log: bytes) -> CompletedProcess[bytes]:
    return CompletedProcess(("docker", "run"), return_code, b"", log)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
