"""Safe Docker LLVM source coverage runner contract tests."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dllm_agent.fuzzing.coverage.execution import coverage_if_reported
from dllm_agent.fuzzing.coverage.runner import CommandRunner, CoverageRequest, run_coverage
from dllm_agent.targets import profiles as target_profiles

VALID_IMAGE = "sha256:" + ("a" * 64)
HARNESS_BYTES = (
    b'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}'
)
LLVM_COV_REPORT = b"""
Filename     Regions    Missed Regions     Cover   Functions  Missed Functions  Executed
TOTAL             10                 2    80.00%           4                 1    75.00%
Lines       Missed Lines     Cover    Branches   Missed Branches     Cover
TOTAL          20            5    75.00%          8                 6    25.00%
"""
LLVM_COV_HEADER = b" ".join(
    (
        b"Filename Regions Missed Regions Cover Functions Missed Functions Executed",
        b"Lines Missed Lines Cover Branches Missed Branches Cover",
    )
)
LLVM_COV_NO_BRANCHES = b"\n".join(
    (LLVM_COV_HEADER, b"TOTAL 1 0 100.00% 1 0 100.00% 1 0 100.00% 0 0 -", b"")
)


def test_run_coverage_rejects_unpinned_image_before_docker_invocation(tmp_path: Path) -> None:
    # Given
    calls: list[tuple[str, ...]] = []
    request = CoverageRequest(
        image_id="llvm:latest",
        source_root=tmp_path,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        timeout_seconds=5,
    )

    # When / Then
    with pytest.raises(ValueError, match="image_id"):
        _ = run_coverage(request, runner=_recording_runner(calls))
    assert calls == []


def test_request_is_immutable(tmp_path: Path) -> None:
    # Given
    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=tmp_path,
        harness_bytes=HARNESS_BYTES,
        corpus=(b"seed",),
        timeout_seconds=5,
    )

    # When / Then
    field_name = "timeout_seconds"
    with pytest.raises(FrozenInstanceError):
        setattr(request, field_name, 10)


def test_coverage_request_rejects_unknown_runtime_profile_before_runner(tmp_path: Path) -> None:
    # Given
    calls: list[tuple[str, ...]] = []

    def run_with_runtime_profile(value: str) -> None:
        request = CoverageRequest(
            image_id=VALID_IMAGE,
            source_root=tmp_path,
            harness_bytes=HARNESS_BYTES,
            corpus=(),
            timeout_seconds=5,
            target_profile=target_profiles.require_target_profile(value),
        )
        _ = run_coverage(request, runner=_recording_runner(calls))

    # When / Then
    with pytest.raises(ValueError, match="target profile"):
        run_with_runtime_profile("manifest-shell-command")
    assert calls == []


def test_coverage_total_treats_zero_branch_dash_as_full_coverage() -> None:
    # Given
    process = CompletedProcess(("llvm-cov",), 0, LLVM_COV_NO_BRANCHES, b"")

    # When
    totals = coverage_if_reported(process)

    # Then
    assert totals is not None
    assert totals.branches == 100.0


def test_openexr_coverage_helper_leaves_only_build_directory_host_cleanable() -> None:
    # Given
    helper = (Path(__file__).parents[1] / "tools/openexr_runtime/build-coverage").read_text()

    # When / Then
    assert "umask 000" in helper
    assert "trap 'chmod -R a+rwX /work/build-coverage 2>/dev/null || true' EXIT" in helper


def test_run_coverage_uses_hardened_docker_argv_and_parses_total_coverage(
    tmp_path: Path,
) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "include").mkdir()
    seen_environments: list[Mapping[str, str]] = []

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        seen_environments.append(environment)
        assert timeout_seconds >= 60
        command_text = " ".join(command)
        if "clang++" in command:
            work_dir = _host_path_after(command, "/work")
            assert work_dir.stat().st_mode & 0o007 == 0o007
            assert (work_dir / "harness.cc").stat().st_mode & 0o004 == 0o004
            assert "-std=c++17" in command
            assert "-O1" in command
            assert "-g" in command
            assert "-fprofile-instr-generate" in command
            assert "-fcoverage-mapping" in command
            assert "-fsanitize=fuzzer" in command
            assert "-I/target/include" in command
            return CompletedProcess(command, 0, b"", b"compiled\n")
        if "llvm-profdata" in command:
            work_dir = _host_path_after(command, "/work")
            _ = (work_dir / "coverage.profdata").write_bytes(b"indexed-profile")
            return CompletedProcess(command, 0, b"", b"merged\n")
        if "llvm-cov" in command:
            assert "report" in command
            assert "/work/coverage_target" in command
            assert "-instr-profile=/work/coverage.profdata" in command
            return CompletedProcess(command, 0, LLVM_COV_REPORT, b"")
        if "/work/coverage_target" in command:
            work_dir = _host_path_after(command, "/work")
            _ = (work_dir / "coverage.profraw").write_bytes(b"raw-profile")
            assert "-runs=0" in command
            return CompletedProcess(command, 0, b"replayed\n", b"")
        raise AssertionError(command_text)

    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(b"seed-a", b"seed-b"),
        timeout_seconds=10,
    )

    # When
    result = run_coverage(request, runner=runner)

    # Then
    for argv in (result.build_argv, result.run_argv, result.merge_argv, result.report_argv):
        _assert_hardened_docker_argv(argv)

    assert result.build_status == "succeeded"
    assert result.run_status == "completed"
    assert result.merge_status == "succeeded"
    assert result.report_status == "succeeded"
    assert result.build_log == b"compiled\n"
    assert result.run_log == b"replayed\n"
    assert result.merge_log == b"merged\n"
    assert result.report_log == LLVM_COV_REPORT
    assert result.coverage is not None
    assert result.coverage.regions == 80.0
    assert result.coverage.functions == 75.0
    assert result.coverage.lines == 75.0
    assert result.coverage.branches == 25.0
    assert seen_environments[-1]["LLVM_PROFILE_FILE"] == "/work/coverage.profraw"


def test_run_coverage_openexr_profile_uses_fixed_helper_and_owned_report_scope(
    tmp_path: Path,
) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        work_dir = _host_path_after(command, "/work")
        if "/opt/dllm/bin/build-coverage" in command:
            return CompletedProcess(command, 0, b"", b"")
        if "/work/coverage_target" in command and "llvm-cov" not in command:
            _ = (work_dir / "coverage.profraw").write_bytes(b"raw-profile")
            return CompletedProcess(command, 0, b"", b"")
        if "llvm-profdata" in command:
            _ = (work_dir / "coverage.profdata").write_bytes(b"indexed-profile")
            return CompletedProcess(command, 0, b"", b"")
        if "llvm-cov" in command:
            return CompletedProcess(command, 0, LLVM_COV_REPORT, b"")
        raise AssertionError(command)

    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(b"seed",),
        timeout_seconds=5,
        target_profile="openexr-v3.4",
    )

    # When
    result = run_coverage(request, runner=runner)

    # Then
    assert result.build_argv[-1:] == ("/opt/dllm/bin/build-coverage",)
    assert result.report_argv[-1:] == ("/opt/openexr-src/src/lib",)
    assert "/target" not in result.report_argv
    assert not any("_deps" in item for item in result.report_argv)
    _assert_hardened_docker_argv(result.build_argv)
    _assert_hardened_docker_argv(result.report_argv)


def test_run_coverage_header_only_profile_retains_exact_build_and_report_argv(
    tmp_path: Path,
) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        timeout_seconds=5,
    )

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        return CompletedProcess(command, 2 if "clang++" in command else 0, b"", b"")

    # When
    result = run_coverage(request, runner=runner)

    # Then
    assert result.build_argv[-11:] == (
        "clang++",
        "-std=c++17",
        "-O1",
        "-g",
        "-fprofile-instr-generate",
        "-fcoverage-mapping",
        "-fsanitize=fuzzer",
        "-I/target/include",
        "/work/harness.cc",
        "-o",
        "/work/coverage_target",
    )
    assert result.report_argv[-5:] == (
        "llvm-cov",
        "report",
        "/work/coverage_target",
        "-instr-profile=/work/coverage.profdata",
        "/target",
    )


def test_run_coverage_skips_replay_merge_and_report_when_build_fails(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[tuple[str, ...]] = []

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        calls.append(command)
        return CompletedProcess(command, 2, b"", b"compile failed\n")

    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        timeout_seconds=5,
    )

    # When
    result = run_coverage(request, runner=runner)

    # Then
    assert len(calls) == 1
    assert result.build_status == "failed"
    assert result.run_status == "skipped"
    assert result.merge_status == "skipped"
    assert result.report_status == "skipped"
    assert result.coverage is None


def test_run_coverage_rejects_symlink_profile_output(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        if "clang++" in command:
            return CompletedProcess(command, 0, b"", b"")
        if "/work/coverage_target" not in command:
            return CompletedProcess(command, 0, b"", b"")
        work_dir = _host_path_after(command, "/work")
        profile = work_dir / "coverage.profraw"
        target = work_dir / "real.profraw"
        _ = target.write_bytes(b"raw-profile")
        profile.symlink_to(target)
        return CompletedProcess(command, 0, b"", b"")

    request = CoverageRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        timeout_seconds=5,
    )

    # When / Then
    with pytest.raises(ValueError, match="unsafe output"):
        _ = run_coverage(request, runner=runner)


def _recording_runner(calls: list[tuple[str, ...]]) -> CommandRunner:
    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        calls.append(command)
        return CompletedProcess(command, 0, b"", b"")

    return runner


def _assert_hardened_docker_argv(argv: tuple[str, ...]) -> None:
    assert argv[:2] == ("docker", "run")
    assert "--rm" in argv
    assert "--network" in argv
    assert "none" in argv
    assert "--read-only" in argv
    assert "--cap-drop" in argv
    assert "ALL" in argv
    assert "--security-opt" in argv
    assert "no-new-privileges=true" in argv
    assert "--pids-limit" in argv
    assert "--memory" in argv
    assert "--cpus" in argv
    assert "--user" in argv
    assert "65534:65534" in argv
    assert "--tmpfs" in argv
    assert "/tmp:rw,noexec,nosuid,nodev,size=64m" in argv  # noqa: S108 - Docker tmpfs.
    assert "--pull" in argv
    assert "never" in argv
    assert not any(";" in item for item in argv)


def _host_path_after(command: tuple[str, ...], container_path: str) -> Path:
    for index, item in enumerate(command):
        if item == "-v":
            host, mounted, mode = command[index + 1].split(":")
            if mounted == container_path and mode == "rw":
                return Path(host)
    raise AssertionError(container_path)
