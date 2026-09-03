"""Compile-once OpenEXR snapshot probe tests."""

from collections.abc import Mapping
from pathlib import Path
from subprocess import CompletedProcess
from typing import override

from dllm_agent.targets.openexr.particle import decode_particle, materialize_particle
from dllm_agent.targets.openexr.probe import (
    ProbeConfig,
    ProbeRequest,
    prepare_openexr_probe,
    provisional_probe_family_key,
)

VALID_IMAGE = "sha256:" + ("a" * 64)
HARNESS_SHA = "b" * 64


class RecordingRunner:
    """Command runner that records probe invocations and writes fake artifacts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fuzz_inputs: list[bytes] = []
        self.coverage_calls: int = 0

    @property
    def fuzz_call_count(self) -> int:
        """Return how many hot probe runs executed."""
        return len(self.fuzz_inputs)

    def commands_named(self, name: str) -> int:
        """Count argv items matching one executable basename."""
        return sum(Path(item).name == name for command in self.calls for item in command)

    def __call__(
        self, command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        """Capture a command and return deterministic libFuzzer-like output."""
        self.calls.append(command)
        assert timeout_seconds >= 30
        if "/usr/local/bin/llvm-profdata" in command:
            work_root = _host_path_after(command, "/work")
            assert (work_root / "coverage.profraw").read_bytes() == b"raw-profile"
            _ = (work_root / "coverage.profdata").write_bytes(b"indexed-profile")
            return CompletedProcess(command, 0, b"merged\n", b"")
        if "/usr/local/bin/llvm-cov" in command:
            work_root = _host_path_after(command, "/work")
            assert (work_root / "coverage.profdata").read_bytes() == b"indexed-profile"
            return CompletedProcess(
                command, 0, b"TOTAL 1 0 100.00% 1 0 100.00% 1 0 100.00% 1 0 100.00%\n", b""
            )
        if "/opt/dllm/bin/poc_probe-coverage" in command:
            self.coverage_calls += 1
            assert _docker_flag(command, "--env") == "LLVM_PROFILE_FILE=/work/coverage.profraw"
            work_root = _host_path_after(command, "/work")
            _ = (work_root / "coverage.profraw").write_bytes(b"raw-profile")
            return CompletedProcess(command, 0, b"coverage-run\n", b"")
        assert environment["ASAN_OPTIONS"] == "detect_leaks=0:abort_on_error=1:symbolize=1"
        assert environment["UBSAN_OPTIONS"] == "halt_on_error=1:print_stacktrace=1"
        work_root = _host_path_after(command, "/work")
        self.fuzz_inputs.append((work_root / "corpus/seed-0000").read_bytes())
        _ = (work_root / "artifacts/unit-000001").write_bytes(b"grown-corpus")
        return CompletedProcess(
            command,
            0,
            b"",
            (
                b"#2 DONE   cov: 7 ft: 11 corp: 2/31b lim: 128 exec/s: 41\n"
                b"stat::number_of_executed_units: 123\n"
            ),
        )


def test_prepared_probe_runs_distinct_snapshot_bytes_without_campaign_cmake() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        first = probe.run(_request(b"first", seed=7))
        second = probe.run(_request(b"second", seed=11))

    # Then
    assert runner.commands_named("cmake") == 0
    assert runner.coverage_calls == 0
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert runner.fuzz_inputs == [b"first", b"second"]
    assert first.cache_hit is False
    assert second.final_units == 123
    assert second.coverage_edges == 7
    assert second.value_profile_features == 11
    assert second.corpus_count == 2
    assert second.executions_per_second == 41


def test_prepared_probe_deduplicates_exact_cache_key() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        first = probe.run(_request(b"same", seed=7))
        second = probe.run(_request(b"same", seed=7))

    # Then
    assert first == second.model_copy(update={"cache_hit": False})
    assert second.cache_hit is True
    assert runner.fuzz_call_count == 1


def test_cache_key_changes_when_probe_budget_changes() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        first = probe.run(_request(b"same", seed=7, duration=1))
        second = probe.run(_request(b"same", seed=7, duration=2))

    # Then
    assert first.cache_key_sha256 != second.cache_key_sha256
    assert runner.fuzz_call_count == 2


def test_probe_argv_is_hardened_and_value_profiled() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        _ = probe.run(_request(b"seed", seed=99, duration=3, max_len=256))

    # Then
    argv = runner.calls[0]
    assert "--network" in argv
    assert "none" in argv
    assert "--read-only" in argv
    assert "--cap-drop" in argv
    assert "ALL" in argv
    assert "--security-opt" in argv
    assert "no-new-privileges=true" in argv
    assert "--user" in argv
    assert "65534:65534" in argv
    assert "--pull" in argv
    assert "never" in argv
    assert "/opt/dllm/bin/poc_probe-asan" in argv
    assert "-use_value_profile=1" in argv
    assert "-seed=99" in argv
    assert "-max_total_time=3" in argv
    assert "-max_len=256" in argv


def test_probe_argv_keeps_every_docker_flag_adjacent_to_its_value() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        _ = probe.run(_request(b"seed", seed=99, duration=3, max_len=256))

    # Then
    argv = runner.calls[0]
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges=true"
    assert argv[argv.index("--pids-limit") + 1] == "128"
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1"
    assert argv[argv.index("--user") + 1] == "65534:65534"
    assert argv[argv.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108
    assert argv[argv.index("--pull") + 1] == "never"


def test_promoted_coverage_is_separate_from_hot_probe_run() -> None:
    # Given
    runner = RecordingRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        request = _request(b"seed", seed=7)
        _ = probe.run(request)
        coverage = probe.promoted_coverage(request)

    # Then
    assert runner.fuzz_call_count == 1
    assert runner.coverage_calls == 1
    assert coverage.report_status == "succeeded"
    assert coverage.run_status == "completed"
    assert coverage.merge_status == "succeeded"
    assert coverage.coverage is not None
    assert coverage.coverage.regions == 100.0
    assert coverage.coverage.functions == 100.0
    assert coverage.coverage.lines == 100.0
    assert coverage.coverage.branches == 100.0
    assert (
        coverage.profraw_sha256
        == "dd3ad5f4b602f305ea0808b6886ca5d4a55ea573703eee4b3e017f68fcc14328"
    )
    assert (
        coverage.profdata_sha256
        == "3beda45cf83afc42abb32b167cd78d347752e7401a91b6a5aca198fb698449ad"
    )
    assert runner.commands_named("llvm-profdata") == 1
    assert runner.commands_named("llvm-cov") == 1
    assert not any("/opt/dllm/bin/poc_probe-coverage" in command for command in runner.calls[:1])


def test_valid_particle_snapshot_has_no_sanitizer_marker() -> None:
    # Given
    runner = RecordingRunner()
    snapshot = materialize_particle(
        decode_particle((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "validity-preserving")
    )

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        result = probe.run(_request(snapshot.content, seed=20260820))

    # Then
    assert result.snapshot_sha256 == snapshot.sha256
    assert result.sanitizer_observed is False
    assert runner.fuzz_inputs == [snapshot.content]


def test_probe_extracts_provisional_family_from_sanitizer_log() -> None:
    """Given ASan output, when probed, then a stable provisional family is attached."""
    # Given
    runner = SanitizerRunner()

    # When
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        result = probe.run(_request(b"asan-seed", seed=20260820))

    # Then
    assert result.sanitizer_observed is True
    assert result.sanitizer_class == "heap-buffer-overflow"
    assert result.root_signature == (
        "READ of size 4|Imf_3_4::InputFile::readPixels|LLVMFuzzerTestOneInput"
    )
    assert result.provisional_family_key == provisional_probe_family_key(
        "heap-buffer-overflow",
        "READ of size 4|Imf_3_4::InputFile::readPixels|LLVMFuzzerTestOneInput",
    )


class SanitizerRunner(RecordingRunner):
    """Command runner that returns a representative sanitizer report."""

    @override
    def __call__(
        self, command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        """Capture a command and emit sanitizer text with unstable addresses."""
        self.calls.append(command)
        work_root = _host_path_after(command, "/work")
        self.fuzz_inputs.append((work_root / "corpus/seed-0000").read_bytes())
        _ = (work_root / "artifacts/crash-000001").write_bytes(b"crash")
        return CompletedProcess(
            command,
            1,
            b"",
            b"\n".join(
                (
                    b"==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xabc",
                    b"READ of size 4 at 0xdef thread T0",
                    b"    #0 0x111 in Imf_3_4::InputFile::readPixels /src/InputFile.cpp:123",
                    b"    #1 0x222 in LLVMFuzzerTestOneInput /src/fuzzer.cpp:45",
                    b"SUMMARY: AddressSanitizer: heap-buffer-overflow /src/InputFile.cpp:123",
                )
            ),
        )


def _config() -> ProbeConfig:
    return ProbeConfig(
        image_id=VALID_IMAGE,
        harness_sha256=HARNESS_SHA,
        rss_limit_mb=512,
        timeout_seconds=5,
    )


def _request(snapshot: bytes, *, seed: int, duration: int = 1, max_len: int = 128) -> ProbeRequest:
    return ProbeRequest(
        snapshot=snapshot,
        fuzzer_seed=seed,
        duration_seconds=duration,
        max_len=max_len,
    )


def _host_path_after(command: tuple[str, ...], mount_point: str) -> Path:
    flag_index = command.index("-v")
    for item in command[flag_index + 1 :]:
        if item.endswith(f":{mount_point}:rw"):
            return Path(item.removesuffix(f":{mount_point}:rw"))
    raise AssertionError(mount_point)


def _docker_flag(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]
