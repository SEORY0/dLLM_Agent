"""Safe Docker libFuzzer runner contract tests."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dllm_agent.fuzzing.runner import CommandRunner, FuzzRunRequest, run_fuzz
from dllm_agent.targets import profiles as target_profiles

VALID_IMAGE = "sha256:" + ("a" * 64)
HARNESS_BYTES = (
    b'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}'
)


def test_run_fuzz_rejects_unpinned_image_before_docker_invocation(tmp_path: Path) -> None:
    # Given
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id="llvm:latest",
        source_root=tmp_path,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When / Then
    with pytest.raises(ValueError, match="image_id"):
        _ = run_fuzz(request, runner=_recording_runner(calls))
    assert calls == []


def test_run_fuzz_rejects_missing_source_root_before_docker_invocation(tmp_path: Path) -> None:
    # Given
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=tmp_path / "missing",
        harness_bytes=b"int x;",
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When / Then
    with pytest.raises(ValueError, match="source_root"):
        _ = run_fuzz(request, runner=_recording_runner(calls))
    assert calls == []


def test_request_is_immutable(tmp_path: Path) -> None:
    # Given
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=tmp_path,
        harness_bytes=b"int x;",
        corpus=(b"seed",),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When / Then
    field_name = "max_len"
    with pytest.raises(FrozenInstanceError):
        setattr(request, field_name, 32)


def test_run_fuzz_passes_requested_seed_to_libfuzzer(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
        fuzzer_seed=4242,
    )

    # When
    _ = run_fuzz(request, runner=_recording_runner(calls))

    # Then
    fuzz_argv = next(
        command for command in calls if "/work/fuzz_target" in command and "/work/corpus" in command
    )
    assert fuzz_argv.count("-seed=4242") == 1


def test_run_fuzz_rejects_negative_fuzzer_seed_before_docker_invocation(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
        fuzzer_seed=-1,
    )

    # When / Then
    with pytest.raises(ValueError, match="fuzzer_seed"):
        _ = run_fuzz(request, runner=_recording_runner(calls))
    assert calls == []


def test_fuzz_request_rejects_unknown_runtime_profile_before_runner(tmp_path: Path) -> None:
    # Given
    calls: list[tuple[str, ...]] = []

    def run_with_runtime_profile(value: str) -> None:
        request = FuzzRunRequest(
            image_id=VALID_IMAGE,
            source_root=tmp_path,
            harness_bytes=HARNESS_BYTES,
            corpus=(),
            duration_seconds=1,
            max_len=16,
            rss_limit_mb=64,
            timeout_seconds=5,
            target_profile=target_profiles.require_target_profile(value),
        )
        _ = run_fuzz(request, runner=_recording_runner(calls))

    # When / Then
    with pytest.raises(ValueError, match="target profile"):
        run_with_runtime_profile("manifest-shell-command")
    assert calls == []


def test_openexr_runtime_dockerfile_hardcodes_clean_pins_and_instrumentation() -> None:
    # Given
    dockerfile = (Path(__file__).parents[1] / "tools/openexr_runtime/Dockerfile").read_text()

    # When / Then
    assert "ARG " not in dockerfile
    assert "777d231a179de1711d2a942810d9559216ab3f4c" in dockerfile
    assert "962a0b40842bc4bc73a81294af5bef6dc974272a" in dockerfile
    assert "a29644c43663d8cad3a3f2f22fa064f1d138df9873e32e410265756d25755087" in dockerfile
    assert "1e480d11cb98b032a2dece9b9a8730512effc7f6" in dockerfile
    assert "status --porcelain=v1 --untracked-files=all" in dockerfile
    assert "unset CFLAGS CXXFLAGS" in dockerfile
    assert dockerfile.count("-fsanitize=fuzzer-no-link") >= 4
    assert "CMAKE_BUILD_TYPE=Release" not in dockerfile
    assert dockerfile.count("CMAKE_BUILD_TYPE=Debug") == 2


def test_search_build_disables_exactly_one_ubsan_check_and_says_so() -> None:
    # Given
    root = Path(__file__).parents[1] / "tools/openexr_runtime"
    dockerfile = (root / "Dockerfile").read_text()
    cmake = (root / "CMakeLists.txt").read_text()

    # When: collect every -fno-sanitize=<check> (not -fno-sanitize-recover)
    def disabled(text: str) -> set[str]:
        return {
            token.split("=", 1)[1]
            for token in text.replace('"', " ").split()
            if token.startswith("-fno-sanitize=")
        }

    # Then only nonnull-attribute is disabled (it aborts every search cell at
    # ImfOpaqueAttribute.cpp:35 within ~2000 execs); anything wider must be a
    # deliberate, reviewed change, and the deviation is recorded as an image label.
    assert disabled(dockerfile) == {"nonnull-attribute"}
    assert disabled(cmake) == {"nonnull-attribute"}
    assert "-fno-sanitize-recover=undefined" in dockerfile
    assert 'ubsan-disabled-checks="nonnull-attribute"' in dockerfile


def test_openexr_fuzz_helper_leaves_only_build_directory_host_cleanable() -> None:
    # Given
    helper = (Path(__file__).parents[1] / "tools/openexr_runtime/build-fuzz").read_text()

    # When / Then
    assert "umask 000" in helper
    assert "trap 'chmod -R a+rwX /work/build-fuzz 2>/dev/null || true' EXIT" in helper


def test_run_fuzz_supports_asan_only_profile(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        return CompletedProcess(command, 0, b"", b"")

    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(b"seed",),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
        sanitizers="address",
    )

    # When
    result = run_fuzz(request, runner=runner)

    # Then
    assert "-fsanitize=fuzzer,address" in result.build_argv
    assert "-fsanitize=fuzzer,address,undefined" not in result.build_argv
    assert result.sanitizers == "address"


def test_run_fuzz_openexr_profile_uses_only_fixed_build_helper(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
        sanitizers="address",
        target_profile="openexr-v3.4",
    )

    # When
    result = run_fuzz(request, runner=_recording_runner(calls))

    # Then
    assert result.build_argv[-1:] == ("/opt/dllm/bin/build-fuzz",)
    assert "clang++" not in result.build_argv[-2:]
    _assert_hardened_docker_argv(result.build_argv)


def test_run_fuzz_openexr_profile_rejects_non_asan_sanitizer_before_docker(
    tmp_path: Path,
) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    calls: list[tuple[str, ...]] = []
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
        sanitizers="address,undefined",
        target_profile="openexr-v3.4",
    )

    # When / Then
    with pytest.raises(ValueError, match="address"):
        _ = run_fuzz(request, runner=_recording_runner(calls))
    assert calls == []


def test_run_fuzz_header_only_profile_retains_exact_build_argv(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When
    result = run_fuzz(request, runner=_recording_runner([]))

    # Then
    assert result.build_argv[-10:] == (
        "clang++",
        "-std=c++17",
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-fsanitize=fuzzer,address,undefined",
        "-I/target/include",
        "/work/harness.cc",
        "-o",
        "/work/fuzz_target",
    )


def test_run_fuzz_truncates_large_corpus_deterministically(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        if "/work/fuzz_target" in command:
            corpus = _host_path_after(command, "/work") / "corpus"
            for index in range(257):
                _ = (corpus / f"generated-{index:04d}").write_bytes(index.to_bytes(2, "big"))
        return CompletedProcess(command, 0, b"", b"")

    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When
    result = run_fuzz(request, runner=runner)

    # Then
    assert len(result.corpus_artifacts) == 256
    assert result.corpus_artifacts[0].name == "generated-0000"
    assert result.corpus_artifacts[-1].name == "generated-0255"


def test_run_fuzz_uses_hardened_docker_argv_and_collects_artifacts(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "include").mkdir()
    seen_environments: list[Mapping[str, str]] = []

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        seen_environments.append(environment)
        command_text = " ".join(command)
        if "--version" in command:
            return CompletedProcess(command, 0, b"clang version 18\n", b"")
        if "clang++" in command:
            assert timeout_seconds >= 30
            work_dir = _host_path_after(command, "/work")
            assert work_dir.stat().st_mode & 0o007 == 0o007
            assert (work_dir / "harness.cc").stat().st_mode & 0o004 == 0o004
            return CompletedProcess(command, 0, b"", b"compiled\n")
        if "/work/fuzz_target" in command:
            assert timeout_seconds >= 33
            work_dir = _host_path_after(command, "/work")
            artifact_dir = work_dir / "artifacts"
            corpus_dir = work_dir / "corpus"
            _ = (artifact_dir / "crash-1").write_bytes(b"boom")
            _ = (corpus_dir / "new-seed").write_bytes(b"grown")
            return CompletedProcess(command, 77, b"fuzzing\n", b"crash\n")
        raise AssertionError(command_text)

    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=HARNESS_BYTES,
        corpus=(b"seed-a", b"seed-b"),
        duration_seconds=3,
        max_len=128,
        rss_limit_mb=256,
        timeout_seconds=10,
    )

    # When
    result = run_fuzz(request, runner=runner)

    # Then
    for argv in (
        result.compiler_version_argv,
        result.fuzzer_version_argv,
        result.build_argv,
        result.fuzz_argv,
    ):
        _assert_hardened_docker_argv(argv)

    assert "-std=c++17" in result.build_argv
    assert "-fsanitize=fuzzer,address,undefined" in result.build_argv
    assert "-I/target/include" in result.build_argv
    assert "-max_total_time=3" in result.fuzz_argv
    assert "-timeout=10" in result.fuzz_argv
    assert "-rss_limit_mb=256" in result.fuzz_argv
    assert "-max_len=128" in result.fuzz_argv
    assert result.fuzz_argv.count("-seed=0") == 1
    assert "-print_final_stats=1" in result.fuzz_argv
    assert result.build_status == "succeeded"
    assert result.fuzz_status == "crashed"
    assert result.build_log == b"compiled\n"
    assert result.fuzz_log == b"fuzzing\ncrash\n"
    assert result.crash_artifacts[0].name == "crash-1"
    assert result.crash_artifacts[0].content == b"boom"
    assert tuple(artifact.name for artifact in result.crash_artifacts) == ("crash-1",)
    assert tuple((artifact.name, artifact.content) for artifact in result.corpus_artifacts) == (
        ("new-seed", b"grown"),
        ("seed-0000", b"seed-a"),
        ("seed-0001", b"seed-b"),
    )
    assert seen_environments[-1]["ASAN_OPTIONS"] == "detect_leaks=0:abort_on_error=1:symbolize=1"
    assert seen_environments[-1]["UBSAN_OPTIONS"] == "halt_on_error=1:print_stacktrace=1"


def test_run_fuzz_rejects_symlink_artifacts(tmp_path: Path) -> None:
    # Given
    source_root = tmp_path / "source"
    source_root.mkdir()

    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        if "--version" in command or "clang++" in command:
            return CompletedProcess(command, 0, b"", b"")
        work_dir = _host_path_after(command, "/work")
        crash = work_dir / "artifacts" / "crash-1"
        _ = crash.write_bytes(b"boom")
        (work_dir / "artifacts" / "unsafe").symlink_to(crash)
        return CompletedProcess(command, 77, b"", b"")

    request = FuzzRunRequest(
        image_id=VALID_IMAGE,
        source_root=source_root,
        harness_bytes=b"int x;",
        corpus=(),
        duration_seconds=1,
        max_len=16,
        rss_limit_mb=64,
        timeout_seconds=5,
    )

    # When / Then
    with pytest.raises(ValueError, match="unsafe artifact"):
        _ = run_fuzz(request, runner=runner)


def _record(calls: list[tuple[str, ...]], command: tuple[str, ...]) -> CompletedProcess[bytes]:
    calls.append(command)
    return CompletedProcess(command, 0, b"", b"")


def _recording_runner(calls: list[tuple[str, ...]]) -> CommandRunner:
    def runner(
        command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
    ) -> CompletedProcess[bytes]:
        _ = environment, timeout_seconds
        return _record(calls, command)

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
    assert "--env" in argv
    assert "ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:symbolize=1" in argv
    assert "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1" in argv
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
