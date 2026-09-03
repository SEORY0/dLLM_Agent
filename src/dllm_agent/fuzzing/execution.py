"""Docker execution helpers for the libFuzzer runner."""

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from dllm_agent.fuzzing.contracts import (
    BuildStatus,
    CommandRunner,
    FuzzArtifact,
    FuzzRunRequest,
    FuzzStatus,
    SanitizerProfile,
)
from dllm_agent.targets.profiles import require_target_profile

_IMAGE_ID_PATTERN: Final = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_UBSAN_OPTIONS: Final = "halt_on_error=1:print_stacktrace=1"
_MAX_CRASH_ARTIFACTS: Final = 64
_MAX_CORPUS_ARTIFACTS: Final = 256
_MAX_ARTIFACT_BYTES: Final = 1_048_576
_MAX_CORPUS_BYTES: Final = 8_388_608
_MIN_RSS_LIMIT_MB: Final = 16
_TIMEOUT_EXIT_CODE: Final = 124
_PIDS_LIMIT: Final = "128"
_CPUS: Final = "1"
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108 - Docker tmpfs target.
_COMPILER_BASE_FLAGS: Final = (
    "-std=c++17",
    "-O1",
    "-g",
    "-fno-omit-frame-pointer",
    "-I/target/include",
    "/work/harness.cc",
    "-o",
    "/work/fuzz_target",
)


def run_command(
    command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    """Run Docker with bounded execution and normalized failure results."""
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, shell disabled, Docker only.
            command,
            capture_output=True,
            check=False,
            env=environment,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            command, _TIMEOUT_EXIT_CODE, error.stdout or b"", error.stderr or b""
        )
    except OSError as error:
        message = (error.strerror or "docker execution failed").encode()
        return subprocess.CompletedProcess(command, 125, b"", message)


def validate_request(request: FuzzRunRequest) -> None:
    """Validate request fields before any Docker invocation."""
    if _IMAGE_ID_PATTERN.fullmatch(request.image_id) is None:
        msg = "image_id must be sha256: followed by 64 lowercase hex characters"
        raise ValueError(msg)
    if not request.source_root.is_dir():
        msg = "source_root must exist and be a directory"
        raise ValueError(msg)
    if request.duration_seconds < 1:
        msg = "duration_seconds must be positive"
        raise ValueError(msg)
    if request.max_len < 1:
        msg = "max_len must be positive"
        raise ValueError(msg)
    if request.rss_limit_mb < _MIN_RSS_LIMIT_MB:
        msg = "rss_limit_mb must be at least 16"
        raise ValueError(msg)
    if request.timeout_seconds <= 0:
        msg = "timeout_seconds must be positive"
        raise ValueError(msg)
    if request.fuzzer_seed < 0:
        msg = "fuzzer_seed must be non-negative"
        raise ValueError(msg)
    if request.sanitizers not in {"address,undefined", "address"}:
        msg = "unsupported sanitizer profile"
        raise ValueError(msg)
    validate_target_profile(request)


def validate_target_profile(request: FuzzRunRequest) -> None:
    """Enforce the reviewed sanitizer boundary for the selected target."""
    match require_target_profile(request.target_profile):
        case "header-only":
            return
        case "openexr-v3.4":
            if request.sanitizers != "address":
                msg = "openexr-v3.4 requires address sanitizer"
                raise ValueError(msg)


def compiler_flags(profile: SanitizerProfile) -> tuple[str, ...]:
    """Return fixed compiler flags for one validated sanitizer profile."""
    return (
        *_COMPILER_BASE_FLAGS[:4],
        f"-fsanitize=fuzzer,{profile}",
        *_COMPILER_BASE_FLAGS[4:],
    )


def stage_inputs(work_root: Path, corpus_root: Path, request: FuzzRunRequest) -> None:
    """Write the harness and seed corpus into the Docker work directory."""
    harness = work_root / "harness.cc"
    _ = harness.write_bytes(request.harness_bytes)
    harness.chmod(0o444)
    for index, seed in enumerate(request.corpus):
        path = corpus_root / f"seed-{index:04d}"
        _ = path.write_bytes(seed)
        path.chmod(0o444)


def docker_prefix(request: FuzzRunRequest, work_root: Path) -> tuple[str, ...]:
    """Build the hardened Docker argv prefix."""
    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        _PIDS_LIMIT,
        "--memory",
        f"{request.rss_limit_mb}m",
        "--cpus",
        _CPUS,
        "--user",
        "65534:65534",
        "--tmpfs",
        _TMPFS,
        "--env",
        f"ASAN_OPTIONS={_ASAN_OPTIONS}",
        "--env",
        f"UBSAN_OPTIONS={_UBSAN_OPTIONS}",
        "-v",
        f"{request.source_root.resolve()}:/target:ro",
        "-v",
        f"{work_root}:/work:rw",
        "-w",
        "/work",
        "--pull",
        "never",
        request.image_id,
    )


def fuzz_environment() -> dict[str, str]:
    """Return sanitizer environment variables passed to Docker."""
    return {
        "ASAN_OPTIONS": _ASAN_OPTIONS,
        "UBSAN_OPTIONS": _UBSAN_OPTIONS,
    }


def run_fuzzer_if_built(
    build: subprocess.CompletedProcess[bytes],
    fuzz_argv: tuple[str, ...],
    environment: Mapping[str, str],
    request: FuzzRunRequest,
    runner: CommandRunner,
) -> subprocess.CompletedProcess[bytes]:
    """Skip fuzzing on build failure, otherwise run with the fuzzing timeout."""
    if build.returncode != 0:
        return subprocess.CompletedProcess(fuzz_argv, 125, b"", b"build failed; fuzzing skipped")
    process_timeout = request.duration_seconds + max(request.timeout_seconds, 30.0)
    return runner(fuzz_argv, environment, process_timeout)


def build_status(build: subprocess.CompletedProcess[bytes]) -> BuildStatus:
    """Map the compiler process exit code to a build status."""
    if build.returncode == 0:
        return "succeeded"
    if build.returncode == _TIMEOUT_EXIT_CODE:
        return "timed_out"
    return "failed"


def fuzz_status(
    build: subprocess.CompletedProcess[bytes], fuzz: subprocess.CompletedProcess[bytes]
) -> FuzzStatus:
    """Map build and fuzz process exit codes to a fuzz status."""
    if build.returncode != 0:
        return "skipped"
    if fuzz.returncode == 0:
        return "completed"
    if fuzz.returncode == _TIMEOUT_EXIT_CODE:
        return "timed_out"
    return "crashed"


def collect_corpus_artifacts(root: Path) -> tuple[FuzzArtifact, ...]:
    """Collect a deterministic bounded prefix of a validated fuzzer corpus."""
    return _collect_artifacts(
        root,
        max_artifacts=_MAX_CORPUS_ARTIFACTS,
        max_total_bytes=_MAX_CORPUS_BYTES,
        strict_count=False,
    )


def collect_crash_artifacts(root: Path) -> tuple[FuzzArtifact, ...]:
    """Collect bounded crash files and reject count overflow."""
    return _collect_artifacts(
        root,
        max_artifacts=_MAX_CRASH_ARTIFACTS,
        max_total_bytes=_MAX_CRASH_ARTIFACTS * _MAX_ARTIFACT_BYTES,
        strict_count=True,
    )


def _collect_artifacts(
    root: Path, *, max_artifacts: int, max_total_bytes: int, strict_count: bool
) -> tuple[FuzzArtifact, ...]:
    children = sorted(root.iterdir())
    sizes: list[int] = []
    for child in children:
        if child.is_symlink() or not child.is_file():
            msg = f"refusing unsafe artifact: {child.name}"
            raise ValueError(msg)
        size = child.stat().st_size
        if size > _MAX_ARTIFACT_BYTES:
            msg = f"artifact size limit exceeded: {child.name}"
            raise ValueError(msg)
        sizes.append(size)
    if strict_count and len(children) > max_artifacts:
        msg = "artifact count limit exceeded"
        raise ValueError(msg)
    artifacts: list[FuzzArtifact] = []
    total_bytes = 0
    for child, size in zip(children[:max_artifacts], sizes[:max_artifacts], strict=True):
        if total_bytes + size > max_total_bytes:
            break
        artifacts.append(FuzzArtifact(name=child.name, content=child.read_bytes()))
        total_bytes += size
    return tuple(artifacts)
