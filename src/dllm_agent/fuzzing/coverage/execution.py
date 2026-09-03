"""Docker execution helpers for LLVM source coverage."""

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from dllm_agent.fuzzing.coverage.contracts import CoverageRequest, CoverageTotals
from dllm_agent.targets.profiles import require_target_profile


def argv(value: str) -> tuple[str, ...]:
    """Split a fixed command template into argv tokens."""
    return tuple(re.findall(r"\S+", value))


_IMAGE_ID_PATTERN: Final = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_MAX_INPUT_BYTES: Final = 1_048_576
_MAX_OUTPUT_BYTES: Final = 67_108_864
_MAX_CORPUS_ITEMS: Final = 4096
_TIMEOUT_EXIT_CODE: Final = 124
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108 - Docker tmpfs target.
PROFILE_FILE: Final = "/work/coverage.profraw"
_COVERAGE_FIELD_COUNT: Final = 4
# fmt: off
COMPILER_FLAGS: Final = argv("-std=c++17 -O1 -g -fprofile-instr-generate -fcoverage-mapping -fsanitize=fuzzer -I/target/include /work/harness.cc -o /work/coverage_target")  # noqa: E501
_DOCKER_HEAD: Final = argv("docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges=true --pids-limit 128 --memory 2g --cpus 1 --user 65534:65534 --tmpfs")  # noqa: E501
# fmt: on
MERGE_TAIL: Final = argv(
    "llvm-profdata merge -sparse /work/coverage.profraw -o /work/coverage.profdata"
)
REPORT_TAIL: Final = argv(
    "llvm-cov report /work/coverage_target -instr-profile=/work/coverage.profdata /target"
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


def validate_request(request: CoverageRequest) -> None:
    """Validate request fields before any Docker invocation."""
    _ = require_target_profile(request.target_profile)
    if _IMAGE_ID_PATTERN.fullmatch(request.image_id) is None:
        msg = "image_id must be sha256: followed by 64 lowercase hex characters"
        raise ValueError(msg)
    if request.source_root.is_symlink() or not request.source_root.is_dir():
        msg = "source_root must be a non-symlink directory"
        raise ValueError(msg)
    if request.timeout_seconds <= 0:
        msg = "timeout_seconds must be positive"
        raise ValueError(msg)
    if len(request.harness_bytes) > _MAX_INPUT_BYTES:
        msg = "harness input size limit exceeded"
        raise ValueError(msg)
    if len(request.corpus) > _MAX_CORPUS_ITEMS:
        msg = "corpus item count limit exceeded"
        raise ValueError(msg)
    if any(len(seed) > _MAX_INPUT_BYTES for seed in request.corpus):
        msg = "corpus input size limit exceeded"
        raise ValueError(msg)


def stage_inputs(work_root: Path, corpus_root: Path, request: CoverageRequest) -> None:
    """Write the harness and replay corpus into the Docker work directory."""
    harness = work_root / "harness.cc"
    _ = harness.write_bytes(request.harness_bytes)
    harness.chmod(0o444)
    for index, seed in enumerate(request.corpus):
        path = corpus_root / f"seed-{index:04d}"
        _ = path.write_bytes(seed)
        path.chmod(0o444)


def docker_prefix(request: CoverageRequest, work_root: Path) -> tuple[str, ...]:
    """Build the hardened Docker argv prefix."""
    return (
        *_DOCKER_HEAD,
        _TMPFS,
        "--env",
        f"LLVM_PROFILE_FILE={PROFILE_FILE}",
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


def skip(command: tuple[str, ...], message: str) -> subprocess.CompletedProcess[bytes]:
    """Return a skipped process result with a stable message."""
    return subprocess.CompletedProcess(command, 125, b"", message.encode())


def require_regular_output(path: Path) -> None:
    """Require a bounded regular output file from the Docker work directory."""
    if path.is_symlink() or not path.is_file():
        msg = f"unsafe output: {path.name}"
        raise ValueError(msg)
    if path.stat().st_size > _MAX_OUTPUT_BYTES:
        msg = f"output size limit exceeded: {path.name}"
        raise ValueError(msg)


def log_bytes(process: subprocess.CompletedProcess[bytes]) -> bytes:
    """Return bounded combined stdout and stderr bytes."""
    log = process.stdout + process.stderr
    if len(log) > _MAX_OUTPUT_BYTES:
        msg = "process output size limit exceeded"
        raise ValueError(msg)
    return log


def status(
    prerequisites: tuple[subprocess.CompletedProcess[bytes], ...],
    process: subprocess.CompletedProcess[bytes],
    success: str,
) -> str:
    """Map process exit codes to status strings with prerequisite skipping."""
    if any(step.returncode != 0 for step in prerequisites):
        return "skipped"
    if process.returncode == 0:
        return success
    if process.returncode == _TIMEOUT_EXIT_CODE:
        return "timed_out"
    return "failed"


def coverage_if_reported(report: subprocess.CompletedProcess[bytes]) -> CoverageTotals | None:
    """Parse LLVM coverage TOTAL rows when the report command succeeded."""
    if report.returncode != 0:
        return None
    percentages: list[float] = []
    for line in report.stdout.decode(errors="replace").splitlines():
        fields = line.split()
        if fields[:1] == ["TOTAL"]:
            percentages.extend(
                float(field.removesuffix("%")) for field in fields if field.endswith("%")
            )
            if fields[-1:] == ["-"]:
                percentages.append(100.0)
    if len(percentages) < _COVERAGE_FIELD_COUNT:
        return None
    return CoverageTotals(
        regions=percentages[0],
        functions=percentages[1],
        lines=percentages[2],
        branches=percentages[3],
    )
