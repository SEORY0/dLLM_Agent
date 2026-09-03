"""Temporary compiler and AddressSanitizer validator."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dllm_agent.core.models import Finding, ValidatorVerdict

_COMPILER_FLAGS = (
    "-std=c11",
    "-O0",
    "-g",
    "-fno-omit-frame-pointer",
    "-fno-pie",
    "-no-pie",
    "-fsanitize=address",
    "-Wall",
    "-Wextra",
)


@dataclass(frozen=True, slots=True)
class ValidatorRequest:
    """Inputs needed for one proof execution of already-trusted source bytes."""

    target_name: str
    target_bytes: bytes
    finding: Finding
    proof_input: bytes
    timeout_seconds: float


def validate(request: ValidatorRequest) -> ValidatorVerdict:
    """Compile and execute only the finalized finding."""
    with tempfile.TemporaryDirectory(prefix="dllm-agent-") as directory:
        source = Path(directory) / Path(request.target_name).name
        binary = Path(directory) / "target"
        try:
            _ = source.write_bytes(request.target_bytes)
        except OSError:
            return ValidatorVerdict(
                status="inconclusive",
                reason="validator could not stage target",
                exit_code=None,
                timed_out=False,
                sanitizer=None,
            )
        compile_result = _compile(source, binary)
        if compile_result.returncode != 0:
            return ValidatorVerdict(
                status="inconclusive",
                reason="compiler rejected target",
                exit_code=compile_result.returncode,
                timed_out=False,
                sanitizer=None,
            )
        return _execute(request, binary)


def _compile(target: Path, binary: Path) -> subprocess.CompletedProcess[bytes]:
    command = ("cc", *_COMPILER_FLAGS, str(target), "-o", str(binary))
    try:
        return subprocess.run(  # noqa: S603 - fixed compiler argv, shell disabled
            command, capture_output=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, 125, b"", b"")


def _execute(request: ValidatorRequest, binary: Path) -> ValidatorVerdict:
    environment = os.environ.copy()
    environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1:abort_on_error=1"
    try:
        result = subprocess.run(  # noqa: S603 - repository-owned temporary binary
            (str(binary),),
            input=request.proof_input,
            capture_output=True,
            check=False,
            timeout=request.timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return ValidatorVerdict(
            status="inconclusive",
            reason="validator timed out",
            exit_code=None,
            timed_out=True,
            sanitizer=None,
        )
    except OSError as error:
        return ValidatorVerdict(
            status="inconclusive",
            reason=error.strerror or "validator execution failed",
            exit_code=None,
            timed_out=False,
            sanitizer=None,
        )
    return _verdict(request, result)


def _verdict(
    request: ValidatorRequest, result: subprocess.CompletedProcess[bytes]
) -> ValidatorVerdict:
    normalized = result.stderr.decode("utf-8", errors="replace")
    sanitizer = "AddressSanitizer: stack-buffer-overflow"
    line_marker = f"{request.target_name}:{request.finding.span.start_line}"
    proved = result.returncode != 0 and sanitizer in normalized and line_marker in normalized
    return ValidatorVerdict(
        status="proved" if proved else "rejected",
        reason="ASan proof matched finding" if proved else "required ASan evidence absent",
        exit_code=result.returncode,
        timed_out=False,
        sanitizer=sanitizer if sanitizer in normalized else None,
    )
