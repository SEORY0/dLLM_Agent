"""OpenEXR proof process execution helpers."""

# ruff: noqa: D103

import json
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, Literal, override

from dllm_agent.core.io import sha256_bytes
from dllm_agent.targets.openexr.particle import FIELD_ORDER, decode_particle, materialize_particle

type ProofCommand = Callable[[tuple[str, ...], Mapping[str, str], float], CompletedProcess[bytes]]
type ProofProcessResult = tuple[CompletedProcess[bytes], tuple[str, ...], tuple[str, ...]]
type MinimizedResult = tuple[bytes, bytes, tuple[str, ...], tuple[str, ...]]
type RecoveredSelectors = tuple[
    tuple[int, ...], Literal["validity-preserving", "boundary-breaking"]
]
type StructuredMinimizationStatus = Literal["unavailable", "unchanged", "reduced"]

OPENEXR_PROOF_IMAGE_DIGEST: Final = (
    "sha256:7591279f51256ef5e80341c788d31145ecf1dd0f7a64db7f0cdeaba527ff5add"
)
OPENEXR_TARGET_COMMIT: Final = "777d231a179de1711d2a942810d9559216ab3f4c"
OPENEXR_PROOF_BINARY: Final = "/opt/dllm/bin/poc_probe-asan"
OPENEXR_PROOF_SEED: Final = 20260820
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_UBSAN_OPTIONS: Final = "halt_on_error=1:print_stacktrace=1"
_ASAN_ENV: Final = {"ASAN_OPTIONS": _ASAN_OPTIONS, "UBSAN_OPTIONS": _UBSAN_OPTIONS}
_TIMEOUT_EXIT_CODE: Final = 124
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108 - Docker tmpfs target.
_MAX_OUTPUT_BYTES: Final = 8_388_608
_MAX_ARTIFACT_BYTES: Final = 1_048_576
_MAX_ARTIFACTS: Final = 8
_MINIMIZED_ARTIFACT_NAME: Final = "minimized"
# fmt: off
_DOCKER_HEAD: Final = ("docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true", "--pids-limit", "128", "--cpus", "1", "--user", "65534:65534")  # noqa: E501
# fmt: on
_ASAN_RE: Final = re.compile(rb"ERROR:\s*AddressSanitizer:\s*([A-Za-z0-9_-]+)")
_FRAME_RE: Final = re.compile(rb"^\s*#\d+\s+0x[0-9a-fA-F]+\s+in\s+(.+)$", re.MULTILINE)
_HEX_RE: Final = re.compile(rb"0x[0-9a-fA-F]+")
_CRASH_RE: Final = re.compile(rb"(Segmentation fault|SIGSEGV|Address not mapped|core dumped)")
_SELECTOR_RANGES: Final = (range(4),) * 4 + (range(1),) * 4 + (range(4),) * 2


@dataclass(frozen=True, slots=True)
class ProofBoundaryError(RuntimeError):
    """Raised when exact proof input or output crosses a safety boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        """Return the boundary rejection reason."""
        return self.detail


@dataclass(frozen=True, slots=True)
class StructuredMinimizationResult:
    """Structured minimization outcome and exact replay evidence."""

    status: StructuredMinimizationStatus
    input_bytes: bytes
    input_sha256: str
    log: bytes
    argvs: tuple[tuple[str, ...], ...]


def run_exact(
    snapshot: bytes, duration_seconds: int, proof_seed: int, runner: ProofCommand
) -> ProofProcessResult:
    with tempfile.TemporaryDirectory(prefix="dllm-openexr-proof-") as directory:
        root = Path(directory)
        _stage_input(root, snapshot)
        argv = _replay_argv(root, proof_seed)
        process = runner(argv, _ASAN_ENV, float(duration_seconds + 30))
        artifacts = _artifact_hashes(root / "artifacts")
        return process, argv, artifacts


def structured_minimize(
    snapshot: bytes,
    duration_seconds: int,
    proof_seed: int,
    runner: ProofCommand,
    signature: tuple[str, str],
) -> StructuredMinimizationResult:
    recovered = _recover_v1_selectors(snapshot)
    snapshot_sha = sha256_bytes(snapshot)
    if recovered is None:
        return StructuredMinimizationResult(
            status="unavailable",
            input_bytes=snapshot,
            input_sha256=snapshot_sha,
            log=b"structured-unavailable\n",
            argvs=(),
        )
    selectors, mode = recovered
    for index, selector in enumerate(selectors):
        if selector == 0:
            continue
        candidate_selectors = (*selectors[:index], 0, *selectors[index + 1 :])
        candidate = materialize_particle(decode_particle(candidate_selectors, mode))
        first, first_argv, _first_artifacts = run_exact(
            candidate.content, duration_seconds, proof_seed, runner
        )
        reject_boundary_process(first)
        second, second_argv, _second_artifacts = run_exact(
            candidate.content, duration_seconds, proof_seed, runner
        )
        reject_boundary_process(second)
        if (
            asan_signature(log_bytes(first)) == signature
            and asan_signature(log_bytes(second)) == signature
        ):
            log = f"reduced:{snapshot_sha}->{candidate.sha256}:{FIELD_ORDER[index]}\n".encode()
            return StructuredMinimizationResult(
                status="reduced",
                input_bytes=candidate.content,
                input_sha256=candidate.sha256,
                log=log,
                argvs=(first_argv, second_argv),
            )
    return StructuredMinimizationResult(
        status="unchanged",
        input_bytes=snapshot,
        input_sha256=snapshot_sha,
        log=f"structured-unchanged:{snapshot_sha}\n".encode(),
        argvs=(),
    )


def try_minimized(
    snapshot: bytes,
    duration_seconds: int,
    proof_seed: int,
    runner: ProofCommand,
    signature: tuple[str, str],
) -> MinimizedResult | None:
    with tempfile.TemporaryDirectory(prefix="dllm-openexr-minimize-") as directory:
        root = Path(directory)
        _stage_input(root, snapshot)
        argv = _minimize_argv(root, duration_seconds, proof_seed)
        process = runner(argv, _ASAN_ENV, float(duration_seconds + 30))
        reject_boundary_process(process)
        log = log_bytes(process)
        minimized = _minimized_artifact(root / "artifacts")
    if minimized is None:
        return None
    rerun, rerun_argv, _artifacts = run_exact(minimized, duration_seconds, proof_seed, runner)
    reject_boundary_process(rerun)
    if asan_signature(log_bytes(rerun)) != signature:
        return None
    return minimized, log, argv, rerun_argv


def reject_boundary_process(process: CompletedProcess[bytes]) -> None:
    if process.returncode in {_TIMEOUT_EXIT_CODE, 125}:
        detail = f"proof boundary process failed: exit {process.returncode}"
        raise ProofBoundaryError(detail)


def crash_signal(process: CompletedProcess[bytes], log: bytes) -> bool:
    return process.returncode in {134, 136, 139} or _CRASH_RE.search(log) is not None


def run_command(
    command: tuple[str, ...], environment: Mapping[str, str], timeout_seconds: float
) -> CompletedProcess[bytes]:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, shell disabled, Docker only.
            command, capture_output=True, check=False, env=environment, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        return CompletedProcess(
            command, _TIMEOUT_EXIT_CODE, error.stdout or b"", error.stderr or b""
        )
    except OSError as error:
        return CompletedProcess(command, 125, b"", (error.strerror or "docker failed").encode())


def log_bytes(process: CompletedProcess[bytes]) -> bytes:
    log = process.stdout + process.stderr
    if len(log) > _MAX_OUTPUT_BYTES:
        detail = f"proof output size limit exceeded: {_MAX_OUTPUT_BYTES}"
        raise ProofBoundaryError(detail)
    return log


def asan_signature(log: bytes) -> tuple[str, str] | None:
    marker = _ASAN_RE.search(log)
    if marker is None:
        return None
    frames = tuple(_normalize_frame(match.group(1)) for match in _FRAME_RE.finditer(log))
    root = "|".join(frame for frame in frames[:2] if frame)
    return marker.group(1).decode(), root


def argv_sha256(argv: tuple[str, ...]) -> str:
    return sha256_bytes(json.dumps(argv, separators=(",", ":")).encode())


def _stage_input(root: Path, snapshot: bytes) -> None:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    root.chmod(0o777)
    artifacts.chmod(0o777)
    input_path = root / "input.exr"
    _ = input_path.write_bytes(snapshot)
    input_path.chmod(0o444)


def _replay_argv(root: Path, proof_seed: int) -> tuple[str, ...]:
    return (
        *_docker_prefix(root),
        OPENEXR_PROOF_BINARY,
        "/work/input.exr",
        "-runs=1",
        f"-seed={proof_seed}",
    )


def _minimize_argv(root: Path, duration_seconds: int, proof_seed: int) -> tuple[str, ...]:
    return (
        *_docker_prefix(root),
        OPENEXR_PROOF_BINARY,
        "/work/input.exr",
        "-minimize_crash=1",
        f"-seed={proof_seed}",
        f"-max_total_time={duration_seconds}",
        "-artifact_prefix=/work/artifacts/",
    )


def _docker_prefix(root: Path) -> tuple[str, ...]:
    return (
        *_DOCKER_HEAD,
        "--tmpfs",
        _TMPFS,
        "--env",
        f"ASAN_OPTIONS={_ASAN_OPTIONS}",
        "--env",
        f"UBSAN_OPTIONS={_UBSAN_OPTIONS}",
        "-v",
        f"{root}:/work:rw",
        "-w",
        "/work",
        "--pull",
        "never",
        OPENEXR_PROOF_IMAGE_DIGEST,
    )


def _normalize_frame(frame: bytes) -> str:
    without_addresses = _HEX_RE.sub(b"0xADDR", frame)
    return without_addresses.decode(errors="replace")


def _artifact_hashes(root: Path) -> tuple[str, ...]:
    children = sorted(root.iterdir())
    if len(children) > _MAX_ARTIFACTS:
        detail = "artifact count limit exceeded"
        raise ProofBoundaryError(detail)
    return tuple(sha256_bytes(_safe_artifact_bytes(child)) for child in children)


def _minimized_artifact(root: Path) -> bytes | None:
    children = sorted(root.iterdir())
    if len(children) == 0:
        return None
    if len(children) > _MAX_ARTIFACTS:
        detail = "artifact count limit exceeded"
        raise ProofBoundaryError(detail)
    return _safe_artifact_bytes(root / _MINIMIZED_ARTIFACT_NAME)


def _safe_artifact_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        detail = f"refusing unsafe artifact: {path.name}"
        raise ProofBoundaryError(detail)
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        detail = f"artifact size limit exceeded: {path.name}"
        raise ProofBoundaryError(detail)
    return path.read_bytes()


def _recover_v1_selectors(snapshot: bytes) -> RecoveredSelectors | None:
    for mode in ("validity-preserving", "boundary-breaking"):
        for selectors in product(*_SELECTOR_RANGES):
            if materialize_particle(decode_particle(selectors, mode)).content == snapshot:
                return selectors, mode
    return None
