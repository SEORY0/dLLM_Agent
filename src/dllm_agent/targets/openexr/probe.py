# ruff: noqa: D101, D102
"""Compile-once OpenEXR snapshot probing."""

import json
import re
import subprocess
import tempfile
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, Literal, Self

from pydantic import Field, computed_field, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    build_long_probe_argv as build_long_probe_argv,  # noqa: PLC0414
)
from dllm_agent.targets.openexr.probe_coverage import (
    CoverageProbeInput,
    PromotedCoverageResult,
    run_promoted_coverage,
)

type ProbeCommand = Callable[[tuple[str, ...], Mapping[str, str], float], CompletedProcess[bytes]]

_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_UBSAN_OPTIONS: Final = "halt_on_error=1:print_stacktrace=1"
_ASAN_ENV: Final = {"ASAN_OPTIONS": _ASAN_OPTIONS, "UBSAN_OPTIONS": _UBSAN_OPTIONS}
_TIMEOUT_EXIT_CODE: Final = 124
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108 - Docker tmpfs target.
_MAX_SNAPSHOT_BYTES: Final = 1_048_576
_MAX_OUTPUT_BYTES: Final = 8_388_608
_MAX_ARTIFACT_BYTES: Final = 1_048_576
_MAX_CRASH_ARTIFACTS: Final = 64
# fmt: off
_DOCKER_HEAD: Final = ("docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true", "--pids-limit", "128", "--cpus", "1", "--user", "65534:65534")  # noqa: E501
# fmt: on
_STAT_UNITS: Final = re.compile(rb"stat::number_of_executed_units:\s*(\d+)")
_DONE_STATS: Final = re.compile(
    rb"\bDONE\b.*?\bcov:\s*(\d+).*?\bft:\s*(\d+).*?\bcorp:\s*(\d+).*?\bexec/s:\s*(\d+)",
    re.DOTALL,
)
_ASAN_CLASS: Final = re.compile(rb"ERROR: AddressSanitizer:\s*([^\s:]+)")
_UBSAN_CLASS: Final = re.compile(rb"runtime error:\s*([^\n]+)")
_HEX_RE: Final = re.compile(rb"0x[0-9a-fA-F]+")
_FRAME_RE: Final = re.compile(rb"^\s*#\d+\s+0x[0-9a-fA-F]+\s+in\s+([^\s]+)")
_ACCESS_RE: Final = re.compile(rb"\b(READ|WRITE) of size \d+")
_ROOT_SIGNATURE_PARTS: Final = 3


class _ProbeBoundaryError(RuntimeError):
    pass


class ProbeConfig(FrozenModel):
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    harness_sha256: Sha256
    rss_limit_mb: int = Field(ge=64)
    timeout_seconds: int = Field(gt=0)


class ProbeRequest(FrozenModel):
    snapshot: bytes = Field(min_length=1, max_length=_MAX_SNAPSHOT_BYTES)
    fuzzer_seed: int = Field(ge=0)
    duration_seconds: int = Field(gt=0, le=30)
    max_len: int = Field(gt=0, le=_MAX_SNAPSHOT_BYTES)

    @computed_field
    @property
    def snapshot_sha256(self) -> str:
        return sha256_bytes(self.snapshot)


class ProbeResult(FrozenModel):
    snapshot_sha256: Sha256
    cache_key_sha256: Sha256
    cache_hit: bool
    run_status: Literal["completed", "crashed", "timed_out"]
    sanitizer_observed: bool
    final_units: int | None
    coverage_edges: int | None
    value_profile_features: int | None
    corpus_count: int | None
    executions_per_second: int | None
    run_log_sha256: Sha256
    run_argv_sha256: Sha256
    crash_artifact_sha256: tuple[Sha256, ...]
    crash_artifact_bytes: tuple[bytes, ...] = Field(default=(), exclude=True)
    sanitizer_class: str | None = None
    root_signature: str | None = None
    provisional_family_key: Sha256 | None = None

    @model_validator(mode="after")
    def _crash_artifact_hashes_match(self) -> Self:
        if tuple(sha256_bytes(item) for item in self.crash_artifact_bytes) != (
            self.crash_artifact_sha256
        ):
            message = "crash artifact hashes do not match payload bytes"
            raise ValueError(message)
        return self


class PreparedProbe:
    def __init__(self, config: ProbeConfig, runner: ProbeCommand) -> None:  # noqa: D107
        self._config: ProbeConfig = config
        self._runner: ProbeCommand = runner
        self._cache: dict[str, ProbeResult] = {}

    def run(self, request: ProbeRequest) -> ProbeResult:
        cache_key = _cache_key(self._config, request)
        if cache_key in self._cache:
            return self._cache[cache_key].model_copy(update={"cache_hit": True})
        with tempfile.TemporaryDirectory(prefix="dllm-openexr-probe-") as directory:
            root = Path(directory)
            _stage_snapshot(root, request.snapshot)
            argv = _probe_argv(self._config, root, request, "poc_probe-asan")
            timeout = float(request.duration_seconds + max(self._config.timeout_seconds, 30))
            process = self._runner(argv, _ASAN_ENV, timeout)
            log = _log_bytes(process)
            artifact_bytes = _artifact_bytes(root / "artifacts")
        final_units, coverage_edges, features, corpus_count, execs = _parse_fuzzer_log(log)
        sanitizer_class, root_signature = sanitizer_signature(log)
        # fmt: off
        status: Literal["completed", "crashed", "timed_out"] = "completed" if process.returncode == 0 else "timed_out" if process.returncode == _TIMEOUT_EXIT_CODE else "crashed"  # noqa: E501
        family = None if sanitizer_class is None or root_signature is None else provisional_probe_family_key(sanitizer_class, root_signature)  # noqa: E501
        # fmt: on
        # fmt: off
        result = ProbeResult(snapshot_sha256=request.snapshot_sha256, cache_key_sha256=cache_key, cache_hit=False, run_status=status, sanitizer_observed=_sanitizer_observed(log), final_units=final_units, coverage_edges=coverage_edges, value_profile_features=features, corpus_count=corpus_count, executions_per_second=execs, run_log_sha256=sha256_bytes(log), run_argv_sha256=_argv_sha256(argv), crash_artifact_sha256=tuple(sha256_bytes(item) for item in artifact_bytes), crash_artifact_bytes=artifact_bytes, sanitizer_class=sanitizer_class, root_signature=root_signature, provisional_family_key=family)  # noqa: E501
        # fmt: on
        self._cache[cache_key] = result
        return result

    def promoted_coverage(self, request: ProbeRequest) -> PromotedCoverageResult:
        with tempfile.TemporaryDirectory(prefix="dllm-openexr-coverage-") as directory:
            # fmt: off
            coverage_request = CoverageProbeInput(image_id=self._config.image_id, snapshot=request.snapshot, snapshot_sha256=request.snapshot_sha256, fuzzer_seed=request.fuzzer_seed, duration_seconds=request.duration_seconds, timeout_seconds=self._config.timeout_seconds, rss_limit_mb=self._config.rss_limit_mb, max_len=request.max_len)  # noqa: E501
            # fmt: on
            return run_promoted_coverage(coverage_request, self._runner, Path(directory))


def _run_command(
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
        message = (error.strerror or "docker execution failed").encode()
        return CompletedProcess(command, 125, b"", message)


@contextmanager
def prepare_openexr_probe(
    config: ProbeConfig, runner: ProbeCommand = _run_command
) -> Generator[PreparedProbe]:
    """Prepare a campaign using an already-built OpenEXR probe image."""
    yield PreparedProbe(config, runner)


def _cache_key(config: ProbeConfig, request: ProbeRequest) -> str:
    payload = {
        "duration_seconds": request.duration_seconds,
        "fuzzer_seed": request.fuzzer_seed,
        "harness_sha256": config.harness_sha256,
        "image_id": config.image_id,
        "max_len": request.max_len,
        "snapshot_sha256": request.snapshot_sha256,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _stage_snapshot(root: Path, snapshot: bytes) -> None:
    corpus = root / "corpus"
    artifacts = root / "artifacts"
    corpus.mkdir()
    artifacts.mkdir()
    root.chmod(0o777)
    corpus.chmod(0o777)
    artifacts.chmod(0o777)
    seed = corpus / "seed-0000"
    _ = seed.write_bytes(snapshot)
    seed.chmod(0o444)


def _probe_argv(
    config: ProbeConfig,
    root: Path,
    request: ProbeRequest,
    binary: Literal["poc_probe-asan", "poc_probe-coverage"],
) -> tuple[str, ...]:
    # fmt: off
    return (*_docker_prefix(config, root), f"/opt/dllm/bin/{binary}", "/work/corpus", "-artifact_prefix=/work/artifacts/", "-use_value_profile=1", f"-max_total_time={request.duration_seconds}", f"-timeout={config.timeout_seconds}", f"-rss_limit_mb={config.rss_limit_mb}", f"-max_len={request.max_len}", f"-seed={request.fuzzer_seed}", "-print_final_stats=1")  # noqa: E501
    # fmt: on


def _docker_prefix(config: ProbeConfig, root: Path) -> tuple[str, ...]:
    # fmt: off
    return (*_DOCKER_HEAD, "--memory", f"{config.rss_limit_mb}m", "--tmpfs", _TMPFS, "--env", f"ASAN_OPTIONS={_ASAN_OPTIONS}", "--env", f"UBSAN_OPTIONS={_UBSAN_OPTIONS}", "-v", f"{root}:/work:rw", "-w", "/work", "--pull", "never", config.image_id)  # noqa: E501
    # fmt: on


def _parse_fuzzer_log(
    log: bytes,
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    units = _STAT_UNITS.search(log)
    done = _DONE_STATS.search(log)
    return (
        _group_int(units, 1),
        _group_int(done, 1),
        _group_int(done, 2),
        _group_int(done, 3),
        _group_int(done, 4),
    )


def _group_int(match: re.Match[bytes] | None, index: int) -> int | None:
    if match is None:
        return None
    return int(match.group(index))


def _sanitizer_observed(log: bytes) -> bool:
    return b"ERROR: AddressSanitizer" in log or b"runtime error:" in log


def sanitizer_signature(log: bytes) -> tuple[str | None, str | None]:
    """Extract a stable sanitizer class and root signature from probe output."""
    sanitizer_class = _sanitizer_class(log)
    if sanitizer_class is None:
        return None, None
    root_signature = _root_signature(log)
    if root_signature is None:
        return sanitizer_class, None
    return sanitizer_class, root_signature


def provisional_probe_family_key(sanitizer_class: str, root_signature: str) -> str:
    """Return a deterministic provisional proof-family key for probe-side caps."""
    payload = {
        "root_signature": _normalize_text(root_signature),
        "sanitizer_class": sanitizer_class.strip(),
    }
    return sha256_bytes(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())


def _sanitizer_class(log: bytes) -> str | None:
    asan = _ASAN_CLASS.search(log)
    if asan is not None:
        return asan.group(1).decode(errors="replace")
    ubsan = _UBSAN_CLASS.search(log)
    if ubsan is None:
        return None
    return _normalize_text(ubsan.group(1).decode(errors="replace"))


def _root_signature(log: bytes) -> str | None:
    parts: list[str] = []
    for line in log.splitlines():
        access = _ACCESS_RE.search(line)
        if access is not None and len(parts) == 0:
            parts.append(_normalize_bytes(access.group(0)))
            continue
        frame = _FRAME_RE.match(line)
        if frame is not None:
            parts.append(_normalize_bytes(frame.group(1)))
        if len(parts) >= _ROOT_SIGNATURE_PARTS:
            return "|".join(parts)
    if len(parts) == 0:
        return None
    return "|".join(parts)


def _normalize_bytes(value: bytes) -> str:
    return _normalize_text(_HEX_RE.sub(b"0xADDR", value).decode(errors="replace"))


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _log_bytes(process: CompletedProcess[bytes]) -> bytes:
    log = process.stdout + process.stderr
    if len(log) > _MAX_OUTPUT_BYTES:
        raise _ProbeBoundaryError(  # noqa: TRY003
            f"probe output size limit exceeded: {_MAX_OUTPUT_BYTES}"  # noqa: EM102
        )
    return log


def _artifact_bytes(root: Path) -> tuple[bytes, ...]:
    children = sorted(root.iterdir())
    if len(children) > _MAX_CRASH_ARTIFACTS:
        raise _ProbeBoundaryError("artifact count limit exceeded")  # noqa: EM101,TRY003
    artifacts: list[bytes] = []
    for child in children:
        if child.is_symlink() or not child.is_file():
            raise _ProbeBoundaryError(  # noqa: TRY003
                f"refusing unsafe artifact: {child.name}"  # noqa: EM102
            )
        if child.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise _ProbeBoundaryError(  # noqa: TRY003
                f"artifact size limit exceeded: {child.name}"  # noqa: EM102
            )
        artifacts.append(child.read_bytes())
    return tuple(artifacts)


def _argv_sha256(argv: tuple[str, ...]) -> str:
    return sha256_bytes(json.dumps(argv, separators=(",", ":")).encode())
