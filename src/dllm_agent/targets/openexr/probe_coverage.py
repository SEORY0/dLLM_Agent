"""Promoted OpenEXR probe LLVM coverage replay.

Module-size opt-out ``# noqa: SIZE_OK`` keeps historical APIs and the required replayer together.
"""

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from subprocess import CompletedProcess
from typing import ClassVar, Final, Literal, final

from pydantic import ConfigDict, Field, ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    CoverageSeriesRequest,
    CoverageStepRequest,
    CoverageStepResult,
    TargetCoverageCounts,
)
from dllm_agent.experiments.openexr_one_day.evidence_backing import (
    ArtifactBackingEvidence,
    EvidenceBackingError,
    persist_backing_bytes,
)

type ProbeCommand = Callable[[tuple[str, ...], Mapping[str, str], float], CompletedProcess[bytes]]

_LLVM_PROFILE_FILE: Final = "/work/coverage.profraw"
_PROFILE_DATA_FILE: Final = "/work/coverage.profdata"
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108 - Docker tmpfs target.
_TIMEOUT_EXIT_CODE: Final = 124
_MAX_SNAPSHOT_BYTES: Final = 1_048_576
_MAX_OUTPUT_BYTES: Final = 8_388_608
_MAX_PROFILE_BYTES: Final = 67_108_864
# fmt: off
_DOCKER_HEAD: Final = ("docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true", "--pids-limit", "128", "--cpus", "1", "--user", "65534:65534")  # noqa: E501
# fmt: on
_COVERAGE_FIELDS: Final = 4


class CoverageProbeInput(FrozenModel):  # noqa: D101
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    snapshot: bytes = Field(min_length=1, max_length=_MAX_SNAPSHOT_BYTES)
    snapshot_sha256: Sha256
    fuzzer_seed: int = Field(ge=0)
    duration_seconds: int = Field(gt=0, le=30)
    timeout_seconds: int = Field(gt=0)
    rss_limit_mb: int = Field(ge=64)
    max_len: int = Field(gt=0, le=_MAX_SNAPSHOT_BYTES)


class CoverageTotals(FrozenModel):  # noqa: D101
    regions: float
    functions: float
    lines: float
    branches: float


class PromotedCoverageResult(FrozenModel):  # noqa: D101
    snapshot_sha256: Sha256
    run_status: Literal["completed", "failed", "timed_out"]
    merge_status: Literal["succeeded", "failed", "timed_out", "skipped"]
    report_status: Literal["succeeded", "failed", "timed_out", "skipped"]
    run_log_sha256: Sha256
    merge_log_sha256: Sha256
    report_log_sha256: Sha256
    run_argv_sha256: Sha256
    merge_argv_sha256: Sha256
    report_argv_sha256: Sha256
    profraw_sha256: Sha256 | None
    profdata_sha256: Sha256 | None
    coverage: CoverageTotals | None


class CoverageBoundaryError(RuntimeError):
    """Raised when external coverage evidence is malformed or unsafe."""

    reason: str

    def __init__(self, reason: str) -> None:  # noqa: D107
        self.reason = reason
        super().__init__(reason)


_CoverageBoundaryError = CoverageBoundaryError


class ProfileMergeRequest(FrozenModel):  # noqa: D101
    prior_profdata_path: Path | None
    new_profraw_paths: tuple[Path, ...] = Field(min_length=1)
    output_profdata_path: Path


class _LlvmExportModel(FrozenModel):
    """Parse-only view of `llvm-cov export` JSON.

    Real exports carry fields we don't use (`percent`, `notcovered`, `functions`,
    `instantiations`, `totals`, `type`, `version`, and `mcdc` on newer LLVM), and
    the set grows between LLVM releases -- so unknown keys are ignored. The fields
    we *do* consume stay strictly required integers, which is what keeps the
    percentage-only / missing-data rejections intact.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")


class _Metric(_LlvmExportModel):
    covered: int = Field(ge=0)
    count: int = Field(ge=0)


class _Summary(_LlvmExportModel):
    branches: _Metric
    regions: _Metric
    lines: _Metric


class _ExportFile(_LlvmExportModel):
    filename: str
    summary: _Summary


class _ExportData(_LlvmExportModel):
    files: tuple[_ExportFile, ...]


class _Export(_LlvmExportModel):
    data: tuple[_ExportData, ...] = Field(min_length=1)


def parse_target_coverage_export(raw: bytes) -> TargetCoverageCounts:
    """Parse `llvm-cov export -summary-only` into target-only integer counts."""
    try:
        report = _Export.model_validate_json(raw)
    except ValidationError as error:
        reason = "coverage export requires integer count fields"
        raise CoverageBoundaryError(reason) from error
    prefix = Path("/opt/openexr-src/src/lib/OpenEXR")
    files = tuple(
        item
        for data in report.data
        for item in data.files
        if _is_target_file(prefix, Path(item.filename))
    )
    if not files:
        reason = "coverage export contains no canonical target file"
        raise CoverageBoundaryError(reason)
    # fmt: off
    counts = TargetCoverageCounts(target_file_count=len(files), branches_covered=sum(item.summary.branches.covered for item in files), branches_total=sum(item.summary.branches.count for item in files), regions_covered=sum(item.summary.regions.covered for item in files), regions_total=sum(item.summary.regions.count for item in files), lines_covered=sum(item.summary.lines.covered for item in files), lines_total=sum(item.summary.lines.count for item in files))  # noqa: E501
    # fmt: on
    if counts.branches_total <= 0:
        reason = "target branch total must be positive"
        raise CoverageBoundaryError(reason)
    return counts


def build_incremental_profile_merge_argv(request: ProfileMergeRequest) -> tuple[str, ...]:
    """Build fixed argv merging new raw profiles with the prior cumulative profile."""
    inputs = () if request.prior_profdata_path is None else (str(request.prior_profdata_path),)
    # fmt: off
    return ("/usr/local/bin/llvm-profdata", "merge", "-sparse", *inputs, *(str(path) for path in request.new_profraw_paths), "-o", str(request.output_profdata_path))  # noqa: E501
    # fmt: on


@final
class DockerCoverageReplayer:
    """Execute incremental target-only coverage replay in the pinned Docker image."""

    def __init__(self, request: CoverageSeriesRequest, runner: ProbeCommand | None = None) -> None:
        """Bind replay evidence to one immutable series request."""
        self._request = request
        self._runner = _run_coverage_command if runner is None else runner
        self._root = request.corpus_index.artifact_root / "coverage"
        if self._root.is_symlink() or (self._root.exists() and not self._root.is_dir()):
            raise CoverageBoundaryError(reason="unsafe coverage evidence root")
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(0o777)

    def replay(self, request: CoverageStepRequest) -> CoverageStepResult:
        """Replay new inputs and persist one cumulative profile/report step."""
        current = self._root / "current.profdata"
        _validate_prior_profile(current, request.prior_profdata_sha256)
        step_root = self._root / f"t{request.scheduled_elapsed_seconds:05d}"
        try:
            step_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise CoverageBoundaryError(reason="coverage step evidence already exists") from error
        step_root.chmod(0o777)
        run_argvs: list[tuple[str, ...]] = []
        run_argv_evidence: list[ArtifactBackingEvidence] = []
        raw_hashes: list[str] = []
        raw_evidence: list[ArtifactBackingEvidence] = []
        raw_paths: list[Path] = []
        for index, (input_path, digest) in enumerate(
            zip(request.new_input_paths, request.new_input_sha256, strict=True)
        ):
            _validate_replay_input(self._request, input_path, digest)
            raw_path = step_root / f"input-{index:04d}.profraw"
            argv = _coverage_run_argv(self._request, input_path, raw_path)
            run_argv_evidence.append(
                _persist_evidence(
                    self._request,
                    step_root / f"run-{index:04d}.argv.json",
                    _argv_bytes(argv),
                )
            )
            process = _run_required(self._runner, argv, "coverage replay")
            _persist_exact(step_root / f"run-{index:04d}.log", _log_bytes(process))
            raw_bytes = _profile_bytes(raw_path)
            raw_hashes.append(sha256_bytes(raw_bytes))
            raw_evidence.append(_persist_evidence(self._request, raw_path, raw_bytes))
            raw_paths.append(raw_path)
            run_argvs.append(argv)
        profile_path = step_root / "profile.profdata"
        merge_argv = _coverage_merge_argv(
            self._request,
            current if request.prior_profdata_sha256 is not None else None,
            tuple(raw_paths),
            profile_path,
        )
        merge_argv_evidence = _persist_evidence(
            self._request,
            step_root / "merge.argv.json",
            _argv_bytes(merge_argv),
        )
        merge = _run_required(self._runner, merge_argv, "coverage profile merge")
        _persist_exact(step_root / "merge.log", _log_bytes(merge))
        profile_bytes = _profile_bytes(profile_path)
        report_argv = _coverage_report_argv(self._request, profile_path)
        report_argv_evidence = _persist_evidence(
            self._request,
            step_root / "report.argv.json",
            _argv_bytes(report_argv),
        )
        report = _run_required(self._runner, report_argv, "coverage export")
        report_bytes = _bounded_report(report.stdout)
        counts = parse_target_coverage_export(report_bytes)
        _persist_exact(step_root / "report.json", report_bytes)
        _persist_exact(step_root / "report.log", report.stderr)
        _replace_current(current, profile_bytes)
        profile_evidence = _persist_evidence(self._request, profile_path, profile_bytes)
        report_evidence = _persist_evidence(self._request, step_root / "report.json", report_bytes)
        return CoverageStepResult(
            counts=counts,
            run_argv_sha256=tuple(_argv_sha256(item) for item in run_argvs),
            profraw_sha256=tuple(raw_hashes),
            merge_argv_sha256=_argv_sha256(merge_argv),
            profdata_sha256=sha256_bytes(profile_bytes),
            report_argv_sha256=_argv_sha256(report_argv),
            report_sha256=sha256_bytes(report_bytes),
            run_argv_evidence=tuple(run_argv_evidence),
            profraw_evidence=tuple(raw_evidence),
            merge_argv_evidence=merge_argv_evidence,
            profdata_evidence=profile_evidence,
            report_argv_evidence=report_argv_evidence,
            report_evidence=report_evidence,
        )


def _is_target_file(prefix: Path, path: Path) -> bool:
    if not path.is_absolute() or ".." in path.parts:
        return False
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return False
    return relative != Path() and len(relative.parts) > 0


def _run_coverage_command(
    argv: tuple[str, ...], environment: Mapping[str, str], timeout: float
) -> CompletedProcess[bytes]:
    _ = environment
    return subprocess.run(  # noqa: S603 - fixed shell-free Docker argv.
        argv, capture_output=True, timeout=timeout, check=False
    )


def _run_required(
    runner: ProbeCommand, argv: tuple[str, ...], label: str
) -> CompletedProcess[bytes]:
    try:
        process = runner(argv, {}, 60.0)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CoverageBoundaryError(reason=f"{label} failed: {error}") from error
    log = _log_bytes(process)
    if process.returncode != 0:
        detail = log.decode(errors="replace").strip()
        raise CoverageBoundaryError(reason=f"{label} exited {process.returncode}: {detail}")
    return process


def _validate_prior_profile(path: Path, expected: str | None) -> None:
    present = path.exists() or path.is_symlink()
    if expected is None:
        if present:
            raise CoverageBoundaryError(reason="unexpected prior profile evidence")
        return
    if not present or _profile_sha(path) != expected:
        raise CoverageBoundaryError(reason="prior profile hash mismatch")


def _validate_replay_input(request: CoverageSeriesRequest, path: Path, expected: str) -> None:
    try:
        _ = path.resolve().relative_to(request.corpus_index.artifact_root.resolve())
    except ValueError as error:
        raise CoverageBoundaryError(reason="coverage input escapes artifact root") from error
    if path.is_symlink() or not path.is_file():
        raise CoverageBoundaryError(reason="coverage input is not a regular file")
    content = path.read_bytes()
    if not content or len(content) > _MAX_SNAPSHOT_BYTES or sha256_bytes(content) != expected:
        raise CoverageBoundaryError(reason="coverage input hash or size mismatch")


def _coverage_run_argv(
    request: CoverageSeriesRequest, input_path: Path, profile_path: Path
) -> tuple[str, ...]:
    command = (
        "/opt/dllm/bin/poc_probe-coverage",
        _container_path(request, input_path),
        "-runs=1",
    )
    environment = (f"LLVM_PROFILE_FILE={_container_path(request, profile_path)}",)
    return _series_docker_argv(request, command, environment)


def _coverage_merge_argv(
    request: CoverageSeriesRequest,
    prior_path: Path | None,
    raw_paths: tuple[Path, ...],
    output_path: Path,
) -> tuple[str, ...]:
    merge = build_incremental_profile_merge_argv(
        ProfileMergeRequest(
            prior_profdata_path=(
                None if prior_path is None else Path(_container_path(request, prior_path))
            ),
            new_profraw_paths=tuple(Path(_container_path(request, path)) for path in raw_paths),
            output_profdata_path=Path(_container_path(request, output_path)),
        )
    )
    return _series_docker_argv(request, merge, ())


def _coverage_report_argv(request: CoverageSeriesRequest, profile_path: Path) -> tuple[str, ...]:
    command = (
        "/usr/local/bin/llvm-cov",
        "export",
        "-summary-only",
        "/opt/dllm/bin/poc_probe-coverage",
        f"-instr-profile={_container_path(request, profile_path)}",
    )
    return _series_docker_argv(request, command, ())


def _series_docker_argv(
    request: CoverageSeriesRequest,
    command: tuple[str, ...],
    environment: tuple[str, ...],
) -> tuple[str, ...]:
    env_argv = tuple(item for value in environment for item in ("--env", value))
    # fmt: off
    return ("docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true", "--pids-limit", "128", "--cpuset-cpus", str(request.cpu_index), "--memory", "2048m", "--tmpfs", _TMPFS, "--user", "65534:65534", *env_argv, "-v", f"{request.corpus_index.artifact_root}:/work:rw", "-w", "/work", "--pull", "never", request.image_digest, *command)  # noqa: E501
    # fmt: on


def _container_path(request: CoverageSeriesRequest, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(request.corpus_index.artifact_root.resolve())
    except ValueError as error:
        raise CoverageBoundaryError(
            reason="coverage evidence path escapes artifact root"
        ) from error
    return f"/work/{relative.as_posix()}"


def _bounded_report(content: bytes) -> bytes:
    if not content or len(content) > _MAX_OUTPUT_BYTES:
        raise CoverageBoundaryError(reason="coverage report output is empty or too large")
    return content


def _profile_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CoverageBoundaryError(reason="unsafe coverage profile output")
    content = path.read_bytes()
    if not content or len(content) > _MAX_PROFILE_BYTES:
        raise CoverageBoundaryError(reason="coverage profile output is empty or too large")
    return content


def _persist_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise CoverageBoundaryError(reason="coverage evidence content mismatch")
        return
    with path.open("xb") as handle:
        _ = handle.write(content)


def _replace_current(path: Path, content: bytes) -> None:
    staged = path.with_suffix(".profdata.next")
    _persist_exact(staged, content)
    _ = staged.replace(path)


def _persist_evidence(
    request: CoverageSeriesRequest, path: Path, content: bytes
) -> ArtifactBackingEvidence:
    try:
        relative = path.relative_to(request.corpus_index.artifact_root).as_posix()
        return persist_backing_bytes(request.corpus_index.artifact_root, relative, content)
    except (EvidenceBackingError, ValueError) as error:
        raise CoverageBoundaryError(reason=str(error)) from error


def run_promoted_coverage(
    request: CoverageProbeInput, runner: ProbeCommand, root: Path
) -> PromotedCoverageResult:
    """Run the fixed coverage binary and LLVM report commands."""
    _stage_snapshot(root, request.snapshot)
    run_argv = _run_argv(request, root)
    run = runner(run_argv, {}, float(max(request.timeout_seconds, 60)))
    run_log = _log_bytes(run)
    profraw_sha = _profile_sha(root / "coverage.profraw") if run.returncode == 0 else None
    merge_argv = _merge_argv(request, root)
    merge = _skip(merge_argv, "coverage run failed; merge skipped")
    if run.returncode == 0:
        merge = runner(merge_argv, {}, 60.0)
    merge_log = _log_bytes(merge)
    profdata_sha = _profile_sha(root / "coverage.profdata") if merge.returncode == 0 else None
    report_argv = _report_argv(request, root)
    report = _skip(report_argv, "coverage report skipped")
    if merge.returncode == 0:
        report = runner(report_argv, {}, 60.0)
    report_log = _log_bytes(report)
    return PromotedCoverageResult(
        snapshot_sha256=request.snapshot_sha256,
        run_status=_run_status(run),
        merge_status=_step_status(run, merge),
        report_status=_step_status(merge, report),
        run_log_sha256=sha256_bytes(run_log),
        merge_log_sha256=sha256_bytes(merge_log),
        report_log_sha256=sha256_bytes(report_log),
        run_argv_sha256=_argv_sha256(run_argv),
        merge_argv_sha256=_argv_sha256(merge_argv),
        report_argv_sha256=_argv_sha256(report_argv),
        profraw_sha256=profraw_sha,
        profdata_sha256=profdata_sha,
        coverage=_coverage_if_reported(report),
    )


def _stage_snapshot(root: Path, snapshot: bytes) -> None:
    corpus = root / "corpus"
    corpus.mkdir()
    root.chmod(0o777)
    corpus.chmod(0o777)
    seed = corpus / "seed-0000"
    _ = seed.write_bytes(snapshot)
    seed.chmod(0o444)


def _docker_prefix(request: CoverageProbeInput, root: Path) -> tuple[str, ...]:
    return (
        *_DOCKER_HEAD,
        "--memory",
        f"{request.rss_limit_mb}m",
        "--tmpfs",
        _TMPFS,
        "--env",
        f"LLVM_PROFILE_FILE={_LLVM_PROFILE_FILE}",
        "-v",
        f"{root}:/work:rw",
        "-w",
        "/work",
        "--pull",
        "never",
        request.image_id,
    )


def _run_argv(request: CoverageProbeInput, root: Path) -> tuple[str, ...]:
    return (
        *_docker_prefix(request, root),
        "/opt/dllm/bin/poc_probe-coverage",
        "/work/corpus",
        "-runs=0",
        f"-max_total_time={request.duration_seconds}",
        f"-timeout={request.timeout_seconds}",
        f"-rss_limit_mb={request.rss_limit_mb}",
        f"-max_len={request.max_len}",
        f"-seed={request.fuzzer_seed}",
    )


def _merge_argv(request: CoverageProbeInput, root: Path) -> tuple[str, ...]:
    return (
        *_docker_prefix(request, root),
        "/usr/local/bin/llvm-profdata",
        "merge",
        "-sparse",
        _LLVM_PROFILE_FILE,
        "-o",
        _PROFILE_DATA_FILE,
    )


def _report_argv(request: CoverageProbeInput, root: Path) -> tuple[str, ...]:
    return (
        *_docker_prefix(request, root),
        "/usr/local/bin/llvm-cov",
        "report",
        "/opt/dllm/bin/poc_probe-coverage",
        f"-instr-profile={_PROFILE_DATA_FILE}",
        "/opt/openexr-src/src/lib/OpenEXR",
    )


def _profile_sha(path: Path) -> str:
    return sha256_bytes(_profile_bytes(path))


def _coverage_if_reported(report: CompletedProcess[bytes]) -> CoverageTotals | None:
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
    if len(percentages) < _COVERAGE_FIELDS:
        return None
    return CoverageTotals(
        regions=percentages[0],
        functions=percentages[1],
        lines=percentages[2],
        branches=percentages[3],
    )


def _run_status(process: CompletedProcess[bytes]) -> Literal["completed", "failed", "timed_out"]:
    if process.returncode == 0:
        return "completed"
    if process.returncode == _TIMEOUT_EXIT_CODE:
        return "timed_out"
    return "failed"


def _step_status(
    prerequisite: CompletedProcess[bytes], process: CompletedProcess[bytes]
) -> Literal["succeeded", "failed", "timed_out", "skipped"]:
    if prerequisite.returncode != 0:
        return "skipped"
    if process.returncode == 0:
        return "succeeded"
    if process.returncode == _TIMEOUT_EXIT_CODE:
        return "timed_out"
    return "failed"


def _skip(command: tuple[str, ...], message: str) -> CompletedProcess[bytes]:
    return CompletedProcess(command, 125, b"", message.encode())


def _log_bytes(process: CompletedProcess[bytes]) -> bytes:
    log = process.stdout + process.stderr
    if len(log) > _MAX_OUTPUT_BYTES:
        raise _CoverageBoundaryError("coverage output too large")  # noqa: EM101,TRY003
    return log


def _argv_sha256(argv: tuple[str, ...]) -> str:
    return sha256_bytes(_argv_bytes(argv))


def _argv_bytes(argv: tuple[str, ...]) -> bytes:
    return json.dumps(argv, separators=(",", ":")).encode()
