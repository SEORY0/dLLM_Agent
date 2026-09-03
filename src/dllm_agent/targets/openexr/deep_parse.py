"""Hardened Docker replay for byte-exact OpenEXR deep-parse evidence.

# noqa: SIZE_OK -- provenance, process execution, and evidence are one trust boundary.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, Literal, Self, final, override

from pydantic import Field, ValidationError, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.post_search import (
    DeepParseEvidenceArtifact,
    DeepParseRequest,
    DeepParseResult,
    DeepParseRunEvidence,
    DeepParseVerdict,
    PostSearchEvidenceError,
    persist_deep_parse_evidence,
    validate_deep_parse_result_evidence,
)

type DeepParseCommand = Callable[[tuple[str, ...], float], CompletedProcess[bytes]]
type InfrastructureFailureKind = Literal["provenance", "timeout", "docker", "sanitizer", "checker"]

_CHECKER_PATH: Final = "/opt/dllm/bin/openexr-deep-parse-check"
_MANIFEST_PATH: Final = "/opt/dllm/evidence/build-manifest.json"
_SEARCH_BINARY_PATH: Final = "/opt/dllm/bin/poc_probe-asan"
_COVERAGE_BINARY_PATH: Final = "/opt/dllm/bin/poc_probe-coverage"
_MAX_INPUT_BYTES: Final = 1_048_576
_MAX_OUTPUT_BYTES: Final = 8_388_608
_SHA256_HEX_LENGTH: Final = 64
_TMPFS: Final = "/tmp:rw,noexec,nosuid,nodev,size=64m"  # noqa: S108
_ASAN_OPTIONS: Final = "detect_leaks=0:abort_on_error=1:symbolize=1"
_UBSAN_OPTIONS: Final = "halt_on_error=1:print_stacktrace=1"
_BUILD_PATHS: Final = (
    _SEARCH_BINARY_PATH,
    _COVERAGE_BINARY_PATH,
    _CHECKER_PATH,
    "/opt/dllm/runtime/deep_parse_check.cc",
    "/opt/dllm/evidence/openexr-asan-compile-commands.json",
    "/opt/dllm/evidence/openexr-coverage-compile-commands.json",
    "/opt/dllm/evidence/runtime-asan-compile-commands.json",
    "/opt/dllm/evidence/runtime-coverage-compile-commands.json",
)
_SANITIZER_MARKERS: Final = (
    b"addresssanitizer",
    b"undefinedbehaviorsanitizer",
    b"runtime error:",
)
_PROVENANCE_FAILURE: Final[InfrastructureFailureKind] = "provenance"
_TIMEOUT_FAILURE: Final[InfrastructureFailureKind] = "timeout"
_DOCKER_FAILURE: Final[InfrastructureFailureKind] = "docker"
_SANITIZER_FAILURE: Final[InfrastructureFailureKind] = "sanitizer"
_CHECKER_FAILURE: Final[InfrastructureFailureKind] = "checker"


class BuildFileEvidence(FrozenModel):
    """Hash-bound file installed in the immutable runtime image."""

    path: str = Field(pattern=r"^/opt/dllm/")
    sha256: Sha256


class OpenExrRuntimeBuildManifest(FrozenModel):
    """Canonical build provenance emitted by the pinned runtime image."""

    schema_version: Literal["dllm-openexr-runtime-build/v1"]
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    deep_parse_checker_sha256: Sha256
    search_sanitizers: tuple[Literal["address"], Literal["undefined"]]
    undefined_recovery: Literal[False]
    coverage_instrumentation: tuple[
        Literal["fuzzer-no-link"],
        Literal["profile-instr-generate"],
        Literal["coverage-mapping"],
    ]
    files: tuple[BuildFileEvidence, ...] = Field(
        min_length=len(_BUILD_PATHS), max_length=len(_BUILD_PATHS)
    )

    @model_validator(mode="after")
    def _canonical_paths(self) -> Self:
        if tuple(item.path for item in self.files) != _BUILD_PATHS:
            message = "deep-parse build manifest file set is not canonical"
            raise ValueError(message)
        return self


class DeepParseInfrastructureError(RuntimeError):
    """Typed failure for Docker, sanitizer, checker, or provenance faults."""

    kind: InfrastructureFailureKind
    reason: str
    evidence_root: Path
    evidence: DeepParseRunEvidence | None

    def __init__(
        self,
        kind: InfrastructureFailureKind,
        reason: str,
        evidence_root: Path,
        evidence: DeepParseRunEvidence | None = None,
    ) -> None:
        """Store the failure classification and any persisted run evidence."""
        self.kind = kind
        self.reason = reason
        self.evidence_root = evidence_root
        self.evidence = evidence
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return f"{self.kind}: {self.reason}"


@dataclass(frozen=True, slots=True)
class _RunCapture:
    """Exact bytes and outcome captured for one checker attempt."""

    index: int
    input_evidence: DeepParseEvidenceArtifact
    argv: tuple[str, ...]
    stdout: bytes
    stderr: bytes
    return_code: int | None
    failure: Literal["timeout", "docker"] | None


@dataclass(frozen=True, slots=True)
class _ValidatedInput:
    """One final-corpus path whose bounded bytes match its request hash."""

    path: Path
    sha256: Sha256
    content: bytes


@final
class DockerDeepParseReplayer:
    """Replay the final corpus with one pinned, offline checker process per input."""

    def __init__(
        self,
        checker_binary_sha256: Sha256,
        runner: DeepParseCommand | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Bind the shell-free process boundary and per-input timeout."""
        if timeout_seconds <= 0:
            reason = "deep-parse timeout must be positive"
            raise _evidence_failure(reason)
        self._runner = _run_command if runner is None else runner
        self._timeout_seconds = timeout_seconds
        self.checker_binary_sha256 = checker_binary_sha256

    def replay(self, request: DeepParseRequest) -> DeepParseResult:
        """Validate provenance, replay every exact input, and persist byte evidence."""
        evidence_root = request.artifact_root / "post-search" / "deep-parse-evidence"
        _prepare_evidence_root(evidence_root)
        if request.checker_binary_sha256 != self.checker_binary_sha256:
            reason = "configured deep-parse checker hash mismatch"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        inputs = _validated_inputs(request)
        manifest_evidence = self._validate_provenance(request, evidence_root)
        verdicts = tuple(
            self._replay_one(request, evidence_root, index, relative_path)
            for index, relative_path in enumerate(inputs)
        )
        result = DeepParseResult(
            replay_kind="docker",
            evidence_root=evidence_root,
            build_manifest_evidence=manifest_evidence,
            verdicts=verdicts,
            success_count=sum(verdict.parse_success for verdict in verdicts),
            total_count=len(verdicts),
        )
        validate_deep_parse_result_evidence(result)
        return result

    def _validate_provenance(
        self, request: DeepParseRequest, evidence_root: Path
    ) -> DeepParseEvidenceArtifact:
        inspect = self._required(
            ("docker", "image", "inspect", "--format={{.Id}}", request.image_digest),
            evidence_root,
        )
        if inspect.stdout.strip() != request.image_digest.encode() or inspect.stderr:
            reason = "Docker image ID mismatch"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        manifest_process = self._required(
            (*_docker_prefix(request), "/bin/cat", _MANIFEST_PATH), evidence_root
        )
        manifest_bytes = _bounded_output(manifest_process.stdout, "build manifest")
        manifest_evidence = persist_deep_parse_evidence(
            evidence_root, Path("provenance/build-manifest.json"), manifest_bytes
        )
        if sha256_bytes(manifest_bytes) != request.build_manifest_sha256:
            reason = "build manifest hash mismatch"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        manifest = _parse_manifest(manifest_bytes, evidence_root)
        expected_hashes = {item.path: item.sha256 for item in manifest.files}
        if (
            manifest.target_commit != request.target_commit
            or manifest.deep_parse_checker_sha256 != request.checker_binary_sha256
            or expected_hashes[_CHECKER_PATH] != request.checker_binary_sha256
        ):
            reason = "requested build provenance mismatch"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        hashes = self._required(
            (*_docker_prefix(request), "/usr/bin/sha256sum", *_BUILD_PATHS), evidence_root
        )
        if hashes.stderr or _parse_hashes(hashes.stdout, evidence_root) != expected_hashes:
            reason = "runtime build file hash mismatch"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        return manifest_evidence

    def _required(self, argv: tuple[str, ...], evidence_root: Path) -> CompletedProcess[bytes]:
        try:
            process = self._runner(argv, self._timeout_seconds)
        except subprocess.TimeoutExpired as error:
            reason = "provenance command timed out"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root) from error
        except OSError as error:
            reason = f"provenance command failed: {error}"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root) from error
        if process.returncode != 0:
            reason = f"provenance command exited {process.returncode}"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        return process

    def _replay_one(
        self, request: DeepParseRequest, evidence_root: Path, index: int, item: _ValidatedInput
    ) -> DeepParseVerdict:
        input_evidence = persist_deep_parse_evidence(
            evidence_root, Path("runs") / f"{index:06d}" / "input.exr", item.content
        )
        staged_path = evidence_root / input_evidence.relative_path
        container_path = staged_path.relative_to(request.artifact_root)
        argv = (
            *_docker_prefix(request, mount_root=True),
            _CHECKER_PATH,
            f"/work/{container_path.as_posix()}",
        )
        try:
            process = self._runner(argv, self._timeout_seconds)
        except subprocess.TimeoutExpired as error:
            evidence = _persist_run(
                evidence_root,
                _RunCapture(
                    index=index,
                    input_evidence=input_evidence,
                    argv=argv,
                    stdout=_exception_bytes(error.stdout),
                    stderr=_exception_bytes(error.stderr),
                    return_code=None,
                    failure="timeout",
                ),
            )
            reason = "deep-parse checker timed out"
            raise _failure(_TIMEOUT_FAILURE, reason, evidence_root, evidence) from error
        except OSError as error:
            evidence = _persist_run(
                evidence_root,
                _RunCapture(
                    index=index,
                    input_evidence=input_evidence,
                    argv=argv,
                    stdout=b"",
                    stderr=str(error).encode(),
                    return_code=None,
                    failure="docker",
                ),
            )
            reason = "deep-parse Docker invocation failed"
            raise _failure(_DOCKER_FAILURE, reason, evidence_root, evidence) from error
        stdout = _bounded_output(process.stdout, "checker stdout")
        stderr = _bounded_output(process.stderr, "checker stderr")
        evidence = _persist_run(
            evidence_root,
            _RunCapture(
                index=index,
                input_evidence=input_evidence,
                argv=argv,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                failure=None,
            ),
        )
        if _has_sanitizer(stdout + stderr):
            reason = "sanitizer output during deep parse"
            raise _failure(_SANITIZER_FAILURE, reason, evidence_root, evidence)
        if process.returncode not in (0, 1):
            reason = f"deep-parse checker exited {process.returncode}"
            raise _failure(
                _CHECKER_FAILURE,
                reason,
                evidence_root,
                evidence,
            )
        return DeepParseVerdict(
            input_evidence=evidence.input_evidence,
            argv_evidence=evidence.argv_evidence,
            stdout_evidence=evidence.stdout_evidence,
            stderr_evidence=evidence.stderr_evidence,
            exit_evidence=evidence.exit_evidence,
            input_path=item.path,
            input_sha256=item.sha256,
            size_bytes=len(item.content),
            parse_success=process.returncode == 0,
        )


def _run_command(argv: tuple[str, ...], timeout: float) -> CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, check=False, timeout=timeout)  # noqa: S603


def _docker_prefix(request: DeepParseRequest, *, mount_root: bool = False) -> tuple[str, ...]:
    mount = () if not mount_root else ("-v", f"{request.artifact_root}:/work:ro", "-w", "/work")
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
        "64",
        "--cpus",
        "1",
        "--memory",
        "2048m",
        "--tmpfs",
        _TMPFS,
        "--user",
        "65534:65534",
        "--env",
        f"ASAN_OPTIONS={_ASAN_OPTIONS}",
        "--env",
        f"UBSAN_OPTIONS={_UBSAN_OPTIONS}",
        *mount,
        "--pull",
        "never",
        request.image_digest,
    )


def _validated_inputs(request: DeepParseRequest) -> tuple[_ValidatedInput, ...]:
    root = request.artifact_root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        reason = "unsafe deep-parse artifact root"
        raise _evidence_failure(reason)
    relative_paths: list[Path] = []
    validated: list[_ValidatedInput] = []
    for path, expected in zip(request.input_paths, request.input_sha256, strict=True):
        try:
            relative = path.relative_to(root)
            _ = path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            reason = "deep-parse input escapes artifact root"
            raise _evidence_failure(reason) from error
        if _path_has_symlink(root, relative) or not path.is_file():
            reason = "deep-parse input is not a regular non-symlink file"
            raise _evidence_failure(reason)
        content = path.read_bytes()
        if not content or len(content) > _MAX_INPUT_BYTES or sha256_bytes(content) != expected:
            reason = "deep-parse input hash or size mismatch"
            raise _evidence_failure(reason)
        relative_paths.append(relative)
        validated.append(_ValidatedInput(path=path, sha256=expected, content=content))
    if len(set(relative_paths)) != len(relative_paths):
        reason = "deep-parse input paths must be unique"
        raise _evidence_failure(reason)
    return tuple(validated)


def _path_has_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _prepare_evidence_root(root: Path) -> None:
    if root.parent.is_symlink() or root.is_symlink() or (root.exists() and not root.is_dir()):
        reason = "unsafe deep-parse evidence root"
        raise _evidence_failure(reason)
    root.mkdir(parents=True, exist_ok=True)


def _parse_manifest(raw: bytes, evidence_root: Path) -> OpenExrRuntimeBuildManifest:
    try:
        manifest = OpenExrRuntimeBuildManifest.model_validate_json(raw)
    except ValidationError as error:
        reason = "invalid build manifest"
        raise _failure(_PROVENANCE_FAILURE, reason, evidence_root) from error
    if raw != canonical_json_bytes(manifest.model_dump(mode="json")):
        reason = "noncanonical build manifest"
        raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
    return manifest


def _parse_hashes(raw: bytes, evidence_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        reason = "invalid build hash output"
        raise _failure(_PROVENANCE_FAILURE, reason, evidence_root) from error
    for line in lines:
        digest, separator, path = line.partition("  ")
        if separator != "  " or len(digest) != _SHA256_HEX_LENGTH or path not in _BUILD_PATHS:
            reason = "invalid build hash output"
            raise _failure(_PROVENANCE_FAILURE, reason, evidence_root)
        result[path] = digest
    return result


def _persist_run(root: Path, capture: _RunCapture) -> DeepParseRunEvidence:
    prefix = Path("runs") / f"{capture.index:06d}"
    exit_payload = canonical_json_bytes(
        {"return_code": capture.return_code}
        if capture.failure is None
        else {"failure": capture.failure, "return_code": capture.return_code}
    )
    return DeepParseRunEvidence(
        input_evidence=capture.input_evidence,
        argv_evidence=persist_deep_parse_evidence(
            root, prefix / "argv.json", canonical_json_bytes(capture.argv)
        ),
        stdout_evidence=persist_deep_parse_evidence(root, prefix / "stdout.bin", capture.stdout),
        stderr_evidence=persist_deep_parse_evidence(root, prefix / "stderr.bin", capture.stderr),
        exit_evidence=persist_deep_parse_evidence(root, prefix / "exit.json", exit_payload),
    )


def _exception_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, str):
        return value.encode()
    return value


def _bounded_output(content: bytes, label: str) -> bytes:
    if len(content) > _MAX_OUTPUT_BYTES:
        reason = f"deep-parse {label} exceeds byte limit"
        raise _evidence_failure(reason)
    return content


def _has_sanitizer(content: bytes) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in _SANITIZER_MARKERS)


def _failure(
    kind: InfrastructureFailureKind,
    reason: str,
    evidence_root: Path,
    evidence: DeepParseRunEvidence | None = None,
) -> DeepParseInfrastructureError:
    return DeepParseInfrastructureError(kind, reason, evidence_root, evidence)


def _evidence_failure(reason: str) -> PostSearchEvidenceError:
    return PostSearchEvidenceError(reason)
