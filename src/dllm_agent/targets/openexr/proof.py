"""Exact-input OpenEXR ASan proof runner.

Module-size opt-out ``# noqa: SIZE_OK`` keeps Task 4 proof behavior in its target module.
"""

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.openexr_one_day.proof_batch import proved_cause_id
from dllm_agent.targets.openexr.proof_execution import (
    OPENEXR_PROOF_BINARY,
    OPENEXR_PROOF_IMAGE_DIGEST,
    OPENEXR_PROOF_SEED,
    OPENEXR_TARGET_COMMIT,
    ProofBoundaryError,
    ProofCommand,
    StructuredMinimizationResult,
    StructuredMinimizationStatus,
    argv_sha256,
    asan_signature,
    crash_signal,
    log_bytes,
    reject_boundary_process,
    run_command,
    run_exact,
    structured_minimize,
    try_minimized,
)

type ProofStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer"]

__all__ = [
    "OPENEXR_PROOF_IMAGE_DIGEST",
    "OPENEXR_PROOF_SEED",
    "OPENEXR_TARGET_COMMIT",
    "OpenExrProofRunner",
    "ProofBoundaryError",
    "ProofCommand",
    "ProofResult",
    "ProofSummaryView",
    "prove_snapshot",
    "proved_cause_id",
]


class ExactProofRequest(Protocol):
    """Structural view of the Task4 exact proof request."""

    @property
    def snapshot(self) -> bytes: ...

    @property
    def snapshot_sha256(self) -> Sha256: ...

    @property
    def duration_seconds(self) -> int: ...

    @property
    def proof_seed(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _ResultEvidence:
    request: ExactProofRequest
    status: ProofStatus
    log: bytes
    argvs: tuple[tuple[str, ...], ...]
    artifacts: tuple[str, ...]
    signature: tuple[str, str] | None
    structured: StructuredMinimizationResult


@dataclass(frozen=True, slots=True)
class _ProvedEvidence:
    request: ExactProofRequest
    log: bytes
    artifacts: tuple[str, ...]
    signature: tuple[str, str]
    replay_argvs: tuple[tuple[str, ...], tuple[str, ...]]
    structured: StructuredMinimizationResult
    minimized: tuple[bytes, bytes, tuple[str, ...], tuple[str, ...]] | None


class ProofSummaryView(FrozenModel):
    """Task4-compatible proof summary fields."""

    oracle_status: ProofStatus
    input_sha256: Sha256 | None
    proof_log_sha256: Sha256 | None


class ProofResult(FrozenModel):
    """Exact proof evidence without vulnerability claims."""

    oracle_status: ProofStatus
    input_bytes: bytes
    input_sha256: Sha256
    proof_log: bytes
    proof_log_sha256: Sha256
    sanitizer: Literal["AddressSanitizer"] | None
    sanitizer_class: str | None
    root_signature: str | None
    run_argv_sha256: tuple[Sha256, ...]
    run_argv_bytes: tuple[bytes, ...]
    image_digest: Literal["sha256:7591279f51256ef5e80341c788d31145ecf1dd0f7a64db7f0cdeaba527ff5add"]
    target_commit: Literal["777d231a179de1711d2a942810d9559216ab3f4c"]
    proof_binary: Literal["/opt/dllm/bin/poc_probe-asan"]
    proof_seed: int = Field(ge=0)
    structured_minimization_status: StructuredMinimizationStatus
    structured_minimization_log_sha256: Sha256
    structured_minimized_input_sha256: Sha256 | None = None
    minimized_input_bytes: bytes | None = None
    minimized_input_sha256: Sha256 | None = None
    minimization_log_sha256: Sha256 | None = None
    crash_artifact_sha256: tuple[Sha256, ...] = ()

    @property
    def result_bytes(self) -> bytes:
        """Return canonical result metadata bytes without embedded raw evidence blobs."""
        payload = self.model_dump(
            mode="json",
            exclude={"input_bytes", "proof_log", "minimized_input_bytes", "run_argv_bytes"},
        )
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()

    def summary(self) -> ProofSummaryView:  # noqa: D102
        return ProofSummaryView(
            oracle_status=self.oracle_status,
            input_sha256=self.input_sha256,
            proof_log_sha256=self.proof_log_sha256,
        )


class OpenExrProofRunner:
    """Production Task4 proof runner for exact OpenEXR snapshot bytes."""

    def __init__(self, command: ProofCommand | None = None) -> None:
        """Create a runner using the production Docker command by default."""
        self._command: ProofCommand = run_command if command is None else command

    def run(self, request: ExactProofRequest) -> ProofResult:
        """Run proof and return rich evidence with a summary compatibility view."""
        return prove_snapshot(request, self._command)


def prove_snapshot(request: ExactProofRequest, runner: ProofCommand | None = None) -> ProofResult:
    """Run exact-byte ASan replay twice and retain only consistent evidence."""
    _check_request(request)
    command = run_command if runner is None else runner
    first_seed, second_seed = _replay_seeds(request.proof_seed)
    first, first_argv, first_artifacts = run_exact(
        request.snapshot, request.duration_seconds, first_seed, command
    )
    reject_boundary_process(first)
    first_log = log_bytes(first)
    first_signature = asan_signature(first_log)
    second, second_argv, second_artifacts = run_exact(
        request.snapshot, request.duration_seconds, second_seed, command
    )
    reject_boundary_process(second)
    second_log = log_bytes(second)
    second_signature = asan_signature(second_log)
    log = first_log + second_log
    replay_argvs = (first_argv, second_argv)
    artifacts = (*first_artifacts, *second_artifacts)
    unavailable = StructuredMinimizationResult(
        status="unavailable",
        input_bytes=request.snapshot,
        input_sha256=request.snapshot_sha256,
        log=b"structured-unavailable\n",
        argvs=(),
    )
    if first_signature is None:
        return _result(
            _ResultEvidence(
                request=request,
                status=(
                    "crash_without_sanitizer"
                    if crash_signal(first, first_log) and crash_signal(second, second_log)
                    else "not_reproduced"
                ),
                log=log,
                argvs=replay_argvs,
                artifacts=artifacts,
                signature=None,
                structured=unavailable,
            )
        )
    if second_signature != first_signature:
        return _result(
            _ResultEvidence(
                request=request,
                status="not_reproduced",
                log=log,
                argvs=replay_argvs,
                artifacts=artifacts,
                signature=first_signature,
                structured=unavailable,
            )
        )
    if not " ".join(first_signature[1].split()):
        return _result(
            _ResultEvidence(
                request=request,
                status="not_reproduced",
                log=log,
                argvs=replay_argvs,
                artifacts=artifacts,
                signature=first_signature,
                structured=unavailable,
            )
        )
    structured = structured_minimize(
        request.snapshot, request.duration_seconds, request.proof_seed, command, first_signature
    )
    minimized = try_minimized(
        structured.input_bytes,
        request.duration_seconds,
        request.proof_seed,
        command,
        first_signature,
    )
    return _proved_result(
        _ProvedEvidence(
            request=request,
            log=log,
            artifacts=artifacts,
            signature=first_signature,
            replay_argvs=replay_argvs,
            structured=structured,
            minimized=minimized,
        )
    )


def _check_request(request: ExactProofRequest) -> None:
    if sha256_bytes(request.snapshot) != request.snapshot_sha256:
        detail = "snapshot sha256 mismatch"
        raise ProofBoundaryError(detail)
    if request.proof_seed < 0:
        detail = "proof seed must be nonnegative"
        raise ProofBoundaryError(detail)


def _replay_seeds(proof_seed: int) -> tuple[int, int]:
    return proof_seed, proof_seed + 1


def _result(evidence: _ResultEvidence) -> ProofResult:
    return ProofResult(
        oracle_status=evidence.status,
        input_bytes=evidence.request.snapshot,
        input_sha256=evidence.request.snapshot_sha256,
        proof_log=evidence.log,
        proof_log_sha256=sha256_bytes(evidence.log),
        sanitizer="AddressSanitizer" if evidence.signature is not None else None,
        sanitizer_class=evidence.signature[0] if evidence.signature is not None else None,
        root_signature=(
            evidence.signature[1]
            if evidence.signature is not None and evidence.status == "proved"
            else None
        ),
        run_argv_sha256=tuple(argv_sha256(argv) for argv in evidence.argvs),
        run_argv_bytes=tuple(_argv_bytes(argv) for argv in evidence.argvs),
        image_digest=OPENEXR_PROOF_IMAGE_DIGEST,
        target_commit=OPENEXR_TARGET_COMMIT,
        proof_binary=OPENEXR_PROOF_BINARY,
        proof_seed=evidence.request.proof_seed,
        structured_minimization_status=evidence.structured.status,
        structured_minimization_log_sha256=sha256_bytes(evidence.structured.log),
        structured_minimized_input_sha256=(
            evidence.structured.input_sha256 if evidence.structured.status == "reduced" else None
        ),
        crash_artifact_sha256=evidence.artifacts,
    )


def _proved_result(evidence: _ProvedEvidence) -> ProofResult:
    if evidence.minimized is None:
        return _result(
            _ResultEvidence(
                request=evidence.request,
                status="proved",
                log=evidence.log,
                argvs=(*evidence.replay_argvs, *evidence.structured.argvs),
                artifacts=evidence.artifacts,
                signature=evidence.signature,
                structured=evidence.structured,
            )
        )
    minimized_bytes, minimization_log, minimize_argv, rerun_argv = evidence.minimized
    return _result(
        _ResultEvidence(
            request=evidence.request,
            status="proved",
            log=evidence.log,
            argvs=(*evidence.replay_argvs, *evidence.structured.argvs, minimize_argv, rerun_argv),
            artifacts=evidence.artifacts,
            signature=evidence.signature,
            structured=evidence.structured,
        )
    ).model_copy(
        update={
            "minimized_input_bytes": minimized_bytes,
            "minimized_input_sha256": sha256_bytes(minimized_bytes),
            "minimization_log_sha256": sha256_bytes(minimization_log),
        }
    )


def _argv_bytes(argv: tuple[str, ...]) -> bytes:
    return json.dumps(argv, separators=(",", ":")).encode()
