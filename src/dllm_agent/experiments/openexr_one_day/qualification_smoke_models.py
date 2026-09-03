"""Qualification smoke schemas and dependency protocols."""

from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
# ruff: noqa: D101, D102, D103, D107
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Self, assert_never, override

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.models import Arm  # noqa: TC001

SMOKE_DEADLINE_SECONDS = 300
RESULT_NAME = "qualification-smoke-result.json"
WRAPPER_NAME = "qualification-smoke-artifact.json"
BINDINGS_NAME = "qualification-smoke-bindings.json"
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class QualificationSmokeError(Exception):
    """Raised when qualification smoke evidence cannot be trusted."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CandidateBankSurface(Protocol):
    @property
    def selected_bank_sha256(self) -> Sha256: ...


class CandidateBankBuilder(Protocol):
    def build(self, repetition: int) -> CandidateBankSurface: ...


class ModelExecutor(Protocol):
    def prepare_corpus(self, cell_id: str, bank_sha256: Sha256) -> SmokeModelPreparation: ...


class SmokeCellRunner(Protocol):
    def run(self, cell_id: str, deadline_monotonic_seconds: float) -> SmokeCellResult: ...


class SmokeEvidenceLoader(Protocol):
    def load_cell_result(self, cell_id: str) -> SmokeCellResult: ...


class SmokePostSearch(Protocol):
    def coverage_and_proof(self, cell_id: str, result_sha256: Sha256) -> SmokePostSearchOutcome: ...
    def load_outcome(self, cell_id: str) -> SmokePostSearchOutcome: ...


@dataclass(frozen=True, slots=True)
class SmokeExecutionDependencies:
    bank_builder: CandidateBankBuilder
    model_executor: ModelExecutor
    cell_runner: SmokeCellRunner
    evidence_loader: SmokeEvidenceLoader
    post_search: SmokePostSearch


class QualificationSmokeBinding(FrozenModel):
    cell_id: str = Field(min_length=1)
    bank_sha256: Sha256
    model_prep_sha256: Sha256 | None


class QualificationSmokeBindings(FrozenModel):
    schema_version: Literal["dllm-openexr-qualification-smoke-bindings/v1"] = (
        "dllm-openexr-qualification-smoke-bindings/v1"
    )
    plan_sha256: Sha256
    bindings: tuple[QualificationSmokeBinding, ...]
    bindings_sha256: Sha256


class QualificationSmokeCell(FrozenModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    cell_id: str = Field(min_length=1)
    pair_id: int = Field(ge=0, le=1)
    repetition: int = Field(ge=0, le=1)
    arm: Arm
    fuzzer_seed: int = Field(ge=0)
    start_monotonic_seconds: float
    deadline_monotonic_seconds: float
    deadline_seconds: Literal[300]
    artifact_root: Path

    @model_validator(mode="after")
    def _fixed_deadline(self) -> Self:
        if self.deadline_monotonic_seconds - self.start_monotonic_seconds != SMOKE_DEADLINE_SECONDS:
            message = "qualification smoke cell deadline must be exactly 300 seconds"
            raise ValueError(message)
        return self


class QualificationSmokePlan(FrozenModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    schema_version: Literal["dllm-openexr-qualification-smoke-plan/v1"] = (
        "dllm-openexr-qualification-smoke-plan/v1"
    )
    config_sha256: Sha256
    smoke_root: Path
    cells: tuple[QualificationSmokeCell, ...] = Field(min_length=10, max_length=10)

    @property
    def wave_count(self) -> Literal[1]:
        return 1

    @property
    def pair_count(self) -> int:
        return len({cell.pair_id for cell in self.cells})

    @property
    def cell_count(self) -> int:
        return len(self.cells)


class SmokeModelPreparation(FrozenModel):
    cell_id: str = Field(min_length=1)
    preparation_sha256: Sha256


class SmokeCellResult(FrozenModel):
    cell_id: str = Field(min_length=1)
    status: Literal["completed", "failed", "deadline", "restart_limit"]
    attempt_count: int = Field(ge=1)
    deadline_monotonic_seconds: float
    result_sha256: Sha256


class SmokePostSearchOutcome(FrozenModel):
    cell_id: str = Field(min_length=1)
    proof_status: Literal["available", "unavailable"]
    deep_parse_status: Literal["available", "unavailable"]
    coverage_envelope_sha256: Sha256
    proof_envelope_sha256: Sha256
    deep_parse_envelope_sha256: Sha256
    proof_result_sha256: Sha256
    deep_parse_result_sha256: Sha256


class QualificationSmokeCellOutcome(FrozenModel):
    cell_id: str = Field(min_length=1)
    pair_id: int = Field(ge=0, le=1)
    repetition: int = Field(ge=0, le=1)
    arm: Arm
    attempt_count: int = Field(ge=1)
    bank_sha256: Sha256
    model_prep_sha256: Sha256 | None
    cell_result_sha256: Sha256
    coverage_envelope_sha256: Sha256
    proof_envelope_sha256: Sha256
    deep_parse_envelope_sha256: Sha256
    proof_result_sha256: Sha256
    deep_parse_result_sha256: Sha256
    cell_outcome_sha256: Sha256


class QualificationSmokeResult(FrozenModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    schema_version: Literal["dllm-openexr-qualification-smoke-result/v1"] = (
        "dllm-openexr-qualification-smoke-result/v1"
    )
    plan_sha256: Sha256
    cells: tuple[QualificationSmokeCellOutcome, ...]
    cell_result_set_sha256: Sha256
    restart_success_pair_ids: tuple[int, ...]
    result_sha256: Sha256
    passed: bool


class QualificationSmokeArtifact(FrozenModel):
    """Hash-bound wrapper around the raw canonical smoke result."""

    schema_version: Literal["dllm-openexr-qualification-smoke-artifact/v1"] = (
        "dllm-openexr-qualification-smoke-artifact/v1"
    )
    plan_sha256: Sha256
    result: QualificationSmokeResult
    result_sha256: Sha256
    artifact_tree_sha256: Sha256


def load_smoke_result_file(plan: QualificationSmokePlan) -> QualificationSmokeResult:
    try:
        raw = _read_artifact(plan.smoke_root / RESULT_NAME)
        result = QualificationSmokeResult.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise QualificationSmokeError(reason="qualification smoke result is unreadable") from error
    if raw != canonical_json_bytes(result.model_dump(mode="json")):
        raise QualificationSmokeError(reason="qualification smoke result is noncanonical")
    validate_qualification_smoke_result(plan, result)
    return result


def load_smoke_artifact_file(
    plan: QualificationSmokePlan, artifact_path: Path | None = None
) -> QualificationSmokeArtifact:
    path = plan.smoke_root / WRAPPER_NAME if artifact_path is None else artifact_path
    try:
        raw = _read_artifact(path)
        artifact = QualificationSmokeArtifact.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise QualificationSmokeError(reason="qualification smoke artifact unavailable") from error
    if raw != canonical_json_bytes(artifact.model_dump(mode="json")):
        raise QualificationSmokeError(reason="qualification smoke artifact is noncanonical")
    validate_qualification_smoke_artifact(plan, artifact, path)
    return artifact


def load_smoke_bindings_file(plan: QualificationSmokePlan) -> QualificationSmokeBindings:
    try:
        raw = _read_artifact(plan.smoke_root / BINDINGS_NAME)
        bindings = QualificationSmokeBindings.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise QualificationSmokeError(
            reason="qualification smoke bindings are unreadable"
        ) from error
    if raw != canonical_json_bytes(bindings.model_dump(mode="json")):
        raise QualificationSmokeError(reason="qualification smoke bindings are noncanonical")
    validate_smoke_bindings(plan, bindings)
    return bindings


def validate_smoke_bindings(
    plan: QualificationSmokePlan, bindings: QualificationSmokeBindings
) -> None:
    if bindings.plan_sha256 != model_sha256(plan):
        raise QualificationSmokeError(reason="qualification smoke binding plan hash mismatch")
    expected = tuple(cell.cell_id for cell in plan.cells)
    observed = tuple(binding.cell_id for binding in bindings.bindings)
    if observed != expected:
        raise QualificationSmokeError(reason="qualification smoke binding cell set mismatch")
    unsigned = bindings.model_copy(update={"bindings_sha256": "0" * 64})
    payload = canonical_json_bytes(unsigned.model_dump(mode="json", exclude={"bindings_sha256"}))
    if bindings.bindings_sha256 != sha256_bytes(payload):
        raise QualificationSmokeError(reason="qualification smoke binding hash mismatch")
    for cell, binding in zip(plan.cells, bindings.bindings, strict=True):
        validate_binding_for_cell(cell, binding)


def validate_qualification_smoke_artifact(
    plan: QualificationSmokePlan,
    artifact: QualificationSmokeArtifact,
    artifact_path: Path,
) -> None:
    if artifact.plan_sha256 != model_sha256(plan):
        raise QualificationSmokeError(reason="qualification smoke artifact hash mismatch")
    if artifact.result_sha256 != model_sha256(artifact.result):
        raise QualificationSmokeError(reason="qualification smoke artifact hash mismatch")
    if artifact.artifact_tree_sha256 != artifact_tree_sha256(
        artifact_path.parent, exclude=artifact_path
    ):
        raise QualificationSmokeError(reason="qualification smoke artifact hash mismatch")
    validate_qualification_smoke_result(plan, artifact.result)


def validate_qualification_smoke_result(
    plan: QualificationSmokePlan,
    result: QualificationSmokeResult,
) -> None:
    if result.plan_sha256 != model_sha256(plan):
        raise QualificationSmokeError(reason="qualification smoke plan hash mismatch")
    if len(result.cells) != len(plan.cells):
        raise QualificationSmokeError(
            reason="qualification smoke requires exactly 10 smoke cell outcomes"
        )
    expected_cell_ids = tuple(cell.cell_id for cell in plan.cells)
    observed_cell_ids = tuple(cell.cell_id for cell in result.cells)
    if observed_cell_ids != expected_cell_ids:
        raise QualificationSmokeError(reason="qualification smoke cell set mismatch")
    _validate_smoke_cell_outcome_hashes(plan, result)
    cell_payloads = [cell.model_dump(mode="json") for cell in result.cells]
    if result.cell_result_set_sha256 != sha256_bytes(canonical_json_bytes(cell_payloads)):
        raise QualificationSmokeError(reason="qualification smoke result-set hash mismatch")
    expected_restart = tuple(sorted({cell.pair_id for cell in plan.cells}))
    if result.restart_success_pair_ids != expected_restart:
        raise QualificationSmokeError(reason="qualification smoke restart audit mismatch")
    if not result.passed:
        raise QualificationSmokeError(reason="qualification smoke did not pass")
    payload = canonical_json_bytes(result.model_dump(mode="json", exclude={"result_sha256"}))
    if result.result_sha256 != sha256_bytes(payload):
        raise QualificationSmokeError(reason="qualification smoke result hash mismatch")


def _validate_smoke_cell_outcome_hashes(
    plan: QualificationSmokePlan,
    result: QualificationSmokeResult,
) -> None:
    for plan_cell, outcome in zip(plan.cells, result.cells, strict=True):
        if (
            outcome.pair_id != plan_cell.pair_id
            or outcome.repetition != plan_cell.repetition
            or outcome.arm != plan_cell.arm
        ):
            raise QualificationSmokeError(reason="qualification smoke cell plan mismatch")
        payload = canonical_json_bytes(
            outcome.model_dump(mode="json", exclude={"cell_outcome_sha256"})
        )
        if outcome.cell_outcome_sha256 != sha256_bytes(payload):
            raise QualificationSmokeError(reason="qualification smoke cell outcome hash mismatch")


def validate_binding_for_cell(
    cell: QualificationSmokeCell, binding: QualificationSmokeBinding
) -> None:
    match cell.arm:
        case "selective-revision" | "final-only" | "full-regenerate" | "structured-one-shot":
            if binding.model_prep_sha256 is None:
                raise QualificationSmokeError(reason="model prep hash missing")
        case "libfuzzer-only":
            if binding.model_prep_sha256 is not None:
                raise QualificationSmokeError(reason="baseline model prep hash forbidden")
        case unreachable:
            assert_never(unreachable)


def write_canonical_once(path: Path, model: BaseModel) -> None:
    reject_unsafe_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(_JSON_OBJECT.validate_python(model.model_dump(mode="json")))
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise QualificationSmokeError(
                reason="qualification smoke artifact already exists"
            ) from error
        _fsync_directory(path.parent)
    except OSError as error:
        raise QualificationSmokeError(
            reason=f"qualification smoke artifact write failed: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def write_smoke_artifact(
    plan: QualificationSmokePlan,
    result: QualificationSmokeResult,
    artifact_path: Path | None = None,
) -> QualificationSmokeArtifact:
    validate_qualification_smoke_result(plan, result)
    path = plan.smoke_root / WRAPPER_NAME if artifact_path is None else artifact_path
    artifact = QualificationSmokeArtifact(
        plan_sha256=model_sha256(plan),
        result=result,
        result_sha256=model_sha256(result),
        artifact_tree_sha256=artifact_tree_sha256(path.parent, exclude=path),
    )
    write_canonical_once(path, artifact)
    return artifact


def ensure_smoke_artifact(
    plan: QualificationSmokePlan,
    result: QualificationSmokeResult,
    artifact_path: Path | None = None,
) -> QualificationSmokeArtifact:
    path = plan.smoke_root / WRAPPER_NAME if artifact_path is None else artifact_path
    if not path.exists():
        return write_smoke_artifact(plan, result, path)
    artifact = load_smoke_artifact_file(plan, path)
    if artifact.result != result:
        raise QualificationSmokeError(reason="qualification smoke artifact result mismatch")
    return artifact


def reject_unsafe_artifact_path(path: Path) -> None:
    if path.is_symlink():
        raise QualificationSmokeError(reason="unsafe smoke artifact path")


def artifact_tree_sha256(root: Path, *, exclude: Path | None = None) -> str:
    reject_unsafe_artifact_path(root)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_exclude = None if exclude is None else exclude.resolve(strict=False)
        files: list[dict[str, JsonValue]] = []
        for path in sorted(resolved_root.rglob("*")):
            reject_unsafe_artifact_path(path)
            if not path.is_file() or path == resolved_exclude:
                continue
            content = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(resolved_root).as_posix(),
                    "sha256": sha256_bytes(content),
                    "size_bytes": len(content),
                }
            )
    except OSError as error:
        raise QualificationSmokeError(
            reason="qualification smoke artifact tree unavailable"
        ) from error
    return sha256_bytes(canonical_json_bytes({"files": files}))


def model_sha256(model: BaseModel) -> str:
    return sha256_bytes(canonical_json_bytes(model.model_dump(mode="json")))


def _read_artifact(path: Path) -> bytes:
    reject_unsafe_artifact_path(path)
    if not path.is_file():
        raise QualificationSmokeError(reason="qualification smoke artifact missing")
    return path.read_bytes()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
