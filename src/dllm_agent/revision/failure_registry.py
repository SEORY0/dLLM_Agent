"""Cross-run persistence for the tri-modal field-outcome signal.

`trajectory_policy.LineageFieldOutcome` already carries the tri-modal failure
signal (`invalid_count`, `duplicate_count`, `proof_failure_count`) this
registry persists -- this module adds no new signal, only a JSONL
append-only file so one run's outcomes can seed a later run's field
selection. A `FailureRecord` is scoped by `target_pin_sha256` (never by
lineage: `lineage_id` is a per-run particle index, not a stable identity
across runs) so seeding never crosses target pins.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

from pydantic import Field, ValidationError

from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.revision.trajectory_policy import LineageFieldOutcome
from dllm_agent.targets.openexr.particle import FieldName

type CauseBucket = Literal["invalid", "duplicate", "proof_failure"]

_CAUSE_BUCKETS: tuple[CauseBucket, ...] = ("invalid", "duplicate", "proof_failure")


class FailureRecord(FrozenModel):
    """One persisted (target, field, cause) outcome for one cause bucket."""

    schema_version: Literal["dllm-failure-registry/v1"] = "dllm-failure-registry/v1"
    target_pin_sha256: Sha256
    field: FieldName
    cause_bucket: CauseBucket
    count: int = Field(ge=1)
    reward_delta_sum: float
    trials: int = Field(gt=0)
    last_seen_run_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class FailureRegistryError(ValueError):
    """Raised when a failure registry file cannot be parsed."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Return a fail-closed registry parse error."""
        return f"{self.path}: {self.reason}"


def load(path: Path) -> tuple[FailureRecord, ...]:
    """Load persisted failure records, or return empty if the file is absent."""
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[FailureRecord] = []
    for line_index, line in enumerate(lines):
        if line == "":
            continue
        try:
            records.append(FailureRecord.model_validate_json(line))
        except ValidationError as error:
            reason = f"line {line_index + 1}: {error}"
            raise FailureRegistryError(path=path, reason=reason) from error
    return tuple(records)


def append(path: Path, record: FailureRecord) -> None:
    """Atomically append one record to the registry file."""
    _write_all(path, (*load(path), record))


def failure_records(
    outcome: LineageFieldOutcome, *, target_pin_sha256: str, run_id: str
) -> tuple[FailureRecord, ...]:
    """Explode one observed outcome into per-cause-bucket persisted records."""
    counts: dict[CauseBucket, int] = {
        "invalid": outcome.invalid_count,
        "duplicate": outcome.duplicate_count,
        "proof_failure": outcome.proof_failure_count,
    }
    return tuple(
        FailureRecord(
            target_pin_sha256=target_pin_sha256,
            field=outcome.field,
            cause_bucket=bucket,
            count=counts[bucket],
            reward_delta_sum=outcome.reward_delta,
            trials=outcome.trials,
            last_seen_run_id=run_id,
        )
        for bucket in _CAUSE_BUCKETS
        if counts[bucket] > 0
    )


def seed_outcomes(
    records: tuple[FailureRecord, ...], lineage_id: str, target_pin_sha256: str
) -> tuple[LineageFieldOutcome, ...]:
    """Convert persisted records for one target pin into `LineageFieldOutcome` seeds."""
    return tuple(
        _outcome(lineage_id, record)
        for record in records
        if record.target_pin_sha256 == target_pin_sha256
    )


def _outcome(lineage_id: str, record: FailureRecord) -> LineageFieldOutcome:
    return LineageFieldOutcome(
        lineage_id=lineage_id,
        field=record.field,
        reward_delta=record.reward_delta_sum,
        trials=record.trials,
        invalid_count=record.count if record.cause_bucket == "invalid" else 0,
        duplicate_count=record.count if record.cause_bucket == "duplicate" else 0,
        proof_failure_count=record.count if record.cause_bucket == "proof_failure" else 0,
    )


def _write_all(path: Path, records: tuple[FailureRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    body = "".join(record.model_dump_json() + "\n" for record in records)
    _ = temporary.write_text(body, encoding="utf-8")
    _ = temporary.replace(path)
