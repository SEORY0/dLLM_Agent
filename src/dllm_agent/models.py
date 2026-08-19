"""Versioned external schemas and internal domain models."""

from typing import Annotated, ClassVar, Literal, NewType, Protocol

from pydantic import BaseModel, ConfigDict, Field

CandidateId = NewType("CandidateId", str)
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class FrozenModel(BaseModel):
    """Immutable strict schema base."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)


class SourceSpan(FrozenModel):
    """One-based source span."""

    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class Trajectory(FrozenModel):
    """Replay trajectory with fields consumed by the ranker."""

    candidate_id: str = Field(min_length=1)
    nfe: int = Field(ge=0)
    vulnerability_probability: Probability
    localization_probability: Probability
    consistency: Probability
    trace: tuple[str, ...] = Field(min_length=2)
    cwe: Literal["CWE-121"]
    span: SourceSpan
    explanation: str = Field(min_length=1)


class ReplayFixture(FrozenModel):
    """Synthetic Stable-DiffCoder-compatible replay fixture."""

    schema_version: Literal["dllm-replay/v1"]
    provenance_kind: Literal["synthetic-deterministic-fixture"]
    compatible_model: Literal["Stable-DiffCoder"]
    compatible_revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"]
    target_sha256: Sha256
    proof_input: Literal["DLC-PROOF-000001"]
    proof_input_sha256: Sha256
    trajectories: tuple[Trajectory, ...] = Field(min_length=4, max_length=4)


class ExperimentConfig(FrozenModel):
    """Checked experiment configuration."""

    schema_version: Literal["dllm-experiment/v1"]
    experiment_id: str = Field(min_length=1)
    candidate_bank_size: Literal[4]
    validator_timeout_seconds: Annotated[float, Field(gt=0.0, le=2.0)]
    expected_cwe: Literal["CWE-121"]
    fixture: str = Field(min_length=1)
    target: str = Field(min_length=1)


class CandidateScore(FrozenModel):
    """Auditable trajectory score."""

    candidate_id: str
    score: float
    nfe: int


class Finding(FrozenModel):
    """Finalized candidate selected for validation."""

    candidate_id: str
    cwe: Literal["CWE-121"]
    span: SourceSpan
    explanation: str


class ValidatorVerdict(FrozenModel):
    """Normalized isolated execution result."""

    status: Literal["proved", "rejected", "inconclusive"]
    reason: str
    exit_code: int | None
    timed_out: bool
    sanitizer: str | None


class Accounting(FrozenModel):
    """Offline inference and validation accounting."""

    candidate_count: int
    total_nfe: int
    validated_candidates: Literal[1]


class Provenance(FrozenModel):
    """SHA-256 inputs and replay origin."""

    target_sha256: Sha256
    fixture_sha256: Sha256
    proof_input_sha256: Sha256
    model_revision: str
    replay_kind: Literal["synthetic-deterministic-fixture"]


class ScanReport(FrozenModel):
    """Auditable scan output schema."""

    schema_version: Literal["dllm-scan-report/v1"]
    finding: Finding
    trajectory_summary: tuple[CandidateScore, ...]
    permutation_control: tuple[str, ...]
    validator: ValidatorVerdict
    evidence: tuple[str, ...]
    accounting: Accounting
    provenance: Provenance


class ExpandedPlan(FrozenModel):
    """Deterministically expanded experiment plan."""

    schema_version: Literal["dllm-plan/v1"]
    experiment_id: str
    candidate_bank_size: Literal[4]
    validator_timeout_seconds: float
    expected_cwe: Literal["CWE-121"]
    target: str
    fixture: str
    target_sha256: Sha256
    fixture_sha256: Sha256


class CandidateSampler(Protocol):
    """Candidate acquisition seam for replay or future live samplers."""

    def sample(self) -> tuple[Trajectory, ...]:
        """Acquire an immutable candidate bank."""
        ...
