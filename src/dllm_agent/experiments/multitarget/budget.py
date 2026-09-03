"""Frozen budget and grid contract for the 24-hour multi-target experiment."""

from dataclasses import dataclass, field
from typing import Final, Literal, assert_never

type Target = Literal["libpng", "libtiff", "libxml2"]
type Arm = Literal[
    "libfuzzer-only",
    "structured-one-shot",
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
]
type FeedbackArm = Literal[
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
]

TARGETS: Final[tuple[Target, ...]] = ("libpng", "libtiff", "libxml2")
ARMS: Final[tuple[Arm, ...]] = (
    "libfuzzer-only",
    "structured-one-shot",
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
)
FEEDBACK_ARMS: Final[tuple[FeedbackArm, ...]] = (
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
)


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    """Worker-second allocation for one target/arm/repetition cell."""

    initial_probe_worker_seconds: int
    revision_probe_worker_seconds: int
    campaign_worker_seconds: int
    search_worker_seconds: int
    proof_worker_seconds: int = 0


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Search budget shared by each experiment cell."""

    total_worker_seconds: int = 86_400
    candidate_count: int = 16
    probe_worker_seconds: int = 30
    feedback_probe_rounds: int = 2
    proof_worker_seconds: int = 0

    def for_arm(self, arm: Arm) -> BudgetAllocation:
        """Return the exact allocation for the selected experiment arm."""
        match arm:
            case "libfuzzer-only" | "structured-one-shot":
                probe_worker_seconds = 0
            case "feedback-no-remask" | "selective-revision" | "pseudo-feedback":
                probe_worker_seconds = self.candidate_count * self.probe_worker_seconds
            case _:
                assert_never(arm)

        return BudgetAllocation(
            initial_probe_worker_seconds=probe_worker_seconds,
            revision_probe_worker_seconds=probe_worker_seconds,
            campaign_worker_seconds=self.total_worker_seconds
            - (probe_worker_seconds * self.feedback_probe_rounds),
            search_worker_seconds=self.total_worker_seconds,
            proof_worker_seconds=self.proof_worker_seconds,
        )


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    """One target, arm, and repetition in the multi-target grid."""

    target: Target
    arm: Arm
    repetition: int


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Frozen multi-target experiment defaults and grid expansion."""

    raw_particle_count: int = 32
    minimum_unique_count: int = 16
    repetitions: int = 1
    budget: SearchBudget = field(default_factory=SearchBudget)

    def cells(self) -> tuple[ExperimentCell, ...]:
        """Expand the target x arm x repetition grid."""
        return tuple(
            ExperimentCell(target=target, arm=arm, repetition=repetition)
            for target in TARGETS
            for arm in ARMS
            for repetition in range(self.repetitions)
        )
