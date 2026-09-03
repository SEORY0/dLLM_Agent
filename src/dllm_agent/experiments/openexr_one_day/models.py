"""Immutable contracts for the OpenEXR one-day screening experiment."""

from dataclasses import dataclass
from typing import Final, Literal, NewType

type Arm = Literal[
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
    "libfuzzer-only",
]
type ModelArm = Literal[
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
]
type PrimaryEndpoint = Literal["target_branch_coverage_auc"]

MonotonicSeconds = NewType("MonotonicSeconds", float)
Seed = NewType("Seed", int)
GpuIndex = NewType("GpuIndex", int)
RepetitionIndex = NewType("RepetitionIndex", int)
WaveIndex = NewType("WaveIndex", int)

ARMS: Final[tuple[Arm, ...]] = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
    "libfuzzer-only",
)
MODEL_ARMS: Final[tuple[ModelArm, ...]] = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
)
REPETITION_COUNT: Final = 10
WAVE_COUNT: Final = 5
REPETITIONS_PER_WAVE: Final = 2
WAVE_DURATION_SECONDS: Final = 10_800
GPU_COUNT: Final = 8
TARGET_COMMIT: Final = "777d231a179de1711d2a942810d9559216ab3f4c"


@dataclass(frozen=True, slots=True)
class PreregistrationManifest:
    """Machine-readable decisions fixed before the one-day campaign begins."""

    target_commit: str = TARGET_COMMIT
    arms: tuple[Arm, ...] = ARMS
    repetitions: int = REPETITION_COUNT
    wave_count: int = WAVE_COUNT
    repetitions_per_wave: int = REPETITIONS_PER_WAVE
    wave_duration_seconds: int = WAVE_DURATION_SECONDS
    confirmatory_treatment: ModelArm = "selective-revision"
    confirmatory_control: ModelArm = "final-only"
    primary_endpoint: PrimaryEndpoint = "target_branch_coverage_auc"


@dataclass(frozen=True, slots=True)
class SeedNamespace:
    """Half-open seed range reserved for one paired repetition."""

    repetition: RepetitionIndex
    start_seed: Seed
    stop_seed_exclusive: Seed


@dataclass(frozen=True, slots=True)
class CellSchedule:
    """One arm/repetition cell bound to resources, seeds, and a deadline."""

    repetition: RepetitionIndex
    arm: Arm
    wave_index: WaveIndex
    seed_namespace: SeedNamespace
    fuzzer_seed: Seed
    gpu_index: GpuIndex | None
    deadline_monotonic_seconds: MonotonicSeconds


@dataclass(frozen=True, slots=True)
class WaveSchedule:
    """Two paired repetitions that execute concurrently for three hours."""

    index: WaveIndex
    repetitions: tuple[RepetitionIndex, RepetitionIndex]
    start_monotonic_seconds: MonotonicSeconds
    deadline_monotonic_seconds: MonotonicSeconds
    cells: tuple[CellSchedule, ...]


@dataclass(frozen=True, slots=True)
class CampaignSchedule:
    """Frozen five-wave execution schedule and preregistered decisions."""

    preregistration: PreregistrationManifest
    seed_namespaces: tuple[SeedNamespace, ...]
    waves: tuple[WaveSchedule, ...]

    @property
    def cells(self) -> tuple[CellSchedule, ...]:
        """Return every arm/repetition cell in deterministic wave order."""
        return tuple(cell for wave in self.waves for cell in wave.cells)

    @property
    def absolute_deadline_monotonic_seconds(self) -> MonotonicSeconds:
        """Return the fixed deadline of the final main-campaign wave."""
        return self.waves[-1].deadline_monotonic_seconds
