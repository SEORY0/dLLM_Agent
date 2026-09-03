"""Pure arm routing and duration contracts for the real campaign."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from dllm_agent.experiments.multitarget.budget import FEEDBACK_ARMS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dllm_agent.experiments.multitarget.budget import Arm
    from dllm_agent.experiments.multitarget.corpus import CandidateInput

_PARTICLE_COUNT = 16


@dataclass(slots=True)
class CampaignCoreError(Exception):
    """Raised when a campaign population or duration is inconsistent."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CampaignDurations:
    """Physical phase durations and an explicit full/smoke label."""

    label: str
    probe_seconds: int
    nonfeedback_main_seconds: int
    feedback_main_seconds: int
    is_full: bool

    @classmethod
    def full(cls) -> CampaignDurations:
        """Return the frozen 24-hour per-cell allocation."""
        return cls(
            label="full-24h",
            probe_seconds=30,
            nonfeedback_main_seconds=86_400,
            feedback_main_seconds=85_440,
            is_full=True,
        )

    @classmethod
    def smoke(cls, *, probe_seconds: int, main_seconds: int) -> CampaignDurations:
        """Return a clearly labelled bounded integration configuration."""
        if probe_seconds < 1 or main_seconds < 1:
            raise CampaignCoreError(reason="smoke durations must be positive")
        return cls(
            label="smoke",
            probe_seconds=probe_seconds,
            nonfeedback_main_seconds=main_seconds,
            feedback_main_seconds=main_seconds,
            is_full=False,
        )

    def main_seconds(self, arm: Arm) -> int:
        """Return the main campaign duration for one arm."""
        return self.feedback_main_seconds if arm in FEEDBACK_ARMS else self.nonfeedback_main_seconds

    def logical_worker_seconds(self, arm: Arm) -> int:
        """Return main plus logically charged sequential micro-probes."""
        probe_seconds = 0 if arm not in FEEDBACK_ARMS else 2 * _PARTICLE_COUNT * self.probe_seconds
        return self.main_seconds(arm) + probe_seconds


def arm_candidate_inputs(
    arm: Arm,
    initial: tuple[CandidateInput, ...],
    revisions: Mapping[str, tuple[CandidateInput, ...]],
) -> tuple[CandidateInput, ...]:
    """Route one cell to baseline, shared initial, or arm-specific revised inputs."""
    if arm == "libfuzzer-only":
        return ()
    if arm == "structured-one-shot":
        return _require_population(initial)
    population = revisions.get(arm)
    if population is None:
        raise CampaignCoreError(reason=f"missing revision population for {arm}")
    return _require_population(population)


def probe_seed(base_seed: int, particle_index: int) -> int:
    """Derive a stable nonnegative seed for one particle probe."""
    if base_seed < 0 or particle_index < 0:
        raise CampaignCoreError(reason="probe seed inputs must be nonnegative")
    digest = hashlib.sha256(f"{base_seed}:{particle_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _require_population(
    population: tuple[CandidateInput, ...],
) -> tuple[CandidateInput, ...]:
    hashes = tuple(item.snapshot_sha256 for item in population)
    if len(population) != _PARTICLE_COUNT or len(set(hashes)) != _PARTICLE_COUNT:
        raise CampaignCoreError(reason="arm requires exactly 16 unique candidate inputs")
    return population
