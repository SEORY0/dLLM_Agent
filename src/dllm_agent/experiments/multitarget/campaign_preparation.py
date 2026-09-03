"""Preparation helpers for live multitarget campaign particle populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from dllm_agent.experiments.multitarget.campaign_core import probe_seed
from dllm_agent.experiments.multitarget.corpus import CandidateInput
from dllm_agent.experiments.multitarget.probe_runner import (
    ProbeRunRequest,
    ProbeRunResult,
    run_candidate_probe,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.experiments.multitarget.bank_generation import RevisedCandidate
    from dllm_agent.experiments.multitarget.candidate_bank import Candidate, ProbeObservation
    from dllm_agent.experiments.multitarget.docker_runtime import RuntimeProfile

_PARTICLE_COUNT = 16
_POPULATION_ERROR = "campaign preparation requires exactly 16 unique particles"


class RunCandidateProbe(Protocol):
    """Narrow seam for tests and live one-candidate probe execution."""

    def __call__(self, request: ProbeRunRequest) -> ProbeRunResult:
        """Run one candidate micro-probe."""
        ...


@dataclass(slots=True)
class CampaignPreparationError(Exception):
    """Raised when campaign preparation inputs are inconsistent."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CandidateProbeBatchRequest:
    """Boundary object for one target's sequential candidate micro-probes."""

    target_label: str
    profile: RuntimeProfile
    image_id: str
    work_root: Path
    candidates: tuple[Candidate, ...]
    duration_seconds: int
    probe_seed_base: int
    cpu_index: int
    poll_seconds: int


@dataclass(frozen=True, slots=True)
class CandidateProbeBatchResult:
    """Ordered probe observations and the full per-particle run evidence."""

    observations: tuple[ProbeObservation, ...]
    results: tuple[ProbeRunResult, ...]


def candidates_to_inputs(candidates: tuple[Candidate, ...]) -> tuple[CandidateInput, ...]:
    """Convert the selected initial 16-particle population into corpus inputs."""
    _ = _require_unique_candidate_population(candidates)
    return tuple(
        CandidateInput(
            snapshot_sha256=candidate.snapshot.sha256,
            content=candidate.snapshot.content,
        )
        for candidate in candidates
    )


def revised_candidates_to_inputs(
    revised: tuple[RevisedCandidate, ...],
) -> tuple[CandidateInput, ...]:
    """Convert an accepted revised 16-particle population into corpus inputs."""
    _ = _require_unique_revised_population(revised)
    return tuple(
        CandidateInput(
            snapshot_sha256=candidate.snapshot.sha256,
            content=candidate.snapshot.content,
        )
        for candidate in revised
    )


def run_candidate_micro_probes(
    request: CandidateProbeBatchRequest,
    *,
    run_probe: RunCandidateProbe = run_candidate_probe,
) -> CandidateProbeBatchResult:
    """Run exactly 16 one-candidate probes sequentially for one target."""
    candidates = _require_unique_candidate_population(request.candidates)
    request.work_root.mkdir(parents=True, exist_ok=True)
    results: list[ProbeRunResult] = []
    for particle_index, candidate in enumerate(candidates):
        work_dir = request.work_root / f"particle-{particle_index:02d}"
        if work_dir.exists():
            raise CampaignPreparationError(
                reason=f"probe work directory already exists: {work_dir}"
            )
        result = run_probe(
            ProbeRunRequest(
                name=f"dllm-{request.target_label}-probe-p{particle_index:02d}",
                profile=request.profile,
                image_id=request.image_id,
                work_dir=work_dir,
                candidate=candidate,
                duration_seconds=request.duration_seconds,
                fuzzer_seed=probe_seed(request.probe_seed_base, particle_index),
                cpu_index=request.cpu_index,
                poll_seconds=request.poll_seconds,
            )
        )
        results.append(result)
    return CandidateProbeBatchResult(
        observations=tuple(result.observation for result in results),
        results=tuple(results),
    )


def _require_unique_candidate_population(
    candidates: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    hashes = tuple(candidate.snapshot.sha256 for candidate in candidates)
    if len(candidates) != _PARTICLE_COUNT or len(set(hashes)) != _PARTICLE_COUNT:
        raise CampaignPreparationError(reason=_POPULATION_ERROR)
    return candidates


def _require_unique_revised_population(
    revised: tuple[RevisedCandidate, ...],
) -> tuple[RevisedCandidate, ...]:
    hashes = tuple(candidate.snapshot.sha256 for candidate in revised)
    if len(revised) != _PARTICLE_COUNT or len(set(hashes)) != _PARTICLE_COUNT:
        raise CampaignPreparationError(reason=_POPULATION_ERROR)
    return revised
