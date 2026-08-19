"""Transparent deterministic trajectory ranking."""

from dataclasses import dataclass

from dllm_agent.models import CandidateScore, Trajectory


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """Replay fields consumed by the preregistered score."""

    vulnerability_probability: float
    localization_probability: float
    consistency: float
    nfe: int


def trajectory_score(inputs: ScoreInputs) -> float:
    """Score replay evidence; lower excess NFE is preferred."""
    penalty = max(inputs.nfe - 8, 0) * 0.005
    score = (
        0.5 * inputs.vulnerability_probability
        + 0.3 * inputs.localization_probability
        + 0.2 * inputs.consistency
        - penalty
    )
    return round(score, 6)


def score_trajectory(trajectory: Trajectory) -> CandidateScore:
    """Project one typed trajectory into its auditable score."""
    inputs = ScoreInputs(
        trajectory.vulnerability_probability,
        trajectory.localization_probability,
        trajectory.consistency,
        trajectory.nfe,
    )
    return CandidateScore(
        candidate_id=trajectory.candidate_id,
        score=trajectory_score(inputs),
        nfe=trajectory.nfe,
    )


def rank_trajectories(trajectories: tuple[Trajectory, ...]) -> tuple[CandidateScore, ...]:
    """Return scores in deterministic descending order."""
    scores = (score_trajectory(trajectory) for trajectory in trajectories)
    return tuple(sorted(scores, key=lambda item: (-item.score, item.candidate_id)))


def no_fixed_point_permutation(trajectory: Trajectory) -> tuple[str, ...]:
    """Rotate one trace without changing its finalized candidate."""
    return trajectory.trace[1:] + trajectory.trace[:1]
