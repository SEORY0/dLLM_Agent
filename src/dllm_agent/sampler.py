"""Replay candidate sampler."""

from dataclasses import dataclass

from dllm_agent.models import ReplayFixture, Trajectory


@dataclass(frozen=True, slots=True)
class ReplaySampler:
    """Deterministic test double for the dLLM-native trajectory capture seam."""

    fixture: ReplayFixture

    def sample(self) -> tuple[Trajectory, ...]:
        """Return the immutable replay candidate bank."""
        return self.fixture.trajectories
