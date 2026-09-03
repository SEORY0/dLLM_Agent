# ruff: noqa: D100,D101

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from tools.poc_revision_experiment_json import JsonObject


class LiveBudgetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LiveBudget:
    epochs: int
    initial_steps: int
    revision_steps: int
    probe_seconds: int
    proof_seconds: int

    def __post_init__(self) -> None:
        """Reject zero or negative live budget values."""
        for name, value in (
            ("epochs", self.epochs),
            ("initial_steps", self.initial_steps),
            ("revision_steps", self.revision_steps),
            ("probe_seconds", self.probe_seconds),
            ("proof_seconds", self.proof_seconds),
        ):
            if value <= 0:
                msg = f"{name} must be a positive integer"
                raise LiveBudgetError(msg)

    def max_model_nfe(self, particles: int, arm: str) -> int:
        """Compute the requested maximum model calls for this budget."""
        if particles <= 0:
            msg = "particles must be a positive integer"
            raise LiveBudgetError(msg)
        initial_steps, revision_steps = self.model_steps(arm)
        return particles * (initial_steps + (self.epochs * revision_steps))

    def manifest(self, particles: int, arm: str) -> JsonObject:
        """Render the live budget fields stored in outer manifests."""
        initial_steps, revision_steps = self.model_steps(arm)
        return {
            "particles": particles,
            "epochs": self.epochs,
            "initial_steps": initial_steps,
            "revision_steps": revision_steps,
            "probe_seconds": self.probe_seconds,
            "proof_seconds": self.proof_seconds,
            "max_model_nfe": self.max_model_nfe(particles, arm),
        }

    def model_steps(self, arm: str) -> tuple[int, int]:
        """Return arm-specific model step provenance."""
        if arm == "libfuzzer-only":
            return (0, 0)
        return (self.initial_steps, self.revision_steps)


DEFAULT_LIVE_BUDGET: Final = LiveBudget(
    epochs=1,
    initial_steps=8,
    revision_steps=2,
    probe_seconds=1,
    proof_seconds=10,
)
