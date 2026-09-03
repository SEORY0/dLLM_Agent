"""Typed campaign-level failure causes for Task 5 coordination."""

from __future__ import annotations

from typing import Literal, override

type ExogenousCause = Literal[
    "gpu-xid",
    "gpu-reset",
    "host-power-loss",
    "storage-io",
    "orchestrator-corruption",
]


class ExogenousCampaignError(Exception):
    """Evidence-bearing external cause that invalidates paired treatment evidence."""

    cause: ExogenousCause
    evidence: str

    def __init__(self, *, cause: ExogenousCause, evidence: str) -> None:
        """Store the exogenous cause and supporting evidence."""
        self.cause = cause
        self.evidence = evidence
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"{self.cause}: {self.evidence}"
