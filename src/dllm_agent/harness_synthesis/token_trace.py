"""Per-position denoising trajectory records for free-form generation.

Sibling to ``experiments/multitarget/slot_sampler_contracts.TrajectoryStep`` — same
field-naming style, deliberately not a subclass — because the two record structurally
different things. ``TrajectoryStep`` records one commit per categorical selector:
``GenericSlotSampler`` never revisits a position once filled, so "permanence" is
trivially the first (and only) step. Harness synthesis remasks and regenerates free-form
code tokens within one generation, so a position can be committed, remasked, and
recommitted several times; ``TokenTrace`` keeps one record per (step, position)
observation instead of one record per commit, and ``permanence_step`` resolves which
commit, if any, actually stuck. See
``docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md`` §4.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenTrace:
    """One position's observed state at one denoising step.

    ``confidence``/``entropy``/``margin`` are the trajectory features spec §5 keeps as
    secondary evidence (only ``confidence`` is the primary RQ1 signal); all three are
    ``None`` while masked and required once committed.
    """

    step_index: int
    position: int
    token_id: int
    masked: bool
    confidence: float | None = None
    entropy: float | None = None
    margin: float | None = None

    def __post_init__(self) -> None:
        """Reject a trace whose confidence contradicts its masked state."""
        if self.masked and self.confidence is not None:
            reason = "a masked position has no commit confidence"
            raise ValueError(reason)
        if not self.masked and self.confidence is None:
            reason = "a committed position must record its commit confidence"
            raise ValueError(reason)


def position_traces(traces: Sequence[TokenTrace], position: int) -> tuple[TokenTrace, ...]:
    """Return one position's traces from a mixed-position stream, in step order."""
    return tuple(
        sorted((trace for trace in traces if trace.position == position), key=_step_index)
    )


def permanence_step(traces: Sequence[TokenTrace]) -> int | None:
    """Return the step a position last became committed and stayed committed.

    ``traces`` must already be one position's records in step order (build it with
    ``position_traces`` from a mixed-position stream). Returns ``None`` when the
    position never committed, or when its last recorded observation is masked — a
    remask with no later commit is not a permanence step, it is an open remask.
    """
    if not traces or traces[-1].masked:
        return None
    last_commit_step: int | None = None
    for trace in traces:
        if trace.masked:
            last_commit_step = None
        elif last_commit_step is None:
            last_commit_step = trace.step_index
    return last_commit_step


def commit_confidence(traces: Sequence[TokenTrace]) -> float | None:
    """Return the confidence recorded at ``permanence_step``, or ``None`` if none."""
    step = permanence_step(traces)
    if step is None:
        return None
    return next(trace.confidence for trace in traces if trace.step_index == step)


def _step_index(trace: TokenTrace) -> int:
    return trace.step_index
