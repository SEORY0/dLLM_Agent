"""Dream commit-trajectory bookkeeping: caller-supplied observations -> TokenTrace.

Deliberately does no tensor math and never imports torch: this repo's convention for
``src/`` modules that touch the optional ``gpu`` extra keeps tensor computation in
``tools/`` (not under strict basedpyright — see ``runtime/ar_model_loader.py``'s
lazy-import Protocol pattern and ``runtime/revision_process.py``'s ``load_torch_runtime``)
and any ``src/`` logic thin and torch-free. The actual softmax/entropy/margin computation
over Dream's per-step logits belongs to a ``tools/`` adapter driving the real
``generation_tokens_hook_func`` (Task 4's GPU smoke script), which converts only the
handful of positions that changed each step to plain floats before calling this module —
converting a full ``[seq_len, vocab]`` distribution in pure Python every step would be far
too slow for the fuzzing-loop timing budget this module ultimately serves.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, override

from dllm_agent.harness_synthesis.token_trace import TokenTrace


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A completed generation (or repair round) and its full commit trajectory."""

    token_ids: tuple[int, ...]
    traces: tuple[TokenTrace, ...]


class TraceableGenerationBackend(Protocol):
    """Diffusion generation boundary that exposes its per-step commit trajectory.

    Named to match ``revision/controller_models.RevisionSampler``'s naming/style (spec
    §7) — a sibling boundary for a structurally different, free-form multi-remask
    generation loop, not a shared implementation.
    """

    def generate(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        seed: int,
        new_token_count: int,
        steps: int,
        init_token_ids: tuple[int, ...] | None = None,
    ) -> GenerationResult:
        """Generate ``new_token_count`` tokens, optionally resuming from a remasked state."""
        ...


@dataclass(frozen=True, slots=True)
class CommitObservation:
    """One position's already-computed trajectory features at the current step."""

    position: int
    token_id: int
    confidence: float
    entropy: float
    margin: float


@dataclass(slots=True)
class TrajectoryRecorderError(Exception):
    """Raised when a caller's commit observations disagree with the token stream."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the bookkeeping failure reason."""
        return self.reason


class TrajectoryRecorder:
    """Assembles a ``TokenTrace`` stream across one ``diffusion_generate`` call.

    Dream calls its tokens hook with ``step=None`` once before the denoising loop, then
    once per step afterward. Within one call a position only ever moves from masked to
    committed and never back — Dream's own sampler only ever writes into positions
    still in ``mask_index`` that step (see ``generation_utils.py``'s
    ``x[row_indices, transfer_index] = ...``, where ``transfer_index`` is drawn from
    currently-masked positions) — so this recorder only needs to detect newly-committed
    positions and cross-check them against what the caller reports. A remask only
    happens ACROSS separate ``generate`` calls, driven by ``remask.py``'s selection
    policies; combining multiple rounds' traces into one ``permanence_step``-ready
    stream is the calling experiment script's job (Task 8-10), not this recorder's.
    """

    def __init__(self, mask_token_id: int) -> None:
        """Bind the mask token id this recorder watches transitions against."""
        self._mask_token_id: int = mask_token_id
        self._previously_masked: frozenset[int] | None = None
        self._traces: list[TokenTrace] = []

    def observe(
        self,
        step: int | None,
        token_ids: Sequence[int],
        commits: Sequence[CommitObservation] = (),
    ) -> None:
        """Feed one hook call and the caller's per-position features for it.

        ``step``/``token_ids`` come from the hook itself; ``commits`` is the caller's own
        feature computation for this step. Raises if ``commits`` does not name exactly
        the positions ``token_ids`` shows as newly committed relative to the previous
        call — catching a caller that forgot a position's features, or reported a stale
        one, rather than silently recording a gap that would corrupt ``permanence_step``
        downstream.
        """
        masked = frozenset(
            position
            for position, token_id in enumerate(token_ids)
            if token_id == self._mask_token_id
        )
        if step is not None:
            if self._previously_masked is None:
                reason = "observe(None, initial_token_ids) must be called first"
                raise TrajectoryRecorderError(reason)
            newly_committed = self._previously_masked - masked
            reported = frozenset(commit.position for commit in commits)
            if reported != newly_committed:
                reason = (
                    f"step {step}: commits report positions {sorted(reported)} but "
                    f"token_ids shows {sorted(newly_committed)} newly committed"
                )
                raise TrajectoryRecorderError(reason)
            self._traces.extend(
                TokenTrace(
                    step_index=step,
                    position=commit.position,
                    token_id=commit.token_id,
                    masked=False,
                    confidence=commit.confidence,
                    entropy=commit.entropy,
                    margin=commit.margin,
                )
                for commit in commits
            )
        self._previously_masked = masked

    @property
    def traces(self) -> tuple[TokenTrace, ...]:
        """Return every commit recorded so far, in the order it was observed."""
        return tuple(self._traces)
