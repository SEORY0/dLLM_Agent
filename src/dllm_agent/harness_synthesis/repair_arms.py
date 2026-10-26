"""RQ4b repair arms: masked continuations, frozen candidate plans, selection and cost.

Pure. Builds no prompt and calls no model. ``tools/harness_synthesis/rq4b_functional_pilot.py``
feeds these sequences to Dream and evaluates what comes back.

Every arm gets the same number of candidates ``K`` per (sample, replicate). Seed slot
``j`` (``seed_for(..., j)``) is shared across arms (common random numbers). Suffix and
resample candidate ``j`` use slot ``j``. The infill arm's candidates are ordered by mask
length (``INFILL_MASK_OFFSETS``), and the slot paired with each length rotates by
replicate: in replicate ``r``, length position ``k`` uses slot ``(k + r) % K``. With
``R = K`` replicates, every length meets every slot exactly once per sample, so a
length difference cannot be a fixed slot effect (``seed_slot_balance`` checks this).
The selection rule is frozen before running: walk the candidates in their frozen order
and stop at the first that passes the functional evaluator. Compute is recorded per
candidate and is never assumed equal across arms. A Dream denoising step is one
forward pass over the whole prompt plus continuation, however many positions are
masked, so ``forward_cost`` is steps times sequence length.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dllm_agent.harness_synthesis.remask import TokenWindow, infill_window

type ArmName = Literal["infill-sweep", "suffix", "resample", "infill-oracle-length"]

# The three compared arms. ``infill-oracle-length`` is a diagnostic: one candidate whose
# mask count equals the *target's* token length, which a deployed repairer would not
# know. It answers "with exactly the right capacity, does infill repair?" and is
# excluded from every arm comparison.
ARMS: tuple[ArmName, ...] = ("infill-sweep", "suffix", "resample")
DIAGNOSTIC_ARM: ArmName = "infill-oracle-length"
# The diagnostic's own seed slot, shared with no compared candidate.
DIAGNOSTIC_SEED_SLOT = 9
K = 5
PILOT_REPLICATES = K
# Candidate j of the infill arm masks L + INFILL_MASK_OFFSETS[j] positions, where L is
# the defective line's own token window (observable, unlike the target's length). The
# order is frozen nearest-first so cost-to-first-success means the same thing per sample.
INFILL_MASK_OFFSETS: tuple[int, ...] = (0, 2, -2, 4, 8)
SUFFIX_SLACK = 8
RESAMPLE_NEW_TOKENS = 96
PILOT_SEED_BASE = 13_000_000
CONFIRMATORY_SEED_BASE = 13_500_000


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """One generation call: which arm, its order index, seed and decoding size."""

    arm: ArmName
    index: int
    seed: int
    masked: int
    steps: int


def seed_for(sample_index: int, replicate: int, candidate: int, *, base: int) -> int:
    """Deterministic, collision-free seed for (sample, replicate, candidate) under ``base``."""
    if not (0 <= candidate < 10 and 0 <= replicate < 100 and 0 <= sample_index < 500):  # noqa: PLR2004
        reason = "seed_for: index out of the block's range"
        raise ValueError(reason)
    return base + sample_index * 1000 + replicate * 10 + candidate


def infill_mask_lengths(window_length: int) -> tuple[int, ...]:
    """The infill arm's frozen mask lengths for an ``L``-token window; raises if any is <= 0.

    Raising rather than dropping a nonpositive length keeps the arms attempt-matched. The
    benchmark validation checks that every sample's ``L`` is at least 3.
    """
    lengths = tuple(window_length + offset for offset in INFILL_MASK_OFFSETS)
    if min(lengths) <= 0:
        reason = f"window of {window_length} tokens gives a nonpositive mask length"
        raise ValueError(reason)
    return lengths


def plan(  # noqa: PLR0913 - the unit's identity plus its two optional sizes.
    window: TokenWindow,
    continuation_length: int,
    *,
    sample_index: int,
    replicate: int,
    base: int,
    oracle_length: int | None = None,
) -> tuple[CandidatePlan, ...]:
    """All ``3 * K`` generation calls for one (sample, replicate), in frozen order.

    With ``oracle_length``, one diagnostic infill candidate is appended, on its own seed
    slot (``DIAGNOSTIC_SEED_SLOT``). It is never part of an arm comparison or a selection.

    One token per denoising step in every arm (``steps == masked``). The RQ4 pilot's
    resample arm decoded two tokens per step, which made it cheaper per token and
    differently sampled. That mismatch is removed here.
    """
    seeds = [seed_for(sample_index, replicate, j, base=base) for j in range(K)]
    suffix_masked = continuation_length - window.start + SUFFIX_SLACK
    plans = [
        CandidatePlan("infill-sweep", k, seeds[(k + replicate) % K], length, length)
        for k, length in enumerate(infill_mask_lengths(window.length))
    ]
    plans += [CandidatePlan("suffix", j, seeds[j], suffix_masked, suffix_masked) for j in range(K)]
    plans += [
        CandidatePlan("resample", j, seeds[j], RESAMPLE_NEW_TOKENS, RESAMPLE_NEW_TOKENS)
        for j in range(K)
    ]
    if oracle_length is not None:
        seed = seed_for(sample_index, replicate, DIAGNOSTIC_SEED_SLOT, base=base)
        plans.append(CandidatePlan(DIAGNOSTIC_ARM, 0, seed, oracle_length, oracle_length))
    return tuple(plans)


def seed_slot(seed: int, *, base: int) -> int:
    """The slot ``j`` a seed from ``seed_for`` was drawn with."""
    return (seed - base) % 10


def seed_slot_balance(plans: Sequence[CandidatePlan], *, base: int) -> dict[int, set[int]]:
    """For the infill arm: length position -> set of seed slots it was paired with.

    Over one sample's ``R = K`` replicates, a balanced design gives every position all
    ``K`` slots.
    """
    balance: dict[int, set[int]] = {}
    for p in plans:
        if p.arm == "infill-sweep":
            balance.setdefault(p.index, set()).add(seed_slot(p.seed, base=base))
    return balance


def infill_continuation(
    token_ids: Sequence[int], window: TokenWindow, *, mask_length: int, mask_token_id: int
) -> list[int]:
    """Left context, ``mask_length`` masks, right context (``remask.infill_window``)."""
    return list(
        infill_window(token_ids, window, mask_length=mask_length, mask_token_id=mask_token_id)
    )


def suffix_continuation(
    token_ids: Sequence[int], start: int, *, mask_token_id: int, slack: int = SUFFIX_SLACK
) -> list[int]:
    """Keep ``token_ids[:start]``; mask the rest plus ``slack`` extra positions."""
    return [*token_ids[:start], *([mask_token_id] * (len(token_ids) - start + slack))]


def outside_span_preserved(
    before: Sequence[int], after: Sequence[int], window: TokenWindow, *, mask_length: int
) -> bool:
    """Whether every token outside the infilled span is unchanged (shifted by the resize)."""
    span_end = window.start + mask_length
    return (
        len(after) == len(before) - window.length + mask_length
        and list(after[: window.start]) == list(before[: window.start])
        and list(after[span_end:]) == list(before[window.end :])
    )


def is_special_piece(piece: str) -> bool:
    """A Dream special-token piece such as ``<|endoftext|>`` or ``<|mask|>``."""
    return piece.startswith("<|") and piece.endswith("|>")


def decode_dropping_span_specials(pieces: Sequence[str], start: int, end: int) -> str:
    """Join pieces, dropping special pieces only inside ``[start, end)``.

    On a fixed-length canvas, surplus masks the model does not need often decode to
    end-of-text. Outside the span, a special piece is real structure (the end of the
    block), and ``extract_body`` already stops at it.
    """
    return "".join(
        piece
        for index, piece in enumerate(pieces)
        if not (start <= index < end and is_special_piece(piece))
    )


def forward_cost(steps: int, *, prompt_length: int, continuation_length: int) -> int:
    """Token-positions processed: ``steps`` forward passes over prompt + continuation."""
    return steps * (prompt_length + continuation_length)


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """What the evaluator and the backend reported for one candidate (frozen order)."""

    index: int
    compiled: bool
    success: bool
    cost: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Selection:
    """The frozen selection rule's result over one arm's ``K`` candidates."""

    pass_at_k: bool
    selected_index: int
    first_success_index: int | None
    compiler_invocations: int
    cost_to_selection: int
    total_cost: int
    compiled_any: bool


def select(outcomes: Sequence[CandidateOutcome]) -> Selection:
    """Walk ``outcomes`` in order and stop at the first functional success.

    With no success, the selection is the first candidate that compiled, otherwise the
    first candidate. That choice only matters for the secondary metrics. Compiler
    invocations and cost count up to and including the selected candidate when there is
    a success, and all ``K`` otherwise, because a deployed loop would try them all.
    """
    if not outcomes:
        reason = "select: no candidates"
        raise ValueError(reason)
    ordered = sorted(outcomes, key=lambda o: o.index)
    first = next((pos for pos, o in enumerate(ordered) if o.success), None)
    if first is not None:
        selected, invocations = first, first + 1
    else:
        selected = next((pos for pos, o in enumerate(ordered) if o.compiled), 0)
        invocations = len(ordered)
    return Selection(
        pass_at_k=first is not None,
        selected_index=ordered[selected].index,
        first_success_index=None if first is None else ordered[first].index,
        compiler_invocations=invocations,
        cost_to_selection=sum(o.cost for o in ordered[:invocations]),
        total_cost=sum(o.cost for o in ordered),
        compiled_any=any(o.compiled for o in ordered),
    )


def success_within_budget(outcomes: Sequence[CandidateOutcome], budget: int) -> tuple[bool, int]:
    """``(any success, candidates afforded)`` over the longest prefix costing <= ``budget``.

    The compute-matched comparison. An arm whose first candidate already costs more than
    the budget affords zero candidates and fails. The analysis reports how often that
    happens rather than hiding it.
    """
    spent, afforded = 0, 0
    for outcome in sorted(outcomes, key=lambda o: o.index):
        if spent + outcome.cost > budget:
            break
        spent += outcome.cost
        afforded += 1
    prefix = sorted(outcomes, key=lambda o: o.index)[:afforded]
    return any(o.success for o in prefix), afforded
