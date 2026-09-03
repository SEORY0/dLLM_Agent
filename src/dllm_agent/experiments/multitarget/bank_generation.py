"""Resident structured-target candidate-bank generation."""

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import override

from dllm_agent.experiments.multitarget.candidate_bank import (
    RAW_SAMPLE_COUNT,
    Candidate,
    CandidateBank,
    ProbeObservation,
    freeze_candidate_bank,
    pseudo_feedback,
    remask_plan,
    select_diverse,
)
from dllm_agent.experiments.multitarget.slot_sampler import (
    GenericSlotSampler,
    SlotModel,
    SlotSample,
    SlotSchema,
    SlotTokenizer,
)
from dllm_agent.targets.structured import SnapshotLike, StructuredTarget, load_structured_target

SELECTED_CANDIDATE_COUNT = 16


class RevisionMode(StrEnum):
    """Revision policy over a frozen initial bank."""

    FULL = "full"
    SELECTIVE = "selective"
    PSEUDO = "pseudo"


@dataclass(slots=True)
class BankGenerationError(Exception):
    """Raised when candidate-bank generation or revision cannot proceed."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class InitialBankRequest:
    """Request for one physical initial-bank generation."""

    target_id: str
    prompt_path: Path
    initial_seeds: tuple[int, ...]
    initial_steps: int
    mask_token_id: int = 5
    block_size: int = 64


@dataclass(frozen=True, slots=True)
class BankRevisionRequest:
    """Request for one revision arm over shared parents."""

    mode: RevisionMode
    revision_steps: int
    observations: tuple[ProbeObservation, ...] = ()
    pseudo_seed: int = 0
    revision_seed_base: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCost:
    """Aggregate model-call cost."""

    nfe: int
    forward_calls: int
    gpu_ms: int


@dataclass(frozen=True, slots=True)
class ModelCostSummary:
    """Physical arm cost and logical cost including the shared bank."""

    physical: ModelCost
    logical: ModelCost

    @property
    def physical_total(self) -> ModelCost:
        """Return physically charged model work for this arm."""
        return self.physical

    @property
    def logical_total(self) -> ModelCost:
        """Return logical model work including the shared initial bank."""
        return self.logical


@dataclass(frozen=True, slots=True)
class GeneratedInitialBank:
    """Frozen target bank reused by revision arms."""

    target_id: str
    prompt_context: str
    prompt_sha256: str
    target: StructuredTarget
    raw_samples: tuple[SlotSample, ...]
    bank: CandidateBank
    selected: tuple[Candidate, ...]
    initial_seeds: tuple[int, ...]
    mask_token_id: int
    block_size: int
    cost: ModelCost


@dataclass(frozen=True, slots=True)
class RevisedCandidate:
    """Accepted child sample or unique-parent fallback."""

    parent: Candidate
    sample: SlotSample
    snapshot: SnapshotLike
    used_revision: bool


@dataclass(frozen=True, slots=True)
class GeneratedBank:
    """Revision arm output with hashes, feedback, and costs."""

    target_id: str
    prompt_sha256: str
    bank: CandidateBank
    selected: tuple[Candidate, ...]
    revised: tuple[RevisedCandidate, ...]
    remask_indexes: tuple[tuple[int, ...], ...]
    feedback_observations: tuple[ProbeObservation, ...]
    cost: ModelCostSummary


def generate_initial_bank(
    request: InitialBankRequest,
    *,
    tokenizer: SlotTokenizer,
    model: SlotModel,
) -> GeneratedInitialBank:
    """Generate exactly 32 raw particles and freeze a 16+ unique bank."""
    if len(request.initial_seeds) != RAW_SAMPLE_COUNT:
        raise BankGenerationError(reason="expected exactly 32 initial seeds")
    if request.initial_steps < 1:
        raise BankGenerationError(reason="initial steps must be positive")
    target = load_structured_target(request.target_id)
    prompt_context = request.prompt_path.read_text(encoding="utf-8")
    sampler = GenericSlotSampler(
        tokenizer,
        model,
        schema=_schema_from_target(target),
        mask_token_id=request.mask_token_id,
        block_size=request.block_size,
        prompt_context=prompt_context,
    )
    raw_samples = tuple(
        sampler.initialize(seed=seed, steps=request.initial_steps) for seed in request.initial_seeds
    )
    bank = freeze_candidate_bank(target, raw_samples)
    selected = select_diverse(bank, SELECTED_CANDIDATE_COUNT)
    return GeneratedInitialBank(
        target_id=target.target_id,
        prompt_context=prompt_context,
        prompt_sha256=sha256(prompt_context.encode()).hexdigest(),
        target=target,
        raw_samples=raw_samples,
        bank=bank,
        selected=selected,
        initial_seeds=request.initial_seeds,
        mask_token_id=request.mask_token_id,
        block_size=request.block_size,
        cost=_cost_from_samples(raw_samples),
    )


def revise_bank(
    initial: GeneratedInitialBank,
    request: BankRevisionRequest,
    *,
    tokenizer: SlotTokenizer,
    model: SlotModel,
) -> GeneratedBank:
    """Apply a full, selective, or pseudo-feedback revision policy."""
    if request.revision_steps < 1:
        raise BankGenerationError(reason="revision steps must be positive")
    sampler = GenericSlotSampler(
        tokenizer,
        model,
        schema=_schema_from_target(initial.target),
        mask_token_id=initial.mask_token_id,
        block_size=initial.block_size,
        prompt_context=initial.prompt_context,
    )
    feedback = _feedback_for_mode(request)
    remask_indexes_by_candidate = _remask_indexes_for_mode(
        request=request,
        selected=initial.selected,
        feedback=feedback,
        mutable_indexes=tuple(
            index for index, field in enumerate(initial.target.fields) if field.mutable
        ),
    )
    revision_samples = tuple(
        sampler.revise(
            candidate.sample,
            remask_indexes=indexes,
            seed=_revision_seed(initial, request, position),
            steps=request.revision_steps,
        )
        for position, (candidate, indexes) in enumerate(
            zip(initial.selected, remask_indexes_by_candidate, strict=True)
        )
    )
    revised = _reconcile_revisions(initial.target, initial.selected, revision_samples)
    revision_cost = _cost_from_samples(revision_samples)
    return GeneratedBank(
        target_id=initial.target_id,
        prompt_sha256=initial.prompt_sha256,
        bank=initial.bank,
        selected=initial.selected,
        revised=revised,
        remask_indexes=remask_indexes_by_candidate,
        feedback_observations=feedback,
        cost=ModelCostSummary(
            physical=revision_cost,
            logical=_add_cost(initial.cost, revision_cost),
        ),
    )


def _schema_from_target(target: StructuredTarget) -> SlotSchema:
    fields = target.fields
    return SlotSchema(
        label=target.target_id,
        field_names=tuple(field.name for field in fields),
        cardinalities=tuple(field.cardinality for field in fields),
        mutable_indexes=tuple(index for index, field in enumerate(fields) if field.mutable),
    )


def _feedback_for_mode(request: BankRevisionRequest) -> tuple[ProbeObservation, ...]:
    match request.mode:
        case RevisionMode.FULL:
            return ()
        case RevisionMode.SELECTIVE:
            return request.observations
        case RevisionMode.PSEUDO:
            return pseudo_feedback(request.observations, seed=request.pseudo_seed)


def _remask_indexes_for_mode(
    *,
    request: BankRevisionRequest,
    selected: tuple[Candidate, ...],
    feedback: tuple[ProbeObservation, ...],
    mutable_indexes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    match request.mode:
        case RevisionMode.FULL:
            return tuple(mutable_indexes for _ in selected)
        case RevisionMode.SELECTIVE | RevisionMode.PSEUDO:
            return tuple((index,) for index in remask_plan(selected, feedback, mutable_indexes))


def _reconcile_revisions(
    target: StructuredTarget,
    selected: tuple[Candidate, ...],
    revision_samples: tuple[SlotSample, ...],
) -> tuple[RevisedCandidate, ...]:
    accepted: set[str] = set()
    parent_hashes = {candidate.snapshot.sha256 for candidate in selected}
    revised: list[RevisedCandidate] = []
    for candidate, sample in zip(selected, revision_samples, strict=True):
        snapshot = target.materialize(sample.selectors)
        conflicts = snapshot.sha256 in accepted or snapshot.sha256 in parent_hashes
        if conflicts:
            revised.append(
                RevisedCandidate(
                    parent=candidate,
                    sample=candidate.sample,
                    snapshot=candidate.snapshot,
                    used_revision=False,
                )
            )
            accepted.add(candidate.snapshot.sha256)
        else:
            revised.append(
                RevisedCandidate(
                    parent=candidate,
                    sample=sample,
                    snapshot=snapshot,
                    used_revision=True,
                )
            )
            accepted.add(snapshot.sha256)
    if len(accepted) != len(selected):
        raise BankGenerationError(reason="revision reconciliation lost uniqueness")
    return tuple(revised)


def _cost_from_samples(samples: tuple[SlotSample, ...]) -> ModelCost:
    return ModelCost(
        nfe=sum(sample.nfe for sample in samples),
        forward_calls=sum(sample.forward_calls for sample in samples),
        gpu_ms=sum(sample.gpu_ms for sample in samples),
    )


def _revision_seed(
    initial: GeneratedInitialBank,
    request: BankRevisionRequest,
    position: int,
) -> int:
    if request.revision_seed_base is not None:
        return request.revision_seed_base + position
    return initial.initial_seeds[position] + RAW_SAMPLE_COUNT


def _add_cost(initial: ModelCost, revision: ModelCost) -> ModelCost:
    return ModelCost(
        nfe=initial.nfe + revision.nfe,
        forward_calls=initial.forward_calls + revision.forward_calls,
        gpu_ms=initial.gpu_ms + revision.gpu_ms,
    )
