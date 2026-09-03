"""Proposal-aware slot filling for selective revision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from dllm_agent.targets.openexr.particle import FIELD_ORDER, selector_class
from tools.runtime.selective_revision_json import TrajectoryStep, WireProtocolError
from tools.runtime.selective_revision_template import (
    BlockDenoiseResult,
    FillResult,
    SlotTokenizer,
    Template,
    selector_for_token,
    selector_slots,
)
from tools.runtime.selective_revision_validation import transfer_counts

_FORWARD_SEED_STRIDE: Final = 104_729


class SlotModel(Protocol):
    """Small model surface for selective slot denoising."""

    def denoise_block(  # noqa: PLR0913, PLR0917 - block denoising context is explicit.
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        """Propose tokens for all supplied masked positions."""
        ...


@dataclass(frozen=True, slots=True)
class FillRequest:
    """One complete masked-slot fill operation."""

    token_ids: tuple[int, ...]
    slot_indexes: tuple[int, ...]
    previous_token_ids: tuple[int | None, ...]
    steps: int
    seed: int


@dataclass(frozen=True, slots=True, order=True)
class _Candidate:
    confidence: float
    position: int
    slot_index: int
    token_id: int
    block_index: int
    forward_ordinal: int


@dataclass(frozen=True, slots=True)
class _ForwardProposal:
    ordinal: int
    block_index: int
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    margins: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SlotFiller:
    """Execute proposal forwards and commit the scheduled highest-confidence tokens."""

    tokenizer: SlotTokenizer
    model: SlotModel
    template: Template
    field_allowed_token_ids: tuple[tuple[int, ...], ...]
    block_size: int

    def fill(self, request: FillRequest) -> FillResult:
        """Fill requested slots while retaining every model proposal row."""
        positions = tuple(self.template.slot_positions[index] for index in request.slot_indexes)
        current = list(request.token_ids)
        selected: list[int] = []
        selected_indexes: list[int] = []
        selected_confidences: list[float] = []
        forward_blocks: list[int] = []
        trajectory: list[TrajectoryStep] = []
        gpu_ms = 0
        next_forward_ordinal = 0
        remaining = dict(
            zip(
                positions,
                zip(request.slot_indexes, request.previous_token_ids, strict=True),
                strict=True,
            )
        )
        for step_index, transfer_count in enumerate(
            transfer_counts(len(remaining), request.steps), start=1
        ):
            if transfer_count <= 0 or not remaining:
                continue
            masked_before = len(remaining)
            forwards, candidates, step_gpu_ms = self._propose(
                current,
                remaining,
                seed=request.seed,
                step_index=step_index,
                first_ordinal=next_forward_ordinal,
            )
            next_forward_ordinal += len(forwards)
            gpu_ms += step_gpu_ms
            forward_blocks.extend(forward.block_index for forward in forwards)
            commits = tuple(
                sorted(candidates, reverse=True)[: min(transfer_count, len(candidates))]
            )
            commits_by_forward: dict[int, list[_Candidate]] = {}
            for commit in commits:
                current[commit.position] = commit.token_id
                selected.append(commit.token_id)
                selected_indexes.append(commit.slot_index)
                selected_confidences.append(commit.confidence)
                commits_by_forward.setdefault(commit.forward_ordinal, []).append(commit)
                del remaining[commit.position]
            for forward in forwards:
                row_commits = tuple(commits_by_forward.get(forward.ordinal, ()))
                trajectory.append(
                    TrajectoryStep(
                        step_index=step_index,
                        block_index=forward.block_index,
                        masked_before=masked_before,
                        committed_positions=tuple(commit.position for commit in row_commits),
                        token_ids=tuple(commit.token_id for commit in row_commits),
                        confidences=tuple(commit.confidence for commit in row_commits),
                        proposal_positions=forward.positions,
                        proposal_token_ids=forward.token_ids,
                        proposal_confidences=forward.confidences,
                        proposal_margins=forward.margins,
                        remaining_masked=len(remaining),
                    )
                )
        selectors = tuple(
            selector_for_token(self.tokenizer, current[position])
            for position in self.template.slot_positions
        )
        return FillResult(
            token_ids=tuple(current),
            selected_slot_token_ids=tuple(selected),
            selected_slot_indexes=tuple(selected_indexes),
            selectors=selector_slots(selectors),
            confidences=tuple(selected_confidences),
            forward_blocks=tuple(dict.fromkeys(forward_blocks)),
            nfe=len(trajectory),
            gpu_ms=gpu_ms,
            trajectory_steps=tuple(trajectory),
        )

    def _propose(
        self,
        current: list[int],
        remaining: dict[int, tuple[int, int | None]],
        *,
        seed: int,
        step_index: int,
        first_ordinal: int,
    ) -> tuple[tuple[_ForwardProposal, ...], tuple[_Candidate, ...], int]:
        forwards: list[_ForwardProposal] = []
        candidates: list[_Candidate] = []
        gpu_ms = 0
        groups = self._groups(remaining)
        for block_index, allowed_token_ids in sorted(groups):
            positions, previous = groups[(block_index, allowed_token_ids)]
            ordinal = first_ordinal + len(forwards)
            result = self.model.denoise_block(
                block_index,
                tuple(current),
                positions,
                allowed_token_ids,
                previous,
                seed + (ordinal * _FORWARD_SEED_STRIDE),
                step_index,
            )
            self._validate_result(result, positions)
            forwards.append(
                _ForwardProposal(
                    ordinal=ordinal,
                    block_index=block_index,
                    positions=positions,
                    token_ids=result.token_ids,
                    confidences=result.confidences,
                    margins=result.proposal_margins,
                )
            )
            gpu_ms += result.gpu_ms
            for position, token_id, confidence in zip(
                positions, result.token_ids, result.confidences, strict=True
            ):
                if token_id not in allowed_token_ids:
                    raise WireProtocolError(reason="model selected a disallowed token")
                candidates.append(
                    _Candidate(
                        confidence=confidence,
                        position=position,
                        slot_index=remaining[position][0],
                        token_id=token_id,
                        block_index=block_index,
                        forward_ordinal=ordinal,
                    )
                )
        return tuple(forwards), tuple(candidates), gpu_ms

    def _groups(
        self, remaining: dict[int, tuple[int, int | None]]
    ) -> dict[tuple[int, tuple[int, ...]], tuple[tuple[int, ...], tuple[int | None, ...]]]:
        grouped: dict[tuple[int, tuple[int, ...]], tuple[list[int], list[int | None]]] = {}
        for position, (slot_index, previous) in remaining.items():
            allowed = self._allowed_for_slot(slot_index, previous)
            key = (position // self.block_size, allowed)
            positions, previous_tokens = grouped.setdefault(key, ([], []))
            positions.append(position)
            previous_tokens.append(previous)
        return {
            key: (tuple(positions), tuple(previous_tokens))
            for key, (positions, previous_tokens) in grouped.items()
        }

    def _allowed_for_slot(self, slot_index: int, previous_token_id: int | None) -> tuple[int, ...]:
        allowed = self.field_allowed_token_ids[slot_index]
        if previous_token_id is None:
            return allowed
        field = FIELD_ORDER[slot_index]
        parent_selector = selector_for_token(self.tokenizer, previous_token_id)
        parent_class = selector_class(field, parent_selector)
        revised = tuple(token_id for index, token_id in enumerate(allowed) if index != parent_class)
        if not revised:
            raise WireProtocolError(reason="remask field has no alternate selector class")
        return revised

    @staticmethod
    def _validate_result(result: BlockDenoiseResult, positions: tuple[int, ...]) -> None:
        expected = len(positions)
        if (
            len(result.token_ids) != expected
            or len(result.confidences) != expected
            or len(result.proposal_margins) != expected
        ):
            raise WireProtocolError(reason="model returned mismatched proposal lengths")
