"""Generic dLLM slot sampler contracts used by all structured targets."""

from dataclasses import dataclass

from dllm_agent.experiments.multitarget.slot_sampler import (
    BlockDenoiseResult,
    GenericSlotSampler,
    SlotSchema,
)


@dataclass(frozen=True, slots=True)
class TinyTokenizer:
    """Tokenizer with four one-token unsigned integer classes."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 + (sum(text.encode("utf-8")) % 8)]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        decoded = {10: "0", 11: "1", 12: "2", 13: "3"}
        return decoded.get(token_ids[0], "<text>")

    def __len__(self) -> int:
        return 20


class TinyModel:
    """Deterministic constrained model that records affected slot positions."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def denoise_block(  # noqa: PLR0913, PLR0917
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        del block_index, token_ids, previous_token_ids
        self.calls.append((mask_positions, seed, step_index))
        selected = tuple(
            allowed_token_ids[(seed + position + step_index) % len(allowed_token_ids)]
            for position in mask_positions
        )
        return BlockDenoiseResult(
            token_ids=selected,
            confidences=tuple(0.75 for _ in selected),
            gpu_ms=3,
        )


def test_initial_denoising_commits_all_ten_typed_slots_with_trajectory_evidence() -> None:
    # Given
    model = TinyModel()
    sampler = GenericSlotSampler(
        TinyTokenizer(),
        model,
        schema=_schema(),
        mask_token_id=99,
        block_size=64,
        prompt_context="target parser context",
    )

    # When
    sample = sampler.initialize(seed=7, steps=8)

    # Then
    assert len(sample.selectors) == 10
    assert len(sample.slot_positions) == 10
    assert len(sample.slot_confidences) == 10
    assert sample.nfe == sample.forward_calls == len(model.calls)
    assert sample.nfe > 0
    assert sample.gpu_ms == sample.nfe * 3
    assert sample.trajectory_steps[-1].remaining_masked == 0
    assert tuple(sorted({step.step_index for step in sample.trajectory_steps})) == tuple(
        range(1, 9)
    )
    assert any(call[2] > 1 for call in model.calls)
    assert len(sample.state_sha256) == 64


def test_selective_revision_reopens_only_requested_slot_and_excludes_parent_class() -> None:
    # Given
    model = TinyModel()
    sampler = GenericSlotSampler(
        TinyTokenizer(),
        model,
        schema=_schema(),
        mask_token_id=99,
        block_size=64,
    )
    parent = sampler.initialize(seed=3, steps=8)
    model.calls.clear()

    # When
    child = sampler.revise(parent, remask_indexes=(2,), seed=19, steps=2)

    # Then
    assert child.selectors[2] != parent.selectors[2]
    assert child.selectors[:2] == parent.selectors[:2]
    assert child.selectors[3:] == parent.selectors[3:]
    changed_positions = {position for positions, _, _ in model.calls for position in positions}
    assert changed_positions == {parent.slot_positions[2]}
    assert child.parent_state_sha256 == parent.state_sha256


def test_full_remask_uses_same_revision_step_budget_but_reopens_every_slot() -> None:
    # Given
    model = TinyModel()
    sampler = GenericSlotSampler(
        TinyTokenizer(),
        model,
        schema=_schema(),
        mask_token_id=99,
        block_size=64,
    )
    parent = sampler.initialize(seed=5, steps=8)
    model.calls.clear()

    # When
    child = sampler.revise(parent, remask_indexes=tuple(range(10)), seed=23, steps=2)

    # Then
    changed_positions = {position for positions, _, _ in model.calls for position in positions}
    assert changed_positions == set(parent.slot_positions)
    assert child.parent_state_sha256 == parent.state_sha256
    assert child.requested_steps == 2
    assert {step.step_index for step in child.trajectory_steps} == {1, 2}


def _schema() -> SlotSchema:
    return SlotSchema(
        label="TEST",
        field_names=tuple(f"field_{index}" for index in range(10)),
        cardinalities=(4, 4, 4, 4, 2, 2, 2, 2, 4, 4),
        mutable_indexes=tuple(range(10)),
    )
