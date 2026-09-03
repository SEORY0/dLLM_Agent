"""Block-level selective revision sampler tests."""

import sys
from pathlib import Path
from typing import override

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.runtime.selective_revision_template import BlockDenoiseResult  # noqa: E402
from tools.selective_revision_sampler import SelectiveRevisionSampler  # noqa: E402

from dllm_agent.targets.openexr.particle import FIELD_ORDER  # noqa: E402
from tests.selective_revision_sampler_fixtures import (  # noqa: E402
    TinyModel,
    initialize_request,
    parent_state,
    revise_request,
    tiny_tokenizer,
)


def test_block_forward_sees_multiple_masks_and_honors_step_cap() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    model = TinyModel((12, 13))
    sampler = SelectiveRevisionSampler(tokenizer, model, mask_token_id=99, block_size=64)

    # When
    result = sampler.revise(
        revise_request(
            parent,
            remask_bitmap=(1 << FIELD_ORDER.index("origin_x"))
            | (1 << FIELD_ORDER.index("origin_y")),
            seed=42,
            revision_steps=1,
        )
    )

    # Then
    assert result.nfe == 2
    assert result.forward_blocks == (0,)
    assert [(block, positions, step) for block, positions, _seed, step in model.denoise_calls] == [
        (0, (parent.slot_positions[1],), 1),
        (0, (parent.slot_positions[0],), 1),
    ]
    assert model.denoise_calls[0][2] == 42
    assert len({call[2] for call in model.denoise_calls}) == 2


def test_seeded_block_model_can_produce_reproducible_diversity() -> None:
    # Given
    tokenizer = tiny_tokenizer()

    class SeedModel(TinyModel):
        @override
        def denoise_block(
            self,
            block_index: int,
            token_ids: tuple[int, ...],
            mask_positions: tuple[int, ...],
            allowed_token_ids: tuple[int, ...],
            previous_token_ids: tuple[int | None, ...],
            seed: int,
            step_index: int,
        ) -> BlockDenoiseResult:
            del block_index, token_ids, previous_token_ids, step_index
            token = allowed_token_ids[seed % len(allowed_token_ids)]
            return BlockDenoiseResult(
                token_ids=tuple(token for _ in mask_positions),
                confidences=tuple(0.8 for _ in mask_positions),
                proposal_margins=tuple(0.5 for _ in mask_positions),
                gpu_ms=1,
            )

    # When
    first = SelectiveRevisionSampler(tokenizer, SeedModel(()), mask_token_id=99, block_size=64)
    second = SelectiveRevisionSampler(tokenizer, SeedModel(()), mask_token_id=99, block_size=64)
    third = SelectiveRevisionSampler(tokenizer, SeedModel(()), mask_token_id=99, block_size=64)

    # Then
    assert (
        first.initialize(initialize_request(seed=1)).selectors
        == second.initialize(initialize_request(seed=1)).selectors
    )
    assert (
        first.initialize(initialize_request(seed=1)).selectors
        != third.initialize(initialize_request(seed=2)).selectors
    )


def test_forward_seeds_are_unique_across_groups_and_steps() -> None:
    """Given a multi-step fill, each model forward receives a unique deterministic seed."""
    # Given
    model = TinyModel((10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(
        tiny_tokenizer(),
        model,
        mask_token_id=99,
        block_size=64,
    )

    # When
    _ = sampler.initialize(initialize_request(seed=9, steps=2))

    # Then
    forward_seeds = tuple(call[2] for call in model.denoise_calls)
    assert len(forward_seeds) > 2
    assert len(forward_seeds) == len(set(forward_seeds))


def test_sampler_records_every_proposal_before_model_selected_commit() -> None:
    """Given A then B model proposals, when B commits, then both forwards remain evident."""
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    width = FIELD_ORDER.index("width")
    height = FIELD_ORDER.index("height")
    width_position = parent.slot_positions[width]

    class ProposalModel(TinyModel):
        @override
        def denoise_block(
            self,
            block_index: int,
            token_ids: tuple[int, ...],
            mask_positions: tuple[int, ...],
            allowed_token_ids: tuple[int, ...],
            previous_token_ids: tuple[int | None, ...],
            seed: int,
            step_index: int,
        ) -> BlockDenoiseResult:
            del block_index, token_ids, previous_token_ids, seed
            position = mask_positions[0]
            is_width = position == width_position
            token_id = (
                allowed_token_ids[-1] if is_width and step_index == 2 else allowed_token_ids[0]
            )
            confidence = 0.1 if is_width and step_index == 1 else 0.9
            margin = -0.3 if is_width and step_index == 1 else 0.6
            return BlockDenoiseResult(
                token_ids=(token_id,),
                confidences=(confidence,),
                proposal_margins=(margin,),
                gpu_ms=1,
            )

    sampler = SelectiveRevisionSampler(
        tokenizer,
        ProposalModel(()),
        mask_token_id=99,
        block_size=64,
    )

    # When
    result = sampler.revise(
        revise_request(
            parent,
            remask_bitmap=(1 << width) | (1 << height),
            seed=44,
            revision_steps=2,
        )
    )

    # Then
    width_steps = tuple(
        step for step in result.trajectory_steps if width_position in step.proposal_positions
    )
    assert len(result.trajectory_steps) == result.nfe == 3
    assert tuple(step.proposal_token_ids for step in width_steps) == ((10,), (13,))
    assert tuple(step.proposal_confidences for step in width_steps) == ((0.1,), (0.9,))
    assert tuple(step.proposal_margins for step in width_steps) == ((-0.3,), (0.6,))
    assert width_steps[0].committed_positions == ()
    assert width_steps[1].committed_positions == (width_position,)
    assert result.token_ids[width_position] == 13
