"""Committed-token selective revision sampler tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.runtime.selective_revision_json import (  # noqa: E402
    InitializeRequest,
    SamplerResponse,
    SelectorState,
    WireProtocolError,
)
from tools.runtime.selective_revision_template import (  # noqa: E402
    build_template,
    numeric_token_ids,
)
from tools.selective_revision_payloads import (  # noqa: E402
    initialize_worker_request,
    parse_response_line,
    request_hash,
    response_with_hash,
)
from tools.selective_revision_sampler import SelectiveRevisionSampler  # noqa: E402
from tools.selective_revision_worker import handle_worker_request  # noqa: E402

from dllm_agent.targets.openexr.particle import FIELD_ORDER  # noqa: E402
from tests.selective_revision_sampler_fixtures import (  # noqa: E402
    TinyModel,
    TinyTokenizer,
    initialize_request,
    parent_state,
    revise_request,
    tiny_tokenizer,
)


def test_template_has_ten_typed_numeric_slots() -> None:
    # Given
    tokenizer = tiny_tokenizer()

    # When
    template = build_template(tokenizer, mask_token_id=99)

    # Then
    assert template.slot_positions == tuple(
        index for index, token_id in enumerate(template.token_ids) if token_id == 99
    )
    assert len(template.slot_positions) == 10
    assert tuple(span.field for span in template.field_spans) == FIELD_ORDER
    assert numeric_token_ids(tokenizer) == (10, 11, 12, 13)


def test_numeric_token_ids_ignore_unicode_digits_that_int_cannot_parse() -> None:
    # Given
    tokenizer = TinyTokenizer(
        {**tiny_tokenizer().pieces, "①": 14},
        {**tiny_tokenizer().decoded, 14: "①"},
    )

    # When / Then
    assert numeric_token_ids(tokenizer) == (10, 11, 12, 13)


def test_initialize_masks_and_fills_all_slots_with_allowed_numeric_tokens() -> None:
    # Given
    model = TinyModel((10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(
        tiny_tokenizer(),
        model,
        mask_token_id=99,
        block_size=64,
    )

    # When
    result = sampler.initialize(initialize_request(seed=7))

    # Then
    assert result.selectors == (0, 1, 2, 3, 0, 0, 0, 0, 0, 1)
    assert result.selected_slot_token_ids == (11, 10, 10, 10, 10, 10, 13, 12, 11, 10)
    assert result.before_remask_token_ids == tuple(99 for _ in FIELD_ORDER)
    assert result.nfe == 12
    assert result.forward_calls == 12
    assert result.forward_blocks == (0,)
    assert result.gpu_ms == 84
    assert len(result.trajectory_steps) == result.nfe
    assert tuple(
        sum(
            len(step.committed_positions)
            for step in result.trajectory_steps
            if step.step_index == step_index
        )
        for step_index in range(1, 9)
    ) == (2, 2, 1, 1, 1, 1, 1, 1)
    assert all(step.proposal_positions for step in result.trajectory_steps)
    assert all(step.block_index == 0 for step in result.trajectory_steps)
    assert model.denoise_calls[0] == (
        0,
        result.slot_positions[4:8],
        7,
        1,
    )
    last_block, last_positions, last_seed, last_step = model.denoise_calls[-1]
    assert (last_block, last_positions, last_step) == (0, (result.slot_positions[0],), 8)
    assert last_seed != 7


def test_worker_handle_binds_response_hash_to_wrapped_worker_request() -> None:
    # Given
    payload = initialize_request(seed=11)
    worker_request = initialize_worker_request(payload)
    sampler = SelectiveRevisionSampler(
        tiny_tokenizer(),
        TinyModel((10, 11, 12, 13, 10, 11, 12, 13, 10, 11)),
        mask_token_id=99,
        block_size=64,
    )

    # When
    response = parse_response_line(handle_worker_request(sampler, worker_request))

    # Then
    assert isinstance(response, SamplerResponse)
    assert response.request_sha256 == request_hash(worker_request)
    assert response.request_sha256 != request_hash(payload)
    assert response_with_hash(response).response_hash == response.response_hash


def test_revision_excludes_parent_selector_class_for_mutable_field() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 10, 10, 10, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((12,)), mask_token_id=99, block_size=64)

    # When
    result = sampler.revise(
        revise_request(parent, remask_bitmap=1 << FIELD_ORDER.index("width"), seed=12)
    )

    # Then
    width_position = parent.slot_positions[FIELD_ORDER.index("width")]
    assert result.status == "revised"
    assert result.token_ids[width_position] != parent.token_ids[width_position]
    width_index = FIELD_ORDER.index("width")
    assert result.selectors[width_index] != parent.selectors[width_index]


def test_revision_commits_each_allowed_model_selected_token() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    origin_x = FIELD_ORDER.index("origin_x")
    parent_a = parent_state(tokenizer, (10, 10, 10, 10, 10, 10, 10, 10, 10, 10))
    parent_b = parent_state(tokenizer, (11, 10, 10, 10, 10, 10, 10, 10, 10, 10))
    # When
    child_a = SelectiveRevisionSampler(
        tokenizer, TinyModel((11,)), mask_token_id=99, block_size=64
    ).revise(revise_request(parent_a, remask_bitmap=1 << origin_x, seed=20))
    child_b = SelectiveRevisionSampler(
        tokenizer, TinyModel((10,)), mask_token_id=99, block_size=64
    ).revise(revise_request(parent_b, remask_bitmap=1 << origin_x, seed=22))
    # Then
    assert child_a.selectors[origin_x] == 1
    assert child_b.selectors[origin_x] == 0
    assert child_a.selected_slot_token_ids == (11,)
    assert child_b.selected_slot_token_ids == (10,)


def test_revise_reopens_only_selected_committed_slot_and_preserves_frozen_tokens() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((13,)), mask_token_id=99, block_size=64)

    # When
    result = sampler.revise(
        revise_request(parent, remask_bitmap=1 << FIELD_ORDER.index("width"), seed=2)
    )

    # Then
    width_index = FIELD_ORDER.index("width")
    assert result.status == "revised"
    assert result.before_remask_token_ids == (12,)
    assert result.remasked_positions == (parent.slot_positions[width_index],)
    assert result.selected_slot_token_ids == (13,)
    assert result.token_ids[parent.slot_positions[width_index]] == 13
    assert result.suggested_field == "width"
    for index, position in enumerate(parent.slot_positions):
        if index != width_index:
            assert result.token_ids[position] == parent.token_ids[position]


def test_revision_parent_hash_or_slot_bitmap_mismatch_fails_closed() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((13,)), mask_token_id=99, block_size=64)

    # When / Then
    with pytest.raises(WireProtocolError, match="parent token hash"):
        _ = sampler.revise(revise_request(parent, parent_token_sha256="0" * 64, seed=2))
    with pytest.raises(WireProtocolError, match="slot positions"):
        _ = sampler.revise(
            revise_request(
                SelectorState(
                    token_ids=parent.token_ids,
                    slot_positions=parent.slot_positions[:-1],
                    selectors=parent.selectors,
                    token_state_sha256=parent.token_state_sha256,
                ),
                seed=2,
            )
        )


def test_unchanged_revision_is_explicit_and_cache_key_is_byte_stable() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((11,)), mask_token_id=99, block_size=64)
    request = revise_request(parent, remask_bitmap=1 << FIELD_ORDER.index("origin_y"), seed=3)

    # When
    first = sampler.revise(request)
    second = sampler.revise(request)

    # Then
    assert first.status == "revised"
    assert first.token_ids != parent.token_ids
    assert second.cache_hit is True
    assert second.response_hash == first.response_hash
    assert request_hash(request) == request_hash(request)
    changed_feedback = revise_request(parent, feedback_digest="1" * 64, seed=3)
    assert request_hash(changed_feedback) != request_hash(request)


def test_selector_slots_across_blocks_keep_exact_commit_attribution() -> None:
    # Given
    tokenizer = tiny_tokenizer()
    sampler = SelectiveRevisionSampler(
        tokenizer,
        TinyModel((10, 11, 12, 13, 10, 11, 12, 13, 10, 11)),
        mask_token_id=99,
        block_size=16,
    )

    # When
    result = sampler.initialize(
        InitializeRequest(request_id="multi-block", particle_id="particle-1", seed=5, steps=1)
    )

    # Then
    committed = tuple(
        position for step in result.trajectory_steps for position in step.committed_positions
    )
    assert len(result.forward_blocks) > 1
    assert len(result.trajectory_steps) == result.nfe
    assert sorted(committed) == sorted(result.slot_positions)
    assert len(committed) == len(set(committed))
    assert all(
        all(position // 16 == step.block_index for position in step.committed_positions)
        for step in result.trajectory_steps
    )
    assert all(
        all(position // 16 == step.block_index for position in step.proposal_positions)
        for step in result.trajectory_steps
    )


def test_revision_rejects_non_complementary_bitmaps() -> None:
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 11, 12, 13, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((13,)), mask_token_id=99, block_size=64)
    with pytest.raises(WireProtocolError, match="cover every slot"):
        _ = sampler.revise(revise_request(parent, remask_bitmap=1, frozen_bitmap=0, seed=6))
    with pytest.raises(WireProtocolError, match="overlap"):
        _ = sampler.revise(revise_request(parent, remask_bitmap=1, frozen_bitmap=1, seed=6))


def test_revision_rejects_fixed_cardinality_remask_field() -> None:
    tokenizer = tiny_tokenizer()
    parent = parent_state(tokenizer, (10, 11, 12, 13, 10, 10, 10, 10, 10, 11))
    sampler = SelectiveRevisionSampler(tokenizer, TinyModel((10,)), mask_token_id=99, block_size=64)
    compression = 1 << FIELD_ORDER.index("compression")

    with pytest.raises(WireProtocolError, match="non-materialized field"):
        _ = sampler.revise(revise_request(parent, remask_bitmap=compression, seed=7))
