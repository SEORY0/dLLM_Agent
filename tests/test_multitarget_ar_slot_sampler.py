"""AR-parent control sampler: strict left-to-right slot decoding contracts."""

from dataclasses import dataclass, field

from dllm_agent.experiments.multitarget.ar_slot_sampler import AutoregressiveSlotSampler
from dllm_agent.experiments.multitarget.slot_sampler_contracts import SlotSchema

_VOCAB_SIZE = 20


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
        return _VOCAB_SIZE


@dataclass(slots=True)
class RecordingArModel:
    """Deterministic next-token model that records every prefix it was given."""

    prefixes: list[tuple[int, ...]] = field(default_factory=list)

    def next_token_logits(self, token_ids: tuple[int, ...]) -> tuple[float, ...]:
        self.prefixes.append(token_ids)
        # Favor the class matching (prefix length % cardinality) so the choice
        # is deterministic and traceable back to how much context was seen.
        return tuple(float(-abs(index - len(token_ids))) for index in range(_VOCAB_SIZE))


def _schema() -> SlotSchema:
    return SlotSchema(
        label="TEST",
        field_names=tuple(f"field_{index}" for index in range(10)),
        cardinalities=(4, 4, 4, 4, 2, 2, 2, 2, 4, 4),
        mutable_indexes=tuple(range(10)),
    )


def test_initialize_commits_all_ten_slots_in_ascending_position_order() -> None:
    # Given
    model = RecordingArModel()
    sampler = AutoregressiveSlotSampler(
        TinyTokenizer(), model, schema=_schema(), mask_token_id=99
    )

    # When
    sample = sampler.initialize(seed=7)

    # Then
    assert len(sample.selectors) == 10
    assert sample.nfe == sample.forward_calls == 10 == len(model.prefixes)
    assert tuple(step.step_index for step in sample.trajectory_steps) == tuple(range(1, 11))
    assert all(len(step.committed_positions) == 1 for step in sample.trajectory_steps)
    assert sample.trajectory_steps[-1].remaining_masked == 0
    assert len(sample.state_sha256) == 64


def test_each_forward_call_only_sees_positions_already_committed() -> None:
    # Given
    model = RecordingArModel()
    sampler = AutoregressiveSlotSampler(
        TinyTokenizer(), model, schema=_schema(), mask_token_id=99
    )

    # When
    sample = sampler.initialize(seed=1)

    # Then: prefix length equals the slot's own position -- so the call never
    # includes that slot's own (not-yet-decided) token or anything after it,
    # and by the time slot N is decoded every earlier slot's mask token has
    # already been replaced by a real committed value.
    for prefix, position in zip(model.prefixes, sample.slot_positions, strict=True):
        assert len(prefix) == position
        assert 99 not in prefix


def test_selected_tokens_are_within_the_slot_catalog_and_confidence_is_a_probability() -> None:
    # Given
    model = RecordingArModel()
    sampler = AutoregressiveSlotSampler(
        TinyTokenizer(), model, schema=_schema(), mask_token_id=99
    )

    # When
    sample = sampler.initialize(seed=42)

    # Then
    schema = _schema()
    for selector, cardinality in zip(sample.selectors, schema.cardinalities, strict=True):
        assert 0 <= selector < cardinality
    for confidence in sample.slot_confidences:
        assert 0.0 < confidence <= 1.0
