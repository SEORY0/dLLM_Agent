"""Shared typed fixtures for selective-revision sampler tests."""

from dataclasses import dataclass

from tools.runtime.selective_revision_json import InitializeRequest, ReviseRequest, SelectorState
from tools.runtime.selective_revision_template import BlockDenoiseResult, build_template
from tools.selective_revision_payloads import sha256_token_ids

from dllm_agent.targets.openexr.particle import FIELD_ORDER, SelectorSlots


@dataclass(frozen=True, slots=True)
class TinyTokenizer:
    """Deterministic tokenizer with one-token fields and numeric IDs."""

    pieces: dict[str, int]
    decoded: dict[int, str]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.pieces[text]]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return self.decoded.get(token_ids[0], "<unused>")

    def __len__(self) -> int:
        return max(self.decoded) + 1


class TinyModel:
    """Tiny slot model that records block forwards and chosen tokens."""

    def __init__(self, selections: tuple[int, ...]) -> None:
        self._selections: tuple[int, ...] = selections
        self.denoise_calls: list[tuple[int, tuple[int, ...], int, int]] = []

    def denoise_block(  # noqa: PLR0913, PLR0917 - mirrors sampler model protocol.
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        del token_ids
        self.denoise_calls.append((block_index, mask_positions, seed, step_index))
        chosen: list[int] = []
        confidences: list[float] = []
        for index, previous_token_id in enumerate(previous_token_ids):
            choice = self._selections[index]
            if choice not in allowed_token_ids:
                choice = allowed_token_ids[0]
            chosen.append(choice)
            confidences.append(0.5 if choice == previous_token_id else 0.9)
        return BlockDenoiseResult(
            token_ids=tuple(chosen),
            confidences=tuple(confidences),
            proposal_margins=tuple(0.4 for _ in chosen),
            gpu_ms=7,
        )


def tiny_tokenizer() -> TinyTokenizer:
    fields = {field: 100 + index for index, field in enumerate(FIELD_ORDER)}
    punctuation = {
        "EXR": 1,
        "=": 2,
        ";": 3,
        "\n": 4,
        "fields": 5,
        "selectors": 6,
        "[": 7,
        "]": 8,
        ",": 9,
    }
    numbers = {"0": 10, "1": 11, "2": 12, "3": 13}
    pieces = fields | punctuation | numbers
    decoded = {token_id: text for text, token_id in pieces.items()}
    decoded[99] = "<mask>"
    return TinyTokenizer(pieces=pieces, decoded=decoded)


def initialize_request(seed: int, *, steps: int = 8) -> InitializeRequest:
    return InitializeRequest(
        request_id=f"init-{seed}",
        particle_id="particle-1",
        seed=seed,
        steps=steps,
    )


def parent_state(tokenizer: TinyTokenizer, slot_tokens: tuple[int, ...]) -> SelectorState:
    template = build_template(tokenizer, mask_token_id=99)
    token_ids = list(template.token_ids)
    for position, token_id in zip(template.slot_positions, slot_tokens, strict=True):
        token_ids[position] = token_id
    selectors = tuple(
        int(tokenizer.decode([token_id], skip_special_tokens=False)) for token_id in slot_tokens
    )
    return SelectorState(
        token_ids=tuple(token_ids),
        slot_positions=template.slot_positions,
        selectors=_selector_slots(selectors),
        token_state_sha256=sha256_token_ids(tuple(token_ids)),
    )


def revise_request(  # noqa: PLR0913 - test builder mirrors wire fields.
    parent: SelectorState,
    *,
    remask_bitmap: int = 1,
    frozen_bitmap: int | None = None,
    parent_token_sha256: str | None = None,
    feedback_digest: str = "f" * 64,
    seed: int,
    revision_steps: int = 8,
) -> ReviseRequest:
    return ReviseRequest(
        request_id=f"revise-{seed}",
        particle_id="particle-1",
        parent=parent,
        parent_token_sha256=parent.token_state_sha256
        if parent_token_sha256 is None
        else parent_token_sha256,
        remask_bitmap=remask_bitmap,
        frozen_bitmap=((1 << len(FIELD_ORDER)) - 1) ^ remask_bitmap
        if frozen_bitmap is None
        else frozen_bitmap,
        revision_steps=revision_steps,
        seed=seed,
        rng_state_sha256="a" * 64,
        feedback_digest=feedback_digest,
        mode="selective-revision/v1",
    )


def _selector_slots(values: tuple[int, ...]) -> SelectorSlots:
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
    )
