"""Immutable contracts for target-agnostic dLLM slot denoising."""

from dataclasses import dataclass
from typing import Protocol, override


@dataclass(slots=True)
class SlotSamplerError(Exception):
    """Raised when a slot schema, model result, or revision is invalid."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the boundary failure reason."""
        return self.reason


class SlotTokenizer(Protocol):
    """Tokenizer operations required by the constrained slot sampler."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode prompt text into token identifiers."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode token identifiers into text."""
        ...

    def __len__(self) -> int:
        """Return the tokenizer vocabulary size."""
        ...


@dataclass(frozen=True, slots=True)
class SlotSchema:
    """Target label, ordered fields, selector cardinalities, and mutable slots."""

    label: str
    field_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    mutable_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BlockDenoiseResult:
    """Tokens and confidence returned by one affected-block forward call."""

    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    gpu_ms: int


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Hash-bound token commitments made during one denoising step."""

    step_index: int
    block_index: int
    masked_before: int
    committed_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    remaining_masked: int
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SlotSample:
    """Completed target selector state and its denoising provenance."""

    token_ids: tuple[int, ...]
    selectors: tuple[int, ...]
    slot_positions: tuple[int, ...]
    slot_confidences: tuple[float, ...]
    selected_slot_token_ids: tuple[int, ...]
    requested_steps: int
    nfe: int
    forward_calls: int
    gpu_ms: int
    trajectory_steps: tuple[TrajectoryStep, ...]
    state_sha256: str
    parent_state_sha256: str | None = None


class SlotModel(Protocol):
    """Model operation required for constrained block-level denoising."""

    def denoise_block(  # noqa: PLR0913, PLR0917 - mirrors the model adapter boundary.
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        """Denoise every supplied position within one causal block."""
        ...
