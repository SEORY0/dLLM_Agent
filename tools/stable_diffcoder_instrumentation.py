"""Stable-DiffCoder generation instrumentation helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Protocol

from tools.stable_diffcoder_payloads import (
    InstrumentedStep,
    TokenTransferEvidence,
    build_token_transfer_evidence,
    sha256_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


class _TensorLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __iter__(self) -> Iterator[_TensorLike]: ...
    def detach(self) -> _TensorLike: ...
    def to(self, device: str) -> _TensorLike: ...
    def reshape(self, *shape: int) -> _TensorLike: ...
    def tolist(self) -> list[int]: ...
    def sum(self) -> _TensorLike: ...
    def item(self) -> int | float: ...
    def clone(self) -> _TensorLike: ...
    def float(self) -> _TensorLike: ...
    def double(self) -> _TensorLike: ...
    def gather(self, dim: int, index: _TensorLike) -> _TensorLike: ...
    def index_select(self, dim: int, index: _TensorLike) -> _TensorLike: ...
    def eq(self, other: int) -> _TensorLike: ...
    def nonzero(self, *, as_tuple: bool = False) -> tuple[_TensorLike, ...]: ...
    def numel(self) -> int: ...
    def mean(self) -> _TensorLike: ...
    def __getitem__(self, key: object) -> _TensorLike: ...
    def __setitem__(self, key: object, value: object) -> None: ...


class _EncodedInputLike(Protocol):
    input_ids: _TensorLike


class _TokenizerLike(Protocol):
    eos_token_id: int | None

    def __len__(self) -> int: ...

    def __call__(self, text: str, *, return_tensors: str) -> _EncodedInputLike: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str: ...

    def decode(
        self, token_ids: _TensorLike | Sequence[int], *, skip_special_tokens: bool
    ) -> str: ...


class _TransferFn(Protocol):
    def __call__(  # noqa: PLR0913, PLR0917
        self,
        logits: _TensorLike,
        temperature: float,
        remasking: str,
        mask_index: _TensorLike,
        x: _TensorLike,
        num_transfer_token: int | None,
        threshold: float | None = None,
        *,
        shift: bool = False,
    ) -> tuple[_TensorLike, _TensorLike]: ...


class _CallOutputLike(Protocol):
    logits: _TensorLike


class _ModelLike(Protocol):
    _get_transfer_index: _TransferFn

    def __call__(
        self, input_ids: _TensorLike, *, attention_mask: _TensorLike
    ) -> _CallOutputLike: ...

    def to(self, device: str) -> None: ...

    def eval(self) -> None: ...

    def generate_block(  # noqa: PLR0913
        self,
        input_ids: _TensorLike,
        *,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float,
        remasking: str,
        tokenizer: _TokenizerLike,
        shift: bool,
        threshold: float | None,
        eos_id: int | None,
    ) -> tuple[_TensorLike, int]: ...


TensorLike = _TensorLike
ModelLike = _ModelLike


@dataclass(frozen=True, slots=True)
class InstrumentationContext:
    """Stable-DiffCoder block geometry needed to map transfer positions."""

    tokenizer: _TokenizerLike
    prompt_length: int
    gen_length: int
    block_length: int


def token_block_sha256(block_tokens: _TensorLike) -> str:
    """Hash a tensor block as little-endian signed token IDs."""
    values = block_tokens.detach().to("cpu").reshape(-1).tolist()
    token_bytes = b"".join(struct.pack("<q", int(token_id)) for token_id in values)
    return sha256_bytes(token_bytes)


def transfer_block_starts(
    *, prompt_length: int, gen_length: int, block_length: int
) -> tuple[int, ...]:
    """Return Stable-DiffCoder block starts matching generate_block residual logic."""
    gen_blocks = gen_length // block_length
    block_sizes = [block_length for _ in range(gen_blocks)]
    remainder = prompt_length % block_length
    if remainder != 0:
        residual = block_length - remainder
        block_sizes = [residual, *block_sizes]
        block_sizes[-1] = block_length - residual
    prefill_length = prompt_length // block_length * block_length
    starts: list[int] = []
    generated_before = 0
    for block_index, block_size in enumerate(block_sizes):
        block_start = prefill_length if block_index == 0 else prompt_length + generated_before
        starts.append(block_start)
        generated_before += block_size
    return tuple(starts)


def select_token_transfer_evidence(  # noqa: PLR0913
    logits: _TensorLike,
    x0: _TensorLike,
    transfer_map: _TensorLike,
    *,
    block_start: int,
    prompt_length: int,
    tokenizer: _TokenizerLike,
) -> tuple[TokenTransferEvidence, ...]:
    """Return position-level selected-token evidence for one transfer hook call."""
    import torch  # noqa: PLC0415

    probabilities = torch.softmax(logits.to("cpu").detach().double(), dim=-1)
    selected_tokens = x0.to("cpu").detach().reshape(-1)
    selected_positions = transfer_map.to("cpu").detach().reshape(-1)
    local_positions = selected_positions.nonzero(as_tuple=True)[0].tolist()
    if not local_positions:
        return ()
    gathered = probabilities.reshape(-1, probabilities.shape[-1]).gather(
        1,
        selected_tokens.reshape(-1, 1),
    )
    token_ids = selected_tokens[selected_positions].tolist()
    confidences = gathered.reshape(-1)[selected_positions].tolist()
    return build_token_transfer_evidence(
        block_start=block_start,
        prompt_length=prompt_length,
        local_positions=local_positions,
        token_ids=token_ids,
        confidences=confidences,
        decode_token=lambda token_id: tokenizer.decode([token_id], skip_special_tokens=False),
    )


def instrument_transfers(
    model: _ModelLike,
    sink: list[InstrumentedStep],
    context: InstrumentationContext,
) -> Callable[[], None]:
    """Wrap the model transfer hook and return a restore callback."""
    original = model._get_transfer_index  # noqa: SLF001
    block_starts = transfer_block_starts(
        prompt_length=context.prompt_length,
        gen_length=context.gen_length,
        block_length=context.block_length,
    )
    block_index = 0

    def wrapped(  # noqa: PLR0913, PLR0917
        _self: _ModelLike,
        logits: _TensorLike,
        temperature: float,
        remasking: str,
        mask_index: _TensorLike,
        x: _TensorLike,
        num_transfer_token: int | None,
        threshold: float | None = None,
        *,
        shift: bool = False,
    ) -> tuple[_TensorLike, _TensorLike]:
        nonlocal block_index
        x0, transfer_map = original(
            logits,
            temperature,
            remasking,
            mask_index,
            x,
            num_transfer_token,
            threshold,
            shift=shift,
        )
        post_transfer = x.detach().clone()
        post_transfer[transfer_map] = x0[transfer_map]
        block_start = block_starts[block_index]
        transfers = select_token_transfer_evidence(
            logits,
            x0,
            transfer_map,
            block_start=block_start,
            prompt_length=context.prompt_length,
            tokenizer=context.tokenizer,
        )
        masked_tokens_before = int(mask_index.sum().item())
        committed_tokens = int(transfer_map.sum().item())
        sink.append(
            InstrumentedStep(
                step_index=len(sink) + 1,
                masked_tokens_before=masked_tokens_before,
                committed_tokens=committed_tokens,
                mean_confidence=_mean_transfer_confidence(transfers),
                post_transfer_block_sha256=token_block_sha256(post_transfer),
                transfers=transfers,
            )
        )
        if masked_tokens_before == committed_tokens and block_index + 1 < len(block_starts):
            block_index += 1
        return x0, transfer_map

    model._get_transfer_index = MethodType(wrapped, model)  # noqa: SLF001

    def restore() -> None:
        model._get_transfer_index = original  # noqa: SLF001

    return restore


def _mean_transfer_confidence(transfers: tuple[TokenTransferEvidence, ...]) -> float:
    if not transfers:
        return 0.0
    return sum(transfer.confidence for transfer in transfers) / len(transfers)
