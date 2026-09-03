"""Stable-DiffCoder-backed slot model adapter behavior."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Self, final

import pytest

from dllm_agent.experiments.multitarget.model_adapter import StableDiffCoderSlotModel
from dllm_agent.experiments.multitarget.slot_sampler import BlockDenoiseResult

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_denoise_block_returns_multitarget_result_with_seeded_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = FakeModel(logits_by_position={3: _vocab_logits()})
    adapter = StableDiffCoderSlotModel(model=model, device="cuda:0", block_size=4)

    # When
    first = adapter.denoise_block(0, (7, 8, 9, 5), (3,), (10, 11, 12), (None,), 4, 2)
    second = adapter.denoise_block(0, (7, 8, 9, 5), (3,), (10, 11, 12), (None,), 4, 2)
    third = adapter.denoise_block(0, (7, 8, 9, 5), (3,), (10, 11, 12), (None,), 5, 2)

    # Then
    assert isinstance(first, BlockDenoiseResult)
    assert first.token_ids == second.token_ids
    assert third.token_ids != first.token_ids
    assert first.confidences == pytest.approx((1.0 / 3.0,))
    assert first.gpu_ms == 17
    assert model.call_count == 3
    assert fake_torch.manual_seeds == [4, 4, 5]
    assert fake_torch.cuda.manual_seeds == [4, 4, 5]


def test_denoise_block_excludes_parent_token_when_revising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = FakeModel(logits_by_position={1: _vocab_logits()})
    adapter = StableDiffCoderSlotModel(model=model, device="cpu", block_size=4)

    # When
    result = adapter.denoise_block(0, (5, 5, 2), (1,), (10, 11, 12), (11,), 0, 1)

    # Then
    assert result.token_ids != (11,)
    assert result.token_ids == (12,)
    assert result.confidences == pytest.approx((0.5,))
    assert result.gpu_ms == 0
    assert model.last_input_tokens == (5, 5, 2)


def test_denoise_block_keeps_single_allowed_class_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = FakeModel(logits_by_position={0: _vocab_logits()})
    adapter = StableDiffCoderSlotModel(model=model, device="cpu", block_size=4)

    # When
    first = adapter.denoise_block(0, (5,), (0,), (10,), (None,), 1, 1)
    second = adapter.denoise_block(0, (5,), (0,), (10,), (None,), 999, 1)

    # Then
    assert first.token_ids == second.token_ids == (10,)
    assert first.confidences == second.confidences == (1.0,)


def test_denoise_block_flattens_allowed_logits_with_frozen_particle_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    logits = list(_vocab_logits())
    logits[10], logits[11], logits[12] = 80.0, 0.0, -80.0
    adapter = StableDiffCoderSlotModel(
        model=FakeModel(logits_by_position={0: tuple(logits)}),
        device="cpu",
        block_size=4,
    )

    _ = adapter.denoise_block(0, (5,), (0,), (10, 11, 12), (None,), 4, 2)

    assert fake_torch.softmax_inputs == [(10.0, 0.0, -10.0)]


@dataclass(slots=True)
class FakeOutput:
    logits: FakeLogits


def _vocab_logits() -> tuple[float, ...]:
    return tuple(0.0 for _ in range(13))


@final
class FakeModel:
    def __init__(self, *, logits_by_position: dict[int, tuple[float, ...]]) -> None:
        self._logits_by_position: dict[int, tuple[float, ...]] = logits_by_position
        self.call_count: int = 0
        self.last_input_tokens: tuple[int, ...] = ()

    def __call__(self, input_ids: FakeInputIds, *, attention_mask: FakeMask) -> FakeOutput:
        self.call_count += 1
        self.last_input_tokens = input_ids.tokens
        return FakeOutput(logits=FakeLogits(self._logits_by_position))


@dataclass(frozen=True, slots=True)
class FakeInputIds:
    tokens: tuple[int, ...]


@final
class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@final
class FakeMask:
    def kron(self, _other: FakeMask) -> Self:
        return self

    def float(self) -> Self:
        return self

    def masked_fill_(self, mask: Self, value: float) -> Self:
        del mask, value
        return self

    def unsqueeze(self, dimension: int) -> Self:
        del dimension
        return self

    def to(self, *, dtype: str) -> Self:
        del dtype
        return self

    def __invert__(self) -> Self:
        return self

    def __getitem__(self, key: tuple[slice, slice]) -> Self:
        del key
        return self


@final
class FakeGenerator:
    def __init__(self, *, device: str) -> None:
        self.device: str = device
        self.seed: int = 0

    def manual_seed(self, seed: int) -> Self:
        self.seed = seed
        return self


@dataclass(frozen=True, slots=True)
class FakeScalar:
    value: int | float

    def item(self) -> int | float:
        return self.value


@final
class FakeIndexVector:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values: tuple[int, ...] = values

    def eq(self, other: int) -> FakeBoolVector:
        return FakeBoolVector(tuple(value == other for value in self._values))

    def nonzero(self, *, as_tuple: bool) -> tuple[Self]:
        assert as_tuple
        return (self,)

    def numel(self) -> int:
        return len(self._values)

    def item(self) -> int:
        assert len(self._values) == 1
        return self._values[0]

    def __getitem__(self, index: int) -> FakeScalar:
        return FakeScalar(self._values[index])

    def __iter__(self) -> Iterator[FakeScalar]:
        return (FakeScalar(value) for value in self._values)


@final
class FakeBoolVector:
    def __init__(self, values: tuple[bool, ...]) -> None:
        self._indexes: tuple[int, ...] = tuple(index for index, value in enumerate(values) if value)

    def nonzero(self, *, as_tuple: bool) -> tuple[FakeIndexVector]:
        assert as_tuple
        return (FakeIndexVector(self._indexes),)


@final
class FakeLogitVector:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values: list[float] = list(values)

    def index_select(self, dim: int, index: FakeIndexVector) -> FakeLogitVector:
        del dim
        return FakeLogitVector(tuple(self._values[int(scalar.item())] for scalar in index))

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __truediv__(self, divisor: float) -> FakeLogitVector:
        return FakeLogitVector(tuple(value / divisor for value in self._values))

    def __setitem__(self, index: FakeScalar, value: float) -> None:
        self._values[int(index.item())] = value


@final
class FakeProbabilityVector:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values: tuple[float, ...] = values

    def item_at(self, index: int) -> float:
        return self._values[index]

    def positive_indexes(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self._values) if value > 0.0)

    def __getitem__(self, index: int) -> FakeScalar:
        return FakeScalar(self._values[index])


@final
class FakeRows:
    def __init__(self, rows: tuple[FakeLogitVector, ...]) -> None:
        self._rows: tuple[FakeLogitVector, ...] = rows

    def detach(self) -> Self:
        return self

    def float(self) -> Self:
        return self

    def __iter__(self) -> Iterator[FakeLogitVector]:
        return iter(self._rows)


@final
class FakeLogits:
    def __init__(self, logits_by_position: dict[int, tuple[float, ...]]) -> None:
        self._logits_by_position: dict[int, tuple[float, ...]] = logits_by_position

    def __getitem__(self, key: tuple[int, list[int], slice]) -> FakeRows:
        _, positions, _ = key
        rows = tuple(FakeLogitVector(self._logits_by_position[position]) for position in positions)
        return FakeRows(rows)


@final
class FakeCuda:
    def __init__(self) -> None:
        self.manual_seeds: list[int] = []
        self.softmax_inputs: list[tuple[float, ...]] = []

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seeds.append(seed)

    def Event(self, *, enable_timing: bool) -> FakeCudaEvent:  # noqa: N802
        return FakeCudaEvent(enable_timing=enable_timing)

    def synchronize(self) -> None:
        return None


@dataclass(slots=True)
class FakeCudaEvent:
    enable_timing: bool

    def record(self) -> None:
        return None

    def elapsed_time(self, other: Self) -> float:
        del other
        assert self.enable_timing
        return 17.9


@final
class FakeTorch(ModuleType):
    long: str = "long"
    bool: str = "bool"
    bfloat16: str = "bfloat16"
    inf: float = float("inf")

    def __init__(self) -> None:
        super().__init__("torch")
        self.cuda: FakeCuda = FakeCuda()
        self.manual_seeds: list[int] = []
        self.softmax_inputs: list[tuple[float, ...]] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seeds.append(seed)

    def Generator(self, *, device: str) -> FakeGenerator:  # noqa: N802
        return FakeGenerator(device=device)

    def no_grad(self) -> FakeNoGrad:
        return FakeNoGrad()

    def tensor(
        self,
        values: list[tuple[int, ...]] | tuple[int, ...],
        *,
        dtype: str,
        device: str,
    ) -> FakeInputIds | FakeIndexVector:
        del dtype, device
        if isinstance(values, list):
            assert len(values) == 1
            return FakeInputIds(tokens=values[0])
        return FakeIndexVector(values)

    def ones(self, shape: tuple[int, int], *, dtype: str, device: str) -> FakeMask:
        del shape, dtype, device
        return FakeMask()

    def tril(self, tensor: FakeMask) -> FakeMask:
        return tensor

    def softmax(self, logits: FakeLogitVector, *, dim: int) -> FakeProbabilityVector:
        del dim
        raw_values: tuple[float, ...] = tuple(logits)
        self.softmax_inputs.append(raw_values)
        live_count = sum(1 for value in raw_values if value != -self.inf)
        return FakeProbabilityVector(
            tuple(0.0 if value == -self.inf else 1.0 / live_count for value in raw_values)
        )

    def multinomial(
        self,
        probabilities: FakeProbabilityVector,
        num_samples: int,
        *,
        generator: FakeGenerator,
    ) -> FakeIndexVector:
        assert num_samples == 1
        live = probabilities.positive_indexes()
        return FakeIndexVector((live[generator.seed % len(live)],))
