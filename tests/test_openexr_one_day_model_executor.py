"""Unit tests for one-day model-arm seed-base derivation."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from dllm_agent.experiments.openexr_one_day.model_executor import _initial_particle_seed_base

if TYPE_CHECKING:
    from dllm_agent.experiments.openexr_one_day.coordinator import ModelArmExecutionRequest


@dataclass(frozen=True, slots=True)
class _FakeFrozenAdapter:
    requested_seed_base: int


@dataclass(frozen=True, slots=True)
class _FakeCell:
    fuzzer_seed: int


@dataclass(frozen=True, slots=True)
class _FakeRequest:
    cell: _FakeCell
    frozen_adapter: object | None = None


def _request(*, fuzzer_seed: int, frozen_adapter: object | None) -> ModelArmExecutionRequest:
    fake = _FakeRequest(cell=_FakeCell(fuzzer_seed=fuzzer_seed), frozen_adapter=frozen_adapter)
    return cast("ModelArmExecutionRequest", cast("object", fake))


def test_frozen_bank_arm_uses_the_bank_shared_seed_base_not_its_own_fuzzer_seed() -> None:
    # Given a frozen bank adapter built with a seed base shared across arms
    request = _request(fuzzer_seed=1_000_003, frozen_adapter=_FakeFrozenAdapter(1_000_000))

    # When
    seed_base = _initial_particle_seed_base(request)

    # Then the bank's shared seed base wins, not this arm's own fuzzer_seed --
    # otherwise FrozenBankAdapter.initial() rejects every arm but whichever
    # one happens to share the bank's seed namespace by coincidence.
    assert seed_base == 1_000_000


def test_non_frozen_bank_arm_falls_back_to_its_own_fuzzer_seed() -> None:
    # Given a direct (non-frozen-bank) revision delegate, e.g. the legacy path
    request = _request(fuzzer_seed=1_000_003, frozen_adapter=None)

    # When
    seed_base = _initial_particle_seed_base(request)

    # Then
    assert seed_base == 1_000_003
