"""Stable-DiffCoder adapter for target-agnostic slot denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dllm_agent.experiments.multitarget.slot_sampler import BlockDenoiseResult

if TYPE_CHECKING:

    class _WorkerRuntime(Protocol):
        """Type-checking facade for the runtime worker instance."""

        def denoise_block(  # noqa: PLR0913, PLR0917
            self,
            block_index: int,
            token_ids: tuple[int, ...],
            mask_positions: tuple[int, ...],
            allowed_token_ids: tuple[int, ...],
            previous_token_ids: tuple[int | None, ...],
            seed: int,
            step_index: int,
        ) -> WorkerDenoiseResult:
            """Run one runtime denoise operation."""
            ...

    class _WorkerFactory(Protocol):
        """Callable facade for constructing runtime workers."""

        def __call__(
            self,
            *,
            model: StableDiffCoderModel,
            device: str,
            block_size: int,
        ) -> _WorkerRuntime:
            """Create a runtime worker around an already-loaded model."""
            ...

    _worker_factory: _WorkerFactory

else:
    from tools.selective_revision_worker import RealSlotModel as _RuntimeWorkerSlotModel

    _worker_factory = _RuntimeWorkerSlotModel


class StableDiffCoderModel(Protocol):
    """Already-loaded Stable-DiffCoder model object owned by the caller."""


class WorkerDenoiseResult(Protocol):
    """Worker result shape converted into the multitarget sampler contract."""

    token_ids: tuple[int, ...]
    confidences: tuple[float, ...]
    gpu_ms: int


@dataclass(frozen=True, slots=True)
class StableDiffCoderSlotModel:
    """Resident Stable-DiffCoder model adapter for constrained selector slots."""

    model: StableDiffCoderModel
    device: str
    block_size: int

    def denoise_block(  # noqa: PLR0913, PLR0917 - mirrors SlotModel protocol.
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        """Run one Stable-DiffCoder block forward and return multitarget evidence."""
        worker_result = self._worker().denoise_block(
            block_index,
            token_ids,
            mask_positions,
            allowed_token_ids,
            previous_token_ids,
            seed,
            step_index,
        )
        return _convert_worker_result(worker_result)

    def _worker(self) -> _WorkerRuntime:
        return _worker_factory(model=self.model, device=self.device, block_size=self.block_size)


def _convert_worker_result(result: WorkerDenoiseResult) -> BlockDenoiseResult:
    return BlockDenoiseResult(
        token_ids=result.token_ids,
        confidences=result.confidences,
        gpu_ms=result.gpu_ms,
    )
