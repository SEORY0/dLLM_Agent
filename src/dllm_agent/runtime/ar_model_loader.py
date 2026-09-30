"""Lazy-loaded standard causal-LM backend for the AR-parent control sampler.

Loads torch/transformers via `importlib.import_module` + validated `getattr`
(never a static `import torch`), exactly like `load_torch_runtime`/
`load_transformers_imports` in `revision_process.py` -- so this module stays
importable, and passes strict basedpyright, whether or not the optional
`gpu` extra (torch/transformers) is installed. A standard
`AutoModelForCausalLM` needs no custom remote code and no diffusion-specific
`generate_block`/transfer-index surface (unlike the Stable-DiffCoder loader
this deliberately does not reuse), so its own small protocols live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ("HuggingFaceArTokenModel", "load_ar_token_model")


class _TensorLike(Protocol):
    def __getitem__(self, key: object) -> _TensorLike: ...
    def tolist(self) -> list[float]: ...


class _NoGradContext(Protocol):
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info: object) -> None: ...


class _CausalLmOutputLike(Protocol):
    logits: _TensorLike


class _CausalLmLike(Protocol):
    def __call__(self, *, input_ids: _TensorLike) -> _CausalLmOutputLike: ...
    def to(self, device: str) -> _CausalLmLike: ...
    def eval(self) -> _CausalLmLike: ...


class _CausalLmFactory(Protocol):
    def from_pretrained(  # noqa: PLR0913
        self,
        model_id: str,
        *,
        revision: str,
        trust_remote_code: bool,
        cache_dir: str | None,
        local_files_only: bool,
        dtype: object,
    ) -> _CausalLmLike: ...


class _TorchExtras(NamedTuple):
    bfloat16: object
    tensor: Callable[..., _TensorLike]
    no_grad: Callable[[], _NoGradContext]


class _TransformersExtras(NamedTuple):
    model_factory: _CausalLmFactory


def _load_torch_extras() -> _TorchExtras:
    module = import_module("torch")
    bfloat16: object | None = getattr(module, "bfloat16", None)
    tensor: Callable[..., _TensorLike] | None = getattr(module, "tensor", None)
    no_grad: Callable[[], _NoGradContext] | None = getattr(module, "no_grad", None)
    if bfloat16 is None or tensor is None or not callable(tensor):
        message = "torch runtime does not expose bfloat16/tensor"
        raise TypeError(message)
    if no_grad is None or not callable(no_grad):
        message = "torch runtime does not expose no_grad"
        raise TypeError(message)
    return _TorchExtras(bfloat16=bfloat16, tensor=tensor, no_grad=no_grad)


def _load_transformers_extras() -> _TransformersExtras:
    module = import_module("transformers")
    model_factory: _CausalLmFactory | None = getattr(module, "AutoModelForCausalLM", None)
    if model_factory is None or not callable(model_factory.from_pretrained):
        message = "transformers runtime does not expose AutoModelForCausalLM"
        raise TypeError(message)
    return _TransformersExtras(model_factory=model_factory)


@dataclass(frozen=True, slots=True)
class HuggingFaceArTokenModel:
    """`ArTokenModel` backed by one loaded standard causal-LM checkpoint."""

    model: _CausalLmLike
    torch: _TorchExtras
    device: str

    def next_token_logits(self, token_ids: tuple[int, ...]) -> tuple[float, ...]:
        """Run one real forward pass and return next-position vocabulary logits."""
        tensor = self.torch.tensor([list(token_ids)], device=self.device)
        with self.torch.no_grad():
            output = self.model(input_ids=tensor)
        return tuple(output.logits[0, -1, :].tolist())


def load_ar_token_model(
    model_id: str,
    *,
    revision: str,
    cache_dir: str | None,
    device: str,
    local_files_only: bool = False,
) -> HuggingFaceArTokenModel:
    """Load one AR-parent checkpoint onto `device` in evaluation mode."""
    torch = _load_torch_extras()
    transformers = _load_transformers_extras()
    model = transformers.model_factory.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dtype=torch.bfloat16,
    )
    ready = model.to(device).eval()
    return HuggingFaceArTokenModel(model=ready, torch=torch, device=device)
