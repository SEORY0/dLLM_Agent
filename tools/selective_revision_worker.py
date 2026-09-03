# pyright: reportUnnecessaryComparison=false
"""Resident JSONL worker for selective-revision/v1."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, assert_never, cast, override

import typer

if __package__ in {None, ""}:
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPOSITORY_ROOT))
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.runtime.revision_process import (
    pin_worker_process_to_cpu_affinity,
    worker_cpu_affinity_from_text,
)
from tools.runtime.selective_revision_json import (
    InitializeRequest,
    ReviseRequest,
    SamplerResponse,
    WireProtocolError,
    WorkerRequest,
)
from tools.runtime.selective_revision_template import BlockDenoiseResult
from tools.selective_revision_payloads import (
    parse_request_line,
    ready_record,
    request_hash,
    response_with_hash,
)
from tools.selective_revision_sampler import SelectiveRevisionSampler
from tools.stable_diffcoder_generation import (
    ReadOnlyCacheError,
    load_model,
    qualify_read_only_cache,
)
from tools.stable_diffcoder_payloads import RunnerConfig

if TYPE_CHECKING:
    from tools.stable_diffcoder_instrumentation import ModelLike, TensorLike

_SELECTOR_TEMPERATURE = 8.0


@dataclass(frozen=True, slots=True)
class WorkerCacheError(ValueError):
    """Raised when the worker cache cannot satisfy its access mode."""

    reason: str

    @override
    def __str__(self) -> str:
        """Return the worker cache failure reason."""
        return self.reason


def prepare_worker_cache(cache_dir: Path, *, read_only_cache: bool) -> Path:
    """Prepare the worker cache according to historical or campaign mode."""
    if read_only_cache:
        try:
            return qualify_read_only_cache(cache_dir).cache_dir
        except ReadOnlyCacheError as error:
            raise WorkerCacheError(reason=str(error)) from error
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class _TemperatureLogits(Protocol):
    def __truediv__(self, divisor: float) -> TensorLike:
        """Scale logits before selector-class softmax."""
        ...


def selected_vs_runner_up_margin(probabilities: tuple[float, ...], selected_index: int) -> float:
    """Return selected probability minus the strongest allowed alternative."""
    selected_probability = probabilities[selected_index]
    runner_up_probability = max(
        (probability for index, probability in enumerate(probabilities) if index != selected_index),
        default=0.0,
    )
    return selected_probability - runner_up_probability


@dataclass(frozen=True, slots=True)
class RealSlotModel:
    """Stable-DiffCoder-backed selector for one masked numeric slot."""

    model: ModelLike
    device: str
    block_size: int

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
        """Run one block-causal forward and commit all masked numeric slots."""
        import torch  # noqa: PLC0415

        _ = torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=self.device)
        _ = generator.manual_seed(seed + (block_index * 1009) + step_index)
        start_event = (
            torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None
        )
        end_event = torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            block_count = (len(token_ids) + self.block_size - 1) // self.block_size
            block_mask = torch.tril(
                torch.ones((block_count, block_count), dtype=torch.bool, device=self.device)
            )
            local_block = torch.ones(
                (self.block_size, self.block_size), dtype=torch.bool, device=self.device
            )
            mask = block_mask.kron(local_block)[: len(token_ids), : len(token_ids)]
            attention_mask = mask.float()
            _ = attention_mask.masked_fill_(~mask, -torch.inf)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).to(dtype=torch.bfloat16)
            if start_event is not None:
                start_event.record()
            output = self.model(input_ids, attention_mask=attention_mask)
            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
            block_logits = output.logits[0, list(mask_positions), :].detach().float()
            allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=self.device)
            selected_tokens: list[int] = []
            confidences: list[float] = []
            proposal_margins: list[float] = []
            for row, previous_token_id in zip(block_logits, previous_token_ids, strict=True):
                allowed_logits = row.index_select(0, allowed)
                if previous_token_id is not None and len(allowed_token_ids) > 1:
                    old = allowed.eq(previous_token_id).nonzero(as_tuple=True)[0]
                    if old.numel() > 0:
                        allowed_logits[old[0]] = float("-inf")
                probabilities = torch.softmax(
                    cast("_TemperatureLogits", cast("object", allowed_logits))
                    / _SELECTOR_TEMPERATURE,
                    dim=0,
                )
                chosen = int(torch.multinomial(probabilities, 1, generator=generator).item())
                distribution = tuple(
                    float(probabilities[index].item()) for index in range(len(allowed_token_ids))
                )
                selected_probability = distribution[chosen]
                selected_tokens.append(int(allowed_token_ids[chosen]))
                confidences.append(selected_probability)
                proposal_margins.append(selected_vs_runner_up_margin(distribution, chosen))
        gpu_ms = 0
        if start_event is not None and end_event is not None:
            gpu_ms = max(0, int(start_event.elapsed_time(end_event)))
        return BlockDenoiseResult(
            token_ids=tuple(selected_tokens),
            confidences=tuple(confidences),
            proposal_margins=tuple(proposal_margins),
            gpu_ms=gpu_ms,
        )


def main(  # noqa: PLR0913, PLR0917 - Typer command surface.
    cache_dir: Annotated[Path, typer.Option("--cache-dir", file_okay=False, dir_okay=True)],
    prompt_path: Annotated[Path, typer.Option("--prompt-path", exists=True, dir_okay=False)],
    prompt_sha256: Annotated[str, typer.Option("--prompt-sha256")],
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    read_only_cache: Annotated[bool, typer.Option("--read-only-cache")] = False,
    cpu_affinity: Annotated[str | None, typer.Option("--cpu-affinity")] = None,
) -> None:
    """Load Stable-DiffCoder once and serve JSONL commands."""
    if cpu_affinity is not None:
        try:
            affinity = worker_cpu_affinity_from_text(cpu_affinity)
            _ = pin_worker_process_to_cpu_affinity(os.getpid(), affinity)
        except HuntSetupError as error:
            _write_stderr(str(error))
            raise typer.Exit(code=2) from error
    try:
        _ = prepare_worker_cache(cache_dir, read_only_cache=read_only_cache)
    except WorkerCacheError as error:
        _write_stderr(str(error))
        raise typer.Exit(code=2) from error
    prompt_bytes = prompt_path.read_bytes()
    actual_prompt_sha256 = sha256(prompt_bytes).hexdigest()
    if actual_prompt_sha256 != prompt_sha256:
        _write_stderr("worker prompt sha256 mismatch")
        raise typer.Exit(code=2)
    config = RunnerConfig(
        prompt=Path(__file__),
        output=prompt_path,
        count=2,
        seed=0,
        steps=8,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        cache_dir=cache_dir,
        device=device,
    )
    tokenizer, model, resolved_device = load_model(config)
    sampler = SelectiveRevisionSampler(
        tokenizer,
        RealSlotModel(model=model, device=resolved_device, block_size=64),
        mask_token_id=5,
        block_size=64,
        prompt_context=prompt_bytes.decode("utf-8"),
    )
    _write_stdout(
        json.dumps(asdict(ready_record(resolved_device, 1, actual_prompt_sha256)), sort_keys=True)
    )
    for line in sys.stdin:
        try:
            request = parse_request_line(line)
            response = handle_worker_request(sampler, request)
        except WireProtocolError as error:
            _write_stderr(f"worker protocol error: {error}")
            break
        _write_stdout(response)


def handle_worker_request(sampler: SelectiveRevisionSampler, request: WorkerRequest) -> str:
    """Handle one parsed worker request and return one JSONL response body."""
    match request.operation:
        case "initialize":
            payload = _initialize_payload(request.payload)
            response = _bind_worker_request(sampler.initialize(payload), request)
            return json.dumps(asdict(response), sort_keys=True)
        case "revise":
            payload = _revise_payload(request.payload)
            response = _bind_worker_request(sampler.revise(payload), request)
            return json.dumps(asdict(response), sort_keys=True)
        case "shutdown":
            ack = {
                "kind": "shutdown",
                "request_id": request.request_id,
                "request_sha256": request_hash(request),
                "status": "ok",
            }
            return json.dumps(ack, sort_keys=True)
        case unreachable:
            assert_never(unreachable)


def _bind_worker_request(response: SamplerResponse, request: WorkerRequest) -> SamplerResponse:
    return response_with_hash(
        replace(response, request_sha256=request_hash(request), response_hash="")
    )


def _initialize_payload(payload: InitializeRequest | ReviseRequest | None) -> InitializeRequest:
    if not isinstance(payload, InitializeRequest):
        raise WireProtocolError(reason="initialize payload mismatch")
    return payload


def _revise_payload(payload: InitializeRequest | ReviseRequest | None) -> ReviseRequest:
    if not isinstance(payload, ReviseRequest):
        raise WireProtocolError(reason="revise payload mismatch")
    return payload


def _write_stdout(line: str) -> None:
    _ = sys.stdout.write(line + "\n")
    _ = sys.stdout.flush()


def _write_stderr(line: str) -> None:
    _ = sys.stderr.write(line + "\n")
    _ = sys.stderr.flush()


if __name__ == "__main__":
    typer.run(main)
