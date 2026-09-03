"""Stable-DiffCoder model loading and generation helpers."""

from __future__ import annotations

import errno
import inspect
import os
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, NamedTuple, override

from dllm_agent.runtime.revision_process import (
    TokenizerLike,
    load_hub_download,
    load_torch_runtime,
    load_transformers_imports,
)
from tools.runtime.selective_revision_json import canonical_json
from tools.stable_diffcoder_instrumentation import (
    InstrumentationContext,
    ModelLike,
    instrument_transfers,
)
from tools.stable_diffcoder_payloads import (
    MODEL_ID,
    MODEL_REVISION,
    REMOTE_CODE_SHA256,
    CandidatePayload,
    GenerationConfig,
    InstrumentedStep,
    JsonValue,
    ModelProvenance,
    RunnerConfig,
    assemble_candidate_bank,
    extract_fenced_cpp,
    sha256_bytes,
    validate_runner_config,
    write_json_atomic,
)

if TYPE_CHECKING:
    from dllm_agent.revision.controller_models import InitialParticleRequest, RevisionResult


class ModelLoadPlan(NamedTuple):  # noqa: D101
    cache_dir: str | None
    device: str
    local_files_only: bool
    use_safetensors: bool
    low_cpu_mem_usage: bool
    weights_only: bool


class CacheQualification(NamedTuple):  # noqa: D101
    cache_dir: Path
    dynamic_modules_dir: Path
    read_only_mount: bool
    write_trap_verified: bool


@dataclass(frozen=True, slots=True)
class ReadOnlyCacheError(ValueError):
    """Raised when a prewarmed cache is writable or escapes its tree."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


class FrozenParent(NamedTuple):  # noqa: D101
    request: InitialParticleRequest
    result: RevisionResult
    content: bytes
    materialized_sha256: str
    raw_parent_sha256: str


class DuplicateParent(NamedTuple):  # noqa: D101
    materialized_sha256: str
    duplicate_raw_parent_sha256: str
    kept_raw_parent_sha256: str


def model_load_plan(config: RunnerConfig, device: str) -> ModelLoadPlan:  # noqa: D103
    return ModelLoadPlan(
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
        device=device,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        weights_only=True,
    )


def qualify_read_only_cache(cache_dir: Path) -> CacheQualification:
    """Fail closed unless the complete prewarmed cache passes a write trap."""
    root = cache_dir.resolve()
    if not root.is_dir() or not any(root.iterdir()):
        raise ReadOnlyCacheError(reason="read-only worker cache must be pre-warmed")
    modules = (root / "modules").resolve()
    if not modules.is_relative_to(root) or not modules.is_dir() or not any(modules.iterdir()):
        raise ReadOnlyCacheError(
            reason="dynamic-module cache must be pre-warmed inside shared cache"
        )
    entries = (root, *root.rglob("*"))
    if any(
        entry.is_symlink() and not entry.resolve(strict=True).is_relative_to(root)
        for entry in entries
    ):
        raise ReadOnlyCacheError(reason="shared cache symlink escapes verified tree")
    read_only_mount = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
    if not read_only_mount and any(
        not entry.is_symlink() and entry.stat().st_mode & 0o222 for entry in entries
    ):
        raise ReadOnlyCacheError(reason="shared cache has a writable descendant")
    _verify_cache_write_trap(root)
    return CacheQualification(root, modules, read_only_mount, write_trap_verified=True)


def remote_code_sha256(model: ModelLike) -> str:  # noqa: D103
    source_path = inspect.getsourcefile(type(model))
    if source_path is None:
        msg = "could not locate remote model source file"
        raise RuntimeError(msg)
    return sha256_bytes(Path(source_path).read_bytes())


def resolve_device(requested: str) -> str:  # noqa: D103
    if requested != "auto":
        return requested
    return "cuda" if load_torch_runtime().cuda.is_available() else "cpu"


def load_model(config: RunnerConfig) -> tuple[TokenizerLike, ModelLike, str]:
    """Load the pinned tokenizer and model."""
    torch = load_torch_runtime()
    hub_download = load_hub_download()
    transformers = load_transformers_imports()
    device = resolve_device(config.device)
    plan = model_load_plan(config, device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    remote_code = hub_download(
        MODEL_ID,
        "modeling_stable_diffcoder.py",
        revision=MODEL_REVISION,
        cache_dir=plan.cache_dir,
        local_files_only=plan.local_files_only,
    )
    if sha256_bytes(Path(remote_code).read_bytes()) != REMOTE_CODE_SHA256:
        msg = "pinned Stable-DiffCoder remote code hash mismatch"
        raise RuntimeError(msg)
    tokenizer = transformers.tokenizer_factory.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        cache_dir=plan.cache_dir,
        local_files_only=plan.local_files_only,
    )
    direct_device_load = find_spec("accelerate") is not None
    model = transformers.model_factory.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        cache_dir=plan.cache_dir,
        local_files_only=plan.local_files_only,
        dtype=torch.bfloat16,
        use_safetensors=plan.use_safetensors,
        low_cpu_mem_usage=plan.low_cpu_mem_usage,
        weights_only=plan.weights_only,
        device_map={"": plan.device} if direct_device_load else None,
    )
    if not direct_device_load:
        model.to(plan.device)
    model.eval()
    return tokenizer, model, device


def generate_one(  # noqa: PLR0913, PLR0917
    prompt: str,
    tokenizer: TokenizerLike,
    model: ModelLike,
    config: RunnerConfig,
    seed: int,
    device: str,
) -> CandidatePayload:
    """Generate one candidate and its denoising trajectory."""
    import torch  # noqa: PLC0415

    _ = torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    steps: list[InstrumentedStep] = []
    restore = instrument_transfers(
        model,
        steps,
        InstrumentationContext(
            tokenizer=tokenizer,
            prompt_length=input_ids.shape[1],
            gen_length=config.gen_length,
            block_length=config.block_length,
        ),
    )
    try:
        output_ids, nfe = model.generate_block(
            input_ids,
            steps=config.steps,
            gen_length=config.gen_length,
            block_length=config.block_length,
            temperature=config.temperature,
            remasking="low_confidence",
            tokenizer=tokenizer,
            shift=False,
            threshold=None,
            eos_id=tokenizer.eos_token_id,
        )
    finally:
        restore()
    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    source = extract_fenced_cpp(generated_text)[0]
    return CandidatePayload(source=source, seed=seed, nfe=int(nfe), steps=tuple(steps))


def run(config: RunnerConfig) -> dict[str, JsonValue]:
    """Load the pinned model, generate candidates, and write the candidate bank."""
    validated = validate_runner_config(config)
    prompt_bytes = validated.prompt.read_bytes()
    prompt_text = prompt_bytes.decode("utf-8")
    tokenizer, model, device = load_model(validated)
    candidates = tuple(
        generate_one(prompt_text, tokenizer, model, validated, validated.seed + offset, device)
        for offset in range(validated.count)
    )
    torch = load_torch_runtime()
    transformers = load_transformers_imports()

    provenance = ModelProvenance(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        remote_code_sha256=remote_code_sha256(model),
        torch_version=torch.version,
        transformers_version=transformers.version,
        device=device,
        generation_config=GenerationConfig(
            steps=validated.steps,
            gen_length=validated.gen_length,
            block_length=validated.block_length,
            temperature=validated.temperature,
        ),
    )
    document = assemble_candidate_bank(prompt_bytes, provenance, candidates)
    write_json_atomic(validated.output, document)
    return document


def _verify_cache_write_trap(root: Path) -> None:
    trap = root / f".dllm-read-only-write-trap-{os.getpid()}-{monotonic_ns()}"
    try:
        descriptor = os.open(trap, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
            return
        reason = f"shared cache write trap failed unexpectedly: {error}"
        raise ReadOnlyCacheError(reason=reason) from error
    os.close(descriptor)
    trap.unlink(missing_ok=True)
    raise ReadOnlyCacheError(reason="shared cache write trap succeeded")


def revision_trajectory_sha256(result: RevisionResult) -> str:
    """Recompute the canonical hash of a model result's full trajectory."""
    document: JsonValue = [step.model_dump(mode="json") for step in result.trajectory_steps]
    return sha256_bytes(canonical_json(document).encode())
