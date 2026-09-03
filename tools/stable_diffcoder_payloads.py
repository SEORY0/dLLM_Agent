"""Stable-DiffCoder candidate-bank payload contracts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

MODEL_ID: Final = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION: Final = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
REMOTE_CODE_SHA256: Final = "298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104"
SCHEMA_VERSION: Final = "dllm-candidate-bank/v1"
TRANSFER_SCHEMA_VERSION: Final = "stable-diffcoder-transfer-evidence/v1"
MIN_CANDIDATES: Final = 2
MAX_CANDIDATES: Final = 8
CPP_FENCE_RE: Final = re.compile(
    r"```(?P<lang>cpp|c\+\+|cc|cxx)\s*\n(?P<body>.*?)```",
    re.DOTALL | re.IGNORECASE,
)

type JsonValue = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation settings recorded in provenance."""

    steps: int
    gen_length: int
    block_length: int
    temperature: float


@dataclass(frozen=True, slots=True)
class TokenTransferEvidence:
    """One committed token with generated-position evidence."""

    generated_position: int
    token_id: int
    text: str
    confidence: float


def build_token_transfer_evidence(  # noqa: PLR0913
    *,
    block_start: int,
    prompt_length: int,
    local_positions: Sequence[int],
    token_ids: Sequence[int],
    confidences: Sequence[float],
    decode_token: Callable[[int], str],
) -> tuple[TokenTransferEvidence, ...]:
    """Map block-local selected tokens into generated-slice positions."""
    ordered = sorted(
        zip(local_positions, token_ids, confidences, strict=True),
        key=lambda item: item[0],
    )
    return tuple(
        TokenTransferEvidence(
            generated_position=block_start + local_position - prompt_length,
            token_id=int(token_id),
            text=decode_token(int(token_id)),
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        for local_position, token_id, confidence in ordered
    )


@dataclass(frozen=True, slots=True)
class InstrumentedStep:
    """One denoising transfer observation."""

    step_index: int
    masked_tokens_before: int
    committed_tokens: int
    mean_confidence: float
    post_transfer_block_sha256: str
    transfers: tuple[TokenTransferEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidatePayload:
    """Generated candidate and trajectory payload."""

    source: str
    seed: int
    nfe: int
    steps: tuple[InstrumentedStep, ...]


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Model and runtime provenance included in every candidate bank."""

    model_id: str
    revision: str
    remote_code_sha256: str
    torch_version: str
    transformers_version: str
    device: str
    generation_config: GenerationConfig
    python_version: str = sys.version.split()[0]


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Parsed CLI configuration."""

    prompt: Path
    output: Path
    count: int
    seed: int
    steps: int
    gen_length: int
    block_length: int
    temperature: float
    cache_dir: Path | None
    device: str


class RunnerConfigError(ValueError):
    """Invalid CLI configuration."""


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def extract_fenced_cpp(generated_text: str) -> tuple[str, ...]:
    """Extract fenced C++ snippets, or treat unfenced output as one snippet."""
    matches = tuple(match.group("body").strip() for match in CPP_FENCE_RE.finditer(generated_text))
    if matches:
        return matches
    stripped = generated_text.strip()
    return (stripped,) if stripped else ()


def validate_runner_config(config: RunnerConfig) -> RunnerConfig:
    """Validate generation arithmetic and count bounds at the CLI boundary."""
    if config.count < MIN_CANDIDATES or config.count > MAX_CANDIDATES:
        msg = "--count must be between 2 and 8"
        raise RunnerConfigError(msg)
    if config.block_length <= 0:
        msg = "--block-length must be positive"
        raise RunnerConfigError(msg)
    if config.gen_length <= 0 or config.gen_length % config.block_length != 0:
        msg = "--gen-length must be positive and divisible by --block-length"
        raise RunnerConfigError(msg)
    generation_blocks = config.gen_length // config.block_length
    if config.steps <= 0 or config.steps % generation_blocks != 0:
        msg = "--steps must be positive and divisible by the generation block count"
        raise RunnerConfigError(msg)
    return config


def assemble_candidate_bank(
    prompt: bytes,
    provenance: ModelProvenance,
    candidates: Sequence[CandidatePayload],
) -> dict[str, JsonValue]:
    """Assemble the strict candidate-bank JSON payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": sha256_bytes(prompt),
        "model": {
            "model_id": provenance.model_id,
            "revision": provenance.revision,
            "remote_code_sha256": provenance.remote_code_sha256,
            "python_version": provenance.python_version,
            "torch_version": provenance.torch_version,
            "transformers_version": provenance.transformers_version,
            "device": provenance.device,
            "generation": {
                "steps": provenance.generation_config.steps,
                "gen_length": provenance.generation_config.gen_length,
                "block_length": provenance.generation_config.block_length,
                "temperature": provenance.generation_config.temperature,
            },
        },
        "candidates": [
            {
                "candidate_id": f"candidate-{index:02d}",
                "source": candidate.source,
                "source_sha256": sha256_bytes(candidate.source.encode()),
                "seed": candidate.seed,
                "nfe": candidate.nfe,
                "steps": [_assemble_step(step) for step in candidate.steps],
            }
            for index, candidate in enumerate(candidates, start=1)
        ],
    }


def _assemble_step(step: InstrumentedStep) -> dict[str, JsonValue]:
    document: dict[str, JsonValue] = {
        "step_index": step.step_index,
        "masked_tokens_before": step.masked_tokens_before,
        "committed_tokens": step.committed_tokens,
        "mean_confidence": step.mean_confidence,
        "state_sha256": step.post_transfer_block_sha256,
    }
    if step.transfers:
        document["transfer_schema_version"] = TRANSFER_SCHEMA_VERSION
        document["transfers"] = [
            {
                "generated_position": transfer.generated_position,
                "token_id": transfer.token_id,
                "text": transfer.text,
                "confidence": transfer.confidence,
            }
            for transfer in step.transfers
        ]
    return document


def write_json_atomic(output: Path, document: Mapping[str, JsonValue]) -> None:
    """Write JSON through a temporary file and atomic replace."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(output)
