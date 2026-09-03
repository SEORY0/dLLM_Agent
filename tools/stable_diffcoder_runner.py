#!/usr/bin/env python3
"""Standalone real Stable-DiffCoder sampler process."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stable_diffcoder_generation import run
from tools.stable_diffcoder_payloads import (
    CandidatePayload,
    GenerationConfig,
    InstrumentedStep,
    JsonValue,
    ModelProvenance,
    RunnerConfig,
    RunnerConfigError,
    TokenTransferEvidence,
    assemble_candidate_bank,
    extract_fenced_cpp,
    sha256_bytes,
    validate_runner_config,
    write_json_atomic,
)

__all__ = [
    "CandidatePayload",
    "GenerationConfig",
    "InstrumentedStep",
    "JsonValue",
    "ModelProvenance",
    "RunnerConfig",
    "RunnerConfigError",
    "TokenTransferEvidence",
    "assemble_candidate_bank",
    "extract_fenced_cpp",
    "main",
    "run",
    "sha256_bytes",
    "validate_runner_config",
    "write_json_atomic",
]


def main(  # noqa: PLR0913, PLR0917
    prompt: Path = typer.Option(..., "--prompt", exists=True, file_okay=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", file_okay=True, dir_okay=False),
    count: int = typer.Option(4, "--count", min=2, max=8),
    seed: int = typer.Option(0, "--seed"),
    steps: int = typer.Option(128, "--steps"),
    gen_length: int = typer.Option(128, "--gen-length"),
    block_length: int = typer.Option(4, "--block-length"),
    temperature: float = typer.Option(0.0, "--temperature", min=0.0),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", file_okay=False, dir_okay=True),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Generate a Stable-DiffCoder candidate bank from a prompt file."""
    config = RunnerConfig(
        prompt=prompt,
        output=output,
        count=count,
        seed=seed,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cache_dir=cache_dir,
        device=device,
    )
    try:
        run(config)
    except RunnerConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    typer.run(main)
