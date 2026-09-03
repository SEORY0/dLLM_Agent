"""Separate-process Stable-DiffCoder sampler tests."""

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dllm_agent.hunt.models import (
    CandidateBank,
    DenoisingStep,
    GenerationEvidence,
    HarnessCandidate,
    LiveModelProvenance,
)
from dllm_agent.runtime.live_sampler import StableDiffCoderProcessSampler


def _bank(prompt: bytes) -> CandidateBank:
    source = (
        '#include <cstddef>\nextern "C" int LLVMFuzzerTestOneInput('
        "const unsigned char*, std::size_t) { return 0; }\n"
    )
    candidate = HarnessCandidate(
        candidate_id="candidate-01",
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        seed=31,
        nfe=2,
        steps=(
            DenoisingStep(
                step_index=1,
                masked_tokens_before=4,
                committed_tokens=4,
                mean_confidence=0.9,
                state_sha256="d" * 64,
            ),
        ),
    )
    return CandidateBank(
        schema_version="dllm-candidate-bank/v1",
        prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        model=LiveModelProvenance(
            model_id="ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
            revision="10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
            remote_code_sha256="e" * 64,
            python_version="3.12.13",
            transformers_version="5.3.0",
            torch_version="2.5.1+cu121",
            device="cuda:0",
            generation=GenerationEvidence(
                steps=64,
                gen_length=64,
                block_length=4,
                temperature=0.2,
            ),
        ),
        candidates=(candidate, candidate.model_copy(update={"candidate_id": "candidate-02"})),
    )


@dataclass(frozen=True, slots=True)
class StubProcessRunner:
    """Write a valid model result at the command's output boundary."""

    payload: CandidateBank
    commands: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self, command: tuple[str, ...], timeout_seconds: float
    ) -> subprocess.CompletedProcess[bytes]:
        del timeout_seconds
        self.commands.append(command)
        output = Path(command[command.index("--output") + 1])
        _ = output.write_text(self.payload.model_dump_json(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, b"", b"")


def test_process_sampler_parses_a_strict_candidate_bank(tmp_path: Path) -> None:
    # Given
    prompt = tmp_path / "prompt.md"
    prompt_bytes = b"generate a harness"
    _ = prompt.write_bytes(prompt_bytes)
    runtime = tmp_path / "python"
    _ = runtime.write_text("runtime", encoding="utf-8")
    runner = StubProcessRunner(_bank(prompt_bytes))
    sampler = StableDiffCoderProcessSampler(
        runtime_python=runtime,
        prompt_path=prompt,
        cache_dir=tmp_path / "cache",
        count=2,
        seed=31,
        steps=64,
        gen_length=64,
        block_length=4,
        temperature=0.2,
        device="cuda:0",
        timeout_seconds=30.0,
        runner=runner,
    )

    # When
    bank = sampler.sample()

    # Then
    assert len(bank.candidates) == 2
    command = runner.commands[0]
    assert command[0] == str(runtime.resolve())
    assert "stable_diffcoder_runner.py" in command[1]
    assert command[command.index("--cache-dir") + 1] == str((tmp_path / "cache").resolve())


def test_process_sampler_preserves_symlinked_venv_interpreter(tmp_path: Path) -> None:
    # Given
    prompt = tmp_path / "prompt.md"
    prompt_bytes = b"generate a harness"
    _ = prompt.write_bytes(prompt_bytes)
    base_runtime = tmp_path / "python-base"
    _ = base_runtime.write_text("runtime", encoding="utf-8")
    venv_runtime = tmp_path / "venv-python"
    venv_runtime.symlink_to(base_runtime)
    runner = StubProcessRunner(_bank(prompt_bytes))
    sampler = StableDiffCoderProcessSampler(
        runtime_python=venv_runtime,
        prompt_path=prompt,
        cache_dir=tmp_path / "cache",
        count=2,
        seed=31,
        steps=64,
        gen_length=64,
        block_length=4,
        temperature=0.2,
        device="cuda:0",
        timeout_seconds=30.0,
        runner=runner,
    )

    # When
    _ = sampler.sample()

    # Then
    assert runner.commands[0][0] == str(venv_runtime.absolute())
