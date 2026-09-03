"""Tests for the standalone Stable-DiffCoder runner helpers."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import tools.stable_diffcoder_generation as generation  # noqa: E402
from tools.stable_diffcoder_runner import (  # noqa: E402
    CandidatePayload,
    InstrumentedStep,
    JsonValue,
    RunnerConfig,
    TokenTransferEvidence,
    assemble_candidate_bank,
    extract_fenced_cpp,
    sha256_bytes,
    validate_runner_config,
    write_json_atomic,
)

from dllm_agent.hunt.models import CandidateBank  # noqa: E402
from tests.revision_client_fakes import (  # noqa: E402
    FakeModel,
    FakeTokenizer,
    FakeTorch,
    LazyHub,
    LazyTransformers,
    legacy_candidate_bank_v1,
    stable_provenance,
)


def test_extract_fenced_cpp_returns_cpp_blocks_when_markdown_contains_multiple_languages() -> None:
    # Given
    generated = """
ignore prose
```python
print("not C++")
```
```cpp
#include <stdint.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  return 0;
}
```
```c++
int helper() { return 1; }
```
"""

    # When
    blocks = extract_fenced_cpp(generated)

    # Then
    assert blocks == (
        (
            "#include <stdint.h>\n"
            'extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {\n'
            "  return 0;\n"
            "}"
        ),
        "int helper() { return 1; }",
    )


def test_extract_fenced_cpp_falls_back_to_whole_text_when_no_cpp_fence_exists() -> None:
    # Given
    generated = (
        'extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {\nreturn 0;\n}'
    )

    # When
    blocks = extract_fenced_cpp(generated)

    # Then
    assert blocks == (generated,)


def test_sha256_bytes_returns_lowercase_hex_digest() -> None:
    # Given
    payload = b"stable-diffcoder"

    # When
    digest = sha256_bytes(payload)

    # Then
    assert digest == "7eef79fa5e151a916c3f95e09062cb5012a9d56f4afac15fb4a2cb5fa7a23fcc"


def test_assemble_candidate_bank_emits_strict_schema_and_hashes_sources() -> None:
    # Given
    prompt = b"write libFuzzer harness"
    provenance = stable_provenance()
    step = InstrumentedStep(
        step_index=1,
        masked_tokens_before=16,
        committed_tokens=4,
        mean_confidence=0.75,
        post_transfer_block_sha256="b" * 64,
    )
    candidate = CandidatePayload(
        source=(
            "int LLVMFuzzerTestOneInput("
            "const unsigned char *Data, unsigned long Size) { return 0; }"
        ),
        seed=123,
        nfe=8,
        steps=(step,),
    )

    # When
    bank = assemble_candidate_bank(prompt, provenance, (candidate,))

    # Then
    assert bank == {
        "schema_version": "dllm-candidate-bank/v1",
        "prompt_sha256": sha256_bytes(prompt),
        "model": {
            "model_id": "ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
            "revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
            "remote_code_sha256": "a" * 64,
            "python_version": provenance.python_version,
            "torch_version": "2.8.0",
            "transformers_version": "4.55.0",
            "device": "cpu",
            "generation": {
                "steps": 8,
                "gen_length": 64,
                "block_length": 16,
                "temperature": 0.2,
            },
        },
        "candidates": [
            {
                "candidate_id": "candidate-01",
                "source": candidate.source,
                "source_sha256": sha256_bytes(candidate.source.encode()),
                "seed": 123,
                "nfe": 8,
                "steps": [
                    {
                        "step_index": 1,
                        "masked_tokens_before": 16,
                        "committed_tokens": 4,
                        "mean_confidence": 0.75,
                        "state_sha256": "b" * 64,
                    }
                ],
            }
        ],
    }


def test_assemble_candidate_bank_round_trips_versioned_position_transfers() -> None:
    # Given
    prompt = b"write libFuzzer harness with transfer evidence"
    provenance = stable_provenance()
    step = InstrumentedStep(
        step_index=1,
        masked_tokens_before=16,
        committed_tokens=2,
        mean_confidence=0.6,
        post_transfer_block_sha256="b" * 64,
        transfers=(
            TokenTransferEvidence(
                generated_position=0,
                token_id=29905,
                text="int",
                confidence=0.75,
            ),
            TokenTransferEvidence(
                generated_position=1,
                token_id=220,
                text=" ",
                confidence=0.45,
            ),
        ),
    )
    candidate = CandidatePayload(
        source="int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) { return 0; }",
        seed=123,
        nfe=8,
        steps=(step,),
    )

    # When
    bank = assemble_candidate_bank(prompt, provenance, (candidate, candidate))
    parsed = CandidateBank.model_validate_json(json.dumps(bank))

    # Then
    candidates = bank["candidates"]
    assert isinstance(candidates, list)
    first_candidate = candidates[0]
    assert isinstance(first_candidate, dict)
    steps = first_candidate["steps"]
    assert isinstance(steps, list)
    serialized_step = steps[0]
    assert isinstance(serialized_step, dict)
    assert serialized_step == {
        "step_index": 1,
        "masked_tokens_before": 16,
        "committed_tokens": 2,
        "mean_confidence": 0.6,
        "state_sha256": "b" * 64,
        "transfer_schema_version": "stable-diffcoder-transfer-evidence/v1",
        "transfers": [
            {
                "generated_position": 0,
                "token_id": 29905,
                "text": "int",
                "confidence": 0.75,
            },
            {
                "generated_position": 1,
                "token_id": 220,
                "text": " ",
                "confidence": 0.45,
            },
        ],
    }
    assert parsed.candidates[0].steps[0].transfers[0].generated_position == 0
    assert parsed.candidates[0].steps[0].transfers[1].text == " "


def test_candidate_bank_still_parses_existing_v1_steps_without_position_transfers() -> None:
    # Given
    document = legacy_candidate_bank_v1()

    # When
    bank = CandidateBank.model_validate_json(json.dumps(document))

    # Then
    assert bank.candidates[0].steps[0].transfers == ()
    assert bank.candidates[0].steps[0].transfer_schema_version is None


@pytest.mark.parametrize(
    ("steps", "gen_length", "message"), [(7, 65, "gen-length"), (10, 64, "steps")]
)
def test_validate_runner_config_rejects_misaligned_generation(
    steps: int, gen_length: int, message: str
) -> None:
    # Given
    config = RunnerConfig(
        prompt=Path("prompt.txt"),
        output=Path("out.json"),
        count=4,
        seed=7,
        steps=steps,
        gen_length=gen_length,
        block_length=16,
        temperature=0.0,
        cache_dir=None,
        device="cpu",
    )

    # When / Then
    with pytest.raises(ValueError, match=message):
        _ = validate_runner_config(config)


def test_write_json_atomic_writes_complete_document(tmp_path: Path) -> None:
    # Given
    output = tmp_path / "nested" / "bank.json"
    document: dict[str, JsonValue] = {
        "schema_version": "dllm-candidate-bank/v1",
        "candidates": [],
    }

    # When
    write_json_atomic(output, document)

    # Then
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert tuple(tmp_path.rglob("*.tmp")) == ()


def test_load_model_accepts_lazy_export_modules_without_runtime_protocol_isinstance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    remote_code = tmp_path / "modeling_stable_diffcoder.py"
    _ = remote_code.write_text("# pinned remote model\n", encoding="utf-8")
    monkeypatch.setattr(generation, "REMOTE_CODE_SHA256", sha256_bytes(remote_code.read_bytes()))
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", LazyHub(remote_code))
    monkeypatch.setitem(sys.modules, "transformers", LazyTransformers("transformers"))
    config = RunnerConfig(
        prompt=tmp_path / "prompt.md",
        output=tmp_path / "unused.json",
        count=2,
        seed=0,
        steps=8,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        cache_dir=tmp_path / "cache",
        device="cuda:0",
    )

    # When
    tokenizer, model, device = generation.load_model(config)

    # Then
    assert isinstance(tokenizer, FakeTokenizer)
    assert isinstance(model, FakeModel)
    assert model.evaluated is True
    assert model.moved_to == "cuda:0"
    assert device == "cuda:0"
