"""Real-target hunt trust-boundary tests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.hunt.models import (
    CandidateBank,
    DenoisingStep,
    GenerationEvidence,
    HarnessCandidate,
    LiveModelProvenance,
)
from dllm_agent.hunt.validation import validate_candidate_bank, validate_harness_source
from dllm_agent.targets.models import HuntManifest, TargetEvidence

MODEL_ID = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
MODEL_REVISION = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"


def _candidate(source: str, *, source_hash: str | None = None) -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="candidate-01",
        source=source,
        source_sha256=source_hash or hashlib.sha256(source.encode()).hexdigest(),
        seed=11,
        nfe=2,
        steps=(
            DenoisingStep(
                step_index=1,
                masked_tokens_before=4,
                committed_tokens=4,
                mean_confidence=0.8,
                state_sha256="a" * 64,
            ),
        ),
    )


def _bank(prompt: bytes, candidate: HarnessCandidate) -> CandidateBank:
    return CandidateBank(
        schema_version="dllm-candidate-bank/v1",
        prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        model=LiveModelProvenance(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            remote_code_sha256="b" * 64,
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
        candidates=(candidate, _candidate(candidate.source.replace("01", "02"))),
    )


def test_candidate_bank_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    # Given
    prompt = b"generate a harness"
    bank = _bank(prompt, _candidate("int x;", source_hash="0" * 64))

    # When / Then
    with pytest.raises(HuntSetupError, match="candidate source SHA-256"):
        validate_candidate_bank(bank, prompt, tmp_path / "bank.json")


def test_harness_boundary_rejects_process_execution() -> None:
    # Given
    signature = '#include <cstdlib>\nextern "C" int LLVMFuzzerTestOneInput('
    body = 'const unsigned char*, unsigned long) { return system("id"); }'
    source = f"{signature}{body}"
    candidate = _candidate(source)

    # When / Then
    with pytest.raises(HuntSetupError, match="forbidden token"):
        validate_harness_source(
            candidate,
            allowed_includes=("cstddef", "cstdint", "jsoncons_ext/ubjson/ubjson.hpp"),
            required_symbols=("LLVMFuzzerTestOneInput",),
        )


def test_hunt_manifest_requires_digest_pinned_image() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError, match="image_id"):
        _ = HuntManifest(
            schema_version="dllm-hunt/v1",
            hunt_id="jsoncons-ubjson",
            target_id="jsoncons",
            repository_url="https://github.com/danielaparker/jsoncons.git",
            commit_sha="a" * 40,
            security_policy_url=(
                "https://github.com/danielaparker/jsoncons/security/advisories/new"
            ),
            security_policy_sha256="c" * 64,
            prompt="prompts/jsoncons-ubjson-harness.md",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            image_id="cybergym/oss-fuzz:latest",
            allowed_includes=("cstddef",),
            required_symbols=("LLVMFuzzerTestOneInput",),
            seed_hex=("5a",),
            fuzz_seconds=10,
            max_len=4096,
            rss_limit_mb=2048,
            timeout_seconds=5,
        )


def test_target_contract_serialization_and_schema_are_stable_after_extraction() -> None:
    # Given
    manifest = _manifest()
    evidence = TargetEvidence(
        repository_url="https://github.com/AcademySoftwareFoundation/openexr.git",
        commit_sha="a" * 40,
        tree_oid="d" * 40,
        source_archive_sha256="e" * 64,
        security_policy_url=(
            "https://github.com/AcademySoftwareFoundation/openexr/security/advisories/new"
        ),
        security_policy_sha256="b" * 64,
    )

    # When
    manifest_schema = json.dumps(
        manifest.model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    evidence_schema = json.dumps(
        evidence.model_json_schema(), sort_keys=True, separators=(",", ":")
    )

    # Then
    assert manifest.model_dump_json() == (
        '{"schema_version":"dllm-hunt/v1","hunt_id":"openexr-v3.4.14",'
        '"target_id":"openexr",'
        '"repository_url":"https://github.com/AcademySoftwareFoundation/openexr.git",'
        '"commit_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"security_policy_url":'
        '"https://github.com/AcademySoftwareFoundation/openexr/security/advisories/new",'
        '"security_policy_sha256":'
        '"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"prompt":"prompts/openexr.md",'
        '"model_id":"ByteDance-Seed/Stable-DiffCoder-8B-Instruct",'
        '"model_revision":"10cdaf9b486f1cf0273ad968459dbe2d21a1482a",'
        '"image_id":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",'
        '"target_profile":"openexr-v3.4","sanitizers":"address",'
        '"allowed_includes":["cstddef","cstdint"],'
        '"required_symbols":["LLVMFuzzerTestOneInput"],"seed_hex":["5a","00ff"],'
        '"fuzz_seconds":10,"max_len":4096,"rss_limit_mb":2048,"timeout_seconds":5}'
    )
    assert evidence.model_dump_json() == (
        '{"repository_url":"https://github.com/AcademySoftwareFoundation/openexr.git",'
        '"commit_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"tree_oid":"dddddddddddddddddddddddddddddddddddddddd",'
        '"source_archive_sha256":'
        '"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",'
        '"security_policy_url":'
        '"https://github.com/AcademySoftwareFoundation/openexr/security/advisories/new",'
        '"security_policy_sha256":'
        '"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}'
    )
    assert '"title":"HuntManifest"' in manifest_schema
    assert '"title":"TargetEvidence"' in evidence_schema
    assert '"schema_version":{"const":"dllm-hunt/v1"' in manifest_schema
    assert '"target_profile":{"$ref":"#/$defs/TargetProfile","default":"header-only"}' in (
        manifest_schema
    )
    assert '"commit_sha":{"pattern":"^[0-9a-f]{40}$"' in evidence_schema


def _manifest() -> HuntManifest:
    return HuntManifest(
        schema_version="dllm-hunt/v1",
        hunt_id="openexr-v3.4.14",
        target_id="openexr",
        repository_url="https://github.com/AcademySoftwareFoundation/openexr.git",
        commit_sha="a" * 40,
        security_policy_url=(
            "https://github.com/AcademySoftwareFoundation/openexr/security/advisories/new"
        ),
        security_policy_sha256="b" * 64,
        prompt="prompts/openexr.md",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        image_id="sha256:" + ("c" * 64),
        target_profile="openexr-v3.4",
        sanitizers="address",
        allowed_includes=("cstddef", "cstdint"),
        required_symbols=("LLVMFuzzerTestOneInput",),
        seed_hex=("5a", "00ff"),
        fuzz_seconds=10,
        max_len=4096,
        rss_limit_mb=2048,
        timeout_seconds=5,
    )
