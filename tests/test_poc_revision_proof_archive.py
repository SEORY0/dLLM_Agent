"""Online proof cause archive tests."""

from typing import Literal

from pydantic import Field

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.revision.proof_archive import CauseArchive, classify_online_proof


class CompletedProof(FrozenModel):
    """Proof result fixture with the fields consumed by the online archive."""

    oracle_status: Literal["proved", "not_reproduced", "crash_without_sanitizer"]
    input_sha256: Sha256
    proof_log_sha256: Sha256
    target_commit: str = Field(min_length=1)
    sanitizer_class: str | None = None
    root_signature: str | None = None


def test_first_stable_rich_cause_is_new() -> None:
    """Given a rich proved proof, when classified, then it earns new-cause novelty."""
    # Given
    archive = CauseArchive()
    proof = _proved(input_bytes=b"first")

    # When
    verdict = classify_online_proof(proof, archive)

    # Then
    assert verdict.oracle_status == "proved_new_cause"
    assert verdict.input_sha256 == sha256_bytes(b"first")
    assert verdict.proof_log_sha256 == proof.proof_log_sha256
    assert verdict.cause_id is not None
    assert verdict.family_key == verdict.cause_id


def test_same_normalized_cause_from_different_input_is_duplicate() -> None:
    """Given a stored cause, when another input proves it, then novelty is withheld."""
    # Given
    archive = CauseArchive()
    first = _proved(input_bytes=b"first", root_signature="reader 0x111|fuzzer 0x222")
    stored = archive.record(classify_online_proof(first, archive))
    duplicate = _proved(input_bytes=b"second", root_signature="reader 0x999|fuzzer 0xabc")

    # When
    verdict = classify_online_proof(duplicate, stored)

    # Then
    assert verdict.oracle_status == "proved_duplicate_cause"
    assert verdict.input_sha256 == sha256_bytes(b"second")
    assert verdict.proof_log_sha256 == duplicate.proof_log_sha256
    assert verdict.cause_id in stored.cause_ids
    assert verdict.family_key == verdict.cause_id


def test_failed_proof_records_provisional_family_without_novelty() -> None:
    """Given a failed proof, when archived, then only its provisional family advances."""
    # Given
    archive = CauseArchive()
    proof = CompletedProof(
        oracle_status="not_reproduced",
        input_sha256=sha256_bytes(b"candidate"),
        proof_log_sha256=sha256_bytes(b"clean\nclean again\n"),
        target_commit="target-a",
    )

    # When
    verdict = classify_online_proof(proof, archive)
    stored = archive.record(verdict)

    # Then
    assert verdict.oracle_status == "not_reproduced"
    assert verdict.cause_id is None
    assert verdict.input_sha256 == sha256_bytes(b"candidate")
    assert verdict.proof_log_sha256 == proof.proof_log_sha256
    assert stored.cause_ids == ()
    assert stored.family_attempt_count(verdict.family_key) == 1


def test_proved_summary_without_rich_signature_uses_provisional_family() -> None:
    """Given legacy proof evidence, when proved, then no new cause reward is emitted."""
    # Given
    archive = CauseArchive()
    proof = CompletedProof(
        oracle_status="proved",
        input_sha256=sha256_bytes(b"legacy"),
        proof_log_sha256=sha256_bytes(b"asan without normalized signature\n"),
        target_commit="target-a",
    )

    # When
    verdict = classify_online_proof(proof, archive)
    stored = archive.record(verdict)

    # Then
    assert verdict.oracle_status == "proved_duplicate_cause"
    assert verdict.cause_id is None
    assert verdict.input_sha256 == sha256_bytes(b"legacy")
    assert verdict.proof_log_sha256 == proof.proof_log_sha256
    assert stored.cause_ids == ()
    assert stored.family_attempt_count(verdict.family_key) == 1


def test_blank_rich_signature_fields_use_provisional_family() -> None:
    """Given incomplete ASan cause fields, when proved, then no new cause is emitted."""
    # Given
    archive = CauseArchive()
    blank_sanitizer = _proved(input_bytes=b"blank-class", sanitizer_class="  ")
    blank_root = _proved(input_bytes=b"blank-root", root_signature=" \n\t ")

    # When
    class_verdict = classify_online_proof(blank_sanitizer, archive)
    root_verdict = classify_online_proof(blank_root, archive)

    # Then
    assert class_verdict.oracle_status == "proved_duplicate_cause"
    assert class_verdict.cause_id is None
    assert class_verdict.input_sha256 == blank_sanitizer.input_sha256
    assert class_verdict.proof_log_sha256 == blank_sanitizer.proof_log_sha256
    assert root_verdict.oracle_status == "proved_duplicate_cause"
    assert root_verdict.cause_id is None
    assert root_verdict.input_sha256 == blank_root.input_sha256
    assert root_verdict.proof_log_sha256 == blank_root.proof_log_sha256


def test_family_attempt_cap_blocks_repeated_proof_family() -> None:
    """Given a capped family, when another proof is considered, then it is blocked."""
    # Given
    archive = CauseArchive()
    proof = CompletedProof(
        oracle_status="crash_without_sanitizer",
        input_sha256=sha256_bytes(b"candidate"),
        proof_log_sha256=sha256_bytes(b"Segmentation fault\n"),
        target_commit="target-a",
    )
    stored = archive.record(classify_online_proof(proof, archive))

    # When
    capped = stored.can_attempt(proof_log_sha256=proof.proof_log_sha256, attempt_cap=1)
    other_family = stored.can_attempt(
        proof_log_sha256=sha256_bytes(b"different proof log\n"), attempt_cap=1
    )

    # Then
    assert not capped
    assert other_family


def test_record_attempt_consumes_budget_without_completed_verdict() -> None:
    """Given skipped proof output, when attempted, then family budget is consumed."""
    # Given
    archive = CauseArchive()
    family_key = sha256_bytes(b"probe-family")

    # When
    stored = archive.record_attempt(family_key)

    # Then
    assert stored.family_attempt_count(family_key) == 1
    assert not stored.can_attempt_family(family_key=family_key, attempt_cap=1)


def test_rich_cause_family_attempt_cap_uses_explicit_family_key() -> None:
    """Given a recorded rich cause, when checked by family key, then the cap applies."""
    # Given
    archive = CauseArchive()
    verdict = classify_online_proof(_proved(input_bytes=b"rich-cause"), archive)
    stored = archive.record(verdict)
    assert verdict.cause_id is not None

    # When
    capped = stored.can_attempt_family(family_key=verdict.cause_id, attempt_cap=1)

    # Then
    assert not capped


def _proved(
    *,
    input_bytes: bytes,
    target_commit: str = "target-a",
    sanitizer_class: str = "heap-buffer-overflow",
    root_signature: str = "reader 0x111|fuzzer 0x222",
) -> CompletedProof:
    return CompletedProof(
        oracle_status="proved",
        input_sha256=sha256_bytes(input_bytes),
        proof_log_sha256=sha256_bytes(b"asan log:" + input_bytes),
        target_commit=target_commit,
        sanitizer_class=sanitizer_class,
        root_signature=root_signature,
    )
