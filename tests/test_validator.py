"""Real compiler and AddressSanitizer integration tests."""

from pathlib import Path

from dllm_agent.core.models import Finding, SourceSpan
from dllm_agent.replay.validator import ValidatorRequest, validate

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "fixtures/cwe121/vuln.c"
TARGET_BYTES = TARGET.read_bytes()
FINDING = Finding(
    candidate_id="candidate-01",
    cwe="CWE-121",
    span=SourceSpan(path="fixtures/cwe121/vuln.c", start_line=13, end_line=13),
    explanation="stack overwrite",
)


def test_validator_proves_owned_stack_overflow_with_asan() -> None:
    # Given
    request = ValidatorRequest(TARGET.name, TARGET_BYTES, FINDING, b"DLC-PROOF-000001", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "proved"


def test_validator_rejects_safe_short_input() -> None:
    # Given
    request = ValidatorRequest(TARGET.name, TARGET_BYTES, FINDING, b"SAFE\n", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "rejected"


def test_validator_returns_inconclusive_when_execution_times_out() -> None:
    # Given
    target_name = "loop.c"
    target_bytes = b"int main(void) { for (;;) {} }\n"
    finding = FINDING.model_copy(
        update={"span": SourceSpan(path=target_name, start_line=1, end_line=1)}
    )
    request = ValidatorRequest(target_name, target_bytes, finding, b"", 0.01)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "inconclusive"
    assert verdict.timed_out


def test_validator_returns_inconclusive_when_compiler_rejects_target() -> None:
    # Given
    request = ValidatorRequest("invalid.c", b"not c", FINDING, b"", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "inconclusive"
    assert verdict.exit_code != 0
