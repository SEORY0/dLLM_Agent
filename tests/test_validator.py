"""Real compiler and AddressSanitizer integration tests."""

from pathlib import Path

from dllm_agent.models import Finding, SourceSpan
from dllm_agent.validator import ValidatorRequest, validate

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "fixtures/cwe121/vuln.c"
FINDING = Finding(
    candidate_id="candidate-01",
    cwe="CWE-121",
    span=SourceSpan(path="fixtures/cwe121/vuln.c", start_line=13, end_line=13),
    explanation="stack overwrite",
)


def test_validator_proves_owned_stack_overflow_with_asan() -> None:
    # Given
    request = ValidatorRequest(TARGET, FINDING, b"DLC-PROOF-000001", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "proved"


def test_validator_rejects_safe_short_input() -> None:
    # Given
    request = ValidatorRequest(TARGET, FINDING, b"SAFE\n", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "rejected"


def test_validator_returns_inconclusive_when_execution_times_out(tmp_path: Path) -> None:
    # Given
    target = tmp_path / "loop.c"
    _ = target.write_text("int main(void) { for (;;) {} }\n", encoding="utf-8")
    finding = FINDING.model_copy(
        update={"span": SourceSpan(path=str(target), start_line=1, end_line=1)}
    )
    request = ValidatorRequest(target, finding, b"", 0.01)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "inconclusive"
    assert verdict.timed_out


def test_validator_returns_inconclusive_when_compiler_rejects_target(tmp_path: Path) -> None:
    # Given
    target = tmp_path / "invalid.c"
    _ = target.write_text("not c", encoding="utf-8")
    request = ValidatorRequest(target, FINDING, b"", 2.0)

    # When
    verdict = validate(request)

    # Then
    assert verdict.status == "inconclusive"
    assert verdict.exit_code != 0
