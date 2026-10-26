"""Functional verdicts from real clang AST + llvm-cov fixtures, and the pure helpers."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from dllm_agent.harness_synthesis.functional_eval import (
    AstFacts,
    AstNode,
    Construction,
    FunctionalVerdict,
    Segment,
    ast_facts,
    coverage_count,
    evaluate,
    line_col,
)
from dllm_agent.harness_synthesis.repair_benchmark import FUNCTIONAL_SPECS

# Captured from the pinned compile-only image (clang 18.1.8, OpenEXR 3.4.14); see the
# file's own "provenance" key.
_FIXTURE = Path(__file__).parents[1] / "fixtures/harness_synthesis/rq4b-functional-eval-cases.json"


class _Case(TypedDict):
    control: str
    example: str
    expected_success: bool
    source: str
    body_start: int
    body_end: int
    exit_code: int
    ast: list[AstNode]
    segments: list[Segment]


def _cases() -> dict[tuple[str, str], _Case]:
    doc = cast("dict[str, list[_Case]]", json.loads(_FIXTURE.read_text()))
    return {(c["example"], c["control"]): c for c in doc["cases"]}


def _verdict(
    case: _Case,
    *,
    compiled: bool = True,
    facts: AstFacts | str | None = "from-case",
    segments: Sequence[Segment] | None = None,
    exit_code: int | str | None = "from-case",
) -> FunctionalVerdict:
    spec = FUNCTIONAL_SPECS[case["example"]]
    case_facts = ast_facts(
        case["ast"], spec, body_start=case["body_start"], body_end=case["body_end"]
    )
    return evaluate(
        spec,
        source=case["source"],
        compiled=compiled,
        facts=case_facts if isinstance(facts, str) else facts,
        segments=case["segments"] if segments is None else segments,
        exit_code=case["exit_code"] if isinstance(exit_code, str) else exit_code,
    )


def test_every_captured_control_gets_its_expected_verdict() -> None:
    for (example, control), case in _cases().items():
        assert _verdict(case).success is case["expected_success"], (example, control)


def test_reference_traces_chain_through_a_reference_variable() -> None:
    verdict = _verdict(_cases()["InputFile", "reference"])
    assert verdict.constructed_from_stream
    assert ("header", "dataWindow") in verdict.executed_chains
    assert verdict.reasons == ()


def test_auto_copies_are_followed_back_to_the_constructed_object() -> None:
    verdict = _verdict(_cases()["InputFile", "alt-correct-auto-copy"])
    assert verdict.success
    assert ("header", "dataWindow") in verdict.executed_chains


def test_dead_branch_call_is_present_in_the_ast_but_not_executed() -> None:
    verdict = _verdict(_cases()["InputFile", "dead-branch"])
    assert verdict.reasons == ("chain-not-executed:header>dataWindow",)


def test_call_on_another_object_does_not_count() -> None:
    assert _verdict(_cases()["InputFile", "wrong-receiver"]).reasons == (
        "chain-missing:header>dataWindow",
    )


def test_exception_reaching_the_scaffold_catch_fails_even_though_lines_count_once() -> None:
    verdict = _verdict(
        _cases()["MultiPartInputFile", "throws-at-runtime (header of part 1 in a one-part file)"]
    )
    assert "exception-reached-scaffold-catch" in verdict.reasons
    assert not verdict.success


def test_body_level_try_is_rejected_conservatively() -> None:
    assert "body-contains-try" in _verdict(_cases()["InputFile", "body-try-swallows"]).reasons


def test_nonzero_exit_and_non_compiling_inputs_fail() -> None:
    case = _cases()["InputFile", "reference"]
    assert _verdict(case, exit_code=1).reasons == ("nonzero-exit",)
    assert _verdict(case, compiled=False).reasons == ("not-compiled",)
    assert _verdict(case, facts=None).reasons == ("not-compiled",)


def test_missing_coverage_means_nothing_executed() -> None:
    verdict = _verdict(_cases()["InputFile", "reference"], segments=[])
    assert not verdict.success
    assert "construction-not-executed" in verdict.reasons


def test_no_construction_from_stream_is_reported() -> None:
    case = _cases()["InputFile", "reference"]
    facts = AstFacts(
        constructions=(Construction("0x1", "file", case["body_start"], from_stream=False),),
        calls=(),
        body_has_try=False,
        scaffold_catch_offset=None,
    )
    reasons = _verdict(case, facts=facts).reasons
    assert "no-construction-from-stream" in reasons
    assert "scaffold-catch-not-found" in reasons


def test_coverage_count_uses_the_last_segment_at_or_before_the_position() -> None:
    segments = [
        [1, 1, 5, True, True, False],
        [2, 3, 0, True, True, False],
        [3, 1, 0, False, False, True],
    ]
    assert coverage_count(segments, 1, 9) == 5
    assert coverage_count(segments, 2, 3) == 0
    assert coverage_count(segments, 3, 4) is None
    assert coverage_count(segments, 0, 1) is None


def test_line_col_is_one_based() -> None:
    assert line_col("ab\ncd", 0) == (1, 1)
    assert line_col("ab\ncd", 4) == (2, 2)


def test_empty_ast_gives_no_facts() -> None:
    facts = ast_facts([], FUNCTIONAL_SPECS["InputFile"], body_start=0, body_end=1)
    assert facts.constructions == ()
    assert facts.scaffold_catch_offset is None
