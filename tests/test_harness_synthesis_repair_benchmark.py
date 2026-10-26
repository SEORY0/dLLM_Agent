"""RQ4b benchmark definitions: single-defect injection, prompts, and the leakage check."""

import pytest

from dllm_agent.harness_synthesis.body_analysis import right_context
from dllm_agent.harness_synthesis.repair_benchmark import (
    MUTATIONS,
    REFERENCE_BODIES,
    BenchmarkSample,
    Mutation,
    apply_mutation,
    continuation_text,
    leaked_fragments,
    repair_prompt,
    samples,
)


def test_every_mutation_changes_exactly_its_own_line() -> None:
    for sample in samples():
        ref = sample.reference_body.split("\n")
        mut = sample.mutated_body.split("\n")
        changed = [i + 1 for i, (a, b) in enumerate(zip(ref, mut, strict=True)) if a != b]
        assert changed == [sample.mutation.line_number], sample.sample_id


def test_repairing_the_mutated_line_alone_restores_the_reference() -> None:
    for sample in samples():
        lines = sample.mutated_body.split("\n")
        lines[sample.mutation.line_number - 1] = sample.target_line
        assert "\n".join(lines) == sample.reference_body


def test_every_mutated_line_has_executable_code_after_it() -> None:
    for sample in samples():
        assert right_context(sample.mutated_body, sample.mutation.line_number) == "code"


def test_sample_ids_are_unique_and_balanced_across_examples() -> None:
    ids = [m.sample_id for m in MUTATIONS]
    assert len(ids) == len(set(ids)) == 12
    assert sum(s.example == "InputFile" for s in samples()) == 6


def test_apply_mutation_rejects_ambiguous_or_misplaced_originals() -> None:
    body = REFERENCE_BODIES["InputFile"]
    with pytest.raises(ValueError, match="once"):
        _ = apply_mutation(body, Mutation("x", "invalid-method", 1, "header", "h"))
    with pytest.raises(ValueError, match="once"):
        _ = apply_mutation(body, Mutation("x", "invalid-method", 1, "file.header()", "f"))


def test_prompt_requires_inspection_and_never_carries_the_fix() -> None:
    for sample in samples():
        prompt = repair_prompt(sample.example)
        assert "optional" not in prompt.lower()
        assert "&stream" not in prompt
        assert leaked_fragments(sample, prompt) == ()


def test_feedback_prompt_appends_the_diagnostic_verbatim() -> None:
    message = "no member named 'readHeader' in 'Imf_3_4::InputFile'"
    prompt = repair_prompt("InputFile", diagnostic=message)
    assert prompt.startswith(repair_prompt("InputFile"))
    assert prompt.endswith(message)


def test_leaked_fragments_flags_the_target_line_and_new_call_fragments() -> None:
    sample = next(s for s in samples() if s.sample_id == "IF-bad-method")
    assert leaked_fragments(sample, "unrelated text") == ()
    assert leaked_fragments(sample, sample.target_line) != ()
    assert "header(" in leaked_fragments(sample, "hint: call header() on it")


def test_leaked_fragments_flags_a_namespace_visible_in_context() -> None:
    # The rejected MP-header-namespace candidate: its fix was visible on line 1.
    sample = BenchmarkSample(
        "MP-header-namespace",
        "MultiPartInputFile",
        REFERENCE_BODIES["MultiPartInputFile"],
        Mutation(
            "MP-header-namespace",
            "namespace-or-type",
            3,
            "const OPENEXR_IMF_NAMESPACE::Header&",
            "const IMATH_NAMESPACE::Header&",
        ),
    )
    first_line = sample.mutated_body.split("\n")[0]
    assert "OPENEXR_IMF_NAMESPACE::" in leaked_fragments(sample, first_line)


def test_continuation_text_closes_the_try_block_after_the_body() -> None:
    body = REFERENCE_BODIES["InputFile"]
    assert continuation_text(body) == body + "}\n```"
