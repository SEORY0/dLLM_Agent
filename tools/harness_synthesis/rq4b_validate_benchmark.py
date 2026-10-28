# ruff: noqa: T201 -- deliberate console report, not a library module
"""Validate the RQ4b benchmark and its functional evaluator before any GPU run (CPU only).

Benchmark checks, per sample:

- the reference body has no syntax error and passes the functional evaluator on the
  validation input;
- the mutant does not compile, and every clang error lands on the mutated line, so it is
  a single-line defect;
- executable code follows the mutated line (``body_analysis.right_context == "code"``);
- the repair target never reaches the model: ``leaked_fragments`` is empty for the
  prompt in both feedback conditions plus the infill arm's visible context (the
  superset of what the suffix arm sees);
- ``L``, the defective line's Dream token window, gives five positive mask lengths, and
  the target's own token length ``L*`` is recorded (oracle information; no arm uses it).

Evaluator controls, per example: hand-written bodies, each with the verdict it must
get. They include correct repairs written in a different form, which must pass, and
deletion, commenting out, dead code, an early return, the wrong receiver, a call that
throws, a ``try`` inside the body, and a call mentioned only inside a string, all of
which must fail. ``use-removed`` is expected to pass. That records a stated limitation
(how values are used is not checked) as a test, not a surprise.

    uv run python tools/harness_synthesis/rq4b_validate_benchmark.py \
      --out docs/assets/harness-rq4b-benchmark-validation-20261026.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _compile import syntax_diagnostics_many
from _functional_eval import evaluate_many
from _replay import tokenize
from kb_v3_fixed_scaffold import assemble, scaffold_prefix

from dllm_agent.harness_synthesis.body_analysis import right_context
from dllm_agent.harness_synthesis.localize import (
    body_line_regions,
    parse_compiler_diagnostics,
    window_covering,
)
from dllm_agent.harness_synthesis.repair_arms import infill_mask_lengths
from dllm_agent.harness_synthesis.repair_benchmark import (
    FUNCTIONAL_SPECS,
    REFERENCE_BODIES,
    continuation_text,
    leaked_fragments,
    repair_prompt,
    samples,
)

_IF_CTOR = "    OPENEXR_IMF_NAMESPACE::InputFile file(stream);\n"
_MP_CTOR = "    OPENEXR_IMF_NAMESPACE::MultiPartInputFile file(stream);\n"

# (name, example, body, expected success). Every body must compile; a control that does
# not compile tests the compiler, not the evaluator, and is reported as invalid.
CONTROLS: tuple[tuple[str, str, str, bool], ...] = (
    ("reference", "InputFile", REFERENCE_BODIES["InputFile"], True),
    (
        "alt-correct-direct-chain",
        "InputFile",
        _IF_CTOR
        + "    const IMATH_NAMESPACE::Box2i dw = file.header().dataWindow();\n"
        + "    static_cast<void>(dw.max.x);\n  ",
        True,
    ),
    (
        "alt-correct-auto-copy",
        "InputFile",
        _IF_CTOR
        + "    auto header = file.header();\n"
        + "    auto dataWindow = header.dataWindow();\n"
        + "    static_cast<void>(dataWindow.min.y);\n  ",
        True,
    ),
    (
        "use-removed (limitation: value use unchecked)",
        "InputFile",
        "\n".join(REFERENCE_BODIES["InputFile"].split("\n")[:3]) + "\n  ",
        True,
    ),
    ("deleted-inspection", "InputFile", _IF_CTOR + "  ", False),
    (
        "commented-out-inspection",
        "InputFile",
        _IF_CTOR
        + "    // const OPENEXR_IMF_NAMESPACE::Header& header = file.header();\n"
        + "    // const IMATH_NAMESPACE::Box2i& dataWindow = header.dataWindow();\n  ",
        False,
    ),
    (
        "string-literal-mention",
        "InputFile",
        _IF_CTOR + '    const char* note = "file.header().dataWindow()";\n'
        "    static_cast<void>(note);\n  ",
        False,
    ),
    (
        "dead-branch",
        "InputFile",
        _IF_CTOR
        + "    if (size == static_cast<size_t>(-1)) {\n"
        + "      static_cast<void>(file.header().dataWindow());\n"
        + "    }\n  ",
        False,
    ),
    (
        "early-return",
        "InputFile",
        _IF_CTOR + "    return 0;\n    static_cast<void>(file.header().dataWindow());\n  ",
        False,
    ),
    (
        "wrong-receiver",
        "InputFile",
        _IF_CTOR
        + "    OPENEXR_IMF_NAMESPACE::Header other;\n"
        + "    static_cast<void>(other.dataWindow().max.x + file.version());\n  ",
        False,
    ),
    (
        "body-try-swallows",
        "InputFile",
        "    try {\n" + REFERENCE_BODIES["InputFile"].rstrip() + "\n    } catch (...) {\n    }\n  ",
        False,
    ),
    ("reference", "MultiPartInputFile", REFERENCE_BODIES["MultiPartInputFile"], True),
    (
        "alt-correct-inline",
        "MultiPartInputFile",
        _MP_CTOR + "    static_cast<void>(file.parts() + file.header(0).dataWindow().min.x);\n  ",
        True,
    ),
    (
        "throws-at-runtime (header of part 1 in a one-part file)",
        "MultiPartInputFile",
        _MP_CTOR
        + "    const int partCount = file.parts();\n"
        + "    const OPENEXR_IMF_NAMESPACE::Header& header = file.header(1);\n"
        + "    static_cast<void>(partCount + header.dataWindow().max.x);\n  ",
        False,
    ),
    (
        "partial (parts only)",
        "MultiPartInputFile",
        _MP_CTOR + "    static_cast<void>(file.parts());\n  ",
        False,
    ),
    ("deleted-inspection", "MultiPartInputFile", _MP_CTOR + "  ", False),
)


def _span(example: str, body: str) -> tuple[str, int, int]:
    prefix = scaffold_prefix(example)
    return assemble(example, body), len(prefix), len(prefix) + len(body)


def _window(body: str, line_number: int, pieces: list[str]) -> int:
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    region = next(r for r in body_line_regions(body) if r.line_number == line_number)
    window = window_covering(region, offsets)
    if window is None:
        reason = f"no token window for line {line_number}"
        raise ValueError(reason)
    return window.length


def main() -> None:
    """Run every check; write the validation record; exit non-zero if any check fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bench = samples()
    syntax = syntax_diagnostics_many(
        [assemble(s.example, s.reference_body) for s in bench]
        + [assemble(s.example, s.mutated_body) for s in bench]
    )
    cases = [(*_span(s.example, s.reference_body), FUNCTIONAL_SPECS[s.example]) for s in bench]
    cases += [(*_span(s.example, s.mutated_body), FUNCTIONAL_SPECS[s.example]) for s in bench]
    cases += [(*_span(ex, body), FUNCTIONAL_SPECS[ex]) for _, ex, body, _ in CONTROLS]
    runs = evaluate_many(cases)
    _, tokens = tokenize(
        [continuation_text(s.mutated_body) for s in bench]
        + [continuation_text(s.reference_body) for s in bench]
    )

    rows: list[dict[str, Any]] = []
    n = len(bench)
    for i, sample in enumerate(bench):
        prefix_lines = scaffold_prefix(sample.example).count("\n")
        mutant_diags = parse_compiler_diagnostics("\n".join(syntax[n + i]))
        mutant_lines = sorted({d.line - prefix_lines for d in mutant_diags})
        visible_infill = "\n".join(
            line
            for number, line in enumerate(sample.mutated_body.split("\n"), start=1)
            if number != sample.mutation.line_number
        )
        first_diag = mutant_diags[0].message if mutant_diags else ""
        model_inputs = (
            repair_prompt(sample.example)
            + repair_prompt(sample.example, diagnostic=first_diag)
            + scaffold_prefix(sample.example)
            + visible_infill
        )
        mut_tok, ref_tok = tokens[i], tokens[n + i]
        if "".join(mut_tok["pieces"]) != continuation_text(sample.mutated_body):
            reason = f"{sample.sample_id}: pieces do not reproduce the continuation"
            raise ValueError(reason)
        window = _window(sample.mutated_body, sample.mutation.line_number, mut_tok["pieces"])
        target_window = _window(
            sample.reference_body, sample.mutation.line_number, ref_tok["pieces"]
        )
        ref_run, mut_run = runs[i], runs[n + i]
        checks = {
            "reference_syntax_clean": not syntax[i],
            "reference_functional_success": ref_run.verdict.success,
            "mutant_does_not_compile": not mut_run.compiled and bool(mutant_diags),
            "all_errors_on_mutated_line": mutant_lines == [sample.mutation.line_number],
            "code_after_mutated_line": right_context(
                sample.mutated_body, sample.mutation.line_number
            )
            == "code",
            "no_leakage": not leaked_fragments(sample, model_inputs),
            "mask_lengths_positive": min(infill_mask_lengths(window)) > 0,
        }
        rows.append(
            {
                "sample_id": sample.sample_id,
                "example": sample.example,
                "kind": sample.mutation.kind,
                "line_number": sample.mutation.line_number,
                "target_line": sample.target_line,
                "defective_line": sample.defective_line,
                "mutant_diagnostics": [d.message for d in mutant_diags],
                "diagnostic_suggests_fix": any("did you mean" in d.message for d in mutant_diags),
                "defect_window_tokens_L": window,
                "target_tokens_L_star": target_window,
                "infill_mask_lengths": list(infill_mask_lengths(window)),
                "L_star_in_sweep": target_window in infill_mask_lengths(window),
                "leaks": list(leaked_fragments(sample, model_inputs)),
                "checks": checks,
                "valid": all(checks.values()),
            }
        )
    controls = []
    for (name, example, _, expected), run in zip(CONTROLS, runs[2 * n :], strict=True):
        controls.append(
            {
                "control": name,
                "example": example,
                "compiled": run.compiled,
                "expected_success": expected,
                "success": run.verdict.success,
                "reasons": list(run.verdict.reasons),
                "as_expected": run.compiled and run.verdict.success == expected,
            }
        )
    report = {
        "status": "offline validation; CPU only; no model inference",
        "all_samples_valid": all(r["valid"] for r in rows),
        "all_controls_as_expected": all(c["as_expected"] for c in controls),
        "samples": rows,
        "evaluator_controls": controls,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for row in rows:
        print(
            f"{row['sample_id']:22} valid={row['valid']} L={row['defect_window_tokens_L']} "
            f"L*={row['target_tokens_L_star']} diag={row['mutant_diagnostics'][0][:60]}"
        )
    for control in controls:
        print(
            f"{control['example']:18} {control['control'][:50]:50} "
            f"expected={control['expected_success']} got={control['success']} "
            f"{control['reasons']}"
        )
    if not (report["all_samples_valid"] and report["all_controls_as_expected"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
