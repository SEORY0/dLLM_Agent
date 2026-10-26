# ruff: noqa: T201 -- deliberate console report, not a library module
"""Read-only retrospective audit of the RQ4 1-round pilot (post-hoc; not confirmatory).

Reads the two committed pilot result files and does four things. It never changes them.

1. Recomputes the frozen analysis with ``rq4_analyze._analyze`` and diffs it against the
   committed analysis JSON.
2. Re-checks every stored harness (eligible references and all four arms) with
   ``clang++ -fsyntax-only -ferror-limit=0`` in the same pinned image, which gives every
   diagnostic rather than the first three the pilot kept. It also checks that this
   verdict matches the stored build result for every harness.
3. Classifies each eligible trial: diagnostic count and distinct body lines, what follows
   the located line (executable code, comments only, or nothing), whether the reference
   makes an active inspection call, and the infill span (original tokens, mask length,
   re-tokenized replacement length, what the surplus was filled with).
4. Classifies each arm: compiled or not; whether the reference's inspection call was
   kept, rewritten, deleted, or commented out; and whether the reference's errors are
   still present or new ones appeared.

Everything in sections 3 and 4 is post-hoc description of a finished pilot. It produces
no p-values, so no subgroup here is a confirmatory result. "Inspection removed" is not
called a failure: the pilot's prompt made inspection optional. Lexical checks are
comment-aware (``body_analysis``), but they cannot see dead code or resolve types.

    uv run python tools/harness_synthesis/rq4_retro_audit.py \
      --out docs/assets/harness-rq4-retro-audit-20261026.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from _compile import syntax_diagnostics_many
from _replay import tokenize
from kb_v3_fixed_scaffold import assemble, scaffold_prefix
from rq4_analyze import _analyze

from dllm_agent.harness_synthesis.body_analysis import (
    active_member_calls,
    comment_text,
    commented_member_calls,
    constructed_object,
    line_kinds,
    mask_comments,
    right_context,
)
from dllm_agent.harness_synthesis.localize import body_line_regions, parse_compiler_diagnostics

_REPO = Path(__file__).resolve().parents[2]
_ASSETS = _REPO / "docs/assets"
_INPUTS = (
    _ASSETS / "harness-rq4-pilot-inputfile-results-20261026.json",
    _ASSETS / "harness-rq4-pilot-multipart-results-20261026.json",
)
_COMMITTED_ANALYSIS = _ASSETS / "harness-rq4-pilot-analysis-20261026.json"
_ARMS = ("located-infill", "random-infill", "located-suffix", "resample")
# The threshold the earlier post-hoc note used; kept so its claim can be checked as stated.
_LONGER_THAN_LINE = 1.3
_DIAG = re.compile(r"harness\.cc:(\d+):\d+: (?:fatal )?error: (.*)$")


def _tokenize(texts: list[str]) -> list[int]:
    """Re-tokenized lengths (tokenizer only, CPU). Not the generated ids: an approximation."""
    _, tokens = tokenize(texts)
    return [len(entry["ids"]) for entry in tokens]


def _messages(diagnostics: list[str], prefix_lines: int) -> list[tuple[int, str]]:
    """``(body line, message)`` per diagnostic; body line <= 0 means in the scaffold."""
    parsed = []
    for line in diagnostics:
        match = _DIAG.search(line)
        if match:
            parsed.append((int(match.group(1)) - prefix_lines, match.group(2).strip()))
    return parsed


def _inspection_outcome(reference: str, candidate: str, class_name: str) -> dict[str, Any]:
    ref_obj = constructed_object(reference, class_name)
    cand_obj = constructed_object(candidate, class_name)
    ref_calls = {c.method for c in active_member_calls(reference) if c.receiver == ref_obj}
    cand_calls = {c.method for c in active_member_calls(candidate) if c.receiver == cand_obj}
    cand_commented = {c.method for c in commented_member_calls(candidate) if c.receiver == cand_obj}
    if cand_obj is None:
        outcome = "construction-removed"
    elif ref_calls:
        if cand_calls == ref_calls:
            outcome = "kept-same-methods"
        elif cand_calls:
            outcome = "rewritten-methods"
        elif cand_commented:
            outcome = "commented-out"
        else:
            outcome = "deleted"
    else:
        outcome = "inspection-added" if cand_calls else "construct-only-as-reference"
    return {
        "inspection_outcome": outcome,
        "reference_methods": sorted(ref_calls),
        "candidate_methods": sorted(cand_calls),
    }


def _surplus(original_line: str, replacement: str, class_name: str) -> list[str]:
    masked_rep, masked_orig = mask_comments(replacement), mask_comments(original_line)
    kinds = []
    if masked_rep.count(";") > masked_orig.count(";"):
        kinds.append("extra-statement")
    if comment_text(replacement).strip() and not comment_text(original_line).strip():
        kinds.append("added-comment")
    if masked_rep.count(class_name) > masked_orig.count(class_name):
        kinds.append("duplicated-type-name")
    return kinds or ["none-detected"]


def _replacement(reference: str, candidate: str, located: int) -> tuple[str, str]:
    """``(regenerated text, method)`` for an infill candidate, anchored on the located line.

    Preserved context is the reference text before the located line and after its
    newline. Infill preservation (asserted in the pilot) guarantees both survive at the
    token level, so the candidate should start and end with them. The text in between is
    what the model wrote for the located line, newline included. Falls back to a
    common-prefix/suffix diff, which undercounts, and labels it if token boundaries make
    the anchors miss.
    """
    lines = reference.split("\n")
    left = "\n".join(lines[: located - 1]) + ("\n" if located > 1 else "")
    right = "\n".join(lines[located:])
    if (
        candidate.startswith(left)
        and candidate.endswith(right)
        and len(left) + len(right) <= len(candidate)
    ):
        return candidate[len(left) : len(candidate) - len(right)], "line-anchored"
    start = 0
    while start < min(len(reference), len(candidate)) and reference[start] == candidate[start]:
        start += 1
    end = 0
    limit = min(len(reference), len(candidate)) - start
    while end < limit and reference[-1 - end] == candidate[-1 - end]:
        end += 1
    return candidate[start : len(candidate) - end], "diff-fallback"


def _oracle_constructor_repair(trial: dict[str, Any]) -> str | None:
    """The reference body with only its located line replaced by a correct constructor.

    Applies only when the located line is meant to construct the object (it names the
    class and ``stream``). Keeps the model's own variable name so later lines still
    refer to it. If errors remain elsewhere, they are independent defects: no
    single-line repair of the located line could make the body compile.
    """
    example: str = trial["example"]
    lines = trial["reference_body"].split("\n")
    located = lines[trial["located_body_line"] - 1]
    code = mask_comments(located)
    if example not in code or "stream" not in code:
        return None
    match = re.search(example + r"\W+([A-Za-z_]\w*)\s*[(=]", code)
    if match is None or match.group(1) in {"stream", example}:
        return None
    lines[trial["located_body_line"] - 1] = (
        f"    OPENEXR_IMF_NAMESPACE::{example} {match.group(1)}(stream);"
    )
    return "\n".join(lines)


def _classify(trial: dict[str, Any], diags: dict[str, list[str]]) -> dict[str, Any]:
    example: str = trial["example"]
    prefix_lines = scaffold_prefix(example).count("\n")
    body: str = trial["reference_body"]
    located: int = trial["located_body_line"]
    ref_msgs = _messages(diags["reference"], prefix_lines)
    body_lines = sorted({line for line, _ in ref_msgs if line > 0})
    located_text = next(r.text for r in body_line_regions(body) if r.line_number == located)
    ref_obj = constructed_object(body, example)
    ref_inspection = [c.method for c in active_member_calls(body) if c.receiver == ref_obj]
    row: dict[str, Any] = {
        "example": example,
        "seed": trial["seed"],
        "reference_body": body,
        "located_body_line": located,
        "located_line_text": located_text,
        "located_line_raw": body.split("\n")[located - 1] + "\n",
        "located_line_kind": line_kinds(body)[located - 1],
        "nominal_has_right_context": trial["has_right_context"],
        "right_context": right_context(body, located),
        "n_diagnostics": len(ref_msgs),
        "n_scaffold_diagnostics": sum(1 for line, _ in ref_msgs if line <= 0),
        "diagnostic_body_lines": body_lines,
        "errors_on_multiple_lines": len(body_lines) > 1,
        "diagnostics": [m for _, m in ref_msgs],
        "reference_has_active_inspection": bool(ref_inspection),
        "reference_inspection_methods": ref_inspection,
        "reference_mentions_inspection_only_in_comments": not ref_inspection
        and bool([c for c in commented_member_calls(body) if c.receiver == ref_obj]),
        "located_window_tokens": trial["located_window"]["length"],
        "mask_length": trial["mask_length"],
        "arms": {},
    }
    oracle = diags.get("oracle-constructor-repair")
    row["oracle_constructor_repair"] = (
        {"applicable": False}
        if oracle is None
        else {
            "applicable": True,
            "remaining_errors": [m for _, m in _messages(oracle, prefix_lines)],
            "single_line_repair_sufficient": not oracle,
        }
    )
    ref_set = {m for _, m in ref_msgs}
    for arm in _ARMS:
        data = trial["arms"].get(arm)
        if data is None:
            continue
        msgs = _messages(diags[arm], prefix_lines)
        arm_set = {m for _, m in msgs}
        entry: dict[str, Any] = {
            "compiled": data["compiled"],
            "syntax_recheck_clean": not msgs,
            "body": data["body"],
            "n_diagnostics": len(msgs),
            "unresolved_reference_errors": sorted(ref_set & arm_set),
            "new_errors": sorted(arm_set - ref_set),
            **_inspection_outcome(body, data["body"], example),
        }
        if arm == "located-infill":
            entry["replacement_text"], entry["replacement_method"] = _replacement(
                body, data["body"], located
            )
        row["arms"][arm] = entry
    infill = row["arms"]["located-infill"]
    infill["surplus"] = _surplus(located_text, infill["replacement_text"], example)
    infill["replacement_chars_vs_line"] = len(infill["replacement_text"]) / len(
        row["located_line_raw"]
    )
    return row


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "trials": len(group),
            "compiled": {
                arm: sum(1 for r in group if r["arms"].get(arm, {}).get("compiled"))
                for arm in _ARMS
            },
            "compiled_inspection_outcomes": {
                arm: dict(
                    Counter(
                        r["arms"][arm]["inspection_outcome"]
                        for r in group
                        if r["arms"].get(arm, {}).get("compiled")
                    )
                )
                for arm in _ARMS
            },
        }
        for name, group in sorted(groups.items())
    }


def _discordance(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """located-infill vs located-suffix compile discordance per group (descriptive only)."""
    out: dict[str, Any] = {}
    for row in rows:
        group = out.setdefault(
            str(row[key]),
            {"trials": 0, "infill_only": 0, "suffix_only": 0, "both": 0, "neither": 0},
        )
        infill = row["arms"]["located-infill"]["compiled"]
        suffix = row["arms"]["located-suffix"]["compiled"]
        group["trials"] += 1
        cell = (
            "both"
            if infill and suffix
            else "infill_only"
            if infill
            else "suffix_only"
            if suffix
            else "neither"
        )
        group[cell] += 1
    return dict(sorted(out.items()))


def _failure_modes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in _ARMS:
        failing = [
            r["arms"][arm] for r in rows if arm in r["arms"] and not r["arms"][arm]["compiled"]
        ]
        out[arm] = {
            "failing": len(failing),
            "some_reference_error_unresolved": sum(
                1 for a in failing if a["unresolved_reference_errors"]
            ),
            "only_new_errors": sum(
                1 for a in failing if a["new_errors"] and not a["unresolved_reference_errors"]
            ),
        }
    return out


def main() -> None:
    """Write the audit JSON; print a short summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_file = {p.name: json.loads(p.read_text()) for p in _INPUTS}
    recomputed: dict[str, Any] = {name: _analyze(t) for name, t in per_file.items()}
    recomputed["pooled"] = _analyze([t for trials in per_file.values() for t in trials])
    committed = json.loads(_COMMITTED_ANALYSIS.read_text())
    # The committed file is keyed by the run-time output names (rq4-pilot-*.json), the
    # same files later copied into docs/assets under dated names; compare in order.
    committed_files = [v for k, v in committed.items() if k != "pooled"]
    recomputation_matches = (
        recomputed["pooled"] == committed["pooled"]
        and [v for k, v in recomputed.items() if k != "pooled"] == committed_files
    )

    eligible = [t for trials in per_file.values() for t in trials if t["eligible"]]
    order: list[tuple[int, str]] = []
    sources: list[str] = []
    for index, trial in enumerate(eligible):
        order.append((index, "reference"))
        sources.append(assemble(trial["example"], trial["reference_body"]))
        for arm in _ARMS:
            if arm in trial["arms"]:
                order.append((index, arm))
                sources.append(trial["arms"][arm]["harness"])
    oracle_index: dict[int, int] = {}
    for index, trial in enumerate(eligible):
        repaired = _oracle_constructor_repair(trial)
        if repaired is not None:
            oracle_index[index] = len(sources)
            order.append((index, "oracle-constructor-repair"))
            sources.append(assemble(trial["example"], repaired))
    results = syntax_diagnostics_many(sources)
    diags: list[dict[str, list[str]]] = [{} for _ in eligible]
    for (index, arm), lines in zip(order, results, strict=True):
        diags[index][arm] = lines

    rows = [_classify(trial, diags[i]) for i, trial in enumerate(eligible)]
    lengths = _tokenize(
        [r["located_line_raw"] for r in rows]
        + [r["arms"]["located-infill"]["replacement_text"] for r in rows]
    )
    for row, orig, rep in zip(rows, lengths[: len(rows)], lengths[len(rows) :], strict=True):
        row["located_line_retokenized"] = orig
        row["arms"]["located-infill"]["replacement_retokenized"] = rep
        row["arms"]["located-infill"]["replacement_vs_mask"] = rep - row["mask_length"]

    mismatches = [
        (r["example"], r["seed"], arm)
        for r in rows
        for arm, a in r["arms"].items()
        if a["compiled"] != a["syntax_recheck_clean"]
    ]
    stored_first3 = [
        (r["example"], r["seed"])
        for r, t in zip(rows, eligible, strict=True)
        if [d.message for d in parse_compiler_diagnostics(t["reference_compile_error"])]
        != r["diagnostics"][:3]
    ]
    for row in rows:
        row["stratum_refined"] = (
            "code-after"
            if row["right_context"] == "code"
            else "comment-only-after"
            if row["right_context"] == "comment-only"
            else "nothing-after"
        )
        row["single_vs_multi_line_errors"] = (
            "multi" if row["errors_on_multiple_lines"] else "single"
        )
        oracle = row["oracle_constructor_repair"]
        row["single_line_repairable_in_principle"] = (
            "not-assessed"
            if not oracle["applicable"]
            else "yes"
            if oracle["single_line_repair_sufficient"]
            else "no"
        )
        row["stratum_x_error_lines"] = (
            f"{row['stratum_refined']}/{row['single_vs_multi_line_errors']}"
        )

    nominal_right = [r for r in rows if r["nominal_has_right_context"]]
    report = {
        "status": "post-hoc audit of a finished pilot; descriptive only, no confirmatory claims",
        "recomputation_matches_committed_analysis": recomputation_matches,
        "syntax_recheck_vs_stored_build_mismatches": mismatches,
        "stored_first_three_messages_disagree_with_recheck": stored_first3,
        "counts": {
            "eligible": len(rows),
            "nominal_right_context": len(nominal_right),
            "nominal_right_context_but_comment_only_after": sum(
                1 for r in nominal_right if r["right_context"] == "comment-only"
            ),
            "errors_on_multiple_lines": sum(1 for r in rows if r["errors_on_multiple_lines"]),
            "errors_on_multiple_lines_within_nominal_right_context": sum(
                1 for r in nominal_right if r["errors_on_multiple_lines"]
            ),
            "reference_has_active_inspection": sum(
                1 for r in rows if r["reference_has_active_inspection"]
            ),
            "infill_replacement_method": dict(
                Counter(r["arms"]["located-infill"]["replacement_method"] for r in rows)
            ),
            "infill_replacement_longer_than_1_3x_located_line_chars": sum(
                1
                for r in rows
                if r["arms"]["located-infill"]["replacement_chars_vs_line"] > _LONGER_THAN_LINE
            ),
            "infill_replacement_shorter_than_mask_retokenized": sum(
                1 for r in rows if r["arms"]["located-infill"]["replacement_vs_mask"] < 0
            ),
            "infill_surplus_kinds": dict(
                Counter(k for r in rows for k in r["arms"]["located-infill"]["surplus"])
            ),
        },
        "by_example": _aggregate(rows, "example"),
        "by_nominal_right_context": _aggregate(rows, "nominal_has_right_context"),
        "by_refined_stratum": _aggregate(rows, "stratum_refined"),
        "by_error_lines": _aggregate(rows, "single_vs_multi_line_errors"),
        "by_reference_active_inspection": _aggregate(rows, "reference_has_active_inspection"),
        "by_single_line_repairable_in_principle": _aggregate(
            rows, "single_line_repairable_in_principle"
        ),
        "infill_vs_suffix_discordance_by_refined_stratum": _discordance(rows, "stratum_refined"),
        "infill_vs_suffix_discordance_by_stratum_x_error_lines": _discordance(
            rows, "stratum_x_error_lines"
        ),
        "infill_vs_suffix_discordance_nominal_right_context": _discordance(
            [r for r in rows if r["nominal_has_right_context"]], "stratum_x_error_lines"
        ),
        "failure_modes_of_non_compiling_arms": _failure_modes(rows),
        "trials": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in list(report)[:5]}, indent=1))


if __name__ == "__main__":
    main()
