# ruff: noqa: T201 -- deliberate console report, not a library module
"""RQ4 step 0 -- can the compiler localize RQ3-v3's failures onto the model-written body?

Offline, no GPU, no Docker: re-reads the compile errors already stored in the three
RQ3-v3 result files (feasibility, region pilot, window pilot -- every non-compiling
*reference* body, i.e. the unmodified first generation, never a remask arm) and runs
them through the production ``localize`` module. Answers the precondition of the
compiler-localized infill direction (``docs/superpowers/specs/
2026-10-13-compiler-localized-span-infill.md``): if clang's first error usually fell in
the fixed scaffold, there would be nothing in the body to infill.

Also records, per failure, whether the located line has model-written code on its left
and/or right inside the body -- bidirectional infill can only beat left-to-right repair
where there *is* right context to condition on.

    uv run python tools/harness_synthesis/rq4_offline_localizability.py \
      --out docs/assets/harness-rq4-offline-localizability-20261013.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kb_v3_fixed_scaffold import scaffold_prefix

from dllm_agent.harness_synthesis.localize import (
    body_line_regions,
    locate_in_body,
    parse_compiler_diagnostics,
)

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets"
_FEASIBILITY = _ASSETS / "harness-rq3-v3-feasibility-results-20260930.json"
_PILOTS = (
    ("region-pilot", _ASSETS / "harness-rq3-v3-pilot-results-20260930.json"),
    ("window-pilot", _ASSETS / "harness-rq3-v3-window-pilot-results-20260930.json"),
)


def _reference_failures() -> list[tuple[str, str, int, str, str, str]]:
    """``(source, example, seed, harness, body, compile_error)`` per failing reference."""
    failures: list[tuple[str, str, int, str, str, str]] = [
        ("feasibility", t["example"], t["seed"], t["harness"], t["body_text"], t["compile_error"])
        for t in json.loads(_FEASIBILITY.read_text())
        if not t["compiled"]
    ]
    for source, path in _PILOTS:
        for t in json.loads(path.read_text()):
            measurement = t["arm_reference"]
            if not measurement["compiled"]:
                failures.append(
                    (
                        source,
                        t["example"],
                        t["seed"],
                        t["arm_harnesses"]["reference"],
                        t["arm_bodies"]["reference"],
                        measurement["compile_error"],
                    )
                )
    return failures


def _analyze(failure: tuple[str, str, int, str, str, str]) -> dict[str, object]:
    source, example, seed, harness, body, error = failure
    prefix = scaffold_prefix(example)
    if not harness.startswith(prefix):
        msg = f"{source} {example} seed={seed}: stored harness does not start with the scaffold"
        raise ValueError(msg)
    diagnostics = parse_compiler_diagnostics(error)
    lines = [region.line_number for region in body_line_regions(body)]
    row: dict[str, object] = {
        "source": source,
        "example": example,
        "seed": seed,
        "diagnostic_count": len(diagnostics),
        "body_nonblank_lines": len(lines),
    }
    if not diagnostics:
        return {**row, "located": "no-harness-diagnostic"}
    first = diagnostics[0]
    region = locate_in_body(first.line, prefix=prefix, body=body)
    if region is None:
        return {**row, "located": "outside-body", "first_message": first.message}
    return {
        **row,
        "located": "body",
        "body_line": region.line_number,
        "first_message": first.message,
        "has_left_context": region.line_number > min(lines),
        "has_right_context": region.line_number < max(lines),
        "all_diagnostics_in_body": all(
            locate_in_body(d.line, prefix=prefix, body=body) is not None for d in diagnostics
        ),
    }


def main() -> None:
    """Analyze every stored failing reference body and write the per-failure table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [_analyze(failure) for failure in _reference_failures()]
    located = [row for row in rows if row["located"] == "body"]
    summary = {
        "failing_references": len(rows),
        "located_counts": dict(Counter(str(row["located"]) for row in rows)),
        "located_with_right_context": sum(1 for row in located if row["has_right_context"]),
        "located_with_left_context": sum(1 for row in located if row["has_left_context"]),
        "located_all_diagnostics_in_body": sum(
            1 for row in located if row["all_diagnostics_in_body"]
        ),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    for row in rows:
        print(
            f"{row['source']:12} {row['example']:18} seed={row['seed']} {row['located']:12} "
            f"line={row.get('body_line', '-')} nonblank={row['body_nonblank_lines']} "
            f"{str(row.get('first_message', ''))[:70]}"
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
