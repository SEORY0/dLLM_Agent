# ruff: noqa: T201 -- deliberate console report, not a library module
"""Analyze RQ4 pilot results exactly as the frozen protocol specifies.

Reads one or more ``rq4_localized_infill_pilot.py`` output files and reports, over
eligible trials only: per-arm compile rate, stop gates (1)/(2), and the exact
two-sided sign test on discordant pairs for H-loc (located-infill vs random-infill),
H-bi (located-infill vs located-suffix, right-context stratum primary; the other
stratum reported separately, never pooled) and the anchor (located-infill vs
resample). Secondary: intent-to-treat mean edges, first-error-left-located-line,
and infill preservation. No new tests or strata beyond the protocol.

    uv run python tools/harness_synthesis/rq4_analyze.py \
      --out docs/assets/harness-rq4-pilot-analysis-20261026.json \
      rq4-pilot-inputfile.json rq4-pilot-multipart.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

_ARMS = ("located-infill", "random-infill", "located-suffix", "resample")
_UNLOCALIZABLE_MAX = 0.25
_MIN_ELIGIBLE = 12


def _sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test p-value on discordant pairs (p=0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _paired(trials: list[dict], first: str, second: str) -> dict[str, object]:
    pairs = [t for t in trials if first in t["arms"] and second in t["arms"]]
    wins = sum(
        1 for t in pairs if t["arms"][first]["compiled"] and not t["arms"][second]["compiled"]
    )
    losses = sum(
        1 for t in pairs if t["arms"][second]["compiled"] and not t["arms"][first]["compiled"]
    )
    return {
        "pairs": len(pairs),
        f"{first}_only": wins,
        f"{second}_only": losses,
        "concordant": len(pairs) - wins - losses,
        "sign_test_p": round(_sign_test(wins, losses), 4),
    }


def _arm_summary(trials: list[dict], arm: str) -> dict[str, object]:
    present = [t["arms"][arm] for t in trials if arm in t["arms"]]
    failing = [a for a in present if not a["compiled"]]
    edges = [sum(a["replicate_edges"]) / len(a["replicate_edges"]) for a in present]
    summary: dict[str, object] = {
        "n": len(present),
        "compiled": sum(1 for a in present if a["compiled"]),
        "itt_mean_edges": round(sum(edges) / len(edges), 1) if edges else None,
        "mean_tokens_masked": round(sum(a["tokens_masked"] for a in present) / len(present), 1)
        if present
        else None,
        "failing_first_error_left_located_line": sum(
            1 for a in failing if a.get("first_error_left_located_line")
        ),
    }
    if arm.endswith("infill"):
        summary["preserved"] = sum(1 for a in present if a.get("preserved"))
    return summary


def _analyze(attempts: list[dict]) -> dict[str, object]:
    eligible = [t for t in attempts if t["eligible"]]
    failures = [t for t in attempts if not t["reference_compiled"]]
    unlocalizable = sum(
        1 for t in failures if t.get("ineligible_reason") == "not-localizable-to-body"
    )
    right = [t for t in eligible if t["has_right_context"]]
    no_right = [t for t in eligible if not t["has_right_context"]]
    return {
        "attempts": len(attempts),
        "reference_compiled": len(attempts) - len(failures),
        "eligible": len(eligible),
        "unlocalizable_failures": unlocalizable,
        "gate_1_pass": not failures or unlocalizable / len(failures) <= _UNLOCALIZABLE_MAX,
        "gate_2_pass": len(eligible) >= _MIN_ELIGIBLE,
        "right_context_trials": len(right),
        "arms": {arm: _arm_summary(eligible, arm) for arm in _ARMS},
        "H_loc": _paired(eligible, "located-infill", "random-infill"),
        "H_bi_right_context": _paired(right, "located-infill", "located-suffix"),
        "H_bi_no_right_context_reported_separately": _paired(
            no_right, "located-infill", "located-suffix"
        ),
        "anchor": _paired(eligible, "located-infill", "resample"),
    }


def main() -> None:
    """Analyze each input file separately and pooled."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    per_file = {Path(p).name: json.loads(Path(p).read_text()) for p in args.inputs}
    report = {name: _analyze(attempts) for name, attempts in per_file.items()}
    report["pooled"] = _analyze([t for attempts in per_file.values() for t in attempts])
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
