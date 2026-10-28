# ruff: noqa: T201 -- deliberate console report, not a library module
"""Analyze RQ4b pilot results exactly as the frozen plan specifies (spec §8).

Inputs are ``rq4b_functional_pilot.py`` outputs from one condition (P1: no feedback, or
P2: feedback). Mixing conditions is refused. Each sample is one cluster; its replicates
are paired across arms by shared seeds.

- Gates: G1 (every infill candidate preserved its context), G2 (some arm's pooled pass@5
  lies strictly between 0.05 and 0.95).
- Pseudoreplication. The 60 units come from only 12 mutations, so every inferential
  quantity is computed on 12 per-mutation differences. Replicates are averaged inside a
  mutation first. Replicate- and candidate-level counts appear only as descriptive
  numbers, and are never treated as independent. The 12 mutations share 2 reference
  bodies. With 2 examples there is no between-example inference, so each example's
  estimate is reported separately.
- H1: infill-sweep vs suffix, functional pass@5 (equal attempts: K = 5 per arm,
  whatever each costs). H2: infill-sweep vs resample. Each is the mean per-sample
  difference, an exact sign-flip p and its inverted 95% CI (bootstrap CI as a
  sensitivity check), Holm-adjusted over {H1, H2}. In the pilot these are estimates for
  sizing, not confirmatory results.
- H3 (secondary, equal compute). Per unit, the budget is m times the forward cost
  (steps x sequence positions) of one resample candidate, the most expensive candidate
  of any arm. Each arm gets the longest prefix of its frozen order that fits
  (``success_within_budget``). Primary m = 1, so every arm affords at least one
  candidate. m = 1..5 is reported as a curve. The same curve is repeated with measured
  denoising seconds (``generate_seconds``; model load excluded) as a sensitivity check.
  A comparison where an arm affords zero candidates in more than half the units is
  marked degenerate and not interpreted.
- Descriptive: per-candidate success by infill mask offset, compile rate, failure
  reasons, unaffected statements kept, edit distances, cost, seconds.
- Sizing: ``required_clusters`` from the 80% upper confidence bound of H1's per-sample
  SD, for an effect of 0.10 and 0.15.
- ``--audit-sample N`` draws N candidates (seeded, stratified by verdict) for gate G3's
  hand check and writes them out. It runs before any contrast is read.

    uv run python tools/harness_synthesis/rq4b_analyze.py --out rq4b-p1-analysis.json \
      rq4b-p1-if.json rq4b-p1-mp.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import fmean, stdev
from typing import Any

from dllm_agent.harness_synthesis.repair_arms import (
    ARMS,
    DIAGNOSTIC_ARM,
    INFILL_MASK_OFFSETS,
    CandidateOutcome,
    success_within_budget,
)
from dllm_agent.harness_synthesis.repair_stats import (
    cluster_differences,
    compare,
    holm,
    pilot_sd_upper,
    required_clusters,
)

_FLOOR, _CEILING = 0.05, 0.95


_BUDGET_MULTIPLES = (1, 2, 3, 4, 5)
_DEGENERATE_SHARE = 0.5


def _cost(candidate: dict[str, Any], measure: str) -> int:
    if measure == "forward":
        return int(candidate["cost"])
    return round(float(candidate["generate_seconds"]) * 1_000_000)  # microseconds


def _outcomes(unit: dict[str, Any], arm: str, measure: str = "forward") -> list[CandidateOutcome]:
    return [
        CandidateOutcome(
            c["index"], c["compiled"], c["success"], _cost(c, measure), c["generate_seconds"]
        )
        for c in unit["candidates"]
        if c["arm"] == arm
    ]


def _budget(unit: dict[str, Any], multiple: int, measure: str) -> int:
    """``multiple`` x the cost of the unit's first resample candidate (the dearest kind)."""
    first = next(c for c in unit["candidates"] if c["arm"] == "resample" and c["index"] == 0)
    return multiple * _cost(first, measure)


def _pairs(  # noqa: PLR0913 - two arms plus the frozen budget definition.
    units: list[dict[str, Any]],
    a: str,
    b: str,
    *,
    matched: bool,
    multiple: int = 1,
    measure: str = "forward",
) -> dict[str, list[tuple[bool, bool]]]:
    pairs: dict[str, list[tuple[bool, bool]]] = {}
    for unit in units:
        if matched:
            budget = _budget(unit, multiple, measure)
            sa, _ = success_within_budget(_outcomes(unit, a, measure), budget)
            sb, _ = success_within_budget(_outcomes(unit, b, measure), budget)
        else:
            sa = unit["selection"][a]["pass_at_k"]
            sb = unit["selection"][b]["pass_at_k"]
        pairs.setdefault(unit["sample_id"], []).append((sa, sb))
    return pairs


def _descriptive(units: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        cands = [c for u in units for c in u["candidates"] if c["arm"] == arm]
        budgets = [_budget(u, 1, "forward") for u in units]
        afforded = [
            success_within_budget(_outcomes(u, arm), b)[1]
            for u, b in zip(units, budgets, strict=True)
        ]
        out[arm] = {
            "units": len(units),
            "pass_at_5": sum(u["selection"][arm]["pass_at_k"] for u in units),
            "compiled_any": sum(u["selection"][arm]["compiled_any"] for u in units),
            "candidates": len(cands),
            "candidate_success_rate": fmean(c["success"] for c in cands) if cands else None,
            "candidate_compile_rate": fmean(c["compiled"] for c in cands) if cands else None,
            "compiled_but_not_functional": sum(
                1 for c in cands if c["compiled"] and not c["success"]
            ),
            "failure_reasons_when_compiled": dict(
                Counter(
                    r for c in cands if c["compiled"] and not c["success"] for r in c["reasons"]
                )
            ),
            "mean_unaffected_statements_kept_fraction": fmean(
                c["unaffected_statements_kept"] / c["unaffected_statements_total"] for c in cands
            )
            if cands
            else None,
            "mean_edit_distance_to_reference_of_successes": fmean(
                c["edit_distance_to_reference"] for c in cands if c["success"]
            )
            if any(c["success"] for c in cands)
            else None,
            "mean_cost_per_candidate": fmean(c["cost"] for c in cands) if cands else None,
            "mean_generate_seconds": fmean(c["generate_seconds"] for c in cands) if cands else None,
            "mean_compiler_invocations_to_selection": fmean(
                u["selection"][arm]["compiler_invocations"] for u in units
            ),
            "cost_per_success": sum(c["cost"] for c in cands)
            / max(1, sum(c["success"] for c in cands)),
            "compute_matched_candidates_afforded": dict(Counter(afforded)),
        }
    infill = [c for u in units for c in u["candidates"] if c["arm"] == "infill-sweep"]
    out["infill-sweep"]["success_by_mask_offset"] = {
        f"L{offset:+d}": fmean(c["success"] for c in infill if c["index"] == j)
        for j, offset in enumerate(INFILL_MASK_OFFSETS)
        if any(c["index"] == j for c in infill)
    }
    out["infill-sweep"]["preserved_all"] = all(
        c["preserved"] for u in units for c in u["candidates"] if "preserved" in c
    )
    oracle = [c for u in units for c in u["candidates"] if c["arm"] == DIAGNOSTIC_ARM]
    out[DIAGNOSTIC_ARM] = {
        "note": "diagnostic only: mask = target token length (oracle); never compared",
        "candidates": len(oracle),
        "success_rate": fmean(c["success"] for c in oracle) if oracle else None,
        "compile_rate": fmean(c["compiled"] for c in oracle) if oracle else None,
    }
    return out


def _leave_one_out(pairs: dict[str, list[tuple[bool, bool]]]) -> dict[str, Any]:
    """Mean per-sample difference with each sample dropped in turn (decision rule, spec §8)."""
    diffs = dict(zip(sorted(pairs), cluster_differences(pairs), strict=True))
    loo = {
        sample: fmean(d for other, d in diffs.items() if other != sample)
        for sample in diffs
        if len(diffs) > 1
    }
    return {
        "mean_difference": fmean(diffs.values()) if diffs else 0.0,
        "per_sample_difference": diffs,
        "leave_one_out_min": min(loo.values()) if loo else None,
        "leave_one_out_max": max(loo.values()) if loo else None,
    }


def _audit_sample(units: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Seeded draw for G3, stratified by (compiled, success), each candidate keeping its body."""
    strata: dict[tuple[bool, bool], list[dict[str, Any]]] = {}
    for u in units:
        for c in u["candidates"]:
            strata.setdefault((c["compiled"], c["success"]), []).append(
                {"sample_id": u["sample_id"], **c}
            )
    rng = Random(20261026)  # noqa: S311 - reproducible audit draw, not security.
    picked: list[dict[str, Any]] = []
    keys = sorted(strata)
    while len(picked) < n and any(strata.values()):
        for key in keys:
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop(rng.randrange(len(strata[key]))))
    return [
        {
            k: c[k]
            for k in ("sample_id", "arm", "index", "seed", "body", "compiled", "success", "reasons")
        }
        for c in picked
    ]


def main() -> None:
    """Write the frozen analysis (or the G3 audit sample) for one condition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-sample", type=int, default=0)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    units = [u for p in args.inputs for u in json.loads(Path(p).read_text())]
    conditions = {u["feedback"] for u in units}
    if len(conditions) != 1:
        reason = "refusing to pool feedback conditions; analyze P1 and P2 separately"
        raise SystemExit(reason)
    if args.audit_sample:
        Path(args.out).write_text(
            json.dumps(_audit_sample(units, args.audit_sample), indent=1) + "\n"
        )
        return

    descriptive = _descriptive(units)
    rates = [descriptive[arm]["pass_at_5"] / len(units) for arm in ARMS]
    primary = {
        "H1_infill_vs_suffix": compare(_pairs(units, "infill-sweep", "suffix", matched=False)),
        "H2_infill_vs_resample": compare(_pairs(units, "infill-sweep", "resample", matched=False)),
    }
    adjusted = holm({name: c.p_value for name, c in primary.items()})
    secondary = {
        "H3_compute_matched_infill_vs_suffix": compare(
            _pairs(units, "infill-sweep", "suffix", matched=True)
        ),
        "H3_compute_matched_infill_vs_resample": compare(
            _pairs(units, "infill-sweep", "resample", matched=True)
        ),
    }
    curves: dict[str, Any] = {}
    for measure in ("forward", "seconds"):
        for multiple in _BUDGET_MULTIPLES:
            row: dict[str, Any] = {}
            for arm in ARMS:
                afforded = [
                    success_within_budget(_outcomes(u, arm, measure), _budget(u, multiple, measure))
                    for u in units
                ]
                row[arm] = {
                    "success_units": sum(ok for ok, _ in afforded),
                    "zero_candidates_share": fmean(n == 0 for _, n in afforded),
                    "mean_candidates_afforded": fmean(n for _, n in afforded),
                }
            row["degenerate"] = any(
                row[arm]["zero_candidates_share"] > _DEGENERATE_SHARE for arm in ARMS
            )
            curves[f"{measure}_x{multiple}"] = row
    per_example = {
        example: {
            name: asdict(
                compare(
                    _pairs(
                        [u for u in units if u["example"] == example],
                        "infill-sweep",
                        other,
                        matched=False,
                    )
                )
            )
            for name, other in (
                ("H1_infill_vs_suffix", "suffix"),
                ("H2_infill_vs_resample", "resample"),
            )
        }
        for example in sorted({u["example"] for u in units})
    }
    robustness = {
        name: _leave_one_out(_pairs(units, "infill-sweep", other, matched=matched))
        for name, other, matched in (
            ("H1_infill_vs_suffix", "suffix", False),
            ("H3_compute_matched_infill_vs_suffix", "suffix", True),
        )
    }
    h1_diffs = cluster_differences(_pairs(units, "infill-sweep", "suffix", matched=False))
    sizing: dict[str, Any] = {}
    if len(h1_diffs) >= 2 and any(h1_diffs):  # noqa: PLR2004
        sd_upper = pilot_sd_upper(h1_diffs)
        sizing = {
            "h1_sd_point": stdev(h1_diffs),
            "h1_sd_80pct_upper": sd_upper,
            "clusters_for_delta_0.10_alpha_0.025": required_clusters(sd_upper, 0.10, alpha=0.025),
            "clusters_for_delta_0.15_alpha_0.025": required_clusters(sd_upper, 0.15, alpha=0.025),
        }
    report = {
        "condition": "feedback" if conditions.pop() else "location-only",
        "units": len(units),
        "samples": len({u["sample_id"] for u in units}),
        "gate_G1_preservation": descriptive["infill-sweep"]["preserved_all"],
        "gate_G2_discriminating": any(_FLOOR < r < _CEILING for r in rates),
        "primary_exploratory_in_pilot": {
            name: {**asdict(c), "holm_p": adjusted[name]} for name, c in primary.items()
        },
        "secondary": {name: asdict(c) for name, c in secondary.items()},
        "secondary_H3_primary_budget_degenerate": curves["forward_x1"]["degenerate"],
        "equal_compute_curves": curves,
        "per_example_descriptive": per_example,
        "pseudoreplication_note": (
            "inference uses one difference per mutation (clusters); replicate/candidate "
            "counts are descriptive only; 2 reference bodies allow no between-example inference"
        ),
        "descriptive": descriptive,
        "decision_inputs": robustness,
        "by_kind_pass_at_5": {
            kind: {
                arm: sum(u["selection"][arm]["pass_at_k"] for u in units if u["kind"] == kind)
                for arm in ARMS
            }
            for kind in sorted({u["kind"] for u in units})
        },
        "by_sample_pass_at_5": {
            sid: {
                arm: sum(u["selection"][arm]["pass_at_k"] for u in units if u["sample_id"] == sid)
                for arm in ARMS
            }
            for sid in sorted({u["sample_id"] for u in units})
        },
        "sizing": sizing,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("condition", "units", "gate_G1_preservation", "gate_G2_discriminating")
            }
        )
    )
    print(json.dumps(report["primary_exploratory_in_pilot"], indent=1))


if __name__ == "__main__":
    main()
