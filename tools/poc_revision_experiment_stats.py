# ruff: noqa: D100,D101,D102,D103,TC003

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Final, Protocol

from tools.poc_revision_experiment_json import (
    JsonArray,
    JsonObject,
    JsonValue,
    require_array,
    require_bool,
    require_number,
    require_number_or_none,
    require_object,
)

BOOTSTRAP_SEED: Final = 202608209999
BOOTSTRAP_RESAMPLES: Final = 10_000
MIN_PAIRED_REPETITIONS: Final = 5


class MetricReport(Protocol):
    @property
    def metrics(self) -> JsonObject: ...


def contrast(by_arm: Mapping[str, Mapping[int, MetricReport]], left: str, right: str) -> JsonObject:
    repetitions = sorted(by_arm[left])
    diffs = [_score(by_arm[left][rep]) - _score(by_arm[right][rep]) for rep in repetitions]
    wins = sum(diff > 0 for diff in diffs)
    losses = sum(diff < 0 for diff in diffs)
    ties = sum(diff == 0 for diff in diffs)
    return {
        "left": left,
        "right": right,
        "paired_n": len(diffs),
        "mean_diff": _bootstrap_stat(diffs, "mean"),
        "median_diff": _bootstrap_stat(diffs, "median"),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rank_biserial": 0.0 if not diffs else (wins - losses) / len(diffs),
    }


def proof_time(reports: tuple[MetricReport, ...]) -> JsonObject:
    observed: list[JsonValue] = []
    right_censored = 0
    for report in reports:
        proof_seconds = require_number_or_none(report.metrics, "proof_time_seconds")
        censored = require_number_or_none(report.metrics, "proof_censored_seconds")
        if proof_seconds is not None:
            observed.append(proof_seconds)
        if censored is not None:
            right_censored += 1
    return {
        "observed_count": len(observed),
        "right_censored": right_censored,
        "observed_seconds": observed,
    }


def verdict(primary: JsonObject) -> str:
    if _paired_n(primary) < MIN_PAIRED_REPETITIONS:
        return "inconclusive"
    mean_diff = require_object(primary, "mean_diff")
    estimate = require_number(mean_diff, "estimate")
    lower = _number_at(mean_diff, "ci95", 0)
    upper = _number_at(mean_diff, "ci95", 1)
    if estimate >= 0.05 and lower > 0:  # noqa: PLR2004
        return "go"
    if upper < 0.05:  # noqa: PLR2004
        return "no-go"
    return "inconclusive"


def verdict_reasons(primary: JsonObject) -> JsonArray:
    if _paired_n(primary) < MIN_PAIRED_REPETITIONS:
        return ["insufficient_paired_repetitions"]
    return []


def _paired_n(primary: JsonObject) -> int:
    value = primary["paired_n"]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = "paired_n: expected integer"
    raise ValueError(message)


def _bootstrap_stat(diffs: list[float], statistic: str) -> JsonObject:
    observed = _statistic(diffs, statistic)
    if not diffs:
        return {"estimate": observed, "ci95": [0.0, 0.0]}
    rng = random.Random(BOOTSTRAP_SEED + (0 if statistic == "mean" else 1))  # noqa: S311
    estimates = [
        _statistic([diffs[rng.randrange(len(diffs))] for _ in diffs], statistic)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    estimates.sort()
    return {
        "estimate": observed,
        "ci95": [
            estimates[int(0.025 * (len(estimates) - 1))],
            estimates[int(0.975 * (len(estimates) - 1))],
        ],
    }


def _statistic(values: list[float], statistic: str) -> float:
    if not values:
        return 0.0
    if statistic == "mean":
        return sum(values) / len(values)
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _score(report: MetricReport) -> float:
    return 1.0 if require_bool(report.metrics, "validated_success") else 0.0


def _number_at(payload: JsonObject, key: str, index: int) -> float:
    values = require_array(payload, key)
    item = values[index]
    if isinstance(item, int | float) and not isinstance(item, bool):
        return float(item)
    message = f"{key}: expected numeric CI"
    raise ValueError(message)
