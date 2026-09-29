# ruff: noqa: T201 -- deliberate console report, not a library module
r"""Task 9 -- RQ1: does confidence localize the real compile error better than chance?

At the pre-registered N from Task 3's power calc (N=17, alpha=0.05, power=0.80, from
the feasibility pilot's 11/16-vs-3/16 marginals).

Purely observational, no KB, no remask, no intervention (spec §9: "RQ1 상관관계는 순수
관측 통계"). Each generation gives one paired trial when it (a) fails to compile and
(b) the first reported error line falls inside a detected region: does
``select_lowest_confidence_region`` pick that region, and does a matched
``select_random_region`` draw? Compiling generations and errors outside any detected
region are excluded (no error region to compare against) -- this script over-samples
seeds until it collects the target number of *valid* pairs, not just N attempts.

    uv run python tools/harness_synthesis/rq1_measurement.py \\
      --gpu 0 --target-n 17 --out rq1-results.json \\
      --ledger /data/dllm-agent/harness-synthesis/rq1-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
from pathlib import Path
from random import Random

from _compile import compile_harness
from _replay import decode_code_block, replay, run_dream_generate

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.regions import region_at_line, statement_regions
from dllm_agent.harness_synthesis.remask import (
    select_lowest_confidence_region,
    select_random_region,
)

_PROMPT = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    'Define extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) '
    "that feeds the fuzz bytes into OpenEXR parsing APIs and returns 0. Output only "
    "one ```cpp code block."
)
_ERROR_LINE_NUMBER = re.compile(r"harness\.cc:(\d+):")
_MAX_ATTEMPT_MULTIPLIER = 4  # safety cap: give up after this many x target-n attempts.


def sign_test_p_value(n_10: int, n_01: int) -> float:
    """Exact two-sided sign-test p-value on the discordant pairs.

    Matches research-report.md's "정확한 two-sided sign tests" convention: exact
    binomial, not a normal approximation.
    """
    n = n_10 + n_01
    if n == 0:
        return 1.0
    k = min(n_10, n_01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    """Collect the pre-registered N of valid paired trials and run the sign test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--target-n", type=int, default=17)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    run_id = f"rq1-gpu{args.gpu}-{int(time.time())}"

    trials: list[dict[str, object]] = []
    valid = 0
    attempt = 0
    max_attempts = args.target_n * _MAX_ATTEMPT_MULTIPLIER
    while valid < args.target_n and attempt < max_attempts:
        seed = 4_000_000 + args.seed_start + attempt
        attempt += 1
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_dream_generate(
                _PROMPT,
                seed,
                Path(tmp) / "dream.json",
                gpu=args.gpu,
                new_tokens=384,
                steps=192,
                # alg_temp=0.0 makes reveal order deterministic-argmax, so different
                # seeds mostly reproduce the same output (confirmed: an alg_temp=0.0
                # run of this exact script only got 4-5 distinct codes out of 9
                # "different-seed" attempts). RQ1 needs real independent draws.
                alg_temp=0.5,
            )
        recorder = replay(payload)
        prompt_length = int(payload["prompt_length"])
        code, code_offsets = decode_code_block(payload)
        regions = statement_regions(code)
        compile_result = compile_harness(code)

        append(
            ledger_path,
            HarnessSynthesisLedgerRecord(
                run_id=run_id,
                iteration=attempt - 1,
                arm="rq1-observation",
                seed=seed,
                harness_source_sha256=sha256_bytes(code.encode()),
                knowledge_base_sha256=sha256_bytes(_PROMPT.encode()),
                usage_example_names=(),
                compiled=compile_result.compiled,
                compile_error=compile_result.error,
                coverage_edges=None,
                recorded_at_unix_seconds=time.time(),
            ),
        )

        if compile_result.compiled or not compile_result.error:
            print(f"seed={seed} compiled={compile_result.compiled} (excluded, no error line)")
            continue
        match = _ERROR_LINE_NUMBER.search(compile_result.error)
        if match is None:
            print(f"seed={seed} error has no parseable line (excluded)")
            continue
        error_region = region_at_line(regions, int(match.group(1)))
        if error_region is None:
            print(f"seed={seed} error line outside any detected region (excluded)")
            continue

        lowest = select_lowest_confidence_region(
            regions, code_offsets, recorder.traces, position_offset=prompt_length
        )
        random_pick = select_random_region(regions, Random(seed))  # noqa: S311
        confidence_correct = lowest is error_region
        random_correct = random_pick is error_region
        valid += 1
        trials.append(
            {
                "seed": seed,
                "error_line": int(match.group(1)),
                "confidence_correct": confidence_correct,
                "random_correct": random_correct,
                "code": code,
            }
        )
        print(
            f"seed={seed} valid_trial={valid}/{args.target_n} "
            f"confidence_correct={confidence_correct} random_correct={random_correct}"
        )
        Path(args.out).write_text(json.dumps(trials, indent=1))

    n_10 = sum(t["confidence_correct"] and not t["random_correct"] for t in trials)
    n_01 = sum(t["random_correct"] and not t["confidence_correct"] for t in trials)
    p_value = sign_test_p_value(n_10, n_01)
    print(
        f"\n{valid} valid trials ({attempt} attempts). "
        f"confidence-correct={sum(t['confidence_correct'] for t in trials)}/{valid} "
        f"random-correct={sum(t['random_correct'] for t in trials)}/{valid} "
        f"discordant n10={n_10} n01={n_01} sign-test p={p_value:.4f}"
    )
    Path(args.out).write_text(
        json.dumps({"trials": trials, "n_10": n_10, "n_01": n_01, "p_value": p_value}, indent=1)
    )


if __name__ == "__main__":
    main()
