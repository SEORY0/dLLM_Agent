# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ3-B0 -- Directed Baseline Stability Audit.

FROZEN PROTOCOL (written before any of the 64 trials run; do not edit after seeing
results -- if a change is genuinely needed, note it as a deviation, do not rewrite
this docstring silently).

Purpose
-------
Determine whether the directed/reference generation substrate is stable enough to
support the planned RQ3 confirmatory comparison, BEFORE spending that budget. This
audit does not estimate or report a confidence-vs-random treatment effect -- it has
no confidence/random arms at all, only the reference (unmodified directed) generation.

Why this exists: Task 8b reported 24/24 compiling directed harnesses (8 seeds x 3
examples, seeds 3,000,000-3,000,007). The RQ3 smoke (Task 10), using different fresh
seeds (5,000,000-5,000,003) for the same generation config, got only 5/12 (42%).
Both numbers come from small samples of the same nominal process; this audit is the
first attempt to measure that process's real compile rate at a size (64) that at
least gives a usable confidence interval, before locking in RQ3's sample size.

Dataset
-------
64 directed/reference generations over fresh seeds 6,000,000-6,000,063 inclusive.
These seeds are reserved for this audit ONLY -- reserve the full block
6,000,000-6,000,999 as permanently excluded from any future RQ3 confirmatory run
(headroom in case a rerun or extension of this audit is needed later).

Stratification: the frozen dataset currently has only 3 verified usage examples
(checkOpenEXRFile, InputFile, MultiPartInputFile -- kb_diversity_experiment._EXAMPLES),
not 8 as an illustrative 8x8 split would need. Seeds are assigned to examples in
that fixed list order, consecutively, with NO regard to any outcome (pre-registered,
not cherry-picked): the first 22 seeds go to checkOpenEXRFile, the next 21 to
InputFile, the next 21 to MultiPartInputFile (22+21+21=64). A design that wanted
broader task/example coverage would need new hand-verified usage examples first (its
own task, with its own independent-compile-verification burden per api_knowledge.py's
design principle) -- not something this audit invents on the fly.

Protocol
--------
Exactly RQ3 smoke's reference-arm configuration, unchanged to "improve" the rate:
  - model: Dream-Coder-v0-Instruct-7B at /data/dllm-agent/cache-dream-coder
  - generation controls: alg="entropy", temperature=0.5, top_p=0.95, alg_temp=0.5
  - ceilings: new_tokens=768, steps=384 (same for every example, not per-example tuned)
  - prompt: kb_diversity_experiment._prompt(example, "directed")
  - compiler/build: tools/openexr_runtime/harness-compile-only image, coverage mode
  - fuzz: compile_and_fuzz(..., fuzz_seconds=20.0) -- same fixed duration as the smoke
  - hardware: one GPU per process, split across GPU 0 and GPU 1 (32 trials each)

Recorded per trial (JSON results file + audit_ledger.py, arm="b0-audit"):
  example, seed, compiled, failure_category, generation_tokens (the configured
  ceiling, constant), effective_token_count (tokens before the closing ``` fence, or
  the full ceiling if no fence was found), generation_steps (ceiling, constant),
  hit_budget (no closing fence found within the ceiling), compile_error (raw
  diagnostic, preserved verbatim), edges (0 if not compiled), wall_time_seconds.

Failure taxonomy (coarse, reproducible; first matching rule wins, checked in this
order so budget exhaustion is never miscategorized as a language error it merely
looks like once truncated):
  1. generation-budget exhaustion/truncation -- hit_budget is True
  2. hallucinated header -- "file not found" in the diagnostic
  3. hallucinated API/symbol -- "no member named" / "undeclared identifier" /
     "unknown type name" / "use of undeclared"
  4. type/signature mismatch -- "cannot initialize" / "no viable conversion" /
     "no matching constructor" / "no matching function"
  5. syntax/incomplete generation -- "expected '}'" / "expected ';'" /
     "expected expression" / "expected member name"
  6. linker/build failure -- returncode nonzero with no matched diagnostic pattern
     AND the harness itself never got a compile attempt (infra-level failure)
  7. other -- fallback

Analysis (descriptive substrate validation only -- see rq3_b0_analysis.py)
---------------------------------------------------------------------------
Overall compile rate + Wilson score 95% CI; compile rate by example; failure-category
counts; fraction with hit_budget=True; tokens/steps distributions (degenerate here
since ceilings are constant, but effective_token_count varies); edges conditional on
compiling. No confidence-vs-random comparison is computed or reported from this data.

Decision gate (thresholds fixed here, before outcomes; see plan doc for rationale)
-----------------------------------------------------------------------------------
Checked in this order (most severe first) -- see the plan doc's own section for the
full rationale behind each number:
  4. persistently low & not truncation-driven -> reconsider substrate:
     overall compile rate < 40% AND hit_budget fraction < 30%
  3. budget-driven -> reassess ceiling before RQ3:
     hit_budget fraction >= 30% (regardless of compile rate)
  2. heterogeneous -> stratify/redesign:
     (max example compile rate - min example compile rate) >= 40 percentage points
  1. broadly stable -> proceed to re-evaluate RQ3 sample size:
     none of the above triggered (overall compile rate >= 40%, hit_budget < 30%,
     spread < 40pp)

THIS SCRIPT IS NOT EXECUTED AS PART OF WRITING THIS PROTOCOL. It is reviewed first.

    uv run python tools/harness_synthesis/rq3_b0_baseline_audit.py \
      --gpu 0 --seed-block-start 6000000 --seed-count 32 \
      --out rq3-b0-gpu0.json --ledger /data/dllm-agent/harness-synthesis/rq3-b0-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

from _compile import compile_and_fuzz
from _replay import decode_code_block, run_dream_generate
from kb_diversity_experiment import _EXAMPLES, _prompt

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append

_DIRECTED = "directed"
_SEED_BLOCK_START = 6_000_000
_SEED_BLOCK_RESERVED_END = 6_000_999  # exclusive-of-use ceiling: RQ3 confirmatory must
# never draw a seed in [_SEED_BLOCK_START, _SEED_BLOCK_RESERVED_END], not just the 64
# this audit actually uses -- headroom for a rerun or extension of this same audit.
_TRIALS_PER_EXAMPLE = (22, 21, 21)  # sums to 64; assigned in _EXAMPLES' fixed order.
_NEW_TOKENS = 768
_STEPS = 384
_FUZZ_SECONDS = 20.0

_BUDGET_TRUNCATION = "generation-budget exhaustion/truncation"
_FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hallucinated header", re.compile(r"file not found")),
    (
        "hallucinated API/symbol",
        re.compile(r"no member named|undeclared identifier|unknown type name"),
    ),
    (
        "type/signature mismatch",
        re.compile(r"cannot initialize|no viable conversion|no matching (constructor|function)"),
    ),
    (
        "syntax/incomplete generation",
        re.compile(r"expected '\}'|expected ';'|expected expression|expected member name"),
    ),
)


def seed_assignments() -> list[tuple[str, int]]:
    """Return the frozen, pre-registered (example_name, seed) list -- 64 pairs.

    Assignment order is fixed by ``_EXAMPLES``' own order and ``_TRIALS_PER_EXAMPLE``,
    never by any outcome. Calling this function is itself part of the frozen record:
    if the code changes, the assignment changes, so this function must not be edited
    after any trial has run against it.
    """
    assignments: list[tuple[str, int]] = []
    seed = _SEED_BLOCK_START
    for example, count in zip(_EXAMPLES, _TRIALS_PER_EXAMPLE, strict=True):
        for _ in range(count):
            assignments.append((example.name, seed))
            seed += 1
    return assignments


def classify_failure(*, compiled: bool, hit_budget: bool, compile_error: str | None) -> str:
    """Return the first matching coarse failure category (order matters -- see docstring)."""
    if compiled:
        return ""
    if hit_budget:
        return _BUDGET_TRUNCATION
    error = compile_error or ""
    for category, pattern in _FAILURE_PATTERNS:
        if pattern.search(error):
            return category
    if not error:
        return "linker/build failure"
    return "other"


def _run_one(
    example_name: str, seed: int, *, gpu: int | None, prompt: str
) -> dict[str, object]:
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_dream_generate(
            prompt,
            seed,
            Path(tmp) / "generate.json",
            gpu=gpu,
            new_tokens=_NEW_TOKENS,
            steps=_STEPS,
            alg_temp=0.5,
        )
    code, code_offsets = decode_code_block(payload)
    pieces: list[str] = payload["generated_token_pieces"]
    full_text = "".join(pieces)
    # decode_code_block truncates at the first closing fence/stop token; if nothing
    # was found, it returns the full text unchanged -- that is exactly "ran out of
    # budget before the model itself signalled it was done".
    hit_budget = code == full_text
    effective_token_count = len(code_offsets)

    fuzz = compile_and_fuzz(code, fuzz_seconds=_FUZZ_SECONDS, seed=seed)
    wall_time = time.time() - start
    failure_category = classify_failure(
        compiled=fuzz.compiled, hit_budget=hit_budget, compile_error=fuzz.compile_error
    )
    return {
        "example": example_name,
        "seed": seed,
        "compiled": fuzz.compiled,
        "failure_category": failure_category,
        "generation_tokens_ceiling": _NEW_TOKENS,
        "effective_token_count": effective_token_count,
        "generation_steps_ceiling": _STEPS,
        "hit_budget": hit_budget,
        "compile_error": fuzz.compile_error,
        "edges": fuzz.edges if fuzz.edges is not None else 0,
        "wall_time_seconds": wall_time,
        "code": code,
    }


def main() -> None:
    """NOT to be run until the frozen protocol above has been reviewed and approved."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed-block-start", type=int, default=_SEED_BLOCK_START)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    assignments = seed_assignments()
    offset_by_seed = {seed: index for index, (_, seed) in enumerate(assignments)}
    start_index = offset_by_seed[args.seed_block_start]
    my_assignments = assignments[start_index : start_index + args.seed_count]

    ledger_path = Path(args.ledger)
    run_id = f"rq3-b0-gpu{args.gpu}-{int(time.time())}"
    prompts = {example.name: _prompt(example, _DIRECTED) for example in _EXAMPLES}

    results: list[dict[str, object]] = []
    for iteration, (example_name, seed) in enumerate(my_assignments):
        record = _run_one(example_name, seed, gpu=args.gpu, prompt=prompts[example_name])
        append(
            ledger_path,
            HarnessSynthesisLedgerRecord(
                run_id=run_id,
                iteration=iteration,
                arm="b0-audit",
                seed=seed,
                harness_source_sha256=sha256_bytes(record["code"].encode()),  # type: ignore[union-attr]
                knowledge_base_sha256=sha256_bytes(prompts[example_name].encode()),
                usage_example_names=(example_name,),
                compiled=record["compiled"],  # type: ignore[arg-type]
                compile_error=record["compile_error"],  # type: ignore[arg-type]
                coverage_edges=record["edges"],  # type: ignore[arg-type]
                recorded_at_unix_seconds=time.time(),
            ),
        )
        print(
            f"{example_name:18} seed={seed} compiled={record['compiled']!s:5} "
            f"category={record['failure_category']!r:35} hit_budget={record['hit_budget']}"
        )
        results.append(record)
        Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} audit trials completed.")


if __name__ == "__main__":
    main()
