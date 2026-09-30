# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ3-B0v2 -- one-round KB/prompt improvement re-audit.

Separate script from ``rq3_b0_baseline_audit.py`` (v1) -- v1's frozen protocol, its
64 results, and its ledger are untouched. This is the ONE improvement round the user
authorized (no iterative re-tuning after seeing v2's own results: if this does not
clear the gate below, the next step is reported as a recommendation, not a v3 script).

What changed vs v1: only the prompt text (``kb_v2_directed_prompt.prompt_v2``), which
adds the IStream virtual-method contract (re-verified against the real installed
header) and an explicit "copy character-for-character" instruction. The example code
itself (the ``MemoryIStream`` class, the API calls) is byte-identical to v1's -- it was
already contractually correct; the change targets the model deviating from a correct
example, not a wrong one. Every other configuration item is identical to v1 and to
the RQ3 smoke's reference arm: model, alg_temp=0.5, new_tokens=768, steps=384,
harness-compile-only image, fuzz_seconds=20.0.

Dataset: 64 fresh generations, seeds 8,000,000-8,000,063 (a new block; does not
overlap 3,000,000-3,000,007 [Task 8/8b], ~4,000,000-4,010,100 [RQ1],
5,000,000-5,000,003 [RQ3 smoke], or 6,000,000-6,000,999 [RQ3-B0 v1, reserved]).
Reserve 8,000,000-8,000,999 the same way v1's block was reserved: excluded from any
future RQ3 confirmatory run.

Split: 8 checkOpenEXRFile (regression control -- its prompt also changed slightly,
so re-checking it is not redundant), 28 InputFile, 28 MultiPartInputFile (the two
components this round actually targets) = 64.

Decision gate: the SAME thresholds as v1's, applied once, before running:
  - Gate 4: any example's compile rate < 40% AND hit_budget < 30% for that example
    -> that example's substrate is still not usable; do not proceed with it in RQ3.
  - Gate 2: spread across examples >= 40 percentage points -> heterogeneity remains.
  - Gate 1 (pass): every example clears >= 40% with hit_budget < 30%, and spread
    stays below 40pp -> proceed to the small paired remask pilot.
No new thresholds invented after seeing v2's numbers.

NOT EXECUTED as part of writing this file. Reviewed first, same as v1.

    uv run python tools/harness_synthesis/rq3_b0_baseline_audit_v2.py \
      --gpu 0 --seed-block-start 8000000 --seed-count 32 \
      --out rq3-b0v2-gpu0.json --ledger /data/dllm-agent/harness-synthesis/rq3-b0v2-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kb_v2_directed_prompt import prompt_v2
from rq3_b0_baseline_audit import _run_one

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append

_SEED_BLOCK_START = 8_000_000
_SEED_BLOCK_RESERVED_END = 8_000_999
_TRIALS_PER_EXAMPLE = (
    ("checkOpenEXRFile", 8),
    ("InputFile", 28),
    ("MultiPartInputFile", 28),
)  # sums to 64; checkOpenEXRFile is the regression control, the other two are targeted.


def seed_assignments() -> list[tuple[str, int]]:
    """Return the frozen, pre-registered (example_name, seed) list for v2 -- 64 pairs."""
    assignments: list[tuple[str, int]] = []
    seed = _SEED_BLOCK_START
    for example_name, count in _TRIALS_PER_EXAMPLE:
        for _ in range(count):
            assignments.append((example_name, seed))
            seed += 1
    return assignments


def main() -> None:
    """NOT to be run until the frozen v2 protocol above has been reviewed."""
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
    run_id = f"rq3-b0v2-gpu{args.gpu}-{int(time.time())}"
    prompts = {name: prompt_v2(name) for name, _ in _TRIALS_PER_EXAMPLE}

    results: list[dict[str, object]] = []
    for iteration, (example_name, seed) in enumerate(my_assignments):
        record = _run_one(example_name, seed, gpu=args.gpu, prompt=prompts[example_name])
        append(
            ledger_path,
            HarnessSynthesisLedgerRecord(
                run_id=run_id,
                iteration=iteration,
                arm="b0v2-audit",
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

    print(f"\n{len(results)} v2 audit trials completed.")


if __name__ == "__main__":
    main()
