# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ3-v3 feasibility check -- does body-only generation give real remask candidates?

FROZEN protocol (written before running; the pass/fail numbers below are not changed
after seeing results):

Purpose: before designing a paired confidence-vs-random remask pilot on the v3 fixed
scaffold, check that (a) body generation reliably terminates (the model writes a
few statements and stops, rather than rambling past budget), and (b) the resulting
body usually has at least 2 detectable statement regions -- otherwise there is
nothing for "confidence-guided" vs "random" remasking to meaningfully choose between,
and running a pilot would not answer anything.

Dataset: 12 fresh generations per example (InputFile, MultiPartInputFile) = 24 total,
seeds 9,000,000-9,000,023 (9,000,000-9,000,999 reserved from any future RQ3
confirmatory run, same convention as the B0/B0v2 blocks). Small N deliberately --
this is a go/no-go check, not a powered measurement.

Protocol: kb_v3_fixed_scaffold.scaffold_prefix(example) as --prefill-suffix,
kb_v3_fixed_scaffold.body_prompt(example) as --prompt, new_tokens=60, steps=30,
alg_temp=0.5 (same model/alg as every prior experiment). Assemble via
kb_v3_fixed_scaffold.assemble, compile+fuzz via compile_and_fuzz(fuzz_seconds=10.0
-- short, this is a feasibility check, not the pilot's own measurement).

Recorded per trial: example, seed, is_complete (extract_body's own signal),
body_text, region_count (statement_regions() run on the body text alone), compiled,
compile_error, edges.

Frozen decision gate (checked in this order; no threshold invented after the fact):
  1. extraction success rate < 70% -> STOP: body generation itself doesn't reliably
     terminate at this new_tokens/steps budget; report and do not proceed.
  2. among extracted bodies, fraction with region_count >= 2 is < 50% -> STOP: even
     when the model does finish, there is usually nothing to choose between for a
     remask comparison; report and do not proceed.
  3. fewer than 3 distinct body texts across all trials -> STOP: no diversity to
     compare arms against; report and do not proceed.
  4. none of the above -> PASS: proceed to design the small paired pilot.

NOT EXECUTED as part of writing this file. Reviewed first.

    uv run python tools/harness_synthesis/rq3_v3_feasibility.py \
      --gpu 0 --example InputFile --seed-start 9000000 --count 12 \
      --out rq3-v3-feas-inputfile.json \
      --ledger /data/dllm-agent/harness-synthesis/rq3-v3-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from _compile import compile_and_fuzz
from _replay import run_dream_generate
from kb_v3_fixed_scaffold import assemble, body_prompt, extract_body, scaffold_prefix

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.regions import statement_regions

_NEW_TOKENS = 60
_STEPS = 30
_FUZZ_SECONDS = 10.0


def _run_one(example: str, seed: int, *, gpu: int | None) -> dict[str, object]:
    prefix = scaffold_prefix(example)
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_dream_generate(
            body_prompt(example),
            seed,
            Path(tmp) / "generate.json",
            gpu=gpu,
            new_tokens=_NEW_TOKENS,
            steps=_STEPS,
            alg_temp=0.5,
            prefill_suffix=prefix,
        )
    full_text = "".join(payload["generated_token_pieces"])
    body_text, is_complete = extract_body(full_text)
    region_count = len(statement_regions(body_text)) if is_complete else 0

    harness = assemble(example, body_text)
    fuzz = compile_and_fuzz(harness, fuzz_seconds=_FUZZ_SECONDS, seed=seed)
    return {
        "example": example,
        "seed": seed,
        "is_complete": is_complete,
        "body_text": body_text,
        "region_count": region_count,
        "compiled": fuzz.compiled,
        "compile_error": fuzz.compile_error,
        "edges": fuzz.edges if fuzz.edges is not None else 0,
        "harness": harness,
    }


def main() -> None:
    """NOT to be run until the frozen protocol above has been reviewed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--example", required=True, choices=("InputFile", "MultiPartInputFile"))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    run_id = f"rq3-v3-feas-{args.example}-gpu{args.gpu}-{int(time.time())}"
    prompt_for_hash = body_prompt(args.example)

    results: list[dict[str, object]] = []
    for iteration in range(args.count):
        seed = args.seed_start + iteration
        record = _run_one(args.example, seed, gpu=args.gpu)
        append(
            ledger_path,
            HarnessSynthesisLedgerRecord(
                run_id=run_id,
                iteration=iteration,
                arm="v3-feasibility",
                seed=seed,
                harness_source_sha256=sha256_bytes(record["harness"].encode()),  # type: ignore[union-attr]
                knowledge_base_sha256=sha256_bytes(prompt_for_hash.encode()),
                usage_example_names=(args.example,),
                compiled=record["compiled"],  # type: ignore[arg-type]
                compile_error=record["compile_error"],  # type: ignore[arg-type]
                coverage_edges=record["edges"],  # type: ignore[arg-type]
                recorded_at_unix_seconds=time.time(),
            ),
        )
        print(
            f"seed={seed} complete={record['is_complete']!s:5} "
            f"regions={record['region_count']} compiled={record['compiled']!s:5} "
            f"edges={record['edges']}"
        )
        results.append(record)
        Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} feasibility trials completed for {args.example}.")


if __name__ == "__main__":
    main()
