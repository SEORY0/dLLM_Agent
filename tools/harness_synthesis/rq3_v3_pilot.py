# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ3-v3 small paired pilot -- confidence-guided vs random remask, on the fixed scaffold.

FROZEN protocol (written after the feasibility check passed, before running any pilot
trial; not changed after seeing pilot outcomes):

Runs only because rq3_v3_feasibility.py cleared all three gates (100% extraction,
66.7% of complete bodies had >=2 regions, 24/24 distinct bodies). This is still a
small, exploratory pilot -- not the N=469-scale confirmatory study, which remains on
hold.

Design: for each trial, generate ONE initial body (same params as the feasibility
check), branch that SAME body into a confidence-remask arm and a random-remask arm
(matched: same detected regions, same regeneration seed -- only the remask start
position differs, exactly like the v1 RQ3 smoke), and keep the unmodified initial
body as the reference arm. Remasking operates on the body's own ~60-token
continuation only (kb_v3_fixed_scaffold.extract_body / statement_regions on the body
text), never on the fixed scaffold, which cannot be part of what a remask round
touches because it is never part of what Dream generates.

Primary outcome (unchanged from the v1 RQ3 smoke protocol): fixed-fuzz-time edge
coverage, intent-to-treat (a non-compiling arm counts as 0 edges, not excluded).
Departure from v1: each arm is fuzzed with the SAME bundle of 3 libFuzzer seeds
(1, 2, 3), each for fuzz_seconds=20, instead of one single fuzz run -- this
estimates the measurement variance the B0 audit's coverage-variance check found
(same code, same corpus, different -seed or even the same -seed re-run, gave
different edge counts) directly, rather than treating it as an unmodeled caveat.
Report the mean and the observed min/max across the 3 replicates per arm.

Secondary/supplementary only (never the primary comparison): one additional
fixed-execution-count run (-runs=100000) per compiling arm.

Dataset: 6 seeds per example (InputFile, MultiPartInputFile) = 12 trials, seeds
10,000,000-10,000,011 (a new block, reserved 10,000,000-10,000,999 from any future
RQ3 confirmatory run, same convention as every prior stage's block).

Recorded per trial: example, seed, region counts and which was picked for each arm,
tokens_remasked/regen_steps per arm (logged, not forced equal -- same audit
requirement as v1), and per arm: compiled, compile_error, the 3 replicate edge
counts, their mean/min/max, and the supplementary fixed-runs edge count.

NOT EXECUTED as part of writing this file. Reviewed first, same as every prior stage.

    uv run python tools/harness_synthesis/rq3_v3_pilot.py \
      --gpu 0 --example InputFile --seed-start 10000000 --count 6 \
      --out rq3-v3-pilot-inputfile.json \
      --ledger /data/dllm-agent/harness-synthesis/rq3-v3-pilot-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from random import Random

from _compile import compile_for_repeated_fuzzing, fuzz_once
from _replay import replay, run_dream_generate
from kb_v3_fixed_scaffold import assemble, body_prompt, extract_body, scaffold_prefix

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.regions import statement_regions
from dllm_agent.harness_synthesis.remask import (
    region_token_positions,
    remask_from_position_to_end,
    select_lowest_confidence_region,
    select_random_region,
)

_NEW_TOKENS = 60
_STEPS = 30
_FUZZ_SECONDS = 20.0
_FUZZ_SEED_BUNDLE = (1, 2, 3)
_SECONDARY_RUNS = 100_000


def _regen_steps(remasked_count: int) -> int:
    """Same convention as the v1 RQ3 smoke: scale steps with the remasked span."""
    return max(8, remasked_count // 2)


def _generate_body(example: str, seed: int, *, gpu: int | None) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        return run_dream_generate(
            body_prompt(example),
            seed,
            Path(tmp) / "generate.json",
            gpu=gpu,
            new_tokens=_NEW_TOKENS,
            steps=_STEPS,
            alg_temp=0.5,
            prefill_suffix=scaffold_prefix(example),
        )


def _remask_arm(  # noqa: PLR0913 - one remask-round step, not worth splitting.
    example: str,
    seed: int,
    *,
    gpu: int | None,
    generated_token_ids: list[int],
    mask_token_id: int,
    start_position: int,
) -> tuple[str, int, int]:
    """Remask the body from ``start_position`` to its end and regenerate.

    Returns ``(body_text, tokens_remasked, steps)``.
    """
    remasked = remask_from_position_to_end(
        generated_token_ids, start_position, mask_token_id=mask_token_id
    )
    tokens_remasked = len(generated_token_ids) - start_position
    steps = _regen_steps(tokens_remasked)
    with tempfile.TemporaryDirectory() as tmp:
        cont_path = Path(tmp) / "continuation.json"
        cont_path.write_text(json.dumps(list(remasked)))
        payload = run_dream_generate(
            body_prompt(example),
            seed,
            Path(tmp) / "regen.json",
            gpu=gpu,
            steps=steps,
            alg_temp=0.5,
            prefill_suffix=scaffold_prefix(example),
            init_continuation_file=str(cont_path),
        )
    full_text = "".join(payload["generated_token_pieces"])
    body_text, _ = extract_body(full_text)
    return body_text, tokens_remasked, steps


def _measure_arm(harness: str) -> dict[str, object]:
    with compile_for_repeated_fuzzing(harness) as (binary_dir, error):
        if binary_dir is None:
            return {
                "compiled": False,
                "compile_error": error,
                "replicate_edges": [0, 0, 0],
                "mean_edges": 0.0,
                "min_edges": 0,
                "max_edges": 0,
                "secondary_fixed_runs_edges": 0,
            }
        replicate_edges = [
            fuzz_once(binary_dir, fuzz_seconds=_FUZZ_SECONDS, seed=s) or 0
            for s in _FUZZ_SEED_BUNDLE
        ]
        secondary = fuzz_once(binary_dir, fuzz_seconds=_FUZZ_SECONDS * 4, seed=_SECONDARY_RUNS)
        return {
            "compiled": True,
            "compile_error": None,
            "replicate_edges": replicate_edges,
            "mean_edges": sum(replicate_edges) / len(replicate_edges),
            "min_edges": min(replicate_edges),
            "max_edges": max(replicate_edges),
            "secondary_fixed_runs_edges": secondary or 0,
        }


def _run_trial(example: str, seed: int, *, gpu: int | None) -> dict[str, object] | None:
    payload = _generate_body(example, seed, gpu=gpu)
    recorder = replay(payload)
    prompt_length = int(payload["prompt_length"])
    mask_token_id = int(payload["mask_token_id"])
    generated_token_ids: list[int] = payload["generated_token_ids"]
    full_text = "".join(payload["generated_token_pieces"])
    reference_body, _ = extract_body(full_text)
    regions = statement_regions(reference_body)

    if len(regions) < 1:
        print(f"{example} seed={seed}: no regions detected in body, skipped")
        return None

    # Per-token char offsets within the body text, needed by both region selection
    # and region_token_positions below.
    pieces: list[str] = payload["generated_token_pieces"]
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    confidence_region = select_lowest_confidence_region(
        regions, offsets, recorder.traces, position_offset=prompt_length
    )
    random_region = select_random_region(regions, Random(seed))  # noqa: S311
    if confidence_region is None:
        print(f"{example} seed={seed}: no scoreable region, skipped")
        return None

    arm_bodies = {"reference": reference_body}
    arm_costs: dict[str, dict[str, int]] = {}
    for arm, region in (("confidence", confidence_region), ("random", random_region)):
        positions = region_token_positions(region, offsets, position_offset=0)
        start_position = min(positions) if positions else 0
        body_text, tokens_remasked, steps = _remask_arm(
            example,
            seed,
            gpu=gpu,
            generated_token_ids=generated_token_ids,
            mask_token_id=mask_token_id,
            start_position=start_position,
        )
        arm_bodies[arm] = body_text
        arm_costs[arm] = {"tokens_remasked": tokens_remasked, "steps": steps}

    return {
        "example": example,
        "seed": seed,
        "confidence_region_line": confidence_region.line_number,
        "random_region_line": random_region.line_number,
        "same_region_picked": confidence_region is random_region,
        "region_count": len(regions),
        "costs": arm_costs,
        "arm_bodies": arm_bodies,
        "arm_harnesses": {arm: assemble(example, body) for arm, body in arm_bodies.items()},
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
    run_id = f"rq3-v3-pilot-{args.example}-gpu{args.gpu}-{int(time.time())}"
    prompt_for_hash = body_prompt(args.example)

    results: list[dict[str, object]] = []
    for iteration in range(args.count):
        seed = args.seed_start + iteration
        trial = _run_trial(args.example, seed, gpu=args.gpu)
        if trial is None:
            continue
        for arm, harness in trial["arm_harnesses"].items():  # type: ignore[union-attr]
            measurement = _measure_arm(harness)
            trial[f"arm_{arm}"] = measurement
            append(
                ledger_path,
                HarnessSynthesisLedgerRecord(
                    run_id=run_id,
                    iteration=iteration,
                    arm=f"{args.example}-{arm}",
                    seed=seed,
                    harness_source_sha256=sha256_bytes(harness.encode()),
                    knowledge_base_sha256=sha256_bytes(prompt_for_hash.encode()),
                    usage_example_names=(args.example,),
                    compiled=measurement["compiled"],  # type: ignore[arg-type]
                    compile_error=measurement["compile_error"],  # type: ignore[arg-type]
                    coverage_edges=int(measurement["mean_edges"]),  # type: ignore[arg-type]
                    recorded_at_unix_seconds=time.time(),
                ),
            )
            print(
                f"{args.example:18} seed={seed} arm={arm:10} "
                f"compiled={measurement['compiled']!s:5} "
                f"replicates={measurement['replicate_edges']} "
                f"secondary_fixed_runs={measurement['secondary_fixed_runs_edges']}"
            )
        results.append(trial)
        Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} pilot trials completed for {args.example}.")


if __name__ == "__main__":
    main()
