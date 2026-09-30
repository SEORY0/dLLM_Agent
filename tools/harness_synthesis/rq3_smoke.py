# ruff: noqa: T201 -- deliberate console report, not a library module
r"""Task 10 -- RQ3 smoke test: confidence-guided remask vs random remask vs reference.

Protocol frozen BEFORE looking at smoke outcomes (this docstring is the frozen
protocol; do not edit it after seeing results):

1. Directed-KB baseline (Task 8b): one initial harness generated per trial. That SAME
   harness branches into a confidence-remask arm and a random-remask arm; the
   unmodified initial harness is recorded as the reference arm.
2. Arms are matched on eligible regions (both draw from the identical detected region
   set of the SAME initial harness) and on regeneration seed (both remask rounds use
   the same --seed, so the only engineered difference is which region gets remasked).
   ``remask_from_position_to_end`` remasks from a start position to the sequence's
   end, so an EARLIER start position costs strictly more regeneration work than a
   later one -- this is not forced equal (the policy needs the tail to re-flow), but
   every trial logs the actual remasked-token count and regeneration step count for
   both arms, so a systematic cost imbalance between confidence's and random's picks
   is auditable rather than silently confounding the comparison.
3. Primary outcome, over ALL attempted trials, never excluding a compile failure:
   edge coverage under a FIXED fuzz duration (--fuzz-seconds, same for every arm),
   with edges=0 substituted when an arm did not compile. Compile success is reported
   separately as a diagnostic. Edges conditional on compiling is secondary. Delta vs
   the reference arm's own edges is the headline comparison, stratified by origin
   example.
4. alg_temp=0.5 throughout (alg_temp=0.0 made --seed nearly a no-op, per the RQ1
   investigation); every trial's exact code is sha256-hashed into the audit ledger
   (Task 6), paired by a shared (run_id, iteration) key across its three arms.
5. This script is the SMOKE ONLY: exploratory, excluded from the main study's
   confirmatory analysis. It exists to verify the remask mechanism end-to-end, check
   cost parity between arms, see whether the coverage-delta outcome has enough spread
   to be worth measuring, and exercise fitness.py's inputs -- not to draw a conclusion.
   The main study's sample size is computed from this smoke's own observed paired
   variance (research-report.md §5 principle), and its success threshold and analysis
   plan are fixed before that run, not after.

    uv run python tools/harness_synthesis/rq3_smoke.py \
      --gpu 0 --seed-start 0 --seeds-per-example 2 --fuzz-seconds 20 \
      --out rq3-smoke-gpu0.json --ledger /data/dllm-agent/harness-synthesis/rq3-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from random import Random

from _compile import compile_and_fuzz
from _replay import decode_code_block, replay, run_dream_generate
from kb_diversity_experiment import _EXAMPLES, _prompt

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.fitness import FitnessInputs, FitnessWeights, score
from dllm_agent.harness_synthesis.regions import statement_regions
from dllm_agent.harness_synthesis.remask import (
    region_token_positions,
    remask_from_position_to_end,
    select_lowest_confidence_region,
    select_random_region,
)

_DIRECTED = "directed"
_SMOKE_WEIGHTS = FitnessWeights(compile=1.0, reach=1.0, edge_coverage=1.0)


def _regen_steps(remasked_count: int) -> int:
    """Scale regeneration steps with how many tokens are actually being remasked.

    Matches ``dream_backend_smoke.py``'s prototype convention
    (``steps=max(8, (NEW-keep)//2)``): a smaller remask needs fewer denoising steps.
    """
    return max(8, remasked_count // 2)


def _remask_arm(
    *,
    gpu: int | None,
    seed: int,
    generated_token_ids: list[int],
    mask_token_id: int,
    start_position: int,
) -> tuple[str, int, int]:
    """Remask from ``start_position`` to the end and regenerate.

    Returns ``(code, tokens_remasked, steps)``.
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
            # --prompt is unused by _dream_generate.py once --init-continuation-file is
            # given (it only needs the prompt to rebuild prompt_ids, which the caller
            # already encoded into the continuation's positions -- see run_dream_generate
            # below, which still sends --prompt for that prompt_ids reconstruction).
            _CURRENT_PROMPT,
            seed,
            Path(tmp) / "regen.json",
            gpu=gpu,
            steps=steps,
            alg_temp=0.5,
            init_continuation_file=str(cont_path),
        )
    code, _ = decode_code_block(payload)
    return code, tokens_remasked, steps


_CURRENT_PROMPT = ""  # set per trial in main(); module-level so _remask_arm can read it.
_MIN_REGIONS_FOR_A_TRIAL = 2


def _evaluate_arms(  # noqa: PLR0913 - one per-trial evaluation step, not worth splitting.
    arm_codes: dict[str, str],
    *,
    example_name: str,
    seed: int,
    fuzz_seconds: float,
    run_id: str,
    ledger_path: Path,
    iteration: int,
) -> tuple[dict[str, dict[str, object]], int]:
    """Compile+fuzz each arm, record it to the ledger, and add the reference delta."""
    arms: dict[str, dict[str, object]] = {}
    for arm, code in arm_codes.items():
        fuzz = compile_and_fuzz(code, fuzz_seconds=fuzz_seconds, seed=seed)
        edges = fuzz.edges if fuzz.edges is not None else 0
        fitness = score(
            _SMOKE_WEIGHTS,
            FitnessInputs(
                compiled=float(fuzz.compiled), reached_target=float(fuzz.compiled),
                edge_coverage=float(edges),
            ),
        )
        append(
            ledger_path,
            HarnessSynthesisLedgerRecord(
                run_id=run_id,
                iteration=iteration,
                arm=f"{example_name}-{arm}",
                seed=seed,
                harness_source_sha256=sha256_bytes(code.encode()),
                knowledge_base_sha256=sha256_bytes(_CURRENT_PROMPT.encode()),
                usage_example_names=(example_name,),
                compiled=fuzz.compiled,
                compile_error=fuzz.compile_error,
                coverage_edges=edges,
                recorded_at_unix_seconds=time.time(),
            ),
        )
        arms[arm] = {
            "compiled": fuzz.compiled,
            "compile_error": fuzz.compile_error,
            "edges": edges,
            "fitness": fitness,
            "code": code,
        }
        iteration += 1
        print(
            f"{example_name:18} seed={seed} arm={arm:10} "
            f"compiled={fuzz.compiled!s:5} edges={edges}"
        )
    reference_edges = arms["reference"]["edges"]
    for arm in ("confidence", "random"):
        arms[arm]["delta_vs_reference"] = arms[arm]["edges"] - reference_edges  # type: ignore[operator]
    return arms, iteration


def main() -> None:
    """Run the frozen smoke protocol and report per-trial + aggregate results."""
    global _CURRENT_PROMPT  # noqa: PLW0603 - see _CURRENT_PROMPT's own comment.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds-per-example", type=int, default=2)
    parser.add_argument("--fuzz-seconds", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    run_id = f"rq3-smoke-gpu{args.gpu}-{int(time.time())}"
    results: list[dict[str, object]] = []
    iteration = 0

    for example in _EXAMPLES:
        _CURRENT_PROMPT = _prompt(example, _DIRECTED)
        for offset in range(args.seeds_per_example):
            seed = 5_000_000 + args.seed_start + offset
            with tempfile.TemporaryDirectory() as tmp:
                payload = run_dream_generate(
                    _CURRENT_PROMPT,
                    seed,
                    Path(tmp) / "initial.json",
                    gpu=args.gpu,
                    new_tokens=768,
                    steps=384,
                    alg_temp=0.5,
                )
            recorder = replay(payload)
            prompt_length = int(payload["prompt_length"])
            mask_token_id = int(payload["mask_token_id"])
            generated_token_ids: list[int] = payload["generated_token_ids"]
            reference_code, code_offsets = decode_code_block(payload)
            regions = statement_regions(reference_code)

            if len(regions) < _MIN_REGIONS_FOR_A_TRIAL:
                print(f"{example.name} seed={seed}: fewer than 2 regions, skipped")
                continue

            confidence_region = select_lowest_confidence_region(
                regions, code_offsets, recorder.traces, position_offset=prompt_length
            )
            random_region = select_random_region(regions, Random(seed))  # noqa: S311
            if confidence_region is None:
                print(f"{example.name} seed={seed}: no scoreable region, skipped")
                continue

            arm_codes: dict[str, str] = {"reference": reference_code}
            arm_costs: dict[str, dict[str, int]] = {}
            for arm, region in (("confidence", confidence_region), ("random", random_region)):
                # Continuation-relative positions (no prompt offset): these index
                # directly into generated_token_ids, unlike the position_offset=
                # prompt_length space the selection above used to match TokenTrace.
                positions = region_token_positions(region, code_offsets, position_offset=0)
                start_position = min(positions) if positions else 0
                code, tokens_remasked, steps = _remask_arm(
                    gpu=args.gpu,
                    seed=seed,
                    generated_token_ids=generated_token_ids,
                    mask_token_id=mask_token_id,
                    start_position=start_position,
                )
                arm_codes[arm] = code
                arm_costs[arm] = {"tokens_remasked": tokens_remasked, "steps": steps}

            arms, iteration = _evaluate_arms(
                arm_codes,
                example_name=example.name,
                seed=seed,
                fuzz_seconds=args.fuzz_seconds,
                run_id=run_id,
                ledger_path=ledger_path,
                iteration=iteration,
            )
            results.append(
                {
                    "example": example.name,
                    "seed": seed,
                    "confidence_region_line": confidence_region.line_number,
                    "random_region_line": random_region.line_number,
                    "same_region_picked": confidence_region is random_region,
                    "costs": arm_costs,
                    "arms": arms,
                }
            )
            Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} trials completed.")


if __name__ == "__main__":
    main()
