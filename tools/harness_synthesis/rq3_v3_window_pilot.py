# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ3-v3 window pilot -- cost-equalized confidence vs random remask on the fixed scaffold.

FROZEN protocol (written after auditing the region-based v3 pilot's confound, before
running any trial; not changed after seeing outcomes):

The region-based v3 pilot (``rq3_v3_pilot.py``) found that whenever confidence and
random picked different statement regions, confidence's region *always* required a
smaller remask span (cheaper regeneration) than random's -- because
``remask_from_position_to_end`` masks everything from the chosen position through the
sequence's end, a region nearer the end is structurally cheaper regardless of why it
was chosen. That pilot's apparent compile-rate edge for confidence cannot be
attributed to *where* it points without also attributing it to *how much less* it had
to fix -- the two effects are inseparable in that design. This version removes the
entanglement: both arms choose among the same set of FIXED-LENGTH, non-overlapping
token windows (``remask.TokenWindow`` / ``fixed_length_windows``) and remask only the
chosen window (``remask_window``) -- every candidate costs exactly the same number of
tokens to regenerate no matter which one either arm picks, and no other span (nor the
fixed scaffold, never part of the generated tokens at all) is touched.

Window length: 8 tokens, fixed before running. Rationale (from the already-reported,
unmodified region-based pilot's own recorded costs): confidence's chosen (typically
single-statement) regions spanned roughly 28-38 raw-generation tokens including
trailing content, and a single short C++ statement is roughly 8-15 tokens -- 8 is
smaller than a typical single statement, so most bodies long enough to have a real
statement region at all should still partition into >=2 windows, without being chosen
by looking at any outcome from this new design.

Candidate set: ``fixed_length_windows(body_token_count, 8)`` computed from the SAME
generated body every prior version used, where ``body_token_count`` is the number of
raw generated tokens whose character span falls inside the extracted body (never the
raw generation's tail past the body's own end, and never the fixed scaffold, which is
prefill text and not part of ``generated_token_ids`` at all). Confidence picks the
window with the lowest mean commit confidence (``select_lowest_confidence_window``);
random picks uniformly among the identical candidate set with the trial's own seed
(``select_random_window``) -- both from the same initial body and the same
regeneration seed, matching every prior arm-matching convention in this plan.

Regeneration cost is now fixed by construction: ``tokens_remasked = 8`` for every arm
of every trial (equal, not merely logged), and ``steps = 8`` for every regen call.

Feasibility / stop criterion (frozen, checked in this order, not adjusted after
seeing results): a trial whose body has fewer than 2 windows at length 8 is skipped
(recorded as insufficient, no arms run) -- there is nothing to choose between. If more
than half of the attempted trials are skipped for this reason, stop after the batch
and report infeasible; do not shrink the window length or otherwise retune to force a
pass.

Verification (frozen, checked automatically on every trial, not merely assumed): after
each arm's regeneration, every position outside its chosen window must be byte-for-byte
identical to the pre-regen generated-token-id array at that same position; the
assembled harness's fixed prefix/suffix must be byte-for-byte the module's own
``scaffold_prefix``/``scaffold_suffix`` output. Both are asserted, not merely claimed.

Primary outcome (unchanged from every prior RQ3 stage): fixed-fuzz-time edge coverage,
intent-to-treat (a non-compiling arm counts as 0 edges, not excluded), each arm fuzzed
with the same libFuzzer seed bundle (1, 2, 3) at fuzz_seconds=20, reporting mean/min/max.
One supplementary fixed-execution-count run (``-runs=100000``) per compiling arm, never
the primary comparison.

Dataset: 6 seeds per example (InputFile, MultiPartInputFile) = 12 trials, seeds
11,000,000-11,000,011 (a new block, reserved 11,000,000-11,000,999 from any future RQ3
confirmatory run -- does not overlap the region-based v3 pilot's 10,000,000 block nor
the v3 feasibility check's 9,000,000 block).

Preserves, unchanged: the region-based v3 feasibility check and pilot (results and
this file's sibling scripts), and every earlier RQ1/RQ3/B0/B0v2 result. This is a
separate experiment version, not a revision of any of those.

NOT EXECUTED as part of writing this file. Reviewed first, same as every prior stage.

    uv run python tools/harness_synthesis/rq3_v3_window_pilot.py \
      --gpu 0 --example InputFile --seed-start 11000000 --count 6 \
      --out rq3-v3-window-pilot-inputfile.json \
      --ledger /data/dllm-agent/harness-synthesis/rq3-v3-window-pilot-ledger.jsonl
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
from kb_v3_fixed_scaffold import (
    assemble,
    body_prompt,
    extract_body,
    scaffold_prefix,
    scaffold_suffix,
)

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.remask import (
    TokenWindow,
    fixed_length_windows,
    remask_window,
    select_lowest_confidence_window,
    select_random_window,
)

_NEW_TOKENS = 60
_STEPS = 30
_WINDOW_LENGTH = 8
_REGEN_STEPS = 8
_FUZZ_SECONDS = 20.0
_FUZZ_SEED_BUNDLE = (1, 2, 3)
_SECONDARY_RUNS = 100_000
_MIN_WINDOWS = 2


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


def _body_token_count(body_text: str, offsets: list[tuple[int, int]]) -> int:
    """How many raw generated tokens fall inside the extracted body's char span."""
    body_len = len(body_text)
    return sum(1 for start, _end in offsets if start < body_len)


def _regen_window(  # noqa: PLR0913 - one remask-round step, not worth splitting.
    example: str,
    seed: int,
    *,
    gpu: int | None,
    generated_token_ids: list[int],
    mask_token_id: int,
    window: TokenWindow,
) -> tuple[list[int], str]:
    """Remask exactly ``window`` and regenerate. Returns ``(new_token_ids, body_text)``."""
    remasked = remask_window(generated_token_ids, window, mask_token_id=mask_token_id)
    with tempfile.TemporaryDirectory() as tmp:
        cont_path = Path(tmp) / "continuation.json"
        cont_path.write_text(json.dumps(list(remasked)))
        payload = run_dream_generate(
            body_prompt(example),
            seed,
            Path(tmp) / "regen.json",
            gpu=gpu,
            steps=_REGEN_STEPS,
            alg_temp=0.5,
            prefill_suffix=scaffold_prefix(example),
            init_continuation_file=str(cont_path),
        )
    new_token_ids: list[int] = payload["generated_token_ids"]
    full_text = "".join(payload["generated_token_pieces"])
    body_text, _ = extract_body(full_text)
    return new_token_ids, body_text


def _outside_window_preserved(
    before: list[int], after: list[int], window: TokenWindow
) -> bool:
    """Verify every position outside ``window`` kept its pre-regen token id."""
    if len(before) != len(after):
        return False
    return all(
        before[index] == after[index]
        for index in range(len(before))
        if not (window.start <= index < window.end)
    )


def _scaffold_preserved(example: str, harness: str) -> bool:
    """Verify the assembled harness's fixed regions are byte-for-byte the module's own."""
    prefix = scaffold_prefix(example)
    suffix = scaffold_suffix()
    return harness.startswith(prefix) and harness.endswith(suffix)


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
    mask_token_id = int(payload["mask_token_id"])
    generated_token_ids: list[int] = payload["generated_token_ids"]
    full_text = "".join(payload["generated_token_pieces"])
    reference_body, _ = extract_body(full_text)

    pieces: list[str] = payload["generated_token_pieces"]
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    body_token_count = _body_token_count(reference_body, offsets)

    local_windows = fixed_length_windows(body_token_count, _WINDOW_LENGTH)
    if len(local_windows) < _MIN_WINDOWS:
        print(
            f"{example} seed={seed}: only {len(local_windows)} window(s) of length "
            f"{_WINDOW_LENGTH} in a {body_token_count}-token body, skipped"
        )
        return None

    prompt_length = int(payload["prompt_length"])
    global_windows = [
        TokenWindow(start=window.start + prompt_length, length=window.length)
        for window in local_windows
    ]
    confidence_global = select_lowest_confidence_window(global_windows, recorder.traces)
    if confidence_global is None:
        print(f"{example} seed={seed}: no scoreable window, skipped")
        return None
    confidence_window = local_windows[global_windows.index(confidence_global)]
    random_window = select_random_window(local_windows, Random(seed))  # noqa: S311

    arm_bodies = {"reference": reference_body}
    arm_harnesses = {"reference": assemble(example, reference_body)}
    arm_windows = {"confidence": confidence_window, "random": random_window}
    arm_preserved: dict[str, bool] = {}
    for arm, window in arm_windows.items():
        new_token_ids, body_text = _regen_window(
            example,
            seed,
            gpu=gpu,
            generated_token_ids=generated_token_ids,
            mask_token_id=mask_token_id,
            window=window,
        )
        arm_bodies[arm] = body_text
        arm_harnesses[arm] = assemble(example, body_text)
        arm_preserved[arm] = _outside_window_preserved(
            generated_token_ids, new_token_ids, window
        ) and _scaffold_preserved(example, arm_harnesses[arm])

    return {
        "example": example,
        "seed": seed,
        "body_token_count": body_token_count,
        "window_count": len(local_windows),
        "confidence_window": {"start": confidence_window.start, "length": confidence_window.length},
        "random_window": {"start": random_window.start, "length": random_window.length},
        "same_window_picked": confidence_window == random_window,
        "outside_window_preserved": arm_preserved,
        "arm_bodies": arm_bodies,
        "arm_harnesses": arm_harnesses,
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
    run_id = f"rq3-v3-window-pilot-{args.example}-gpu{args.gpu}-{int(time.time())}"
    prompt_for_hash = body_prompt(args.example)

    results: list[dict[str, object]] = []
    skipped = 0
    for iteration in range(args.count):
        seed = args.seed_start + iteration
        trial = _run_trial(args.example, seed, gpu=args.gpu)
        if trial is None:
            skipped += 1
            continue
        for arm, harness in trial["arm_harnesses"].items():  # type: ignore[union-attr]
            measurement = _measure_arm(harness)
            trial[f"arm_{arm}"] = measurement
            append(
                ledger_path,
                HarnessSynthesisLedgerRecord(
                    run_id=run_id,
                    iteration=iteration,
                    arm=f"{args.example}-window-{arm}",
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
                f"preserved={trial['outside_window_preserved'].get(arm)}"  # type: ignore[union-attr]
            )
        results.append(trial)
        Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\n{len(results)} window-pilot trials completed, {skipped} skipped for {args.example}.")
    if skipped > args.count / 2:
        print(
            "STOP: more than half of trials skipped for insufficient windows -- "
            "infeasible at this window length."
        )


if __name__ == "__main__":
    main()
