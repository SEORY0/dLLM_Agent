# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ4 pilot -- compiler-localized, bidirectional span infill on the RQ3-v3 fixed scaffold.

FROZEN protocol (written before running any trial; not changed after seeing outcomes).
Design rationale: ``docs/superpowers/specs/2026-10-13-compiler-localized-span-infill.md``.

Question. RQ1/RQ3 asked the model *where* to repair (its own confidence) and found no
edge over random. RQ4 takes the location from the compiler instead and asks only what
a diffusion model is structurally positioned to do: regenerate *just that span*
conditioned on the code on both sides of it. Two separable effects:

- H-loc (does the compiler's location matter?): ``located-infill`` vs ``random-infill``.
- H-bi (does right-hand context matter, given the location?): ``located-infill`` vs
  ``located-suffix``. Primary analysis only in the stratum where the located line has
  model-written code after it (``has_right_context``); offline step 0 found 14/27 such
  failures. Where it does not, the two arms differ only in slack and are reported
  separately, never pooled into H-bi.
- Anchor (is any of this better than trying again?): ``located-infill`` vs ``resample``.

Population. Same scaffold, prompt and generation settings as the RQ3-v3 window pilot
(``kb_v3_fixed_scaffold``, ``new_tokens=60, steps=30, alg_temp=0.5``). Each attempt
generates one reference body and compiles it. Eligible = it does NOT compile AND
clang's first ``harness.cc`` error maps onto a non-blank body line
(``localize.locate_in_body``). Compiling or non-localizable attempts are recorded and
get no arms. Generation continues until 24 eligible trials or 96 attempts, whichever
comes first.

Arms (all from the same eligible body, all with the trial's own regeneration seed):

1. ``located-infill``: mask the token window covering the located line
   (``localize.window_covering``) and replace it with ``M`` masks
   (``remask.infill_window``); everything before and after is kept as context.
2. ``random-infill``: identical, but on a line drawn uniformly (``Random(seed)``) from
   the body's OTHER non-blank lines. Excluding the located line is deliberate: a draw
   that lands on it makes the two arms identical and carries no information about
   H-loc. Skipped (recorded) when the body has one non-blank line.
3. ``located-suffix``: mask from the located window's start through the end of the
   continuation (``remask.remask_from_position_to_end``) -- the repair a left-to-right
   decoder can make from the same location. Costs more tokens by construction; the
   count is recorded, not equalized (equalizing would mean truncating its context).
4. ``resample``: a fresh body from scratch, seed ``seed + 500_000``, same budget.

Slack/cost. ``M = max(located window, random window) + 8`` tokens, identical for both
infill arms, ``steps = M``. Slack positions the model fills with special tokens (any
piece of the form ``<|...|>``) inside the infilled span are dropped before decoding.
Frozen at 8 before running: the largest single-line repair seen in the offline audit is
a renamed member call, a few tokens longer than the wrong one.

Verification (asserted on every infill arm, not assumed): every token before the window
and every token after the masked span is identical to the pre-regen token at its
(shifted) position; the assembled harness starts with ``scaffold_prefix`` and ends with
``scaffold_suffix``.

Outcomes. Primary: compiles (binary), per arm. Secondary: whether the first error left
the located line (for arms that still fail); intent-to-treat edge coverage with the same
libFuzzer seed bundle (1, 2, 3) at 20 s (0 when not compiling); tokens masked; steps.

Analysis. Exact two-sided sign test on discordant pairs for H-loc, H-bi (right-context
stratum) and the anchor, each separately, alpha 0.05, no multiplicity claim -- this is a
pilot. Its job is the discordance rate that a later confirmatory N is computed from
(``power_calc.required_n_from_discordant_pairs``); it is not itself confirmatory.

Stop gates (checked in order after the run, not tuned afterwards):
(1) more than 25% of non-compiling references not localizable to the body -> the
    offline 27/27 did not transfer; stop and report.
(2) fewer than 12 eligible trials in 96 attempts -> infeasible at this compile rate.

Dataset: seeds 12,000,000-12,000,095 (InputFile), 12,000,100-12,000,195
(MultiPartInputFile); resample arm uses seed + 500,000. 12,000,000-12,999,999 reserved
from any future confirmatory run.

Does NOT claim anything about dLLM vs AR: ``located-suffix`` is the same Dream model
decoding left-to-right, not an AR model. A fair AR comparison for H-bi needs an AR model
with fill-in-the-middle (spec §5).

NOT EXECUTED as part of writing this file. Reviewed first, same as every prior stage.

    uv run python tools/harness_synthesis/rq4_localized_infill_pilot.py \
      --gpu 0 --example InputFile --seed-start 12000000 --max-attempts 96 --target 24 \
      --out rq4-pilot-inputfile.json \
      --ledger /data/dllm-agent/harness-synthesis/rq4-pilot-ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from random import Random
from typing import Any

from _compile import compile_for_repeated_fuzzing, fuzz_once
from _replay import run_dream_generate
from kb_v3_fixed_scaffold import (
    assemble,
    body_prompt,
    extract_body,
    scaffold_prefix,
    scaffold_suffix,
)

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.localize import (
    body_line_regions,
    locate_in_body,
    parse_compiler_diagnostics,
    window_covering,
)
from dllm_agent.harness_synthesis.remask import (
    TokenWindow,
    infill_window,
    remask_from_position_to_end,
)

_NEW_TOKENS = 60
_STEPS = 30
_SLACK = 8
_FUZZ_SECONDS = 20.0
_FUZZ_SEED_BUNDLE = (1, 2, 3)
_RESAMPLE_SEED_OFFSET = 500_000


def _generate(example: str, seed: int, *, gpu: int | None, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    with tempfile.TemporaryDirectory() as tmp:
        return run_dream_generate(
            body_prompt(example),
            seed,
            Path(tmp) / "generate.json",
            gpu=gpu,
            alg_temp=0.5,
            prefill_suffix=scaffold_prefix(example),
            **kwargs,
        )


def _regen(example: str, seed: int, continuation: list[int], *, gpu: int | None, steps: int):  # noqa: ANN202
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "continuation.json"
        path.write_text(json.dumps(continuation))
        return _generate(example, seed, gpu=gpu, steps=steps, init_continuation_file=str(path))


def _offsets(pieces: list[str]) -> list[tuple[int, int]]:
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return offsets


def _is_special(piece: str) -> bool:
    return piece.startswith("<|") and piece.endswith("|>")


def _decode_dropping_span_specials(pieces: list[str], start: int, end: int) -> str:
    """Join pieces, dropping special-token pieces only inside the infilled span."""
    return "".join(
        piece
        for index, piece in enumerate(pieces)
        if not (start <= index < end and _is_special(piece))
    )


def _first_error(harness: str) -> tuple[bool, str | None]:
    with compile_for_repeated_fuzzing(harness) as (binary_dir, error):
        return binary_dir is not None, error


def _measure(harness: str) -> dict[str, object]:
    with compile_for_repeated_fuzzing(harness) as (binary_dir, error):
        if binary_dir is None:
            return {"compiled": False, "compile_error": error, "replicate_edges": [0, 0, 0]}
        edges = [
            fuzz_once(binary_dir, fuzz_seconds=_FUZZ_SECONDS, seed=s) or 0
            for s in _FUZZ_SEED_BUNDLE
        ]
        return {"compiled": True, "compile_error": None, "replicate_edges": edges}


def _infill_arm(  # noqa: PLR0913 - one arm of one trial.
    example: str,
    seed: int,
    *,
    gpu: int | None,
    token_ids: list[int],
    mask_token_id: int,
    window: TokenWindow,
    mask_length: int,
) -> dict[str, object]:
    continuation = list(
        infill_window(token_ids, window, mask_length=mask_length, mask_token_id=mask_token_id)
    )
    payload = _regen(example, seed, continuation, gpu=gpu, steps=mask_length)
    new_ids: list[int] = payload["generated_token_ids"]
    span_end = window.start + mask_length
    preserved = (
        len(new_ids) == len(continuation)
        and new_ids[: window.start] == token_ids[: window.start]
        and new_ids[span_end:] == token_ids[window.end :]
    )
    body, _ = extract_body(
        _decode_dropping_span_specials(payload["generated_token_pieces"], window.start, span_end)
    )
    harness = assemble(example, body)
    preserved = (
        preserved
        and harness.startswith(scaffold_prefix(example))
        and harness.endswith(scaffold_suffix())
    )
    return {
        "body": body,
        "harness": harness,
        "tokens_masked": mask_length,
        "steps": mask_length,
        "preserved": preserved,
    }


def _run_trial(example: str, seed: int, *, gpu: int | None) -> dict[str, object]:
    payload = _generate(example, seed, gpu=gpu, new_tokens=_NEW_TOKENS, steps=_STEPS)
    token_ids: list[int] = payload["generated_token_ids"]
    mask_token_id = int(payload["mask_token_id"])
    pieces: list[str] = payload["generated_token_pieces"]
    body, _ = extract_body("".join(pieces))
    reference = assemble(example, body)
    trial: dict[str, object] = {"example": example, "seed": seed, "reference_body": body}

    compiled, error = _first_error(reference)
    trial["reference_compiled"] = compiled
    trial["reference_compile_error"] = error
    if compiled:
        return {**trial, "eligible": False, "ineligible_reason": "reference-compiles"}
    diagnostics = parse_compiler_diagnostics(error or "")
    located = (
        locate_in_body(diagnostics[0].line, prefix=scaffold_prefix(example), body=body)
        if diagnostics
        else None
    )
    offsets = _offsets(pieces)
    located_window = window_covering(located, offsets) if located is not None else None
    if located is None or located_window is None:
        return {**trial, "eligible": False, "ineligible_reason": "not-localizable-to-body"}

    lines = body_line_regions(body)
    others = [region for region in lines if region.line_number != located.line_number]
    random_region = Random(seed).choice(others) if others else None  # noqa: S311
    random_window = window_covering(random_region, offsets) if random_region is not None else None
    mask_length = max(located_window.length, random_window.length if random_window else 0) + _SLACK

    arms: dict[str, dict[str, object]] = {}
    arms["located-infill"] = _infill_arm(
        example,
        seed,
        gpu=gpu,
        token_ids=token_ids,
        mask_token_id=mask_token_id,
        window=located_window,
        mask_length=mask_length,
    )
    if random_window is not None:
        arms["random-infill"] = _infill_arm(
            example,
            seed,
            gpu=gpu,
            token_ids=token_ids,
            mask_token_id=mask_token_id,
            window=random_window,
            mask_length=mask_length,
        )
    suffix = list(
        remask_from_position_to_end(token_ids, located_window.start, mask_token_id=mask_token_id)
    )
    suffix_masked = len(suffix) - located_window.start
    suffix_payload = _regen(example, seed, suffix, gpu=gpu, steps=suffix_masked)
    suffix_body, _ = extract_body("".join(suffix_payload["generated_token_pieces"]))
    arms["located-suffix"] = {
        "body": suffix_body,
        "harness": assemble(example, suffix_body),
        "tokens_masked": suffix_masked,
        "steps": suffix_masked,
    }
    resample_payload = _generate(
        example, seed + _RESAMPLE_SEED_OFFSET, gpu=gpu, new_tokens=_NEW_TOKENS, steps=_STEPS
    )
    resample_body, _ = extract_body("".join(resample_payload["generated_token_pieces"]))
    arms["resample"] = {
        "body": resample_body,
        "harness": assemble(example, resample_body),
        "tokens_masked": _NEW_TOKENS,
        "steps": _STEPS,
    }

    for arm in arms.values():
        measurement = _measure(str(arm["harness"]))
        arm.update(measurement)
        if not measurement["compiled"]:
            still = parse_compiler_diagnostics(str(measurement["compile_error"] or ""))
            arm["first_error_left_located_line"] = not still or still[0].line != diagnostics[0].line
    return {
        **trial,
        "eligible": True,
        "located_body_line": located.line_number,
        "located_message": diagnostics[0].message,
        "has_right_context": located.line_number < max(r.line_number for r in lines),
        "random_body_line": random_region.line_number if random_region else None,
        "mask_length": mask_length,
        "located_window": {"start": located_window.start, "length": located_window.length},
        "arms": arms,
    }


def main() -> None:
    """NOT to be run until the frozen protocol above has been reviewed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--example", required=True, choices=("InputFile", "MultiPartInputFile"))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=96)
    parser.add_argument("--target", type=int, default=24)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    run_id = f"rq4-pilot-{args.example}-gpu{args.gpu}-{int(time.time())}"
    kb_sha = sha256_bytes(body_prompt(args.example).encode())
    attempts: list[dict[str, object]] = []
    eligible = 0
    for iteration in range(args.max_attempts):
        if eligible >= args.target:
            break
        seed = args.seed_start + iteration
        trial = _run_trial(args.example, seed, gpu=args.gpu)
        attempts.append(trial)
        arms: dict[str, dict[str, object]] = trial.get("arms", {})  # type: ignore[assignment]
        records = {
            "reference": {
                "harness": assemble(args.example, str(trial["reference_body"])),
                "compiled": trial["reference_compiled"],
                "compile_error": trial["reference_compile_error"],
                "replicate_edges": [0, 0, 0],
            }
        }
        records.update(arms)
        for arm, data in records.items():
            edges = data.get("replicate_edges") or [0]
            append(
                ledger_path,
                HarnessSynthesisLedgerRecord(
                    run_id=run_id,
                    iteration=iteration,
                    arm=f"{args.example}-rq4-{arm}",
                    seed=seed,
                    harness_source_sha256=sha256_bytes(str(data["harness"]).encode()),
                    knowledge_base_sha256=kb_sha,
                    usage_example_names=(args.example,),
                    compiled=bool(data["compiled"]),
                    compile_error=data["compile_error"],  # type: ignore[arg-type]
                    coverage_edges=int(sum(edges) / len(edges)),  # type: ignore[arg-type]
                    recorded_at_unix_seconds=time.time(),
                ),
            )
        if trial["eligible"]:
            eligible += 1
            summary = {arm: data["compiled"] for arm, data in arms.items()}
            print(
                f"{args.example} seed={seed} line={trial['located_body_line']} "
                f"right_ctx={trial['has_right_context']} {summary}"
            )
        else:
            print(f"{args.example} seed={seed} ineligible: {trial['ineligible_reason']}")
        Path(args.out).write_text(json.dumps(attempts, indent=1))

    failures = [t for t in attempts if not t["reference_compiled"]]
    unlocalizable = sum(
        1 for t in failures if t.get("ineligible_reason") == "not-localizable-to-body"
    )
    print(
        f"\n{len(attempts)} attempts, {eligible} eligible, {unlocalizable}/{len(failures)} "
        "non-compiling references not localizable to the body."
    )
    if failures and unlocalizable / len(failures) > 0.25:  # noqa: PLR2004 - frozen gate (1).
        print("STOP gate (1): localization did not transfer from the offline audit.")
    elif eligible < 12:  # noqa: PLR2004 - frozen gate (2).
        print("STOP gate (2): fewer than 12 eligible trials -- infeasible at this compile rate.")


if __name__ == "__main__":
    main()
