# ruff: noqa: T201 -- deliberate console report, not a library module
r"""RQ4b pilot P1: compiler-localized infill vs suffix vs resample on controlled single defects.

FROZEN protocol, written before any GPU trial. Hashes of this file and of everything it
depends on are recorded in ``docs/assets/harness-rq4b-p1-freeze-20261026.json``
(``rq4b_freeze.py``). Every GPU run (P1 or ``--smoke``) re-checks them first and
refuses to start on any difference. Rationale, hypotheses, analysis plan and the scope
of any conclusion: ``docs/superpowers/specs/2026-10-26-rq4b-functional-repair-benchmark.md``.
NOT EXECUTED as part of writing this file. It runs only after explicit approval.

Population. The 12 samples of ``repair_benchmark.samples()``, validated offline by
``rq4b_validate_benchmark.py``. Each is a hand-written body that passes the functional
evaluator, with exactly one line broken so that every clang error lands on that line,
and executable code after it. The location is the mutated line, so P1 tests repair
given a correct location, not localization.

Arms. K = 5 candidates each per (sample, replicate) unit. Seed slot j is
``seed_for(sample, replicate, j, base=13_000_000)`` = 13,000,000 + 1000*sample +
10*replicate + j, shared across arms (common random numbers).

- ``infill-sweep``: mask the defective line's token window (L tokens) with L + o_k
  masks, o = (0, +2, -2, +4, +8) in that frozen selection order. Length position k uses
  seed slot (k + replicate) % 5, so over the 5 replicates every length meets every slot
  exactly once per sample. Length effects are therefore not confounded with fixed slot
  effects. Left and right context are fixed. Special pieces inside the span are
  dropped, and outside-span tokens are asserted unchanged (G1).
- ``suffix``: keep everything before the window; mask the rest of the continuation plus
  8. Candidate j uses slot j.
- ``resample``: a fresh 96-token body. Candidate j uses slot j.

Decoding is the same everywhere: ``alg_temp=0.5``, temperature 0.5, top_p 0.95, entropy
reveal order, one token per step (steps = masked positions), scaffold prefix as prefill,
and the same prompt ``repair_prompt(example)`` with inspection required. No compiler
message is given (that is P2, not frozen, refused here).

Selection (frozen). Per arm, walk the candidates in frozen order (infill: by mask
length as above; suffix and resample: by slot) and select the first functional success.
pass@5 = any success. Otherwise select the first compiled candidate (that only matters
for secondary metrics). The diagnostic below never enters any selection.

Diagnostic (never compared, never selected). One extra infill candidate per unit masked
to the target's own token length L* (oracle information), on its own seed slot 9.

Outcome. Primary: functional success (``functional_eval``). Secondary, per candidate:
compile, failure reasons, unaffected statements kept, lexical edit distances, executed
chains, masked tokens, steps, forward cost (steps x sequence length), denoising seconds
(model load excluded) and subprocess seconds. There is no fuzzing.

Size. 12 samples x R = 5 replicates (0..4) = 60 units; 16 generations per unit (15
compared + 1 diagnostic) = 960. Seeds 13,000,000-13,011,049 (P1 accepts no other base).

Gates. G1 (runtime abort): any infill candidate's outside-span tokens changed. G2, G3:
see ``rq4b_analyze.py`` and the spec.

Hardware. GPU 1 only. GPU 0 is occupied by another service (vLLM). One Dream process at
a time, about 16 GB. Check ``nvidia-smi`` before starting and leave other processes
alone.

    # smoke test (approval required; not a result)
    uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 1 --smoke \
      --seed-base 19000000 --samples IF-ctor-pointer --replicates 1 \
      --out /data/dllm-agent/harness-synthesis/rq4b-smoke/smoke.json \
      --ledger /data/dllm-agent/harness-synthesis/rq4b-smoke/smoke-ledger.jsonl
    # P1 (approval required)
    uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 1 \
      --out /data/dllm-agent/harness-synthesis/rq4b-p1/p1.json \
      --ledger /data/dllm-agent/harness-synthesis/rq4b-p1/p1-ledger.jsonl

CPU plumbing check (no model, fake generator; never a result):

    uv run python tools/harness_synthesis/rq4b_functional_pilot.py --dry-run oracle \
      --samples IF-ctor-pointer --replicates 1 --out /tmp/x.json --ledger /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _compile import syntax_diagnostics_many
from _freeze import RESERVED_SEED_BLOCKS, verify_freeze
from _functional_eval import evaluate_many
from _replay import run_dream_generate, tokenize
from kb_v3_fixed_scaffold import assemble, extract_body, scaffold_prefix

from dllm_agent.core.io import sha256_bytes
from dllm_agent.harness_synthesis.audit_ledger import HarnessSynthesisLedgerRecord, append
from dllm_agent.harness_synthesis.body_analysis import (
    lexical_edit_distance,
    unaffected_statements_preserved,
)
from dllm_agent.harness_synthesis.localize import (
    body_line_regions,
    parse_compiler_diagnostics,
    window_covering,
)
from dllm_agent.harness_synthesis.repair_arms import (
    ARMS,
    DIAGNOSTIC_ARM,
    PILOT_REPLICATES,
    PILOT_SEED_BASE,
    CandidateOutcome,
    CandidatePlan,
    decode_dropping_span_specials,
    forward_cost,
    infill_continuation,
    outside_span_preserved,
    plan,
    seed_for,
    seed_slot,
    select,
    suffix_continuation,
)
from dllm_agent.harness_synthesis.repair_benchmark import (
    FUNCTIONAL_SPECS,
    BenchmarkSample,
    continuation_text,
    leaked_fragments,
    repair_prompt,
    samples,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dllm_agent.harness_synthesis.remask import TokenWindow

_ALG_TEMP = 0.5
_FAKE_PAD_ID = -1
_FAKE_PAD_PIECE = "<|endoftext|>"


@dataclass(frozen=True, slots=True)
class Prepared:
    """Per-sample inputs shared by every replicate: tokens, window, diagnostic."""

    sample: BenchmarkSample
    ids: list[int]
    pieces: list[str]
    window: TokenWindow
    reference_ids: list[int]
    reference_window: TokenWindow
    mask_token_id: int
    first_diagnostic: str


type Generator = Callable[[Prepared, CandidatePlan, str, list[int] | None], dict[str, Any]]


def _offsets(pieces: list[str]) -> list[tuple[int, int]]:
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return offsets


def _line_window(body: str, line_number: int, pieces: list[str]) -> TokenWindow:
    region = next(r for r in body_line_regions(body) if r.line_number == line_number)
    window = window_covering(region, _offsets(pieces))
    if window is None:
        reason = f"no token window for body line {line_number}"
        raise ValueError(reason)
    return window


def prepare(bench: list[BenchmarkSample]) -> list[Prepared]:
    """Tokenize every sample's mutant/reference continuation; compile mutants for diagnostics."""
    mask_id, tokens = tokenize(
        [continuation_text(s.mutated_body) for s in bench]
        + [continuation_text(s.reference_body) for s in bench]
    )
    diags = syntax_diagnostics_many([assemble(s.example, s.mutated_body) for s in bench])
    prepared = []
    for i, sample in enumerate(bench):
        mut, ref = tokens[i], tokens[len(bench) + i]
        if "".join(mut["pieces"]) != continuation_text(sample.mutated_body):
            reason = f"{sample.sample_id}: token pieces do not reproduce the continuation"
            raise ValueError(reason)
        parsed = parse_compiler_diagnostics("\n".join(diags[i]))
        if not parsed:
            reason = f"{sample.sample_id}: mutant has no diagnostic (benchmark not validated?)"
            raise ValueError(reason)
        line = sample.mutation.line_number
        prepared.append(
            Prepared(
                sample=sample,
                ids=list(mut["ids"]),
                pieces=list(mut["pieces"]),
                window=_line_window(sample.mutated_body, line, mut["pieces"]),
                reference_ids=list(ref["ids"]),
                reference_window=_line_window(sample.reference_body, line, ref["pieces"]),
                mask_token_id=mask_id,
                first_diagnostic=parsed[0].message,
            )
        )
    return prepared


def _initial(prep: Prepared, candidate: CandidatePlan) -> list[int] | None:
    if candidate.arm in {"infill-sweep", DIAGNOSTIC_ARM}:
        return infill_continuation(
            prep.ids, prep.window, mask_length=candidate.masked, mask_token_id=prep.mask_token_id
        )
    if candidate.arm == "suffix":
        return suffix_continuation(prep.ids, prep.window.start, mask_token_id=prep.mask_token_id)
    return None


def dream_generator(gpu: int | None) -> Generator:
    """The real backend: one ``_dream_generate.py`` subprocess per candidate (GPU)."""

    def generate(
        prep: Prepared, candidate: CandidatePlan, prompt: str, initial: list[int] | None
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            init_file = None
            if initial is not None:
                init_file = str(Path(tmp) / "continuation.json")
                Path(init_file).write_text(json.dumps(initial))
            return run_dream_generate(
                prompt,
                candidate.seed,
                Path(tmp) / "generate.json",
                gpu=gpu,
                new_tokens=candidate.masked,
                steps=candidate.steps,
                alg_temp=_ALG_TEMP,
                init_continuation_file=init_file,
                prefill_suffix=scaffold_prefix(prep.sample.example),
            )

    return generate


def fake_generator(mode: str, pieces_of: dict[int, str]) -> Generator:
    """CPU stand-in for plumbing checks. ``oracle`` fills masks with the reference's tokens.

    ``delete`` fills them with end-of-text. Neither is a model, and nothing produced
    through it is ever a result.
    """

    def generate(
        prep: Prepared, candidate: CandidatePlan, _prompt: str, initial: list[int] | None
    ) -> dict[str, Any]:
        if initial is None:
            fill = prep.reference_ids if mode == "oracle" else []
            out = [*fill, *[_FAKE_PAD_ID] * max(0, candidate.masked - len(fill))][
                : candidate.masked
            ]
        else:
            if candidate.arm in {"infill-sweep", DIAGNOSTIC_ARM}:
                rw = prep.reference_window
                fill = prep.reference_ids[rw.start : rw.end] if mode == "oracle" else []
            else:
                fill = prep.reference_ids[prep.reference_window.start :] if mode == "oracle" else []
            masked = [i for i, t in enumerate(initial) if t == prep.mask_token_id]
            out = list(initial)
            for k, position in enumerate(masked):
                out[position] = fill[k] if k < len(fill) else _FAKE_PAD_ID
        return {
            "prompt_length": 0,
            "mask_token_id": prep.mask_token_id,
            "generated_token_ids": out,
            "generated_token_pieces": [
                _FAKE_PAD_PIECE if t == _FAKE_PAD_ID else pieces_of[t] for t in out
            ],
            "generate_seconds": 0.0,
        }

    return generate


def _candidate(
    prep: Prepared, candidate: CandidatePlan, prompt: str, generator: Generator, *, base: int
) -> dict[str, Any]:
    initial = _initial(prep, candidate)
    started = time.perf_counter()
    payload = generator(prep, candidate, prompt, initial)
    wall = time.perf_counter() - started
    if int(payload["mask_token_id"]) != prep.mask_token_id:
        # The continuation was built with the tokenizer's mask id; a mismatch with the
        # model config's would silently turn every "mask" into a literal token.
        reason = (
            f"mask id mismatch: model {payload['mask_token_id']} vs tokenizer {prep.mask_token_id}"
        )
        raise SystemExit(reason)
    ids: list[int] = payload["generated_token_ids"]
    pieces: list[str] = payload["generated_token_pieces"]
    record: dict[str, Any] = {
        **asdict(candidate),
        "seed_slot": seed_slot(candidate.seed, base=base),
    }
    if candidate.arm in {"infill-sweep", DIAGNOSTIC_ARM}:
        start, end = prep.window.start, prep.window.start + candidate.masked
        record["preserved"] = outside_span_preserved(
            prep.ids, ids, prep.window, mask_length=candidate.masked
        )
        text = decode_dropping_span_specials(pieces, start, end)
        record["span_text"] = "".join(p for p in pieces[start:end])
    else:
        text = "".join(pieces)
    body, complete = extract_body(text)
    record.update(
        {
            "body": body,
            "body_complete": complete,
            "harness": assemble(prep.sample.example, body),
            "prompt_length": int(payload["prompt_length"]),
            "continuation_length": len(ids),
            "cost": forward_cost(
                candidate.steps,
                prompt_length=int(payload["prompt_length"]),
                continuation_length=len(ids),
            ),
            "generate_seconds": float(payload.get("generate_seconds", 0.0)),
            "subprocess_seconds": wall,
        }
    )
    return record


def run_unit(
    prep: Prepared, replicate: int, *, feedback: bool, generator: Generator, base: int
) -> dict[str, Any]:
    """All 15 candidates of one (sample, replicate), evaluated, with per-arm selection."""
    sample = prep.sample
    prompt = repair_prompt(sample.example, diagnostic=prep.first_diagnostic if feedback else None)
    if leaked_fragments(sample, prompt):
        reason = f"{sample.sample_id}: prompt leaks the repair target"
        raise ValueError(reason)
    index = [s.sample_id for s in samples()].index(sample.sample_id)
    plans = plan(
        prep.window,
        len(prep.ids),
        sample_index=index,
        replicate=replicate,
        base=base,
        oracle_length=prep.reference_window.length,
    )
    candidates = [_candidate(prep, p, prompt, generator, base=base) for p in plans]
    bad = [c for c in candidates if "preserved" in c and not c["preserved"]]
    if bad:
        reason = f"G1: {sample.sample_id} r{replicate}: infill changed outside-span tokens"
        raise SystemExit(reason)
    prefix = len(scaffold_prefix(sample.example))
    runs = evaluate_many(
        [
            (c["harness"], prefix, prefix + len(c["body"]), FUNCTIONAL_SPECS[sample.example])
            for c in candidates
        ]
    )
    for c, run in zip(candidates, runs, strict=True):
        kept, total = unaffected_statements_preserved(
            sample.reference_body, c["body"], defect_line=sample.mutation.line_number
        )
        c.update(
            {
                "compiled": run.compiled,
                "compile_error": run.compile_error,
                "exit_code": run.exit_code,
                "success": run.verdict.success,
                "reasons": list(run.verdict.reasons),
                "executed_chains": [list(ch) for ch in run.verdict.executed_chains],
                "unaffected_statements_kept": kept,
                "unaffected_statements_total": total,
                "edit_distance_to_mutant": lexical_edit_distance(sample.mutated_body, c["body"]),
                "edit_distance_to_reference": lexical_edit_distance(
                    sample.reference_body, c["body"]
                ),
            }
        )
    selection = {}
    for arm in ARMS:
        outcomes = [
            CandidateOutcome(
                c["index"], c["compiled"], c["success"], c["cost"], c["generate_seconds"]
            )
            for c in candidates
            if c["arm"] == arm
        ]
        selection[arm] = asdict(select(outcomes))
    # The oracle-length diagnostic never enters a selection or a comparison; it is
    # reported as its single candidate's own verdict.
    diagnostic = next((c for c in candidates if c["arm"] == DIAGNOSTIC_ARM), None)
    return {
        "sample_id": sample.sample_id,
        "example": sample.example,
        "kind": sample.mutation.kind,
        "replicate": replicate,
        "feedback": feedback,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "window": {"start": prep.window.start, "length": prep.window.length},
        "target_tokens": prep.reference_window.length,
        "candidates": candidates,
        "selection": selection,
        "diagnostic": None
        if diagnostic is None
        else {"compiled": diagnostic["compiled"], "success": diagnostic["success"]},
    }


def main() -> None:
    """NOT to be run on a GPU until the frozen protocol above has been approved."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--samples", default="all")
    parser.add_argument("--replicates", type=int, default=PILOT_REPLICATES)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--feedback", choices=("none", "message"), default="none")
    parser.add_argument("--seed-base", type=int, default=PILOT_SEED_BASE)
    parser.add_argument("--dry-run", choices=("oracle", "delete"), default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    _check_run_mode(args)
    for path in (args.out, args.ledger):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    bench = list(samples())
    if args.samples != "all":
        wanted = args.samples.split(",")
        bench = [s for s in bench if s.sample_id in wanted]
        if len(bench) != len(wanted):
            reason = f"unknown sample id in {wanted}"
            raise SystemExit(reason)
    prepared = prepare(bench)
    if args.dry_run is not None:
        pieces_of = {
            token: piece
            for p in prepared
            for token, piece in zip(
                [*p.ids, *p.reference_ids],
                [*p.pieces, *_reference_pieces(p)],
                strict=True,
            )
        }
        generator = fake_generator(args.dry_run, pieces_of)
    else:
        generator = dream_generator(args.gpu)

    backend = f"dry-{args.dry_run}" if args.dry_run else f"gpu{args.gpu}"
    run_id = f"rq4b-{'smoke' if args.smoke else 'p1'}-{backend}-{int(time.time())}"
    units: list[dict[str, Any]] = []
    for prep in prepared:
        for replicate in range(args.replicate_start, args.replicate_start + args.replicates):
            unit = run_unit(
                prep,
                replicate,
                feedback=args.feedback == "message",
                generator=generator,
                base=args.seed_base,
            )
            units.append(unit)
            for c in unit["candidates"]:
                append(
                    Path(args.ledger),
                    HarnessSynthesisLedgerRecord(
                        run_id=run_id,
                        iteration=replicate,
                        arm=f"{unit['sample_id']}-rq4b-{c['arm']}-{c['index']}",
                        seed=c["seed"],
                        harness_source_sha256=sha256_bytes(c["harness"].encode()),
                        knowledge_base_sha256=unit["prompt_sha256"],
                        usage_example_names=(unit["example"],),
                        compiled=c["compiled"],
                        compile_error=c["compile_error"],
                        coverage_edges=0,
                        recorded_at_unix_seconds=time.time(),
                    ),
                )
            print(
                f"{unit['sample_id']} r{replicate} "
                + " ".join(f"{arm}={s['pass_at_k']}" for arm, s in unit["selection"].items())
            )
            Path(args.out).write_text(json.dumps(units, indent=1))


def _check_run_mode(args: argparse.Namespace) -> None:
    """Refuse any GPU run that is not exactly the frozen P1 or a labelled smoke test.

    - P1: frozen files unchanged (``verify_freeze``), the pilot seed block, all R
      replicates from 0, no feedback. P2 is not frozen yet.
    - ``--smoke``: frozen files unchanged too (the smoke test exercises the frozen code),
      but on a seed base outside every reserved block, and never writing into
      ``docs/assets``.
    - ``--dry-run``: a fake generator on CPU; no checks, never a result.
    """
    if args.dry_run is not None:
        return
    if args.gpu is None:
        reason = "a GPU run needs an explicit --gpu"
        raise SystemExit(reason)
    problems = verify_freeze()
    if problems:
        raise SystemExit("frozen protocol changed:\n  " + "\n  ".join(problems))
    if args.smoke:
        lo, hi = args.seed_base, seed_for(499, 99, 9, base=args.seed_base)
        if any(lo <= b_hi and b_lo <= hi for b_lo, b_hi in RESERVED_SEED_BLOCKS):
            reason = f"--smoke seed range {lo}-{hi} overlaps a reserved block"
            raise SystemExit(reason)
        if any("docs/assets" in str(Path(p).resolve()) for p in (args.out, args.ledger)):
            reason = "--smoke output must not go to docs/assets"
            raise SystemExit(reason)
        return
    if (args.seed_base, args.replicate_start, args.replicates, args.feedback) != (
        PILOT_SEED_BASE,
        0,
        PILOT_REPLICATES,
        "none",
    ):
        reason = "P1 is frozen at --seed-base 13000000, replicates 0..4, --feedback none"
        raise SystemExit(reason)


def _reference_pieces(prep: Prepared) -> list[str]:
    # The fake generator needs a piece for every reference id. Re-tokenizing is cheap,
    # and it keeps Prepared free of fields that only a dry run uses.
    _, tokens = tokenize([continuation_text(prep.sample.reference_body)])
    return list(tokens[0]["pieces"])


if __name__ == "__main__":
    main()
