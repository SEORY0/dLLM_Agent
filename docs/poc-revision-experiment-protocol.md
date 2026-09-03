# PoC Revision Experiment Protocol

This protocol defines Task 8 tooling for deterministic paired OpenEXR PoC revision experiments. The checked-in fixture mode validates the runner and analyzer only; it is not model efficacy evidence and makes no vulnerability claim.

## Arms

Primary arms are `libfuzzer-only`, `structured-one-shot`, `feedback-no-remask`, `selective-revision`, and `pseudo-feedback`.

`native-hunt-reference` is recorded only as a non-isomorphic engineering reference and is excluded from the primary causal contrast.

## Pairing And Seeds

For repetition `r` and particle `p`:

- particle seed: `202608200000 + 1000*r + p`
- fuzzer seeds: `202608210000 + 1000*r`, `202608210001 + 1000*r`
- retrieval seed: `202608220000 + r`
- permutation seed: `202608230000 + r`
- proof seed: `202608240000 + r`
- bootstrap seed: `202608209999`

Pseudo-feedback uses a deterministic cyclic derangement derived from the permutation seed. It has no fixed points and does not alter candidates, snapshot bytes, probe seeds, or budget.

## Default Pilot Budget

The default pilot is 5 repetitions, 4 particles, 128 initial NFE per particle, 3 revision epochs, and 16 revision NFE per particle per epoch. Each model arm is capped at 704 model-arm NFE.

Each arm/repetition requests two 3-second probes per snapshot and a 30-second independent proof budget, for 126 requested fuzz/proof seconds. Setup cost is recorded separately from inclusive campaign accounting.

## Artifacts

Each arm/repetition writes:

- `run-manifest.json` with target/model/image pins, seeds, fixed CLI argv, and budget.
- `revision-report.json` with `dllm-poc-revision-report/v1` schema version.
- `arm-report.json` with immutable manifest, report, and evidence-tree references plus hashes.
- `evidence/` with hashable regular-file evidence.

The runner also writes `raw-arm-reports.jsonl` and `fixture-index.json`. The analyzer validates referenced hashes and schemas before computing `paired-summary.json`.

## Analysis

The analyzer is stdlib-only and deterministic. It computes 10,000-resample paired bootstrap CIs for mean and median paired differences, win/loss/tie counts, rank-biserial effect, and a right-censored proof-time table.

The primary contrast is `selective-revision_vs_pseudo-feedback`.

Verdict rules:

- `go`: point estimate is at least +0.05 and the lower CI bound is above 0.
- `no-go`: upper CI bound is below +0.05.
- `inconclusive`: all other cases.

No independent proof means no vulnerability outcome.

## Live Campaign Boundary

Task 8 does not run live OpenEXR/model campaigns. Later live execution must invoke the installed CLI with fixed argv and no shell, stop arms only at declared budgets, and fail closed on report schema or hash mismatch.
