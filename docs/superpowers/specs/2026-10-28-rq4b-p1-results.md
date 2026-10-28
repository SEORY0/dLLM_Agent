# RQ4b P1 — results (exploratory pilot, frozen protocol)

Date: 2026-10-28. Branch `feat/rq4-functional-repair-benchmark`.

- **Protocol:** frozen in [`2026-10-26-rq4b-functional-repair-benchmark.md`](2026-10-26-rq4b-functional-repair-benchmark.md) (code commit `a715164`; manifest `docs/assets/harness-rq4b-p1-freeze-20261026.json`, sha256 `6dbc3129…`). This document changes nothing in the protocol; the spec and every frozen file are unmodified.
- **Artefacts:**
  - per-candidate results: [`harness-rq4b-p1-results-20261028.json`](../../assets/harness-rq4b-p1-results-20261028.json)
  - append-only ledger, 960 rows: [`harness-rq4b-p1-ledger-20261028.jsonl`](../../assets/harness-rq4b-p1-ledger-20261028.jsonl)
  - frozen analysis output: [`harness-rq4b-p1-analysis-20261028.json`](../../assets/harness-rq4b-p1-analysis-20261028.json)
  - G3 hand-check sample: [`harness-rq4b-p1-g3-audit-20261028.json`](../../assets/harness-rq4b-p1-g3-audit-20261028.json)

Every statement is labelled **observed** (computed from the artefacts), **inferred** (a reading of them) or **unverified**. All p-values are **exploratory**: P1 is not confirmatory (spec §8), and its seeds are never reused.

## 1. Execution (observed)

- **Smoke test** (one unit, seeds 19,000,000+, outside every reserved block, kept outside the repo): 16/16 generations, freeze check passed, mask id 151666 matched, preservation held, 156 s, peak 17.6 GB on GPU 1.
- **First P1 launch (10:35):** aborted before any ledger row was written. A shared vLLM service (`solar-open2-250b`, tensor-parallel over both GPUs) started in the same second and grew to 88 GB on GPU 1, and the first Dream subprocess hit CUDA out-of-memory. Per protocol there was no retry. Its log is kept as `/data/dllm-agent/harness-synthesis/rq4b-p1/run-oom-20261028-1035.log`. No candidate from it exists, so nothing was excluded.
- **P1 run (11:16–13:46, about 2.5 h, GPU 1 only, one Dream process at a time):** 60 units, 960 generations, no errors. The freeze check passed at start, and G1 (outside-span preservation) held for every infill candidate. A `hybrid-document-search` job ran on GPU 0 for part of the run; GPU 1 held only P1 and an idle Sinsan process (772 MiB).
- **Analysis:** the frozen commands from spec §7, step 3, unchanged.

## 2. Gates

| Gate | Result |
|---|---|
| G1 preservation | passed: 300/300 infill and 60/60 diagnostic candidates preserved outside the span |
| G2 discriminating | passed: infill-sweep pooled pass@5 = 35/60 |
| G3 hand check, 24 candidates stratified by verdict | 24/24 agree (11 functional successes, 2 compiled-not-functional, 11 not compiled; every compile failure is a nonexistent API or type, such as `numberOfParts`, `FileHeader`, `DataWindow`, `Box2`, `OPENEXR_IMF_NAMESPACE::Box2i`, or a truncated body) |

**Protocol deviation (observed).** The spec says G3 comes before reading any contrast. The analysis script was run immediately after the G3 sample was drawn, and the hand check was done afterwards, by Claude, not a human. G3 found no disagreement, so no re-scoring was needed and the contrasts below are the frozen pipeline's output. A human review of the 24 candidates is still **outstanding**.

## 3. Primary contrasts (exploratory)

Unit of inference: 12 per-mutation differences of mean pass@5 over 5 replicates. Exact sign-flip test; CI by test inversion.

| Comparison | Mean difference | 95% CI (exact) | Bootstrap CI | p | Holm p | Replicates: infill-only / other-only |
|---|---:|---|---|---:|---:|---|
| H1 infill-sweep vs suffix | **+0.57** | 0.28 – 0.88 | 0.33 – 0.80 | 0.0039 | 0.0078 | 34 / 0 |
| H2 infill-sweep vs resample | **+0.58** | 0.30 – 0.90 | 0.33 – 0.82 | 0.0039 | 0.0078 | 35 / 0 |
| H3 compute-matched (m = 1) vs suffix | +0.55 | 0.27 – 0.84 | 0.32 – 0.78 | 0.0039 | (secondary) | 33 / 0 |
| H3 compute-matched (m = 1) vs resample | +0.55 | 0.27 – 0.84 | 0.32 – 0.78 | 0.0039 | (secondary) | 33 / 0 |

- **Observed.** 9 of 12 mutations have a positive difference, 3 are zero, none negative. Leave-one-mutation-out H1 lies in 0.53 – 0.62.
- **Observed.** At m = 1 the H3 budget affords on average 4.7 infill, 1.4 suffix and 1.0 resample candidates; it is not degenerate. On the measured-seconds curve infill succeeds in 26/60 units at m = 1 and 35/60 from m = 2, while suffix reaches 1/60 and resample 0/60 at every m ≤ 5.
- **Observed (per body, descriptive only, 6 clusters each).** InputFile H1 +0.47, MultiPartInputFile H1 +0.67. With 6 clusters the exact CI spans 0 – 1 in both, so neither body alone is informative.

## 4. Descriptive

| Arm | pass@5 (units) | Candidate success | Compiled but not functional | Unaffected statements kept | Mean forward cost per candidate | Mean decode s |
|---|---:|---:|---:|---:|---:|---:|
| infill-sweep | 35/60 | 58/300 (19%) | 0 | 100% | 9,655 | 1.7 |
| suffix | 1/60 | 1/300 | 2 (`dataWindow` call dropped) | 39% | 29,879 | 4.0 |
| resample | 0/60 | 0/300 | 0 | 14% | 54,480 | 6.8 |
| *infill-oracle-length (diagnostic, 1 candidate/unit, outside every comparison)* | *29/60* | *29/60 (48%)* | *0* | *100%* | | |

**Per mutation, infill-sweep pass@5 out of 5 replicates** (suffix and resample are 0 everywhere except IF-extra-arg, where suffix has 1):

| Mutation | Kind | infill-sweep | oracle-length diagnostic |
|---|---|---:|---:|
| IF-no-semicolon | syntax | 5 | 5 |
| IF-extra-arg | argument count | 5 | 5 |
| IF-bad-method | invalid method | 4 | 4 |
| IF-ctor-pointer | constructor argument | 1 | 5 |
| IF-box-namespace | namespace/type | 0 | 0 |
| IF-uncalled-member | missing call parentheses | 0 | 0 |
| MP-bad-method | invalid method | 5 | 0 |
| MP-header-type | namespace/type | 5 | 1 |
| MP-missing-arg | argument count | 5 | 2 |
| MP-string-arg | argument type | 4 | 2 |
| MP-no-semicolon | syntax | 1 | 0 |
| MP-ctor-pointer | constructor argument | 0 | 5 |

**Success by mask length** (per candidate, balanced over seed slots): L+0 18%, L+2 10%, L−2 27%, L+4 8%, L+8 33%. The curve is not monotone.

## 5. Reading (inferred, scoped to this benchmark, model and evaluator)

1. **The baselines are at the floor, so the contrasts mostly measure "keep the known-good code".**
   - Suffix and resample regenerate the downstream lines without seeing the reference. They keep 39% and 14% of the unaffected statements, and replace them with OpenEXR APIs and types that do not exist. This is the same failure seen in the smoke test and the RQ4 audit.
   - Infill keeps every statement outside the span by construction.
   - So H1/H2 show that **localized infill beats regenerate-from-here repair here**. They do **not** isolate the effect of right-hand context: no arm edits only the located line while seeing only the left context. That arm is needed before any claim about bidirectionality.
2. **Where infill fails, it reproduces the model's own prior.**
   - In IF-ctor-pointer and MP-ctor-pointer the sweep regenerates `file(&stream)` or `file(&stream, true)`.
   - In IF-box-namespace and IF-uncalled-member it writes `OPENEXR_IMF_NAMESPACE::Box2i`, `Box2` or `DataWindow`.
   - These are the defects the model tends to write unprompted. Bidirectional context does not override the prior when the prior is itself the bug.
3. **Canvas length still matters, but is not the only bottleneck.**
   - The oracle length, a single candidate, succeeds in 48% of units, above every sweep length (8–33%).
   - It decides the constructor defects: 5/5 vs 1/5 and 0/5.
   - But on MP-bad-method it scores 0/5 where the sweep scores 5/5, so the right length depends on what the model wants to write, not on the reference's length.
   - Inferred: a variable-length infill model (spec §8 canvas diagnostic) would likely help the constructor cases. It would not help the namespace/type cases.
4. **Long-slack successes fill the surplus harmlessly.** At L+4 and L+8 the surplus in the G3 successes becomes a comment or blank line (#7, #15, #17, #23), which explains why L+8 is the best sweep length despite RQ4's canvas problem.

## 6. Prespecified decision (spec §8)

| Criterion for "proceed to a confirmatory design" | Value | Met |
|---|---|---|
| H1 or primary H3-vs-suffix ≥ +0.10 | H1 +0.57; H3 +0.55 | yes |
| Advantage in ≥ 2 mutation kinds | 6 kinds with positive differences | yes |
| Advantage in both reference bodies | InputFile +0.47, MultiPartInputFile +0.67 | yes |
| Leave-one-mutation-out ≥ +0.05 | min +0.53 | yes |

**Decision: proceed to a confirmatory design (expanded benchmark), scoped as in spec §8.** Sizing from P1 (SD(d_s) = 0.44, 80% upper bound 0.55): 132 mutations for δ = 0.15, or 294 for δ = 0.10, at α = 0.025 per hypothesis. It needs new mutations and new reference bodies.

Recommended before freezing the confirmatory protocol (inferred from §5, not prespecified):
- Add a **left-context-only located-line arm** (regenerate only the located line, stop at its newline, then splice the original suffix back). It is the arm that separates "bidirectional context" from "preserve the rest".
- Keep the oracle-length diagnostic, and consider a variable-length model for the constructor cases.

## 7. What P1 does and does not establish

- **Established (observed, exploratory, this setting only):**
  - Given the correct line, Dream-Coder-7B with a 5-length mask sweep repaired 35/60 single-defect units functionally (verified by AST and coverage).
  - Suffix and resample repaired 1/60 and 0/60, at equal attempts and at equal compute.
- **Not established:** that right-hand context causes the advantage (no left-only located-line arm).
- **Not established:** anything beyond these 12 mutations of 2 OpenEXR bodies, this model, this decoding configuration and this evaluator.
- **Not established:** anything about dLLM vs AR (no AR or FIM-AR model was run).
- **Unverified:** a human review of the G3 sample.
