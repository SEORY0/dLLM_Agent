# RQ4b — functional repair on controlled single defects (design, frozen, NOT executed)

Date: 2026-10-26. Branch `feat/rq4-functional-repair-benchmark`, based on PR #4 (`c932c1b`).
Prerequisite reading: [the RQ4 retrospective audit](2026-10-26-rq4-retrospective-audit.md).

**Status.** The design, benchmark, evaluator, harness and analysis are implemented and tested
offline (CPU only). **No GPU run has happened.** Every number from a model below is a plan or an
estimate, never a result.

## 1. Why RQ4 needs a second design

The audit shows the RQ4 pilot cannot separate the factors it intended to test.

- Right-hand context was mostly absent: 17/32 of its stratum had only comments after the line.
- Defects spanned several lines: 13 trials, and in 10 of them one-line repair is impossible even with an oracle.
- Canvas length and editable span differed between arms.
- Decoding budgets were unequal.
- Compile success counted commenting-out as repair, under a prompt that made inspection optional.

RQ4b removes each of these by construction rather than by stratifying afterwards.

| RQ4 confounder | RQ4b control |
|---|---|
| Model-generated defects, uncontrolled number and kind | Hand-written reference bodies plus exactly one recorded mutation, validated so that every clang error lands on the mutated line |
| "Right context" often a comment | Every mutated line has executable code after it (asserted) |
| Inspection optional, metric = compiles | Inspection required by the prompt. Primary metric = functional success (AST + execution) |
| One fixed canvas (window + 8) | Mask-length sweep L + {0, +2, −2, +4, +8}, plus an oracle-length diagnostic L* |
| Unequal attempts and decoding | K = 5 candidates in every arm, common seeds, one token per denoising step everywhere, cost recorded per candidate |
| Compiler message only implicit | A separate factor (P2), given identically to every arm |

## 2. Research question and hypotheses

**RQ4b.** Given a correct compiler-reported location for a single known defect, does
regenerating only that span with both sides fixed (Dream infill) achieve functional repair
better than regenerating from that location onward (suffix) or from scratch (resample), at
matched attempts or matched compute, while preserving the unaffected code?

Three outcomes are kept apart:

1. **Syntactic validity**: compiles.
2. **Functional repair**: constructs the object from `stream` and executes every required call chain on the validation input, with no exception.
3. **Preservation**: the other statements of the reference survive; the executed call chains match the reference.

Hypotheses (two-sided, functional endpoint):

- **H1 (primary).** pass@5(infill-sweep) − pass@5(suffix) ≠ 0.
- **H2 (primary).** pass@5(infill-sweep) − pass@5(resample) ≠ 0.
- **H3 (secondary, compute-matched).** H1 and H2 again, where each arm may spend only the infill arm's total forward cost for that unit.
- **H4 (secondary, descriptive).** Among functional successes: unaffected statements kept, and token edit distance to the reference.
- **H5 (P2 only).** Within each arm, compiler message vs location only. Arm × feedback interaction is exploratory.

**What this does not test.** dLLM vs AR: every arm is Dream. A fair H-bi against an AR model needs a FIM-capable AR model under the same benchmark, which is out of scope here. Localization: the location is correct by construction. Generalization beyond these two OpenEXR classes.

## 3. Benchmark (implemented: `src/dllm_agent/harness_synthesis/repair_benchmark.py`)

Reference bodies were checked against the installed OpenEXR 3.4.14 headers in the pinned image:

- `InputFile(IStream&, int numThreads = …)` and `const Header& header() const` (`ImfInputFile.h`).
- `MultiPartInputFile(IStream&, int, bool)`, `int parts() const` and `const Header& header(int) const` (`ImfMultiPartInputFile.h`).
- `const Box2i& dataWindow() const` (`ImfHeader.h`).

```cpp
// InputFile reference                        // MultiPartInputFile reference
OPENEXR_IMF_NAMESPACE::InputFile file(stream);        OPENEXR_IMF_NAMESPACE::MultiPartInputFile file(stream);
const OPENEXR_IMF_NAMESPACE::Header& header =         const int partCount = file.parts();
    file.header();                                    const OPENEXR_IMF_NAMESPACE::Header& header = file.header(0);
const IMATH_NAMESPACE::Box2i& dataWindow =            static_cast<void>(partCount + header.dataWindow().max.x);
    header.dataWindow();
static_cast<void>(dataWindow.max.x - dataWindow.min.x);
```

Required chains: InputFile `header → dataWindow`. MultiPartInputFile `parts` and `header → dataWindow`.

| Sample | Kind | Line | Defect → target | L | L* |
|---|---|---:|---|---:|---:|
| IF-ctor-pointer | constructor argument | 1 | `file(&stream)` → `file(stream)` | 14 | 13 |
| IF-bad-method | invalid method | 2 | `file.readHeader()` → `file.header()` | 17 | 16 |
| IF-extra-arg | argument count | 2 | `file.header(0)` → `file.header()` | 18 | 16 |
| IF-box-namespace | namespace/type | 3 | `OPENEXR_IMF_NAMESPACE::Box2i` → `IMATH_NAMESPACE::Box2i` | 20 | 17 |
| IF-no-semicolon | syntax | 1 | missing `;` | 13 | 13 |
| IF-uncalled-member | missing call parens | 3 | `header.dataWindow` → `header.dataWindow()` | 17 | 17 |
| MP-ctor-pointer | constructor argument | 1 | `file(&stream)` → `file(stream)` | 16 | 15 |
| MP-bad-method | invalid method | 2 | `file.numParts()` → `file.parts()` | 10 | 9 |
| MP-missing-arg | argument count | 3 | `file.header()` → `file.header(0)` | 16 | 18 |
| MP-header-type | namespace/type | 3 | `PartHeader&` → `Header&` | 19 | 18 |
| MP-no-semicolon | syntax | 2 | missing `;` | 9 | 9 |
| MP-string-arg | argument type | 3 | `file.header("0")` → `file.header(0)` | 18 | 18 |

L is the Dream token window of the defective line (observable). L* is the target line's
window (oracle; it covers the line's trailing newline token, like L).

**Validation (observed, `rq4b_validate_benchmark.py` → [`harness-rq4b-benchmark-validation-20261026.json`](../../assets/harness-rq4b-benchmark-validation-20261026.json)).** All 12 samples pass every check:

- the reference is syntax-clean and functionally successful;
- the mutant does not compile;
- every error is on the mutated line;
- executable code follows it;
- there is no leakage;
- all five mask lengths are positive.

Two kinds of candidate were rejected because their errors spread to other lines: a class-name typo, and an unclosed call. One more was rejected for leakage: `IMATH_NAMESPACE::Header`, whose fix `OPENEXR_IMF_NAMESPACE::` is visible in the prompt and on line 1.

**Leakage control.** `leaked_fragments` fails a sample when the target line, or any call- or
scope-shaped fragment the target has and the defective line lacks, appears in the model's
input. Here the input is the prompt in both feedback conditions, the scaffold prefix, and the
infill arm's visible context (a superset of the suffix arm's). The prompt describes the
required operations in words. The RQ4 prompt's "pass `stream`, not `&stream`" was removed,
because it would hand the `*-ctor-pointer` samples their fix.

**Feedback-condition note.** Three diagnostics contain clang fix-it text ("did you mean …"):
IF-box-namespace, IF-uncalled-member and MP-header-type. Under P2 that text reaches every arm
equally. It is part of the feedback factor, recorded per sample as `diagnostic_suggests_fix`.

## 4. Repair arms (implemented: `repair_arms.py`, `tools/harness_synthesis/rq4b_functional_pilot.py`)

| Arm | Masked | Steps | Candidates |
|---|---|---|---|
| infill-sweep | defective-line window replaced by L + o_k masks, o = (0, +2, −2, +4, +8); both sides fixed | = masked | 5 (one per length) |
| suffix | from the window start to the end of the continuation, + 8 | = masked | 5 (seed slots 0–4) |
| resample | fresh body, 96 tokens | 96 | 5 (seed slots 0–4) |
| *infill-oracle-length* (diagnostic) | L* masks | L* | 1 (seed slot 9), never compared or selected |

- **Common to all arms.** The same prompt and scaffold prefill. `alg_temp` 0.5, temperature 0.5, top_p 0.95, entropy reveal order.
- **Seed allocation (frozen).**
  - Seed slot j of a unit is `13,000,000 + 1000·sample_index + 10·replicate + j`, shared across arms (common random numbers).
  - Suffix and resample candidate j use slot j.
  - Infill length position k (o_k above) uses slot **(k + replicate) mod 5**. Over the 5 replicates of a sample, every mask length is paired with every slot exactly once, so a difference between mask lengths cannot be a fixed slot effect. `seed_slot_balance` and `rq4b_freeze.py` check this.
  - The diagnostic uses slot 9, which no compared candidate uses.
  - The complete table (960 rows: sample, replicate, arm, order index, seed, slot, masked, steps) is in the freeze manifest.
- **Continuation.** Benchmark bodies were never generated, so the continuation is `body + "}\n```"`, tokenized by Dream's own tokenizer (`_dream_tokenize.py`, CPU, tokenizer files only). The pieces must reproduce the text exactly (asserted).
- **Fixed-canvas handling.** Special pieces inside the infilled span are dropped, as in RQ4. Outside-span tokens must be unchanged. Any violation aborts the run (G1). `extract_body` cuts at the try block's `}` or a stray fence or stop token.
- **Selection rule (frozen).**
  - Per compared arm, walk the candidates in frozen order: infill by mask length in the order o above (whatever slot it has), suffix and resample by slot. The first functional success is selected, and pass@5 is whether one exists.
  - Otherwise the first compiled candidate is selected, then the first candidate. This only matters for secondary metrics.
  - Compiler invocations and cost count up to the selected success, or all K.
  - The oracle-length diagnostic is **outside every selection**: it is recorded as its own verdict (`unit["diagnostic"]`), never mixed into any arm's candidates, pass@5, budget or comparison.
- **Compute accounting.**
  - Forward cost = steps × (prompt + continuation length), in token-positions. Dream runs one full forward pass per denoising step, whatever the number of masks. This proxy is linear in sequence length, so it ignores attention's quadratic term, which is small at about 550–600 positions.
  - The prompt is about 470 tokens (task prompt + `cpp` fence + scaffold) plus the chat template, so it dominates the sequence. Measured on the benchmark, suffix masks 2.3–5.9× the positions of the infill L candidate, and resample 4.8–10.7×.
  - Also recorded per candidate: masked positions, steps, `generate_seconds` (denoising only, timed around `diffusion_generate` with a CUDA sync, model load excluded) and subprocess seconds (including model load).
  - Model load is the same constant for every candidate in this one-process-per-candidate design. It is excluded from compute matching because a persistent server would remove it.
- **Mask-length policy.** The sweep raises (rather than drops) a nonpositive length, so arms stay attempt-matched. Every sample has L ≥ 9.

## 5. Functional evaluator (implemented: `functional_eval.py` + `tools/harness_synthesis/_functional_eval.py`)

For each candidate, the evaluator works in one container shared by the batch.

1. **Build.** Coverage binary, same CMake project and flags as `build-harness coverage`.
2. **Run.** Once on the pinned C0 seed (a valid single-part EXR).
3. **Coverage.** llvm-cov region segments for `harness.cc`.
4. **AST.** clang's JSON AST of `LLVMFuzzerTestOneInput`.

The verdict, pure and unit-tested on captured real clang output, is **success** iff all of these hold:

- it compiled and the process exited 0;
- some body variable of the target class is constructed from `stream`, and that construction executed;
- every required chain exists as member calls traced to that variable, directly or through variables (references, `auto` copies);
- every call in the chain has an execution count > 0;
- the scaffold's `catch` block has count 0, so no exception escaped;
- the body contains no `try` of its own.

The exception rule exists because clang does not start a coverage region after a call. Without it, code after a throwing call would still read as executed.

**Controls, run against the real toolchain (observed, all as expected):**

| Must pass | Must fail (reason reported) |
|---|---|
| reference (×2) | deleted inspection (×2) → `chain-missing` |
| direct chain `file.header().dataWindow()` | commented-out inspection → `chain-missing` |
| `auto` copies | call only inside a string literal → `chain-missing` |
| inline MP chain | dead branch → `chain-not-executed` |
| *use of the value removed* (a stated limitation) | early `return` → `chain-not-executed` |
| | call on another object → `chain-missing` |
| | `header(1)` on a one-part file → `exception-reached-scaffold-catch` |
| | body-level `try/catch(...)` → `body-contains-try` |
| | parts only → `chain-missing:header>dataWindow` |

**Limitations (documented, not hidden):**

- Argument values are not checked. A wrong value fails only if it throws on the validation input.
- How a returned value is used is not checked (the "use-removed" control passes).
- One validation input, single-part. A multi-part-only behaviour would not be exercised.
- A body-level `try` is rejected even when harmless (conservative).
- Coverage proves execution, not correctness of the result.
- Fuzzing edge counts are deliberately not used. Scaffold-level coverage does not show that the required API ran.

**Preservation metrics.** `unaffected_statements_preserved` is a multiset of the reference's
other code lines in normalized form, so equivalent rewrites do not count; it is a textual lower
bound. The executed call chains, compared with the reference's, are the behavioural side. The
lexical token edit distance is measured against both the mutant and the reference.

## 6. Compiler-feedback factor (P2; implemented, not run)

`repair_prompt(example, diagnostic=…)` appends `A previous version of this body failed to
compile with: <first clang message>`. `rq4b_functional_pilot.py --feedback message` applies it to
**every arm** in the run, so within one condition all arms receive the same diagnostic
information. P2 reuses P1's mask-length policy, K, seeds and samples unchanged. Only the prompt
differs, so feedback is not mixed with a mask-length change. `rq4b_analyze.py` refuses to pool
conditions. P2 runs only after P1 has been analyzed and the user approves.

## 7. Pilot P1 execution plan (frozen, NOT scheduled)

- **Population.** 12 samples × R = 5 replicates = 60 units. Each unit has 15 compared candidates plus 1 diagnostic: **960 generations**. R was raised from 4 to 5 so the infill length/seed-slot rotation is exactly balanced (§4).
- **Seeds.** 13,000,000–13,011,049. The pilot refuses any other base for P1. Reserved blocks: 9,000,000–12,999,999 (RQ1/RQ3/RQ4), 13,000,000–13,499,999 (RQ4b pilots), 13,500,000–13,999,999 (RQ4b confirmatory).
- **Hardware.** **GPU 1 only.** At freeze time GPU 0 was fully used by a vLLM engine (≈85 GB, 100% utilization). GPU 1 had about 97 GB free, holding only a small Sinsan training process (≈0.8 GB, idle). One Dream process at a time uses about 16 GB bf16.
  - Do not start if GPU 1's free memory is below 24 GB, or its utilization is sustained above 20% (that would mean another workload is active).
  - Nothing on GPU 0 or any other service is touched.
- **Estimated workload (unverified until the smoke test).** About 10–20 s per generation subprocess (mostly model load), so 960 generations ≈ 2.7–5.3 h on GPU 1. The evaluator takes ≈ 1.7 s per candidate on CPU (≈ 30 min in total, interleaved per unit). Disk use is a few MB of JSON.
- **Freeze.** `rq4b_freeze.py` writes `docs/assets/harness-rq4b-p1-freeze-20261026.json` containing:
  - the SHA-256 of every protocol file;
  - the allocation table and its hash;
  - the compile image id;
  - the model config and safetensors-index hashes, plus weight file sizes;
  - the Dream venv's transformers/torch versions;
  - the code commit.

  `rq4b_functional_pilot.py` re-verifies those hashes before any GPU run and refuses on a mismatch.
- **Commands (each requires explicit approval).**
  ```bash
  # 1. one-unit smoke test (throwaway seeds; output outside docs/assets; not a result)
  uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 1 --smoke \
    --seed-base 19000000 --samples IF-ctor-pointer --replicates 1 \
    --out /data/dllm-agent/harness-synthesis/rq4b-smoke/smoke.json \
    --ledger /data/dllm-agent/harness-synthesis/rq4b-smoke/smoke-ledger.jsonl
  # 2. P1
  uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 1 \
    --out /data/dllm-agent/harness-synthesis/rq4b-p1/p1.json \
    --ledger /data/dllm-agent/harness-synthesis/rq4b-p1/p1-ledger.jsonl
  # 3. G3 hand check first, then the frozen analysis
  uv run python tools/harness_synthesis/rq4b_analyze.py --audit-sample 24 \
    --out /data/dllm-agent/harness-synthesis/rq4b-p1/g3-audit.json /data/dllm-agent/harness-synthesis/rq4b-p1/p1.json
  uv run python tools/harness_synthesis/rq4b_analyze.py \
    --out /data/dllm-agent/harness-synthesis/rq4b-p1/p1-analysis.json /data/dllm-agent/harness-synthesis/rq4b-p1/p1.json
  ```
- **Stop conditions.**
  - G1 aborts automatically (outside-span change).
  - The run also aborts automatically on a mask-id mismatch between tokenizer and model, or a frozen-file change.
  - Stop by hand and report if the first unit takes more than 15 min (more than 3× the estimate), if a Dream subprocess fails twice on one candidate, or if GPU 1's free memory falls below 24 GB because another workload started.
- **Smoke test.** One unit: 16 generations, about 3–6 min on GPU 1. It confirms the real backend's payload shape, mask-id agreement, preservation, timing, and that the freeze check passes.
  - It is not a result and never feeds the analysis.
  - Its seeds (19,000,000–19,000,009) lie outside every reserved block, and the pilot refuses `--smoke` output under `docs/assets`.

## 8. Frozen statistical plan

- **Unit of inference: the mutation (pseudoreplication control).**
  - The 60 units come from only 12 mutations. Replicates of one mutation share its defect, prompt and context, so they are repeated measures, not independent observations.
  - Every inferential quantity (test, CI, sizing) is computed on 12 per-mutation differences d_s = mean over its 5 replicates of (pass@5_a − pass@5_b).
  - Replicate-level and candidate-level counts (discordant replicate pairs, per-candidate success rates, success by mask length) are reported as **descriptive only**. They are never used as independent trials.
  - The 12 mutations also share 2 reference bodies, a second level of clustering. With 2 bodies there is no between-body inference, so each body's estimate is reported separately (`per_example_descriptive`), and no claim extends beyond these two bodies.
- **Eligibility.** Every sample of the validated benchmark. No exclusion after the run, except a G1 abort of the whole run.
- **Primary endpoint.** Functional pass@5 per (unit, arm), from the frozen selection rule.
- **Equal-attempt comparison (H1, H2), operational definition.** Each arm gets exactly K = 5 candidates per unit, whatever each costs. Steps, masked tokens and seconds differ between arms by design, and are reported (§4) but not equalized here.
- **Equal-compute comparison (H3), operational definition.**
  - Per unit, the budget is B_m = m × the forward cost of that unit's first resample candidate. That is the most expensive single candidate of any arm, so at m = 1 every arm affords at least one.
  - Each arm uses the longest prefix of its frozen candidate order whose cumulative forward cost fits within B_m. Success means any functional success in that prefix.
  - The primary H3 uses m = 1. With the measured lengths that is roughly 5 infill, 1–2 suffix and 1 resample candidates.
  - A curve over m = 1…5 is reported, along with the same curve using measured `generate_seconds` instead of forward cost (sensitivity to the proxy).
  - A comparison where any arm affords zero candidates in more than half the units is marked degenerate and not interpreted.
  - Cost per functional success is also reported per arm.
- **Strata, prespecified and descriptive only.** Example (IF/MP); mutation kind; L* ≤ L vs L* > L; mask length (success per length, balanced over seed slots).
- **Estimation.** Mean of d_s. The reported 95% CI inverts the exact sign-flip test (every shift it does not reject; exact under symmetry of d_s about its mean). A percentile cluster bootstrap (10,000 resamples, seed 0) is given as a sensitivity check only: it under-covers at n = 12.
- **Test.** Exact two-sided sign-flip permutation over the 12 clusters (4096 patterns).
- **Multiplicity.** Holm over {H1, H2}. H3, H4 and P2 are secondary, reported without confirmatory claims.
- **Gates.**
  - G1: preservation, enforced at runtime.
  - G2: some arm's pooled pass@5 lies strictly between 0.05 and 0.95. Otherwise the benchmark does not discriminate: redesign it, with no confirmatory run.
  - G3: a hand check of 24 candidates, stratified by verdict, before reading any contrast. Any disagreement means fixing the evaluator and re-scoring all candidates first.
- **P1 is not confirmatory.** It estimates SD(d_s) and checks the benchmark and evaluator. Its p-values are labelled exploratory and its seeds are never reused.
- **Confirmatory sizing.**
  - n = ⌈((z₀.₉₈₇₅ + z₀.₈) · SD₈₀ / δ)²⌉ + 2 mutations, where SD₈₀ is the 80% upper confidence bound on P1's SD(d_s) (Browne 1995). α = 0.025 per hypothesis.
  - δ = 0.10 is the minimum effect worth pursuing; δ = 0.15 is also reported.
  - Twelve mutations cannot reach this. A confirmatory run needs new mutations, and new reference bodies so the body-level clustering can be estimated, validated by the same script. The sample size is not chosen to fit the hardware.
- **Scope of every decision below.** It covers this model (Dream-Coder-v0-Instruct-7B), this decoding configuration, this fixed scaffold, these 12 single-line mutations of two OpenEXR bodies, a correct location given for free, and this evaluator. **No outcome of P1 is a general statement about bidirectional infilling, diffusion models, or code repair.** A null or negative result says only that this implementation shows no measurable advantage in this setting.
- **Decision after P1 (prespecified, scoped as above).**
  - **Stop pursuing this implementation** if the H1 point estimate ≤ 0 **and** the primary H3-vs-suffix (m = 1) point estimate ≤ 0. That means no advantage at equal attempts or at equal compute in this setting, and it is reported as such.
  - **Proceed to a confirmatory design** (expanded benchmark) only if H1 or primary H3-vs-suffix ≥ +0.10. The advantage must also appear in at least 2 mutation kinds and in both reference bodies, and the leave-one-mutation-out estimate must stay ≥ +0.05.
  - **Otherwise inconclusive.** Run P2 (feedback, separately frozen) or study the mask-length curve. No claim is made.
  - **Canvas diagnostic (descriptive).** If the oracle-length candidate succeeds far more often than every sweep length, the bottleneck in this setting is canvas length rather than context. That points to a variable-length model rather than more of this sweep.

## 9. Verification status

- **Implemented and tested on CPU:**
  - offline audit;
  - benchmark and validation;
  - evaluator (pure + Docker runner);
  - arms, the seed-slot rotation, selection and budget;
  - cluster statistics with the exact inverted CI;
  - freeze manifest and verification;
  - run-mode guards (P1 parameters fixed; smoke seeds and output location restricted; GPU runs refused without a matching manifest);
  - pilot harness with a CPU `--dry-run` fake generator;
  - analysis script.
- **Not tested:**
  - the real Dream backend path on GPU;
  - the `generate_seconds` field;
  - timing estimates;
  - the G3 hand-check procedure, which needs P1 output.
