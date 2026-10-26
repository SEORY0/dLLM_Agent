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
| infill-sweep | defective-line window replaced by L + {0, +2, −2, +4, +8}[j] masks; both sides fixed | = masked | 5 (one per length) |
| suffix | from the window start to the end of the continuation, + 8 | = masked | 5 (seeds) |
| resample | fresh body, 96 tokens | 96 | 5 (seeds) |
| *infill-oracle-length* (diagnostic) | L* masks | L* | 1, never compared |

- **Common to all arms.** The same prompt and scaffold prefill. `alg_temp` 0.5, temperature 0.5, top_p 0.95, entropy reveal order. Candidate j's seed is shared across arms (common random numbers): `13,000,000 + 1000·sample + 10·replicate + j`.
- **Continuation.** Benchmark bodies were never generated, so the continuation is `body + "}\n```"`, tokenized by Dream's own tokenizer (`_dream_tokenize.py`, CPU, tokenizer files only). The pieces must reproduce the text exactly (asserted).
- **Fixed-canvas handling.** Special pieces inside the infilled span are dropped, as in RQ4. Outside-span tokens must be unchanged. Any violation aborts the run (G1). `extract_body` cuts at the try block's `}` or a stray fence or stop token.
- **Selection rule (frozen).** Walk the candidates in frozen order. The first functional success is selected. Otherwise the first compiled candidate is selected, then the first candidate. Selection matters only for secondary metrics. Compiler invocations and cost count up to the selected success, or all K.
- **Compute.** Cost = steps × (prompt + continuation length), counted in forward token-positions. Equal attempts do not mean equal compute: in the benchmark, suffix masks 2.3–5.9× the positions of the infill L candidate, and resample 4.8–10.7×. The record also holds denoising seconds (new `generate_seconds` field in `_dream_generate.py`, model load excluded) and subprocess seconds.
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

## 7. Pilot P1 execution plan (NOT scheduled)

- **Population.** 12 samples × R = 4 replicates = 48 units. Each unit has 15 compared candidates plus 1 diagnostic: **768 generations**.
- **Seeds.** 13,000,000–13,011,034. The 13,000,000–13,499,999 block is pilot-only. Confirmatory runs use 13,500,000+. None overlaps RQ4's 12,000,000 block or earlier blocks.
- **Resources.** One Dream subprocess per candidate, about 16 GB bf16 on one GPU.
  - **Estimate, unverified.** Each subprocess takes 10–20 s, mostly model load, so 768 generations ≈ 2–4 GPU-hours. Split over the two GPUs (IF on GPU 0, MP on GPU 1) that is ≈ 1–2 h of wall time.
  - **Observed.** The evaluator takes about 1.7 s per candidate on CPU, ≈ 25 min in total.
  - Before launching, check `nvidia-smi` and leave other workloads alone.
- **Commands.**
  ```bash
  uv run python tools/harness_synthesis/rq4b_validate_benchmark.py \
    --out docs/assets/harness-rq4b-benchmark-validation-20261026.json      # must exit 0
  uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 0 \
    --samples IF-ctor-pointer,IF-bad-method,IF-extra-arg,IF-box-namespace,IF-no-semicolon,IF-uncalled-member \
    --replicates 4 --out rq4b-p1-if.json --ledger /data/dllm-agent/harness-synthesis/rq4b-p1-ledger-gpu0.jsonl
  uv run python tools/harness_synthesis/rq4b_functional_pilot.py --gpu 1 \
    --samples MP-ctor-pointer,MP-bad-method,MP-missing-arg,MP-header-type,MP-no-semicolon,MP-string-arg \
    --replicates 4 --out rq4b-p1-mp.json --ledger /data/dllm-agent/harness-synthesis/rq4b-p1-ledger-gpu1.jsonl
  uv run python tools/harness_synthesis/rq4b_analyze.py --audit-sample 24 --out rq4b-p1-g3-audit.json rq4b-p1-*.json
  # hand-check G3 first, then:
  uv run python tools/harness_synthesis/rq4b_analyze.py --out rq4b-p1-analysis.json rq4b-p1-if.json rq4b-p1-mp.json
  ```
- **Stop conditions during the run.**
  - G1 (automatic abort): any change to outside-span tokens.
  - Stop and report if the first unit takes more than 15 minutes, which would mean the timing estimate is wrong by more than 3×.
  - Stop and report if a Dream subprocess fails twice on the same candidate.
- **Smoke test first.** One unit on throwaway seeds outside every reserved block: `--seed-base 19000000 --samples IF-ctor-pointer --replicates 1`, with `--out` and `--ledger` under a scratch directory, never `docs/assets`. It confirms the real backend returns the expected payload shape. It is not a result, and it does not feed the analysis.

## 8. Frozen statistical plan

- **Unit.** The benchmark sample (cluster). Its R replicates share defect, prompt and context, so they are not independent. Each sample contributes d_s = mean over replicates of (success_a − success_b).
- **Eligibility.** Every sample of the validated benchmark. No exclusion after the run, except a G1 abort of the whole run.
- **Primary endpoint.** Functional pass@5 per (unit, arm), from the frozen selection rule.
- **Strata, prespecified and descriptive only.** Example (IF/MP); mutation kind; L* ≤ L vs L* > L.
- **Estimation.** Mean of d_s, with a 95% percentile cluster-bootstrap CI (10,000 resamples, seed 0).
- **Test.** Exact sign-flip permutation over clusters (2^12 patterns), two-sided. The replicate-level discordance counts are reported as well; they are never used as independent pairs.
- **Multiplicity.** Holm over {H1, H2}. H3, H4 and P2 are secondary, reported without confirmatory claims.
- **Gates.**
  - G1: preservation, enforced at runtime.
  - G2: some arm's pooled pass@5 lies strictly between 0.05 and 0.95. Otherwise the benchmark does not discriminate: redesign it, with no confirmatory run.
  - G3: a hand check of 24 candidates, stratified by verdict, before reading any contrast. Any disagreement means fixing the evaluator and re-scoring all candidates first.
- **P1 is not confirmatory.** It estimates SD(d_s) and checks the benchmark and evaluator. Its p-values are labelled exploratory and its seeds are never reused.
- **Confirmatory sizing.**
  - n = ⌈((z₀.₉₈₇₅ + z₀.₈) · SD₈₀ / δ)²⌉ + 2, where SD₈₀ is the 80% upper confidence bound on P1's SD(d_s) (Browne 1995; `repair_stats.pilot_sd_upper`). α = 0.025 per hypothesis.
  - The minimum effect worth pursuing is δ = 0.10 on pass@5; δ = 0.15 is also reported.
  - The 12 samples cannot reach this n. A confirmatory run requires expanding the benchmark to n samples (more mutations, more reference bodies or APIs), validated by the same script. The sample size is not chosen to fit available hardware.
- **Decision after P1 (prespecified).**
  - **Terminate the "local infill advantage" claim for this setting** if the H1 point estimate ≤ 0 **and** the compute-matched H3-vs-suffix point estimate ≤ 0. Then infill shows no advantage at equal attempts or at equal compute, and that is reported as such.
  - **Proceed to a confirmatory design** (expanded benchmark) only if H1 or H3-vs-suffix ≥ +0.10. The advantage must also appear in at least 2 mutation kinds, and the leave-one-sample-out estimate must stay ≥ +0.05.
  - **Otherwise inconclusive.** Run P2 (feedback) or study the mask-length curve. No claim is made.
  - **Canvas diagnostic, independent of the above.** If infill-oracle-length succeeds far more often than every sweep length, the bottleneck is canvas length, not context. That points to a variable-length model (DreamOn-style), not more of this sweep.

## 9. Verification status

Implemented and tested: offline audit; benchmark and validation; evaluator (pure + Docker runner);
arms, selection and budget; cluster statistics; pilot harness, including a CPU `--dry-run`
fake generator; analysis script.

Not tested:

- the real Dream backend path (`dream_generator`) on GPU;
- the `generate_seconds` field;
- timing estimates;
- the G3 hand-check procedure, which needs P1 output.
