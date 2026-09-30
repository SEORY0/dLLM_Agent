# Trajectory-guided harness synthesis: feasibility simulation (2026-09-20)

Scope: a small check that the idea in
`docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md` is possible at all,
before any implementation plan. Model: Dream-Coder-7B (bf16, one RTX PRO 6000 Blackwell).
Target: pinned OpenEXR runtime image (`dllm-openexr-runtime`), compile via its own `build-coverage`.
Scripts: `tools/harness_synthesis_feasibility/`. Raw results: `docs/assets/harness-feasibility-*.json`.
Small n (8-16 seeds), exploratory, no significance claims.

## Mechanics (all confirmed possible)

- Dream's `diffusion_generate` exposes `generation_tokens_hook_func(step, x, logits)`. The returned `x`
  feeds the next step, so mid-run remasking and whole-state infill (step `None`) are native.
- Tokens are never remasked inside `_sample`; "permanence" is only the commit step, so any
  revision logic must come through the hook.
- Dream's remote code needs `transformers==4.46.2`; the project pins 5.x. It ran in a separate venv
  layered over the project's torch 2.11+cu128 (nvidia wheel index was unreachable for a fresh torch).
- The runtime image already compiles a volume-mounted `/work/harness.cc` (`build-coverage`), so no new
  compile container is required for the harness-identity design (per-iteration SHA-256 in the ledger).

## Result 1: confidence localizes the error

16 samples, one round, statement/#include regions. The lowest mean commit-confidence region equalled
the line of the first compiler error in **11/16**; a random region in **3/16**.

## Result 2: blind remasking gives 0% compile

Prompt-only generation: 0/48 compiled (full regeneration, low-confidence remask, random remask).
Failures are hallucinated APIs/headers (`ImfStream.h`, `ImfMemory.h`, `Imf::MemoryStream`,
`using OpenEXR;`). One-region remasking swaps one hallucination for another. Reference harness compiles
(about 1.9k edges in 20 s). The MVP as specced would give a null result.

## Result 3: compile-error feedback with a shared knowledge base

Loop: syntax-check in the image, remask from the first failing line to the end (fixed-span infill cannot
resize a line), prompt carries the error, up to 3 repair rounds. 8 seeds per arm.

| Arm | Compiled | Final failure |
|---|---|---|
| resample (fresh sample + error, no remask) | 0/8 | nonexistent header |
| feedback (error-localized remask) | 0/8 | nonexistent header |
| + static KB (relevant header shortlist, API decl, rule-based did-you-mean from the real header index) | 0/8 | C++ language errors |
| + shared KB (verified snippet, known-bad headers) | 0/8 | C++ language errors |
| + KB with a verified usage example | **8/8, all in round 0** | none |

Reading:
- A header/decl KB removes header hallucination completely (all included headers exist) but Dream-Coder
  then fails on plain C++ (`using OPENEXR_IMF_NAMESPACE;`, `streambuf` misuse) and does not fix the
  offending line across 3 rounds even with the error in the prompt.
- A verified usage example is what makes it compile, matching CKGFuzzer / PromeFuzz. The harnesses are
  near-copies of the example call: 8/8 compile, coverage 1793 edges (7 of 8 measured; one binary
  produced no `cov:` line and was not investigated) vs 1921 for the reference, because they make one
  `checkOpenEXRFile` call, not the reference's two configurations.
- The shared-KB arm never received an update: no arm-4 run ever passed, and its runs never hit a
  missing-header error, so this design does not yet test cross-sample knowledge sharing.

## Verdict

Possible, with changes. Constraints for the spec: (1) compile-error feedback alone does not work with
this model; (2) verified API usage examples are the necessary ingredient; (3) the remaining research
question is diversity beyond the example, because an example-fed harness collapses to it.
Confidence-guided localization works, but its value is unmeasured once errors are compiler-localized.

## Related work

- OSS-Fuzz-Gen: compile-error repair with rule-based context. https://google.github.io/oss-fuzz/research/llms/target_generation/
- CKGFuzzer: code knowledge graph, API-usage repair KB. https://arxiv.org/pdf/2411.11532
- PromeFuzz: knowledge-driven harness generation with RAG. https://dl.acm.org/doi/10.1145/3719027.3765222
- HarnessAgent: tool-augmented harness construction. https://arxiv.org/html/2512.03420
- PromptFuzz: prompt-mutation fuzz loop. https://arxiv.org/pdf/2312.17677
- RemeDi (self-reflective remasking): https://arxiv.org/html/2509.23653
- Token-to-Mask refinement: https://arxiv.org/html/2604.18738v3
