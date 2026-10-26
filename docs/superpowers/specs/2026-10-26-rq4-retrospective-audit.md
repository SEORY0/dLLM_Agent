# RQ4 pilot — retrospective audit (read-only, post-hoc)

Date: 2026-10-26. Branch `feat/rq4-functional-repair-benchmark`, based on the PR #4 head
(`c932c1b`). This changes nothing about the RQ4 pilot. Its frozen protocol
(`tools/harness_synthesis/rq4_localized_infill_pilot.py`), result files and analysis
(`docs/assets/harness-rq4-pilot-*-20261026.*`) are untouched.

- Tool: `tools/harness_synthesis/rq4_retro_audit.py` (CPU only, using the pinned compile-only image and the Dream tokenizer files; no model weights)
- Output: [`docs/assets/harness-rq4-retro-audit-20261026.json`](../../assets/harness-rq4-retro-audit-20261026.json), with one classified row per eligible trial

Everything below is **post-hoc description**: no p-values are computed for any subgroup.
Each statement is labelled **observed** (computed from the stored artefacts), **inferred**
(a reading of the observed numbers), or **unverified**.

## 1. Consistency checks (observed)

| Check | Result |
|---|---|
| Frozen analysis recomputed with `rq4_analyze._analyze`, compared with the committed `harness-rq4-pilot-analysis-20261026.json` | identical, per file and pooled |
| Every stored harness re-checked with `clang++ -fsyntax-only -ferror-limit=0` in the pinned image (48 references, 192 arm harnesses, and the oracle repairs of §3), verdict vs stored build result | 0 mismatches |
| Stored `compile_error` (first 3 lines) vs the first 3 messages of the full re-check | identical for all 48 |

The pilot's reported numbers reproduce: 10/0/22/16 of 48; H-bi right-context 2 vs 12; H-loc 10 vs 0; anchor 7 vs 13.

## 2. The colleague's two manual counts, re-derived

| Claim | Independent result | Status |
|---|---|---|
| 17 of the 32 nominal right-context trials have only comments after the flagged line | **17/32** (10 InputFile, 7 MultiPart) | confirmed |
| 12 of 48 trials have diagnostics on several distinct lines, all within the nominal right-context stratum | **13/48**, all 13 within it | corrected |

The 13th case is MultiPartInputFile seed 12000112. Its first three errors are all on body line 2 and the fourth is on line 3. The pilot kept only three error lines, so any count based on stored errors finds 12. The un-truncated re-check finds 13.

## 3. Strata by what actually follows the flagged line (observed)

The pilot's `has_right_context` was true whenever any non-blank line followed, including a lone `// Optionally inspect…` comment. Split by comment-aware classification (`body_analysis.right_context`):

| Refined stratum | Trials | Errors on several lines | located-infill compiled | located-suffix compiled | resample compiled |
|---|---:|---:|---:|---:|---:|
| executable code after | 15 | 13 | 0 | 7 | 6 |
| comments only after | 17 | 0 | 5 | 8 | 7 |
| nothing after | 16 | 0 | 5 | 7 | 3 |

For every multi-line trial whose located line is a constructor, an **oracle single-line repair** was compiled: the located line replaced by a correct `OPENEXR_IMF_NAMESPACE::<Class> <var>(stream);`, with everything else unchanged. In **10 of the 13** multi-line trials, errors remain on other lines. No repair confined to the located line could compile them. The other 3 have a located line that is not a constructor, so they were not assessed.

How the H-bi discordance (2 infill-only vs 12 suffix-only) splits:

| Within the nominal right-context stratum | infill only | suffix only | both | neither |
|---|---:|---:|---:|---:|
| code after, errors on several lines (13) | 0 | 5 | 0 | 8 |
| code after, error on one line (2) | 0 | 2 | 0 | 0 |
| comments only after (17) | 2 | 5 | 3 | 7 |

**Inferred.** Only **2 of 32** trials match the case H-bi meant to test: executable code after an error confined to one line. Of the rest, 17 have no right-hand code, so infill and suffix differ only in slack and masking. In 13, the defect spans lines that a one-line infill cannot reach (confirmed for 10). The 12-vs-2 result therefore measures mostly "can this arm edit more than one line", not "does right-hand context help". **No conclusion about bidirectional context can be drawn from this stratum**, in either direction.

## 4. What compile success consisted of (observed; lexical, comment-aware)

Per compiled arm, what happened to the reference body's inspection call on the constructed object:

| Arm | Compiled | inspection commented out | deleted | reference had none, still none | call kept, same method | call kept, method changed | call newly added |
|---|---:|---:|---:|---:|---:|---:|---:|
| located-infill | 10 | 5 | 0 | 5 | 0 | 0 | 0 |
| located-suffix | 22 | 9 | 1 | 7 | 2 | 2 | 1 |
| resample | 16 | 6 | 1 | 3 | 1 | 2 | 3 |

- **Observed.** No located-infill success contains an active call on the constructed object. Five succeeded by turning the flagged inspection line into a comment (all in "nothing after", where the flagged line *was* the inspection). The other five are bodies whose reference never had an active inspection call, so fixing the constructor (`&stream` → `stream`) was the whole repair.
- **Observed.** Every compiled body with an active inspection call is an InputFile body: suffix 5, resample 6. No MultiPartInputFile arm compiled while keeping one.
- **Observed.** 14 of the 48 reference bodies had no active inspection call; they mention one only in comments. The pilot prompt said "and optionally inspect it". Deleting or commenting out an inspection was therefore **not a violation of the RQ4 task**, and is not relabelled as failure here. It only shows that compile rate does not measure the inspection.

Representative pairs (verbatim bodies, `docs/assets/harness-rq4-retro-audit-20261026.json`):

```cpp
// InputFile seed 12000005, located-infill — compiles by commenting the inspection out
// reference:
    OPENEXR_IMF_NAMESPACE::InputFile file(stream);
    // Optionally inspect the header, e.g., file.getHeader()
    auto header = file.getHeader();
// after infill of line 3:
    OPENEXR_IMF_NAMESPACE::InputFile file(stream);
    // Optionally inspect the header, e.g., file.getHeader()
    // Uncomment the line below to inspect the header
    // const auto& header = file.getHeader();

// InputFile seed 12000013, located-suffix — compiles by correcting the API use
// reference:
    Imf::FileHeader header = file.header();     // no such type
// after suffix regeneration:
    file.header();

// InputFile seed 12000001, resample — compiles and adds an inspection the reference lacked
    OPENEXR_IMF_NAMESPACE::InputFile file(stream);
    // Access the header of file
    auto header = file.header();
```

## 5. Fixed-length canvas (observed)

Replacement text is taken line-anchored: the reference before the located line and after its newline must bracket the candidate. That held in 48/48, consistent with the asserted token preservation.

- The replacement is longer than the original line in **47/48** trials by more than 1.3× in characters (median 1.66×). This confirms the earlier post-hoc figure.
- Re-tokenized with Dream's tokenizer, the median replacement length equals the mask length exactly (median replacement − mask = 0). It is shorter than the mask in only 4/48. **Inferred:** the model treats the whole canvas as content to fill; only rarely does it fill the slack with end-of-text (which the pilot dropped).
- What surplus content was detected (lexical; several kinds per trial possible): an added comment in 13, an extra statement in 7, a duplicated type name (such as `InputFile inputFile OPENEXR_IMF_NAMESPACE::InputFile(stream)`) in 6, nothing detected in 24.
- Re-tokenization is an approximation. The generated token ids were not stored, and re-encoding decoded text can differ at token boundaries.

## 6. Failures of non-compiling arms (observed)

| Arm | Not compiling | Some reference error left unresolved | Only new errors |
|---|---:|---:|---:|
| located-infill | 38 | 18 | 20 |
| random-infill | 48 | 47 | 1 |
| located-suffix | 26 | 8 | 18 |
| resample | 32 | 9 | 23 |

Message identity is exact clang text. An error that moved to a different line but kept its text counts as unresolved.

## 7. Confounders this audit confirms

1. **The right-context stratum mostly lacks right-hand code.** Of its 32 trials, 17 have only comments after the flagged line (§3).
2. **Multi-line defects.** 13 trials have errors on several lines, all in that stratum. In 10 of them an oracle one-line repair still fails (§3).
3. **Length and slack.** Infill had a fixed canvas of the window plus 8 tokens, which the model filled. Suffix had all remaining positions. The arms differed in editable span, output length and number of masked tokens together (§5; 22.1 vs 48.5 masked tokens on average, in the pilot's own report).
4. **Unequal decoding budgets.** Resample decoded 60 tokens in 30 steps (two per step). The remask arms decoded one token per step.
5. **The metric.** Compile success counted commenting-out and deletion as repair (§4). The prompt made inspection optional, so the metric and the task both allowed it.
6. **Two examples, no shared defect.** The pilot gives each trial one model-generated defect. The defect kind (constructor vs. API misuse vs. syntax) is not controlled and differs between strata.

## 8. What the RQ4 pilot does and does not establish

- **Established (observed).** Clang's first error mapped onto a model-written body line in 48/48 failing references.
- **Established (observed).** On this pilot's population and metric, located-infill compiled less often than located-suffix (10 vs 22) and resample (16). None of its compiled bodies kept an active inspection call.
- **Not established.** Whether right-hand context helps or hurts a single-line repair: only 2 trials test that case.
- **Not established.** Whether infill with a correct span length repairs API misuse: the canvas was never sized to the target.
- **Not established.** Anything about functional repair: inspection was optional and never measured.
- **Not established.** Anything about dLLM vs AR: no AR model was run.
