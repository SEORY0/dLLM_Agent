# Task 6 implementation report

## Outcome

Implemented the preregistered OpenEXR one-day paired analyzer, fail-closed campaign
audit, deterministic evidence tree, atomic result transaction, and CLI. The analyzer
consumes the frozen Task 5 campaign boundary and writes only canonical JSON/JSONL and
Markdown.

Owned files:

- `src/dllm_agent/experiments/openexr_one_day/analysis.py`
- `tools/analyze_openexr_one_day.py`
- `tests/test_openexr_one_day_analysis.py`
- `tests/test_openexr_one_day_evidence.py`
- `tests/openexr_one_day_task6_fixtures.py`
- `task-6-report.md`

No Task 1–5 production file was edited.

## Implemented contracts

- Literal trapezoid integration and normalized `T=10800` excess-coverage AUC.
- Exact C0-at-scheduled-`t=0` validation across all five arms, with no model-arm
  coverage credit before atomic corpus release and an exact
  `corpus_release_elapsed_seconds` knot. `bank_ready_elapsed_seconds` is retained as
  shared-bank cost/readiness evidence and is not used as executable readiness.
- Paired log ratios, strict win/loss/tie accounting, exact two-sided sign tests,
  rank-biserial effects, deterministic Holm adjustment, and an exactly 10,000-draw
  paired bootstrap using seed `202608310001` and bounds 249/9749.
- Preregistered GO, NO-GO, INCONCLUSIVE, and selective-revision efficiency
  precedence and thresholds.
- Explicit administrative censoring at 10,800 seconds for coverage-threshold and
  first-proved-cause endpoints. First-proved-cause time is derived from the exact
  matching search `CellEvent`, not post-search proof completion.
- Canonical Task 5 manifest/outcome/raw-report loading, complete 50-row schedule and
  provenance cross-checks, C0/release/deep-parse/input dereferencing, and rejection
  of symlinks, path escapes, unknown/partial sealed artifacts, stale evidence, and
  byte mutations. Every coverage run-argv, profraw, merge-argv, profdata,
  report-argv, and report descriptor is reopened through the producer's canonical
  coverage validator. Every proof-log, fresh-replay argv, and result descriptor is
  likewise reopened and checked against its declared hash.
- Missing resident model-load duration/RSS/VRAM or its byte-exact measurement
  evidence sets `resource_integrity=false`, making GO impossible even when every
  statistical threshold passes.
- Exact evidence-tree preimage
  `{"schema_version":"dllm-openexr-evidence-tree-content/v1","entries":[...]}`
  over sorted POSIX paths, SHA-256, and byte sizes, excluding only
  `results/evidence-tree.sha256`.
- Sibling-directory staging followed by atomic result-directory installation.
  Existing results must be byte-identical on rerun; validation failure leaves no
  partial result directory.

## TDD evidence

Initial RED, before the production module existed:

```text
uv run pytest tests/test_openexr_one_day_analysis.py tests/test_openexr_one_day_evidence.py -q
exit 2
2 collection errors: ModuleNotFoundError for
dllm_agent.experiments.openexr_one_day.analysis
```

Additional RED for the search-event-derived proved-cause endpoint:

```text
uv run pytest tests/test_openexr_one_day_analysis.py -q
exit 2
ImportError: cannot import name 'time_to_first_proved_cause'
```

Final focused GREEN:

```text
uv run pytest tests/test_openexr_one_day_analysis.py tests/test_openexr_one_day_evidence.py -q
39 passed in 226.80s
```

The evidence suite builds a complete synthetic 50-cell campaign through the real
Task 5 `coordinate_campaign` producer boundary. It verifies canonical output,
byte-identical reruns, exact evidence-tree reproduction, one-byte raw mutation
rejection without output changes, unknown-file/symlink rejection, partial-result
rejection, and CLI exit 2 without partial outputs.

Review-fix RED for missing resident-load evidence:

```text
uv run pytest tests/test_openexr_one_day_evidence.py::test_audit_and_analysis_write_only_complete_canonical_results -q
1 failed in 57.46s
expected INCONCLUSIVE, got GO-to-scale
```

The fixture had 10 complete pairs, 10 selective wins, `+131.4%` paired median
uplift, `103.8%` final-coverage retention, and zero percentage-point deep-parse
drop, while all three resident-load fields were absent. The regression therefore
distinguishes statistical success from missing resource evidence.

## Static and policy gates

```text
uv run ruff format --check <five owned Python files>
5 files already formatted

uv run ruff check <five owned Python files> --output-format concise
All checks passed!

uv run basedpyright <five owned Python files>
0 errors, 0 warnings, 0 notes

uv run .../programming/scripts/python/check-no-excuse-rules.py <five owned Python files>
no violations in 5 file(s)

git diff --check
exit 0

git diff --no-index --check /dev/null <each owned untracked Python file>
owned untracked whitespace checks passed
```

## Fresh CLI determinism smoke

Fixture root: `/data/dllm-agent/task6-cli-backingfix-20260902-v1`

This is generated test evidence, not a live OpenEXR campaign result. The CLI was run
twice against the same frozen fixture. Both runs returned exit 0 and the same root:

```text
evidence_tree_sha256=277d3d2d1d6f1580b50a55605725072ed1fd386dac9e31bf3f9e90a73e629404
```

Both runs produced these identical file hashes:

```text
277d3d2d1d6f1580b50a55605725072ed1fd386dac9e31bf3f9e90a73e629404  evidence-tree.sha256
276871f87f2dfbdc75f84f62589e6ccd6edf1b54c3c4b8c8000adc7ba214b59d  paired-summary.json
10684fbeda687ffc1873a1b63317fceafec5974d0681ca260bdb699d4d7b8940  plots-data.json
bb0c27378b178489f9b3aadb7ee3bcc86049a150138fec62b27bbc1874c8e6a9  raw-arm-reports.jsonl
c3b8be9e1d0dc968e89b3bd2e415eb4bb991ec3ef5abfd9b62802f0b6541073b  report.md
```

Both generated reports contain `Classification: INCONCLUSIVE` and
`Resource integrity: false`; the statistical GO thresholds still pass in the
synthetic fixture, so this smoke exercises the missing-telemetry precedence rather
than a weak-statistics branch.

## Exact remaining Task 5 schema patch required

Task 6 cannot honestly set `resource_integrity=true` until Task 5 persists the
already-collected resident-pool telemetry in the campaign outcome. The shared
`ArtifactBackingEvidence`, `ResidentWorkerLoadEvidence`, and
`ResidentModelLoadEvidence` types and their canonical persistence/loader functions
now exist in `evidence_backing.py` and `resident_evidence.py`. The remaining producer
shape is exactly:

```text
CampaignAccounting additions
  resident_model_load: ResidentModelLoadEvidence
  resident_model_load_evidence: ArtifactBackingEvidence
```

The coordinator must call `build_resident_model_load_evidence` on the complete,
GPU-ordered `ResidentWorkerPool.load_telemetry`, persist the canonical bytes at
`provenance/resident-model-load.json`, and place both typed values in
`CampaignAccounting`. `load_complete_campaign_outcome` must reopen the descriptor,
validate its path/size/SHA and aggregate equalities, and bind the ordered worker PIDs
to `CampaignOutcome.resident_worker_pids`. Task 6 already detects both absent schema
fields and present-but-empty values; it keeps `resource_integrity=false` and the
primary result `INCONCLUSIVE` until both are present, then independently reopens the
backing bytes and rejects any mismatch.

Coverage and proof no longer need an envelope-level schema addition. `CoveragePoint`
now carries ordered descriptors for run argv and profraw plus singular descriptors
for merge argv, profdata, report argv, and report. Task 6 calls
`validate_coverage_series_backings(..., reopen=True)` for every audited series.
`ProofVerdict` now carries `proof_log_evidence`, ordered `run_argv_evidence`, and
`result_evidence`; Task 6 validates their count/order/hash and reopens every declared
file below `proof_request.proof_root`. Missing or mutated backing therefore fails the
audit instead of being represented by a fabricated placeholder.

## Compatibility evidence and explicit gaps

The broader Task 1–6 compatibility command was attempted while the Task 5 producer
was being repaired concurrently:

```text
uv run pytest tests/test_openexr_one_day_*.py tests/test_poc_revision_pipeline.py -q
interrupted after 874.62s: 170 passed, 15 failed, at 42%
```

The 15 reported failures were in unowned Task 1–5 coordinator, recovery,
post-search, qualification, and production-factory tests. Failure traces referenced
`coordinator.py`, `recovery.py`, `post_search.py`, and `tools/run_openexr_one_day.py`;
none referenced an owned Task 6 implementation file. The run was stopped after the
failure pattern was established so the parent could integrate and rerun against the
settled Task 5 producer. A full repository pytest run was therefore not claimed.

Task 5's current `CampaignOutcome` schema does not expose resident model-load
duration, RSS, VRAM, or their measurement descriptor. Task 6 reports those values as
unavailable, sets `resource_integrity=false`, forces the primary decision to
`INCONCLUSIVE`, and does not fabricate them or duplicate them into per-arm
efficiency denominators.

No live campaign was executed, and this report makes no live-result, vulnerability,
zero-day, dLLM-over-AR, general-libFuzzer, or 24-hour/7-day ranking claim.
