# dLLM-native Fuzzing Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic trajectory-aware controller that repeatedly micro-fuzzes real Stable-DiffCoder harness candidates, selects a proof candidate from executable evidence, and emits hash-bound JSON artifacts.

**Architecture:** Keep the existing `run_hunt` path as the proof baseline. Add a separate native controller that extracts stock-Stable trajectory features, runs fixed-seed probe stages through the existing Docker fuzz/coverage runners, promotes candidates with a deterministic tournament, and then delegates the selected candidate to the existing proof path.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest, Clang 18, libFuzzer, ASan/UBSan, LLVM source coverage, digest-pinned Docker.

**Spec:** `docs/dllm-native-fuzzing-harness.md`

## Global Constraints

- Preserve every pre-existing dirty-worktree change; do not revert, delete, or commit user-owned edits.
- Use stock Stable-DiffCoder trajectory evidence only; do not claim committed-token remasking.
- Probe and proof evidence remain separate; only the existing proof report may carry the final sanitizer verdict.
- Every probe stage applies the same duration and at least two fixed seeds to every active candidate.
- All controller decisions are deterministic and use sequence numbers, not timestamps.
- No new dependencies.
- Documentation stays in Markdown with Mermaid; machine evidence stays in versioned JSON/JSONL.

---

### Task 1: Deterministic libFuzzer seed contract

**Files:**
- Modify: `src/dllm_agent/fuzz_types.py`
- Modify: `src/dllm_agent/fuzz_runner.py`
- Modify: `tests/test_fuzz_runner.py`

**Interfaces:**
- Produces: `FuzzRunRequest.fuzzer_seed: int` with default `0`.
- Produces: every `run_fuzz()` invocation includes `-seed=<fuzzer_seed>` in `fuzz_argv`.

- [ ] **Step 1: Write the failing test**

Add a focused test that constructs `FuzzRunRequest(fuzzer_seed=4242)`, invokes the real `run_fuzz` adapter with the existing recording runner, and asserts `-seed=4242` is present in the observed fuzzer argv.

- [ ] **Step 2: Run the test to verify RED**

Run: `uv run pytest tests/test_fuzz_runner.py -q`

Expected: failure because `FuzzRunRequest` does not accept `fuzzer_seed` or the argv omits it.

- [ ] **Step 3: Implement the minimum contract**

Add the frozen dataclass field and append exactly `f"-seed={request.fuzzer_seed}"` to the libFuzzer argv. Validate `fuzzer_seed >= 0` at the existing request boundary.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_fuzz_runner.py -q`

Expected: all tests pass and existing requests continue to use seed `0`.

### Task 2: Stock-Stable trajectory features and tournament policy

**Files:**
- Create: `src/dllm_agent/native_models.py`
- Create: `src/dllm_agent/native_scoring.py`
- Create: `tests/test_native_scoring.py`

**Interfaces:**
- Produces: `NativeHuntConfig`, `ProbeStage`, `TrajectoryFeatures`, `ProbeEvent`, `StageEvent`, `SelectionEvent`, `ControllerEvidence`, and `NativeHuntReport` as frozen strict Pydantic schemas.
- Produces: `extract_trajectory_features(candidate: HarnessCandidate) -> TrajectoryFeatures`.
- Produces: `rank_stage(events: tuple[ProbeEvent, ...], features: tuple[TrajectoryFeatures, ...]) -> tuple[str, ...]`.

- [ ] **Step 1: Write trajectory RED tests**

Use hand-calculated two-step candidates to assert committed-token weighted confidence, confidence floor, adjacent-step volatility, commitment progress AUC, and the combined quality. Add a mutation-sensitive assertion showing swapped trajectories change a dynamic tie.

- [ ] **Step 2: Run the trajectory tests to verify RED**

Run: `uv run pytest tests/test_native_scoring.py -q`

Expected: import failure because the native models and scoring functions do not exist.

- [ ] **Step 3: Implement frozen schemas and pure scoring**

Use only observed `DenoisingStep` fields. The quality components are bounded to `[0, 1]`; missing dynamic coverage sorts below observed coverage; stage ranking compares sanitizer observation, build success rate, median branch/line/feature/edge coverage, trajectory quality, median exec/s, and candidate ID in that order.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_native_scoring.py -q`

Expected: all feature and deterministic ranking tests pass.

### Task 3: Probe controller, evidence ledger, and proof handoff

**Files:**
- Create: `src/dllm_agent/native_pipeline.py`
- Modify: `src/dllm_agent/hunt_pipeline.py`
- Modify: `src/dllm_agent/hunt_artifacts.py`
- Create: `tests/test_native_pipeline.py`
- Modify: `tests/test_hunt_pipeline.py`

**Interfaces:**
- Produces: `run_native_hunt(manifest_path, candidate_bank_path, controller_path, source_root, *, fuzz_runner, coverage_runner, artifact_dir) -> NativeHuntReport`.
- Extends: `run_hunt(..., selected_candidate_id: str | None = None, fuzz_seconds: int | None = None)` while preserving default one-shot behavior.
- Produces: deterministic `controller-events.jsonl` whose SHA-256 is embedded in `NativeHuntReport`.

- [ ] **Step 1: Write controller RED tests**

Create a real temporary pinned Git target and two valid candidates. Use narrow in-memory fuzz/coverage adapters that return different executable results. Assert every candidate receives every fixed seed in stage 1, a lower static-score candidate can win from dynamic evidence, dropped candidates receive no later-stage probes, and proof handoff uses the promoted candidate.

- [ ] **Step 2: Write evidence RED tests**

Assert probe directories and proof directory are separate; the JSONL sequence is byte-stable across identical runs; every raw fuzz/coverage log and argv is hash-bound; and an all-build-failed stage raises `HuntSetupError` without a final report.

- [ ] **Step 3: Run controller tests to verify RED**

Run: `uv run pytest tests/test_native_pipeline.py tests/test_hunt_pipeline.py -q`

Expected: failure because `run_native_hunt` and proof selection override do not exist.

- [ ] **Step 4: Implement the minimum orchestration**

Parse the three JSON boundaries once, reuse existing candidate/target validation, stage one pinned archive for probes, run each configured seed from an identical stage-start corpus, preserve each probe through the existing artifact writer, halve according to `ProbeStage.survivors`, and call `run_hunt` with the selected ID and proof duration. Do not add a revision sampler.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_native_pipeline.py tests/test_hunt_pipeline.py -q`

Expected: all controller and baseline proof tests pass.

### Task 4: OpenEXR target profile and prebuilt instrumented image

**Files:**
- Create: `src/dllm_agent/target_profiles.py`
- Modify: `src/dllm_agent/hunt_models.py`
- Modify: `src/dllm_agent/hunt_pipeline.py`
- Modify: `src/dllm_agent/native_pipeline.py`
- Modify: `src/dllm_agent/fuzz_types.py`
- Modify: `src/dllm_agent/fuzz_execution.py`
- Modify: `src/dllm_agent/fuzz_runner.py`
- Modify: `src/dllm_agent/coverage_types.py`
- Modify: `src/dllm_agent/coverage_execution.py`
- Modify: `src/dllm_agent/coverage_runner.py`
- Modify: `tests/test_fuzz_runner.py`
- Modify: `tests/test_coverage_runner.py`
- Modify: `tests/test_hunt_pipeline.py`
- Modify: `tests/test_native_pipeline.py`
- Create: `tools/openexr_runtime/Dockerfile`
- Create: `tools/openexr_runtime/CMakeLists.txt`
- Create: `tools/openexr_runtime/build-fuzz`
- Create: `tools/openexr_runtime/build-coverage`

**Interfaces:**
- Produces: a closed `header-only | openexr-v3.4` target profile carried from
  `HuntManifest` into fuzz and coverage requests.
- Produces: fixed OpenEXR build/report commands selected only by the reviewed profile;
  manifests cannot inject arbitrary argv.
- Produces: a digest-pinned local image with separate ASan and LLVM coverage installs
  of OpenEXR commit `777d231a179de1711d2a942810d9559216ab3f4c`.

- [ ] **Step 1: Write profile RED tests**

Assert the existing profile remains byte-for-byte compatible, the OpenEXR profile invokes
only `/opt/dllm/bin/build-fuzz` and `/opt/dllm/bin/build-coverage`, coverage reports only
the pinned OpenEXR library source root, and unsupported profiles fail before Docker.

- [ ] **Step 2: Run profile tests to verify RED**

Run: `uv run pytest tests/test_fuzz_runner.py tests/test_coverage_runner.py tests/test_hunt_pipeline.py tests/test_native_pipeline.py -q`

Expected: failure because the target profile does not exist.

- [ ] **Step 3: Implement the minimum closed profile**

Keep the header-only argv unchanged. For OpenEXR, call fixed trusted build helpers from the
image and use target-aware LLVM report argv. Pass the literal profile through both baseline
and native controller paths. Do not accept build commands, shell fragments, or link flags
from JSON.

- [ ] **Step 4: Build and smoke-test the pinned target image**

Build the image once from the clean pinned checkout using the Dockerfile and a named local
build context. Both instrumented installs must derive from the same source bytes. Run the
official `openexr_exrcorecheck_fuzzer.cc` through the image's ASan helper and one repository
fixture through the resulting binary before recording the immutable image ID.

### Task 5: CLI, live Stable-DiffCoder candidates, and actual OpenEXR run

**Files:**
- Modify: `src/dllm_agent/cli.py`
- Modify: `tests/test_cli_e2e.py`
- Create: `prompts/openexr-check-harness.md`
- Create: `configs/openexr-v3.4.14-hunt.json`
- Create: `configs/openexr-v3.4.14-native-controller.json`
- Create: `docs/dllm-native-fuzz-results-20260819.md`

**Interfaces:**
- Produces CLI: `dllm-agent native-hunt MANIFEST --candidate-bank PATH --controller PATH --source PATH --artifact-dir PATH --output PATH`.
- Produces live candidate bank: `artifacts/live/openexr-v3.4.14-candidate-bank.json`.
- Produces live report: `artifacts/live/openexr-v3.4.14-native-report.json`.
- Produces live evidence root: `artifacts/live/openexr-v3.4.14-native-run-20260819/`.

- [ ] **Step 1: Write CLI RED tests**

Assert `native-hunt --help` exposes candidate bank, controller, source, artifact directory,
and output options. Add a malformed-controller boundary case that exits `2` and writes no
report or ledger.

- [ ] **Step 2: Implement the CLI and fixed OpenEXR configuration**

Wire the command to `run_native_hunt`. Configure one stage with two fixed seeds and short
micro-fuzz durations, one survivor, and a bounded proof duration. Pin the OpenEXR commit,
SECURITY hash, prompt, Stable-DiffCoder revision, built image ID, in-memory
`checkOpenEXRFile` API, ASan-only proof, and small repository-derived corpus.

- [ ] **Step 3: Generate real candidates and verify the bank**

Use the installed `sample` command with the cached pinned Stable-DiffCoder runtime. Record
generation wall-clock, NFE, trajectory steps, exact prompt hash, and candidate source hashes.
Do not substitute an official or hand-written harness for model output.

- [ ] **Step 4: Run code quality and automated verification**

Run:

```bash
uv run ruff check src tests
uv run basedpyright
uv run pytest -q
```

Expected: all commands exit `0`.

- [ ] **Step 5: Run the actual target through the installed surface**

Run:

```bash
uv run dllm-agent native-hunt configs/openexr-v3.4.14-hunt.json \
  --candidate-bank artifacts/live/openexr-v3.4.14-candidate-bank.json \
  --controller configs/openexr-v3.4.14-native-controller.json \
  --source /home/seory0/.cache/dllm-agent/targets/openexr-v3.4.14-pinned \
  --artifact-dir artifacts/live/openexr-v3.4.14-native-run-20260819 \
  --output artifacts/live/openexr-v3.4.14-native-report.json
```

Expected: exit `0`; both candidates receive the fixed probe schedule; one candidate is promoted; the proof report and hash-bound event ledger are written. A sanitizer result is reported exactly as observed, without a zero-day claim.

- [ ] **Step 6: Record measured results**

Write only measured target pin, candidate/trajectory summary, probe schedule, selection, coverage, sanitizer verdict, command/artifact hashes, and limitations to `docs/dllm-native-fuzz-results-20260819.md`.
