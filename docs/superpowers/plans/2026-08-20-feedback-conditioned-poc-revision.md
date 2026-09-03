# Feedback-conditioned PoC Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a Stable-DiffCoder runtime that materializes executable structured OpenEXR PoC snapshots, feeds short fuzz feedback back to the controller, re-masks only attributed committed field slots, and independently proves sanitizer findings.

**Architecture:** A typed fixed-grammar EXR particle uses one constrained numeric model token per variable field and deterministic catalogs/materialization for raw bytes. A resident Stable-DiffCoder worker fills or reopens only selected slots; a compile-once OpenEXR probe runner evaluates snapshots asynchronously, while a deterministic epoch router attributes reward, selects the next field group, deduplicates work, and hands only reproducible crashes to a separate proof runner.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, PyTorch 2.5.1, Transformers 5.3.0, Stable-DiffCoder-8B-Instruct, Docker, Clang/libFuzzer, ASan, LLVM coverage, pytest, Ruff, basedpyright.

**Spec:** `docs/superpowers/specs/2026-08-20-feedback-conditioned-poc-revision-design.md`

## Global Constraints

- Human-facing documents remain Markdown/Mermaid only; do not create PDF or DOCX.
- Preserve `dllm-native-hunt-report/v1`; the new report is `dllm-poc-revision-report/v1`.
- Call the new model path `selective-revision/v1`, not stock Stable-DiffCoder remasking.
- OpenEXR stays pinned to commit `777d231a179de1711d2a942810d9559216ab3f4c` and the existing SECURITY/tree/image boundary.
- Harness and target are built once per campaign; snapshot probes vary only input bytes.
- Frozen grammar/anchor tokens and materialized anchor bytes must remain bit-identical.
- Probe sanitizer signals never become a vulnerability claim without an independent ASan proof.
- All decisions are deterministic over fixed inputs even when execution completes asynchronously.
- Every new Python module stays below 250 pure LOC; strict types, no `Any`, `object`, casts, type-ignore, broad exception handling, or shell interpolation.
- No new dependency unless existing stdlib/Pydantic/PyTorch surfaces cannot implement the contract.
- The worktree contains user-owned uncommitted work. Do not commit or revert; each task writes an evidence report instead.

---

## File Structure

New focused modules:

- `src/dllm_agent/targets/openexr/particle.py`: typed field catalogs, particle decoding, deterministic minimal scanline EXR materialization, field-to-byte spans.
- `src/dllm_agent/revision/models.py`: strict configs, snapshot/probe/attribution/remask/event/report schemas.
- `src/dllm_agent/revision/policy.py`: reward deltas, field credit, UCB remask decision, deterministic arm behavior.
- `src/dllm_agent/revision/artifacts.py`: byte-stable JSONL and hash-bound artifact persistence.
- `src/dllm_agent/targets/openexr/probe.py`: compile-once Docker probe campaign and input-only libFuzzer execution.
- `src/dllm_agent/targets/openexr/proof.py`: exact-input ASan reproduction and proof evidence.
- `src/dllm_agent/revision/pipeline.py`: fixed-epoch particle orchestration, cache/dedup, promotion and proof handoff.
- `src/dllm_agent/runtime/revision_client.py`: typed resident worker process protocol.
- `tools/selective_revision_payloads.py`: strict JSON request/response parsing for the model environment without project dependencies.
- `tools/selective_revision_sampler.py`: constrained field-slot initial denoising and committed-token reopening.
- `tools/selective_revision_worker.py`: one-load JSONL resident model worker.
- `tools/openexr_runtime/poc_probe.cc`: fixed OpenEXR in-memory fuzz harness built into the target image.
- `tools/run_poc_revision_experiment.py`: paired arm runner and fixed seed schedule.
- `tools/analyze_poc_revision_experiment.py`: paired summaries, bootstrap CI, effect sizes, go/no-go verdict.

Tests mirror responsibility:

- `tests/test_openexr_particle.py`
- `tests/test_poc_revision_models.py`
- `tests/test_poc_revision_policy.py`
- `tests/test_poc_revision_artifacts.py`
- `tests/test_poc_probe.py`
- `tests/test_poc_proof.py`
- `tests/test_poc_revision_pipeline.py`
- `tests/test_selective_revision_sampler.py`
- `tests/test_revision_client.py`
- `tests/test_poc_revision_cli.py`
- `tests/test_poc_revision_experiment.py`

---

### Task 1: Typed OpenEXR particle and deterministic materializer

**Files:**
- Completed: `src/dllm_agent/targets/openexr/particle.py`
- Create: `tests/test_openexr_particle.py`
- Create: `fixtures/poc_revision/owned_exr_particle_v1.json`

**Interfaces:**
- Produces: `FieldName`, `ParticleMode`, `ExrParticle`, `FieldByteSpan`, `MaterializedSnapshot`.
- Produces: `decode_particle(selectors: tuple[int, ...], mode: ParticleMode) -> ExrParticle`.
- Produces: `materialize_particle(particle: ExrParticle) -> MaterializedSnapshot`.
- Snapshot invariants: magic/version anchors fixed; deterministic bytes/hash; every variable field maps to a byte span; derived offsets/sizes are not model-controlled.

- [ ] **Step 1: Write particle/materializer RED tests**

```python
def test_materialize_particle_keeps_openexr_anchors_and_derives_offsets() -> None:
    particle = decode_particle((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "validity-preserving")

    snapshot = materialize_particle(particle)

    assert snapshot.content[:8] == bytes.fromhex("762f310102000000")
    assert snapshot.sha256 == sha256_bytes(snapshot.content)
    assert {span.field for span in snapshot.field_spans} == set(FieldName)
    assert all(span.end <= len(snapshot.content) for span in snapshot.field_spans)


def test_boundary_breaking_changes_only_declared_derived_condition() -> None:
    valid = materialize_particle(decode_particle((0,) * 10, "validity-preserving"))
    broken = materialize_particle(decode_particle((0,) * 10, "boundary-breaking"))

    assert valid.anchor_sha256 == broken.anchor_sha256
    assert valid.content != broken.content
    assert broken.fault_field == "chunk_offset"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_openexr_particle.py -q`
Expected: collection fails because `dllm_agent.targets.openexr.particle` does not exist.

- [ ] **Step 3: Implement the minimal typed particle**

Use frozen dataclasses or existing `FrozenModel`; use exactly ten selector slots:

```python
type FieldName = Literal[
    "origin_x",
    "origin_y",
    "width",
    "height",
    "compression",
    "pixel_type",
    "line_order",
    "x_sampling",
    "payload_pattern",
    "chunk_offset",
]
type ParticleMode = Literal["validity-preserving", "boundary-breaking"]


@dataclass(frozen=True, slots=True)
class ExrParticle:
    selectors: tuple[int, int, int, int, int, int, int, int, int, int]
    mode: ParticleMode


@dataclass(frozen=True, slots=True)
class MaterializedSnapshot:
    particle: ExrParticle
    content: bytes
    sha256: str
    anchor_sha256: str
    field_spans: tuple[FieldByteSpan, ...]
    fault_field: FieldName | None
```

Materialize a minimal single-channel, uncompressed scanline EXR. Derive header length, offset-table position, chunk offset, chunk byte count and pixel bytes. Catalogs include small dimensions and boundary origins; cap emitted payload at 64 KiB. `boundary-breaking` first builds valid bytes, then corrupts only the catalog-selected derived condition.

- [ ] **Step 4: Verify GREEN and real parser acceptance**

Run: `uv run pytest tests/test_openexr_particle.py -q`
Expected: PASS.

Then replay one validity-preserving snapshot through the pinned OpenEXR target image with the existing in-memory `checkOpenEXRFile` harness. Expected: process exits normally with no ASan marker.

- [ ] **Step 5: Quality gate and task report**

Run:

```sh
uv run ruff check src/dllm_agent/targets/openexr/particle.py tests/test_openexr_particle.py
uv run basedpyright src/dllm_agent/targets/openexr/particle.py tests/test_openexr_particle.py
```

Write `.superpowers/sdd/2026-08-20-poc-revision/task-1-report.md` with RED/GREEN commands, snapshot hashes and parser smoke evidence.

---

### Task 2: Revision schemas, deterministic ledger and field policy

**Files:**
- Completed: `src/dllm_agent/revision/models.py`
- Completed: `src/dllm_agent/revision/policy.py`
- Completed: `src/dllm_agent/revision/artifacts.py`
- Create: `tests/test_poc_revision_models.py`
- Create: `tests/test_poc_revision_policy.py`
- Create: `tests/test_poc_revision_artifacts.py`

**Interfaces:**
- Produces: `RevisionConfig`, `ParticleState`, `SnapshotEvidence`, `ProbeFeedback`, `FieldAttribution`, `RemaskDecision`, `RevisionEvent`, `PocRevisionReport`.
- Produces: `attribute_feedback(parent: ProbeFeedback, child: ProbeFeedback, changed: tuple[FieldName, ...]) -> tuple[FieldAttribution, ...]`.
- Produces: `decide_remask(...) -> RemaskDecision`.
- Produces: `revision_events_bytes(...) -> bytes` and `revision_events_sha256(...) -> str`.

- [ ] **Step 1: Write strict-schema RED tests**

```python
def test_revision_config_rejects_budget_that_cannot_form_an_epoch() -> None:
    with pytest.raises(ValidationError):
        RevisionConfig(
            schema_version="dllm-poc-revision-config/v1",
            particle_count=4,
            epochs=3,
            initial_steps=128,
            revision_steps=16,
            probe_seconds=3,
            fuzzer_seeds=(7,),
            proof_seconds=30,
        )


def test_feedback_changes_selected_remask_field() -> None:
    chunk = decide_remask(_history(stalled_field="chunk_offset"), epoch=1)
    window = decide_remask(_history(stalled_field="width"), epoch=1)
    assert chunk.fields == ("chunk_offset",)
    assert window.fields == ("width",)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_revision_models.py tests/test_poc_revision_policy.py tests/test_poc_revision_artifacts.py -q`
Expected: collection fails for missing revision modules.

- [ ] **Step 3: Implement schemas and lexicographic reward policy**

`ProbeFeedback` records build/run status, sanitizer presence, edge/feature coverage, target branch count when available, exec/s, corpus artifacts, `wall_ms`, and exact log/argv hashes. Reward ordering is sanitizer, new target branches, rare/new edges, value-profile features, validity, exec/s, then trajectory uncertainty. Field credit is a controlled one-field delta. UCB uses deterministic `sqrt(2 * log(total_trials) / field_trials)` and field-name tie-breaking.

Event kinds are a strict literal union:

```python
type RevisionEventKind = Literal[
    "particle_started",
    "snapshot_materialized",
    "probe_completed",
    "field_attribution",
    "remask_decided",
    "revision_completed",
    "particle_promoted",
    "particle_dropped",
    "proof_completed",
]
```

- [ ] **Step 4: Implement byte-stable evidence and verify GREEN**

Serialize one JSON object per line with monotonic sequence and no decision timestamp. Assert two identical runs produce identical bytes and the embedded SHA-256 equals raw ledger bytes.

Run: `uv run pytest tests/test_poc_revision_models.py tests/test_poc_revision_policy.py tests/test_poc_revision_artifacts.py -q`
Expected: PASS.

- [ ] **Step 5: Quality gate and task report**

Run Ruff, basedpyright and focused tests; write `task-2-report.md`. Do not change existing native report models.

---

### Task 3: Compile-once OpenEXR snapshot probe runner

**Files:**
- Create: `src/dllm_agent/targets/openexr/probe.py`
- Create: `tests/test_poc_probe.py`
- Create: `tools/openexr_runtime/poc_probe.cc`
- Modify: `tools/openexr_runtime/Dockerfile`
- Modify: `tools/openexr_runtime/CMakeLists.txt`

**Interfaces:**
- Produces: `PreparedProbe`, `ProbeRequest`, `ProbeResult`, `prepare_openexr_probe(...) -> ContextManager[PreparedProbe]`.
- `PreparedProbe.run(request: ProbeRequest) -> ProbeResult` never invokes CMake.
- Cache key: target image, harness hash, snapshot hash, fuzzer seed, duration, max length.

- [ ] **Step 1: Write compile-once and cache RED tests**

```python
def test_prepared_probe_builds_once_and_runs_distinct_snapshot_bytes() -> None:
    runner = RecordingRunner()
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        first = probe.run(_request(b"first", seed=7))
        second = probe.run(_request(b"second", seed=11))

    assert runner.commands_named("cmake") == 0
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert runner.fuzz_inputs == [b"first", b"second"]


def test_prepared_probe_deduplicates_exact_request() -> None:
    with prepare_openexr_probe(_config(), runner=runner) as probe:
        first = probe.run(_request(b"same", seed=7))
        second = probe.run(_request(b"same", seed=7))
    assert first == second.model_copy(update={"cache_hit": True})
    assert runner.fuzz_call_count == 1
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_probe.py -q`
Expected: collection fails for missing `targets.openexr.probe`.

- [ ] **Step 3: Add fixed harness and hardened command construction**

`poc_probe.cc` calls both `checkOpenEXRFile(..., runCoreCheck=false)` and `runCoreCheck=true` on input bytes. Build it once while creating the pinned image and bind its source/binary hash in image labels. A campaign never invokes CMake. Probe argv includes `-use_value_profile=1`, fixed `-seed`, bounded `-max_total_time`, `-timeout`, `-max_len`, ASan environment and existing Docker hardening (`network none`, read-only root, UID/GID 65534, cap drop, no-new-privileges).

- [ ] **Step 4: Implement cheap feedback parsing**

Parse libFuzzer final units, `cov`, `ft`, corpus count, exec/s, ASan marker and crash artifacts. Do not run `run_coverage` in `PreparedProbe.run`; expose a separate promoted-snapshot coverage method.

- [ ] **Step 5: Verify GREEN and live build-count smoke**

Run focused tests, rebuild the image once from the same clean pinned OpenEXR source, inspect labels/digest, then run two snapshots. Assert zero campaign CMake calls, two fuzz invocations, and no ASan marker for the valid fixture. Report the one-time image build separately.

- [ ] **Step 6: Quality gate and task report**

Run `sh -n` where applicable, Ruff, basedpyright and focused tests; record immutable image ID and build/run hashes in `task-3-report.md`.

---

### Task 4: Deterministic revision controller with sampler double

**Files:**
- Create: `src/dllm_agent/revision/pipeline.py`
- Create: `tests/test_poc_revision_pipeline.py`

**Interfaces:**
- Consumes: Task 1 materializer; Task 2 schemas/policy/ledger; Task 3 `PreparedProbe`.
- Defines protocol: `RevisionSampler.initial(request: InitialParticleRequest) -> RevisionResult` and `RevisionSampler.revise(request: RevisionRequest) -> RevisionResult`.
- Produces: `run_poc_revision(config, sampler, probe, proof_runner, artifact_dir) -> PocRevisionReport`.

- [ ] **Step 1: Write controller RED tests**

```python
def test_controller_remasks_attributed_slot_and_probes_new_snapshot() -> None:
    sampler = DeterministicSampler(
        (0,) * 10,
        revisions={"width": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)},
    )
    probe = FeedbackProbe(stalled_field="width")

    report = run_poc_revision(_config(epochs=1), sampler, probe, _proof(), tmp_path)

    decision = report.remask_decisions[0]
    assert decision.fields == ("width",)
    assert sampler.requests[0].remask_bitmap != 0
    assert report.snapshots[1].sha256 != report.snapshots[0].sha256
    assert probe.snapshot_hashes[-1] == report.snapshots[1].sha256


def test_probe_sanitizer_does_not_override_independent_failed_proof() -> None:
    report = run_poc_revision(
        _config(), _sampler(), SanitizerProbe(), ProofRunner(oracle="not_observed"), tmp_path
    )
    assert report.probe_feedback[-1].sanitizer_observed is True
    assert report.proof.oracle_status == "not_observed"
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_revision_pipeline.py -q`
Expected: collection fails for missing pipeline.

- [ ] **Step 3: Implement fixed epochs, dedup and arm semantics**

Arms are strict literals:

```python
type ExperimentArm = Literal[
    "libfuzzer-only",
    "structured-one-shot",
    "feedback-no-remask",
    "selective-revision",
    "pseudo-feedback",
]
```

All particles start from fixed seeds. One-shot never calls `revise`; no-remask can promote or full-resample within the same NFE budget; selective reopens the chosen field group; pseudo-feedback applies a no-fixed-point rotation before attribution. Use AnyIO task groups and a capacity limiter to overlap independent particle probes. Blocking Docker probes run through `anyio.to_thread.run_sync`. Decision epochs await the fixed required result set and sort by particle ID, not completion order. `run_poc_revision` is the synchronous CLI boundary and calls an internal typed async epoch function through `anyio.run`.

- [ ] **Step 4: Add async-order, cache and failure tests**

Assert out-of-order probe completion produces byte-identical ledger; duplicate snapshot skips sampler/probe; invalid materialization never calls probe; all-particle failure prevents proof; selected remask span exceeds no configured field span.

- [ ] **Step 5: Verify GREEN and existing native regressions**

Run:

```sh
uv run pytest tests/test_poc_revision_pipeline.py -q
uv run pytest tests/test_native_pipeline.py tests/test_native_scoring.py -q
```

Expected: all PASS.

- [ ] **Step 6: Quality gate and independent task review**

Run Ruff/basedpyright, write `task-4-report.md`, then dispatch a read-only reviewer for proof separation, determinism and cache correctness.

---

### Task 5: Real committed-token selective revision and resident model worker

**Files:**
- Create: `tools/selective_revision_payloads.py`
- Create: `tools/selective_revision_sampler.py`
- Create: `tools/selective_revision_worker.py`
- Create: `src/dllm_agent/runtime/revision_client.py`
- Create: `tests/test_selective_revision_sampler.py`
- Create: `tests/test_revision_client.py`

**Interfaces:**
- Consumes Task 4 sampler protocol.
- Worker commands: `initialize`, `revise`, `shutdown` over one JSON object per line.
- `RevisionRequest` contains parent suffix token IDs, slot positions, remask fields/bitmap, frozen bitmap, feedback text/hash, steps, seed and mode.
- `RevisionResult` returns selected slot token IDs, token-state SHA-256, transfer evidence, NFE, forward calls, GPU milliseconds and cache hit.

- [ ] **Step 1: Write pure sampler RED tests with a tiny model double**

```python
def test_revise_reopens_committed_slot_and_preserves_frozen_tokens() -> None:
    parent = _state(slot_tokens=(101, 102, 103), frozen=(True, False, True))
    result = revise_slots(_model(select_token=909), parent, remask_positions=(1,), steps=4)
    assert result.slot_tokens == (101, 909, 103)
    assert result.remasked_before == (1,)
    assert result.frozen_unchanged is True


def test_revision_recomputes_from_earliest_changed_block_only() -> None:
    result = revise_slots(_recording_model(), _state_in_blocks(), (9,), steps=4)
    assert result.invalidated_from_block == 2
    assert result.forward_blocks == (0, 1, 2)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_selective_revision_sampler.py tests/test_revision_client.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement fixed-grammar slot construction and constrained logits**

Build the generated suffix from frozen key/separator token IDs plus one numeric selector token per field. Precompute tokenizer IDs whose decoded text matches `[0-9]+`; mask logits at slot positions to this set. Initial generation masks all ten slots. Revision starts from parent suffix IDs, sets only selected committed slot positions to mask ID, and never writes frozen positions.

- [ ] **Step 4: Implement block-aware revision**

Adapt the pinned upstream block loop without modifying cached remote code. Invalidate cache from the earliest changed block. Process affected blocks in order, using the existing block-causal mask and `_get_transfer_index`; record each transfer and final token-state hash. Require actual before/after committed token difference for a successful revision or record `unchanged_revision`.

- [ ] **Step 5: Implement resident JSONL worker/client**

The worker verifies the pinned remote code SHA, loads model/tokenizer once, reports ready provenance, serves multiple initial/revise requests, and shuts down cleanly. The client uses fixed argv, minimal environment, bounded line size, request IDs, timeouts and exact payload hashes. No shell.

- [ ] **Step 6: Verify GREEN with live Stable-DiffCoder**

First run pure doubles. Then load the real model once on `cuda:0`, initialize four particles, revise an attributed field for one particle, and assert:

- selected parent committed token became mask internally;
- at least one selected slot token or selector value changed;
- all frozen slot/grammar token IDs are identical;
- model load count is one;
- result includes NFE/forward/GPU-time evidence.

- [ ] **Step 7: Quality gate and independent sampler review**

Run Ruff/basedpyright/focused tests. Write `task-5-report.md` with real request/result hashes and dispatch a read-only reviewer focused on honest stock-vs-new-sampler claims and frozen-token correctness.

---

### Task 6: Exact-input ASan proof and minimization boundary

**Files:**
- Create: `src/dllm_agent/targets/openexr/proof.py`
- Create: `tests/test_poc_proof.py`

**Interfaces:**
- Produces: `ProofRequest`, `ProofResult`, `prove_snapshot(request, runner) -> ProofResult`.
- Proof input is exact snapshot/crash bytes, not a mutable campaign corpus.
- Final status: `proved | not_reproduced | crash_without_sanitizer`.

- [ ] **Step 1: Write proof RED tests**

```python
def test_probe_signal_requires_new_process_reproduction() -> None:
    result = prove_snapshot(_request(probe_sanitizer=True), runner=_clean_runner())
    assert result.oracle_status == "not_reproduced"


def test_proved_result_binds_minimized_input_and_asan_log() -> None:
    result = prove_snapshot(_request(), runner=_asan_runner())
    assert result.oracle_status == "proved"
    assert result.input_sha256 == sha256_bytes(result.input_bytes)
    assert result.sanitizer == "AddressSanitizer"
    assert result.proof_log_sha256 == sha256_bytes(result.proof_log)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_proof.py -q`
Expected: missing module failure.

- [ ] **Step 3: Implement separate reproduction and minimization**

Run exact bytes in a new ASan process at least twice with the fixed proof seed. Only consistent sanitizer class/stack signature is `proved`. Perform structured field minimization first, then libFuzzer `-minimize_crash=1`; re-run the minimized bytes. Persist original/minimized byte hashes, logs, argv, target/image pins and root-cause signature.

- [ ] **Step 4: Verify GREEN and negative OpenEXR proof smoke**

Run tests and one known-clean OpenEXR snapshot. Expected: `not_reproduced`, no crash artifact and no vulnerability claim.

- [ ] **Step 5: Quality gate and task report**

Run Ruff/basedpyright/focused tests and write `task-6-report.md`.

---

### Task 7: Installed CLI and live OpenEXR revision smoke

**Files:**
- Modify: `src/dllm_agent/cli.py`
- Create: `configs/openexr-v3.4.14-poc-revision.json`
- Create: `prompts/openexr-poc-revision.md`
- Create: `tests/test_poc_revision_cli.py`
- Modify: `README.md`

**Interfaces:**
- Adds installed command `dllm-agent poc-revision`.
- Inputs: pinned target manifest, revision config, source checkout, runtime Python, cache dir, artifact dir, output report.
- Output: atomically written `dllm-poc-revision-report/v1` JSON and evidence tree.

- [ ] **Step 1: Write installed CLI RED tests**

```python
def test_installed_poc_revision_help_exposes_required_boundaries() -> None:
    result = subprocess.run((CLI, "poc-revision", "--help"), capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--config", "--source", "--runtime-python", "--artifact-dir", "--output"):
        assert option in result.stdout


def test_malformed_revision_config_writes_no_report_or_ledger(tmp_path: Path) -> None:
    result = _run_cli(config="{")
    assert result.returncode == 2
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "run").exists()
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_revision_cli.py -q`
Expected: CLI command absent.

- [ ] **Step 3: Wire CLI without widening errors**

Parse strict JSON before creating artifacts or starting Docker/model. Use existing atomic `write_model`; map only known boundary/setup/value errors to exit 2. Keep all paths server-side and fixed argv.

- [ ] **Step 4: Run live smoke**

Use four particles, one initial epoch plus one revision epoch, two fixed fuzzer seeds, 1-second probes and a 10-second proof budget. Required evidence:

- four distinct snapshot byte hashes or an explicit diversity failure;
- at least one real committed-field remask request;
- at least one changed snapshot hash after revision;
- one-time image build count one and per-campaign harness build count zero;
- no full LLVM coverage in every snapshot probe;
- separate proof report;
- no vulnerability claim without reproduced ASan.

- [ ] **Step 5: Verify GREEN and full regression**

Run installed CLI tests, full pytest, Ruff, format check and basedpyright. Write `task-7-report.md` and independently review live hashes/logs.

---

### Task 8: Paired experiment runner and research analysis

**Files:**
- Create: `tools/run_poc_revision_experiment.py`
- Create: `tools/analyze_poc_revision_experiment.py`
- Create: `tests/test_poc_revision_experiment.py`
- Create: `configs/openexr-v3.4.14-poc-experiment.json`
- Create: `docs/poc-revision-experiment-protocol.md`

**Interfaces:**
- Arms: libFuzzer-only, structured-one-shot, feedback-no-remask, selective-revision, pseudo-feedback; current native tournament is a separately labeled non-isomorphic engineering reference.
- Produces one immutable manifest/report/evidence tree per arm/repetition plus `paired-summary.json` and Markdown tables.

- [ ] **Step 1: Write experiment/accounting RED tests**

```python
def test_seed_schedule_is_paired_and_pseudo_feedback_has_no_fixed_points() -> None:
    schedule = build_schedule(repetitions=5, particles=4)
    assert schedule[0].particle_seeds == (202608200000, 202608200001, 202608200002, 202608200003)
    assert all(a != b for a, b in zip(range(4), schedule[0].pseudo_permutation, strict=True))


def test_analysis_reports_paired_effect_and_censored_proof_time() -> None:
    summary = analyze(_synthetic_reports())
    assert summary.contrasts["selective-revision_vs_pseudo-feedback"].paired_n == 5
    assert summary.proof_time.right_censored >= 1
    assert summary.go_no_go in {"go", "no-go", "inconclusive"}
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/test_poc_revision_experiment.py -q`
Expected: missing experiment functions.

- [ ] **Step 3: Implement fixed schedule and budget accounting**

Default pilot:

```text
repetitions = 5
particles = 4
initial NFE = 128 per particle
revision epochs = 3
revision NFE = 16 per particle per epoch
fuzzer seeds = 2 per snapshot
probe = 3 seconds
proof = 30 seconds
max dLLM NFE = 704 per dLLM arm
requested fuzz seconds = 126 per arm
```

Record inclusive wall-clock, CPU/GPU-seconds, NFE, forward calls, model loads, build/probe/coverage/proof calls and cache/dedup hits. Report setup image/model costs separately, but never omit campaign overhead.

- [ ] **Step 4: Implement analysis without adding a statistics dependency**

Use stdlib deterministic bootstrap with a fixed analysis seed for median/mean paired difference and 95% percentile CI. Compute paired win/loss/tie, rank-biserial effect and right-censored proof-time table. Primary mechanism contrast is selective revision vs pseudo-feedback; efficacy contrasts include one-shot and no-remask. Label native tournament as a different harness-generation surface.

- [ ] **Step 5: Run deterministic fixture experiment**

Run 30 cheap repetitions across all logical arms using deterministic fake probe/sampler. Expected: selective revision reaches the owned target field; pseudo-feedback and no-remask do not receive the same credited field under equal accounting. This validates the analyzer, not model efficacy.

- [ ] **Step 6: Run OpenEXR live smoke then five-repetition pilot**

First run one paired repetition and audit artifact volume. If smoke gates pass and disk budget remains safe, run five paired repetitions. Stop an arm only at its predeclared budget. Store artifacts outside the Git worktree under a validated explicit experiment root, with summary/report hashes copied into repository Markdown/JSON.

- [ ] **Step 7: Quality gate and experiment audit**

Run focused tests, full tests, Ruff, basedpyright and evidence-tree hash verification. Independently audit equal budgets, raw logs, ASan markers, cache counts and statistical calculations.

---

### Task 9: Research result document and final comparison

**Files:**
- Create: `docs/poc-revision-openexr-results-20260820.md`
- Modify: `docs/dllm-native-fuzzing-harness.md`
- Modify: `docs/experiment-protocol.md`
- Modify: `.superpowers/sdd/2026-08-20-poc-revision/progress.md`

**Interfaces:**
- Consumes immutable Task 8 summaries and raw evidence hashes.
- Produces Markdown/Mermaid-only research result with measured/inferred/future labels.

- [ ] **Step 1: Write measured tables from machine output**

Include per-arm and paired results for branch/edge/feature coverage, time, NFE/GPU/CPU, build count, cache/dedup, plateau escape and independent proof. Do not manually transcribe values without checking report hashes.

- [ ] **Step 2: State the causal claim boundary**

The strongest allowed mechanism statement requires selective revision to beat pseudo-feedback under paired equal budgets. One live smoke or a confidence interval crossing zero is `inconclusive`. No sanitizer proof means no vulnerability/zero-day claim.

- [ ] **Step 3: Add architecture/result Mermaid diagrams**

Show snapshot→probe→attribution→remask→revision, and a compact paired comparison chart/table. Do not create PDF/DOCX.

- [ ] **Step 4: Final full verification and manual QA**

Run:

```sh
uv run ruff format --check src tests tools
uv run ruff check src tests tools
uv run basedpyright
uv run pytest -q
git diff --check
uv run dllm-agent poc-revision --help
```

Recompute every published report, ledger, snapshot and evidence-tree hash. Search all probe/proof logs for ASan/UBSan markers and ensure the document matches the independent proof verdict.

- [ ] **Step 5: Independent completion review**

Review implementation, live behavior, evidence and research claims. Mark complete only when no required work remains; record limitations if the five-repetition pilot cannot run or remains statistically inconclusive.
