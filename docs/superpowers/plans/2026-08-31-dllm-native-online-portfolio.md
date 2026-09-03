# dLLM-native Online Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current child-replacement PoC revision loop into an online, cause-aware parent-child portfolio that selectively revises only plateaued lineages through one resident dLLM worker.

**Architecture:** Preserve the historical multitarget runner and its artifacts, but make canaries evaluation-only. Extend the installed `revision` runtime with pure archive, proof-cause, portfolio, and trajectory-policy modules; then compose them in `run_poc_revision` while retaining existing Protocol seams and deterministic JSONL evidence.

**Tech Stack:** Python 3.12, Pydantic v2, anyio, pytest, Ruff, BasedPyright, Stable-DiffCoder resident JSONL worker, ASan exact replay.

**Spec:** `docs/superpowers/specs/2026-08-31-dllm-native-online-portfolio-design.md`

## Global Constraints

- Preserve historical experiment artifacts and serialized `canary_triggered` fields.
- No new dependency and no PDF/DOCX output.
- Use exact two-replay ASan proof before any proved cause classification.
- Keep model residency at one worker lifecycle per run.
- Controller decisions are deterministic and timestamp-free.
- Keep target bug patches and Magma canary IDs out of generation prompts and policies.

---

### Task 1: Canary-neutral historical reward

**Files:**
- Modify: `src/dllm_agent/experiments/multitarget/candidate_bank.py`
- Modify: `tests/test_multitarget_candidate_bank.py`

**Interfaces:**
- Produces: `remask_plan(...)` whose result is invariant to `canary_triggered` when all executable metrics are fixed.

- [x] **Step 1: Write the failing test**

Construct candidates whose selector association exists only through canary values and assert that changing only those values does not change the selected field.

- [x] **Step 2: Run RED**

```bash
uv run pytest tests/test_multitarget_candidate_bank.py -q
```

Expected: the canary-only association selects the wrong field.

- [x] **Step 3: Implement the minimum change**

Remove only the canary term from `_reward`; retain `ProbeObservation.canary_triggered`, probe parsing, pseudo-feedback, and artifacts.

- [x] **Step 4: Run GREEN**

```bash
uv run pytest tests/test_multitarget_candidate_bank.py tests/test_multitarget_probe_runner.py -q
```

### Task 2: Parent-child Pareto archive

**Files:**
- Create: `src/dllm_agent/revision/archive.py`
- Create: `tests/test_poc_revision_archive.py`

**Interfaces:**
- Produces: `ProofNovelty`, `ArchiveCandidate`, `ArchiveSelection`.
- Produces: `select_parent_child(parents, children, *, capacity, protected_ids) -> ArchiveSelection`.
- Produces: deterministic non-dominated selection with protected anchors and diversity tie-breaking.

- [x] **Step 1: Write archive RED tests**

Use literal feedback vectors to prove an inferior child cannot replace its parent, a trade-off keeps both when capacity permits, and protected anchors survive truncation.

- [x] **Step 2: Run RED**

```bash
uv run pytest tests/test_poc_revision_archive.py -q
```

Expected: import failure because `revision.archive` does not exist.

- [x] **Step 3: Implement the pure archive**

Compare proof novelty, branches, rare/new edges, value features, validity and exec/s while minimizing model/probe cost. Use snapshot hash for deterministic final ties.

- [x] **Step 4: Run GREEN**

```bash
uv run pytest tests/test_poc_revision_archive.py -q
```

### Task 3: Online proof cause archive and family caps

**Files:**
- Create: `src/dllm_agent/revision/proof_archive.py`
- Modify: `src/dllm_agent/targets/openexr/proof.py`
- Create: `tests/test_poc_revision_proof_archive.py`
- Modify: `tests/test_poc_proof.py`

**Interfaces:**
- Produces: `OnlineProofVerdict`, `CauseArchive`, `classify_online_proof(...)`.
- Extends: `OpenExrProofRunner.run(...) -> ProofResult`, retaining the existing summary properties.
- Cause key: canonical hash of target commit, sanitizer class and normalized root signature.

- [x] **Step 1: Write proof archive RED tests**

Assert the first stable cause is new, a different input with the same normalized cause is duplicate, failed proof increments a provisional family, and the configured cap blocks another proof attempt.

- [x] **Step 2: Run RED**

```bash
uv run pytest tests/test_poc_revision_proof_archive.py tests/test_poc_proof.py -q
```

Expected: archive import failure or missing rich runner result.

- [x] **Step 3: Implement proof classification**

Consume only a completed proof result; never infer cause novelty from probe logs. Preserve proof log and input hashes in every verdict.

- [x] **Step 4: Run GREEN**

```bash
uv run pytest tests/test_poc_revision_proof_archive.py tests/test_poc_proof.py -q
```

### Task 4: Plateau portfolio and per-lineage trajectory mask

**Files:**
- Create: `src/dllm_agent/revision/portfolio.py`
- Create: `src/dllm_agent/revision/trajectory_policy.py`
- Create: `tests/test_poc_revision_portfolio.py`
- Create: `tests/test_poc_revision_trajectory_policy.py`

**Interfaces:**
- Produces: `LineageState`, `PortfolioPolicy`, `update_lineage(...)`, `revision_due(...)`.
- Produces: `FieldTrajectory`, `LineageFieldOutcome`, `choose_lineage_field(...)`.
- A lineage is revision-eligible only after its configured consecutive non-improving epochs.

- [x] **Step 1: Write portfolio and mask RED tests**

Assert improvement resets the plateau, two stalls trigger only the stalled lineage, family caps suppress work, and distinct trajectory commit/confidence histories select distinct fields.

- [x] **Step 2: Run RED**

```bash
uv run pytest tests/test_poc_revision_portfolio.py tests/test_poc_revision_trajectory_policy.py -q
```

Expected: import failures because the modules do not exist.

- [x] **Step 3: Implement the pure policies**

Calculate trajectory features from real `RevisionTrajectoryStep` records. Combine uncertainty with observed field outcomes; never pick from global first-particle suggestion alone.

- [x] **Step 4: Run GREEN**

```bash
uv run pytest tests/test_poc_revision_portfolio.py tests/test_poc_revision_trajectory_policy.py -q
```

### Task 5: Compose the online runtime

**Files:**
- Modify: `tools/selective_revision_sampler.py`
- Modify: `tests/test_selective_revision_sampler.py`
- Modify: `src/dllm_agent/revision/models.py`
- Modify: `src/dllm_agent/revision/controller_models.py`
- Modify: `src/dllm_agent/revision/controller_helpers.py`
- Modify: `src/dllm_agent/revision/pipeline.py`
- Modify: `tests/test_poc_revision_pipeline.py`

**Interfaces:**
- Extends: `RevisionConfig` with defaulted `archive_size`, `plateau_epochs`, and `proof_family_attempt_cap` controls.
- Adds deterministic online proof/cause and parent promotion/drop evidence to the revision ledger/report.
- Retains: `run_poc_revision(...) -> PocRevisionReport` and existing CLI inputs.

- [x] **Step 1: Write integration RED tests**

Run real in-memory sampler/probe/proof seams. Assert proof occurs before the next revision, a worse child leaves the parent active, only plateaued lineages are revised, duplicate causes do not receive novelty twice, and event order is byte-stable.

Also force the model seam to return two different allowed revision tokens for the same parent and assert
the committed selector follows the returned model token rather than a controller-selected successor.

- [x] **Step 2: Run RED**

```bash
uv run pytest tests/test_poc_revision_pipeline.py -q
```

Expected: current child-only loop replaces the parent and emits proof only after all epochs.

- [x] **Step 3: Implement runtime composition**

At every epoch barrier: probe, classify proof triggers, update cause/family memory, select `parent ∪ child`, update lineage plateau, choose lineage-specific fields, then call the still-resident sampler only for eligible scouts.

- [x] **Step 4: Run GREEN**

```bash
uv run pytest tests/test_poc_revision_pipeline.py tests/test_revision_client.py tests/test_revision_client_validation.py -q
```

### Task 6: Installed-surface evidence and documentation

**Files:**
- Modify: `docs/dllm-fuzzing-feedback-loop.md`
- Create: `docs/dllm-native-online-portfolio-harness.md`

**Interfaces:**
- Documents exact implemented flow in Mermaid, the maximum-use verdict, limitations, and required ablations.

- [x] **Step 1: Run all relevant verification**

```bash
uv run pytest -q
uv run ruff check src tests
uv run basedpyright
uv build
```

- [x] **Step 2: Exercise the installed surface**

Run a bounded in-memory end-to-end driver through `run_poc_revision` and run `uv run dllm-agent poc-revision --help`; record model lifecycle and event ordering without claiming measured vulnerability gains.

- [x] **Step 3: Write measured implementation notes**

Record only observed test/smoke results and clearly separate architectural expectations from an unrun 24-hour efficacy experiment.
