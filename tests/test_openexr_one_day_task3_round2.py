"""Round-two integration regressions for the shared bank and resident pool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tools.selective_revision_payloads import sha256_bytes

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.experiments.openexr_one_day.candidate_bank import (
    RAW_PARENT_COUNT,
    ArmBankBinding,
    build_candidate_bank,
    build_frozen_bank_adapters,
)
from dllm_agent.experiments.openexr_one_day.models import (
    MODEL_ARMS,
    GpuIndex,
    RepetitionIndex,
    Seed,
    SeedNamespace,
)
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    GIBIBYTE,
    ResidentWorkerBinding,
    ResidentWorkerPool,
    ResourceLimits,
    ResourceSnapshot,
)
from dllm_agent.revision.controller_models import RevisionRunRequest
from dllm_agent.revision.models import RevisionConfig
from dllm_agent.revision.pipeline import run_poc_revision
from dllm_agent.runtime.revision_client import RevisionWorkerClient
from tests.revision_client_fakes import (
    FakeFactory,
    FakePipe,
    FakeProcess,
    FakeResidentWorker,
    FakeTelemetryProbe,
)
from tests.test_openexr_one_day_arm_equivalence import RequestRecordingProbe
from tests.test_openexr_one_day_candidate_bank import (
    DeterministicInitialSampler,
    RevisionDelegate,
)
from tests.test_poc_revision_pipeline import RecordingProof

if TYPE_CHECKING:
    from pathlib import Path


def test_failed_real_client_startup_retains_reaped_pid_in_pool_telemetry(
    tmp_path: Path,
) -> None:
    # Given
    cache = _read_only_cache(tmp_path)
    prompt = tmp_path / "prompt.md"
    _ = prompt.write_text("prompt context", encoding="utf-8")
    process = FakeProcess(stdout=FakePipe(("not-json\n",)), pid=54_321)
    client = RevisionWorkerClient(
        runtime_python=tmp_path / "python",
        cache_dir=cache,
        prompt_path=prompt,
        prompt_sha256=sha256_bytes(prompt.read_bytes()),
        device="cuda:0",
        timeout_seconds=1.0,
        read_only_cache=True,
        factory=FakeFactory(process),
    )
    later_workers = tuple(FakeResidentWorker(f"cuda:{index}", cache, []) for index in range(1, 8))
    bindings = (
        ResidentWorkerBinding(GpuIndex(0), client),
        *tuple(
            ResidentWorkerBinding(GpuIndex(index), worker)
            for index, worker in enumerate(later_workers, start=1)
        ),
    )
    before = ResourceSnapshot(12 * GIBIBYTE, 0, 22 * GIBIBYTE, 0)
    after = ResourceSnapshot(12 * GIBIBYTE, 0, 4 * GIBIBYTE, GIBIBYTE)
    pool = ResidentWorkerPool(
        bindings,
        FakeTelemetryProbe(before, after),
        ResourceLimits(max_process_rss_bytes=8 * GIBIBYTE),
    )

    # When / Then
    with pytest.raises(HuntSetupError, match="malformed"):
        _ = pool.__enter__()
    assert client.process_id_if_started is None
    assert client.is_alive is False
    assert process.terminated is True
    assert process.wait_calls == 1
    assert process.stdin.was_closed is True
    assert process.stdout.was_closed is True
    assert process.stderr.was_closed is True
    assert len(pool.load_telemetry) == 1
    assert pool.load_telemetry[0].process_id == 54_321


def test_frozen_bank_runs_all_selected_parents_through_contiguous_pipeline_seeds(
    tmp_path: Path,
) -> None:
    # Given
    requested_seed_base = 1_000_000
    namespace = SeedNamespace(
        repetition=RepetitionIndex(0),
        start_seed=Seed(requested_seed_base),
        stop_seed_exclusive=Seed(1_001_000),
    )
    bank = build_candidate_bank(DeterministicInitialSampler(RAW_PARENT_COUNT), namespace, steps=8)
    bindings = tuple(
        ArmBankBinding(arm, RevisionDelegate([]), tmp_path / arm) for arm in MODEL_ARMS
    )
    adapter = next(
        item
        for item in build_frozen_bank_adapters(bank, bindings)
        if item.arm == "structured-one-shot"
    )
    expected_requested_seeds = tuple(requested_seed_base + index for index in range(16))
    source_seeds = tuple(parent.request.seed for parent in bank.selected_parents)
    assert source_seeds != expected_requested_seeds
    config = RevisionConfig(
        schema_version="dllm-poc-revision-config/v1",
        particle_count=16,
        epochs=1,
        initial_steps=8,
        revision_steps=2,
        probe_seconds=1,
        fuzzer_seeds=(7, 11),
        proof_seconds=5,
        archive_size=16,
    )
    request = RevisionRunRequest(
        schema_version="dllm-poc-revision-run/v1",
        arm="structured-one-shot",
        particle_seed_base=requested_seed_base,
    )
    probe = RequestRecordingProbe()

    # When
    report = run_poc_revision(
        config,
        request,
        adapter,
        probe,
        RecordingProof("not_reproduced"),
        tmp_path / "pipeline",
    )

    # Then
    lineages = bank.selected_lineages
    assert tuple(lineage.particle_id for lineage in lineages) == tuple(
        f"p{index:04d}" for index in range(16)
    )
    assert tuple(lineage.requested_seed for lineage in lineages) == expected_requested_seeds
    assert tuple(lineage.source_seed for lineage in lineages) == source_seeds
    assert tuple(lineage.steps for lineage in lineages) == (8,) * 16
    assert len(report.snapshots) == 16
    assert len(probe.requests) == 32


def _read_only_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "shared-cache"
    modules = cache / "modules"
    modules.mkdir(parents=True)
    _ = (modules / "ready.py").write_text("# prewarmed\n", encoding="utf-8")
    for path in sorted(cache.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    cache.chmod(0o555)
    return cache
