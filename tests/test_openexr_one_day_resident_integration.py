"""Coordinator integration for canonical resident model-load accounting."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pytest

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcomeEnvelope,
    CoordinatorError,
    build_campaign_manifest,
    coordinate_campaign,
    load_complete_campaign_outcome,
)
from dllm_agent.experiments.openexr_one_day.models import GpuIndex
from dllm_agent.experiments.openexr_one_day.resident_evidence import (
    build_resident_model_load_evidence,
    load_resident_model_load_evidence,
)
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    GIBIBYTE,
    ResourceSnapshot,
    WorkerLoadTelemetry,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeResidentWorker,
)
from tests.test_openexr_one_day_coordinator import _config_json, _passing_gates

if TYPE_CHECKING:
    from types import TracebackType

    from dllm_agent.experiments.openexr_one_day.models import CellSchedule


def test_coordinator_persists_production_resident_telemetry_in_gpu_pid_order(
    tmp_path: Path,
) -> None:
    config = _qualified_config(tmp_path)
    pool, expected_pids = _telemetry_pool()

    outcome = coordinate_campaign(
        config,
        resident_pool=pool,
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    load = outcome.accounting.resident_model_load
    backing = outcome.accounting.resident_model_load_evidence
    assert load is not None
    assert backing is not None
    assert outcome.resident_worker_pids == expected_pids
    assert backing.relative_path == "provenance/resident-model-load.json"
    assert (
        load_resident_model_load_evidence(
            config.artifact_root,
            backing,
            expected_worker_pids=expected_pids,
        )
        == load
    )
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    reloaded = load_complete_campaign_outcome(
        config.artifact_root,
        build_campaign_manifest(config, schedule),
    )
    assert reloaded == outcome


def test_completed_reload_rejects_resident_descriptor_aggregate_or_pid_mismatch(
    tmp_path: Path,
) -> None:
    config = _qualified_config(tmp_path)
    pool, _expected_pids = _telemetry_pool()
    _ = coordinate_campaign(
        config,
        resident_pool=pool,
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = build_campaign_manifest(config, schedule)
    path = config.artifact_root / "campaign-outcome.json"
    original = CampaignOutcomeEnvelope.model_validate_json(path.read_bytes())
    accounting = original.outcome.accounting
    backing = accounting.resident_model_load_evidence
    assert backing is not None

    bad_descriptor = accounting.model_copy(
        update={"resident_model_load_evidence": backing.model_copy(update={"sha256": "0" * 64})}
    )
    alternative = build_resident_model_load_evidence(_telemetry(load_offset=10.0))
    bad_aggregate = accounting.model_copy(update={"resident_model_load": alternative})
    corrupt_outcomes = (
        original.outcome.model_copy(update={"accounting": bad_descriptor}),
        original.outcome.model_copy(update={"accounting": bad_aggregate}),
        original.outcome.model_copy(
            update={"resident_worker_pids": tuple(reversed(original.outcome.resident_worker_pids))}
        ),
    )
    for corrupt in corrupt_outcomes:
        envelope = original.model_copy(update={"outcome": corrupt})
        _ = path.write_bytes(canonical_json_bytes(envelope.model_dump(mode="json")))
        with pytest.raises(CoordinatorError, match="resident model-load"):
            _ = load_complete_campaign_outcome(config.artifact_root, manifest)


def test_fake_pool_without_telemetry_keeps_legacy_resident_accounting_none(
    tmp_path: Path,
) -> None:
    config = _qualified_config(tmp_path)

    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    assert outcome.accounting.resident_model_load is None
    assert outcome.accounting.resident_model_load_evidence is None
    assert not (config.artifact_root / "provenance/resident-model-load.json").exists()


@dataclass(slots=True)
class _TelemetryPool:
    delegate: FakeResidentPool
    load_telemetry: tuple[WorkerLoadTelemetry, ...]

    def __enter__(self) -> Self:
        _ = self.delegate.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.delegate.__exit__(exc_type, exc, tb)

    def worker_for(self, cell: CellSchedule) -> FakeResidentWorker:
        return self.delegate.worker_for(cell)


def _telemetry_pool() -> tuple[_TelemetryPool, tuple[int, ...]]:
    pids = tuple(20_000 - gpu for gpu in range(8))
    events: list[str] = []
    workers = {gpu: FakeResidentWorker(GpuIndex(gpu), pids[gpu], events=events) for gpu in range(8)}
    delegate = FakeResidentPool(events=events, workers=workers)
    return _TelemetryPool(delegate, _telemetry(pids=pids)), pids


def _telemetry(
    *, pids: tuple[int, ...] | None = None, load_offset: float = 0.0
) -> tuple[WorkerLoadTelemetry, ...]:
    worker_pids = pids or tuple(20_000 - gpu for gpu in range(8))
    before = ResourceSnapshot(96 * GIBIBYTE, 0, 23 * GIBIBYTE, 0)
    after = ResourceSnapshot(78 * GIBIBYTE, 0, 5 * GIBIBYTE, 16 * GIBIBYTE)
    return tuple(
        WorkerLoadTelemetry(
            gpu_index=GpuIndex(gpu),
            requested_device=f"cuda:{gpu}",
            ready_device=f"cuda:{gpu}",
            load_seconds=load_offset + gpu + 0.25,
            process_id=worker_pids[gpu],
            before=before,
            after=after,
            samples=(before, after),
            minimum_host_available_bytes=78 * GIBIBYTE,
            maximum_swap_used_bytes=0,
            minimum_gpu_free_bytes=5 * GIBIBYTE,
            peak_process_rss_bytes=16 * GIBIBYTE,
        )
        for gpu in range(8)
    )


def _qualified_config(tmp_path: Path) -> CampaignConfig:
    suffix = sha256(str(tmp_path).encode()).hexdigest()[:12]
    root = Path("/data/dllm-agent") / f"task5-resident-integration-{suffix}"
    return CampaignConfig.model_validate(
        _config_json(str(root)) | {"qualification_state": "qualified"}
    )
