"""Resident model-load evidence regressions for the OpenEXR campaign."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from dllm_agent.experiments.openexr_one_day.models import GpuIndex
from dllm_agent.experiments.openexr_one_day.resident_evidence import (
    ResidentEvidenceError,
    build_resident_model_load_evidence,
    load_resident_model_load_evidence,
    persist_resident_model_load_evidence,
)
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    GIBIBYTE,
    ResourceSnapshot,
    WorkerLoadTelemetry,
)


def test_resident_load_evidence_persists_exact_eight_gpu_telemetry(tmp_path: Path) -> None:
    telemetry = _telemetry()

    evidence = build_resident_model_load_evidence(telemetry)
    backing = persist_resident_model_load_evidence(tmp_path, evidence)
    loaded = load_resident_model_load_evidence(
        tmp_path,
        backing,
        expected_worker_pids=tuple(1000 + index for index in range(8)),
    )

    assert loaded == evidence
    assert tuple(worker.gpu_index for worker in loaded.workers) == tuple(range(8))
    assert loaded.total_load_seconds == sum(item.load_seconds for item in telemetry)
    assert loaded.peak_process_rss_bytes == 16 * GIBIBYTE
    assert loaded.peak_vram_bytes == 18 * GIBIBYTE
    assert backing.relative_path == "provenance/resident-model-load.json"


def test_resident_load_evidence_rejects_mutated_backing_bytes(tmp_path: Path) -> None:
    backing = persist_resident_model_load_evidence(
        tmp_path,
        build_resident_model_load_evidence(_telemetry()),
    )
    path = tmp_path / backing.relative_path
    _ = path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ResidentEvidenceError, match="backing evidence"):
        _ = load_resident_model_load_evidence(tmp_path, backing)


def test_resident_load_evidence_requires_all_eight_workers() -> None:
    with pytest.raises(ResidentEvidenceError, match="exactly eight"):
        _ = build_resident_model_load_evidence(_telemetry()[:-1])


def test_resident_load_evidence_binds_campaign_worker_pids(tmp_path: Path) -> None:
    backing = persist_resident_model_load_evidence(
        tmp_path,
        build_resident_model_load_evidence(_telemetry()),
    )

    with pytest.raises(ResidentEvidenceError, match="worker PID"):
        _ = load_resident_model_load_evidence(
            tmp_path,
            backing,
            expected_worker_pids=tuple(2000 + index for index in range(8)),
        )


def _telemetry() -> tuple[WorkerLoadTelemetry, ...]:
    before = ResourceSnapshot(96 * GIBIBYTE, 0, 23 * GIBIBYTE, 0)
    middle = ResourceSnapshot(80 * GIBIBYTE, 0, 8 * GIBIBYTE, 12 * GIBIBYTE)
    after = ResourceSnapshot(78 * GIBIBYTE, 0, 5 * GIBIBYTE, 16 * GIBIBYTE)
    return tuple(
        WorkerLoadTelemetry(
            gpu_index=GpuIndex(index),
            requested_device=f"cuda:{index}",
            ready_device=f"cuda:{index}",
            load_seconds=index + 0.25,
            process_id=1000 + index,
            before=before,
            after=after,
            samples=(before, middle, after),
            minimum_host_available_bytes=78 * GIBIBYTE,
            maximum_swap_used_bytes=0,
            minimum_gpu_free_bytes=5 * GIBIBYTE,
            peak_process_rss_bytes=16 * GIBIBYTE,
        )
        for index in range(8)
    )
