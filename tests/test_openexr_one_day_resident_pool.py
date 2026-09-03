from __future__ import annotations

# ruff: noqa: I001

import errno
import os
from typing import TYPE_CHECKING

import pytest
from tools.selective_revision_worker import WorkerCacheError, prepare_worker_cache
from tools.stable_diffcoder_generation import (
    model_load_plan,
    qualify_read_only_cache,
)
from tools.stable_diffcoder_payloads import RunnerConfig

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.experiments.openexr_one_day.models import GpuIndex, MonotonicSeconds
from dllm_agent.experiments.openexr_one_day.resident_pool import (
    GIBIBYTE,
    ResidentPoolError,
    ResidentWorkerPool,
    ResourceLimits,
    ResourceSnapshot,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.revision_client_fakes import (
    FakeResidentWorker,
    FakeTelemetryProbe,
    ReadOnlyStatVfs,
    ScriptedLoadMonitor,
    resident_bindings as _bindings,
    resource_snapshot as _snapshot,
)

if TYPE_CHECKING:
    from pathlib import Path


def _pool(
    workers: list[FakeResidentWorker],
    probe: FakeTelemetryProbe | None = None,
    monitor: ScriptedLoadMonitor | None = None,
) -> ResidentWorkerPool:
    probe = probe or FakeTelemetryProbe(_snapshot(12, 22, 0, 0), _snapshot(12, 4, 0, 1))
    args = (_bindings(workers), probe, ResourceLimits(max_process_rss_bytes=8 * GIBIBYTE))
    if monitor is None:
        return ResidentWorkerPool(*args)
    return ResidentWorkerPool(*args, load_monitor=monitor)


@pytest.fixture
def workers(tmp_path: Path) -> list[FakeResidentWorker]:
    return _workers(_read_only_cache(tmp_path), [])


def test_pool_starts_eight_workers_sequentially_and_reuses_them_across_five_waves(
    tmp_path: Path,
) -> None:
    # Given
    cache = _read_only_cache(tmp_path)
    events: list[str] = []
    workers = _workers(cache, events)
    probe = FakeTelemetryProbe(_snapshot(12, 22, 1, 0), _snapshot(12, 4, 1, 8))
    pool = _pool(workers, probe)
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(100.0))

    # When
    with pool as opened:
        routed = tuple(
            opened.worker_for(cell)
            for wave in schedule.waves
            for cell in wave.cells
            if cell.gpu_index is not None
        )

    # Then
    assert events[:8] == [f"start:cuda:{index}" for index in range(8)]
    assert len(routed) == 40
    assert all(worker.start_count == 1 for worker in workers)
    assert all(worker.close_count == 1 for worker in workers)
    assert len(pool.load_telemetry) == 8
    assert tuple(item.gpu_index for item in pool.load_telemetry) == tuple(
        GpuIndex(index) for index in range(8)
    )
    assert all(item.before.host_available_bytes == 12 * GIBIBYTE for item in pool.load_telemetry)
    assert all(item.after.process_rss_bytes == 8 * GIBIBYTE for item in pool.load_telemetry)
    assert len(probe.calls) == 16


def test_pool_closes_every_started_worker_after_partial_startup_failure(
    workers: list[FakeResidentWorker],
) -> None:
    # Given
    workers[3].fail_start = True
    pool = _pool(workers)

    # When / Then
    with pytest.raises(HuntSetupError, match="synthetic startup failure"):
        _ = pool.__enter__()
    assert [worker.close_count for worker in workers] == [1, 1, 1, 1, 0, 0, 0, 0]


def test_pool_records_in_load_extrema_and_fails_on_a_transient_recovered_breach(
    workers: list[FakeResidentWorker],
) -> None:
    # Given
    transient = _snapshot(11, 3, 2, 9)
    recovered = _snapshot(12, 4, 1, 8)
    pool = _pool(
        workers,
        FakeTelemetryProbe(_snapshot(12, 22, 1, 0), recovered),
        ScriptedLoadMonitor((transient, recovered)),
    )

    # When / Then
    with pytest.raises(ResidentPoolError, match="in-load host MemAvailable"):
        _ = pool.__enter__()
    assert len(pool.load_telemetry) == 1
    attempted = pool.load_telemetry[0]
    assert attempted.minimum_host_available_bytes == 11 * GIBIBYTE
    assert attempted.maximum_swap_used_bytes == 2 * GIBIBYTE
    assert attempted.minimum_gpu_free_bytes == 3 * GIBIBYTE
    assert attempted.peak_process_rss_bytes == 9 * GIBIBYTE
    assert attempted.samples == (_snapshot(12, 22, 1, 0), transient, recovered)


def test_pool_retains_samples_for_the_worker_whose_startup_fails(
    workers: list[FakeResidentWorker],
) -> None:
    # Given
    workers[0].fail_start = True
    peak = _snapshot(12, 8, 0, 7)
    pool = _pool(
        workers, FakeTelemetryProbe(_snapshot(12, 22, 0, 0), peak), ScriptedLoadMonitor((peak,))
    )

    # When / Then
    with pytest.raises(HuntSetupError, match="synthetic startup failure"):
        _ = pool.__enter__()
    assert len(pool.load_telemetry) == 1
    assert pool.load_telemetry[0].peak_process_rss_bytes == 7 * GIBIBYTE
    assert pool.load_telemetry[0].samples[-1] == peak


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("ready_device", "cuda:7", "ready device"),
        ("pid_after_start", 0, "positive PID"),
        ("alive", False, "exited during load"),
    ],
)
def test_pool_rejects_invalid_loaded_worker(
    workers: list[FakeResidentWorker],
    attribute: str,
    value: str | int | bool,
    message: str,
) -> None:
    # Given
    setattr(workers[0], attribute, value)
    pool = _pool(workers)

    # When / Then
    with pytest.raises(ResidentPoolError, match=message):
        _ = pool.__enter__()
    assert workers[0].close_count == 1
    assert all(worker.start_count == 0 for worker in workers[1:])


def test_pool_checks_worker_liveness_before_every_cell_route(
    workers: list[FakeResidentWorker],
) -> None:
    # Given
    pool = _pool(workers)
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(100.0))
    model_cell = next(cell for cell in schedule.waves[0].cells if cell.gpu_index is not None)
    assert model_cell.gpu_index is not None

    # When / Then
    with pool as opened:
        assigned = workers[int(model_cell.gpu_index)]
        assigned.alive = False
        with pytest.raises(ResidentPoolError, match="no longer alive"):
            _ = opened.worker_for(model_cell)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (_snapshot(12, 22 - (1 / GIBIBYTE), 0, 0), _snapshot(12, 4, 0, 1), "pre-load GPU"),
        (_snapshot(12 - (1 / GIBIBYTE), 22, 0, 0), _snapshot(12, 4, 0, 1), "MemAvailable"),
        (_snapshot(12, 22, 0, 0), _snapshot(12, 4 - (1 / GIBIBYTE), 0, 1), "post-load GPU"),
        (_snapshot(12, 22, 0, 0), _snapshot(12 - (1 / GIBIBYTE), 4, 0, 1), "MemAvailable"),
        (_snapshot(12, 22, 0, 0), _snapshot(12, 4, 1 / GIBIBYTE, 1), "swap"),
        (_snapshot(12, 22, 0, 0), _snapshot(12, 4, 0, 8 + (1 / GIBIBYTE)), "RSS"),
    ],
)
def test_pool_fails_below_each_memory_gate(
    workers: list[FakeResidentWorker],
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    message: str,
) -> None:
    # Given
    pool = _pool(workers, FakeTelemetryProbe(before, after))

    # When / Then
    with pytest.raises(ResidentPoolError, match=message):
        _ = pool.__enter__()


def test_published_pool_telemetry_is_read_only(workers: list[FakeResidentWorker]) -> None:
    # Given
    pool = _pool(workers)
    with pool:
        published = pool.load_telemetry

    # When / Then
    with pytest.raises(AttributeError):
        pool.__setattr__("load_telemetry", ())
    assert pool.load_telemetry == published


def test_model_load_plan_is_pinned_local_safetensor_and_direct_to_requested_device(
    tmp_path: Path,
) -> None:
    # Given
    config = RunnerConfig(
        prompt=tmp_path / "prompt.md",
        output=tmp_path / "unused.json",
        count=2,
        seed=0,
        steps=8,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        cache_dir=tmp_path / "shared-cache",
        device="cuda:3",
    )

    # When
    plan = model_load_plan(config, "cuda:3")

    # Then
    assert plan[:2] == (str(tmp_path / "shared-cache"), "cuda:3")
    assert all(plan[2:])


def test_read_only_cache_rejects_any_writable_descendant(tmp_path: Path) -> None:
    # Given
    cache = _writable_cache(tmp_path)

    # When / Then
    with pytest.raises(WorkerCacheError, match="writable descendant"):
        _ = prepare_worker_cache(cache, read_only_cache=True)


def test_read_only_cache_qualification_executes_an_actual_write_trap(tmp_path: Path) -> None:
    # Given
    cache = _read_only_cache(tmp_path)

    # When
    qualification = qualify_read_only_cache(cache)

    # Then
    assert qualification.cache_dir == cache.resolve()
    assert qualification.dynamic_modules_dir == (cache / "modules").resolve()
    assert qualification.write_trap_verified is True
    with pytest.raises(PermissionError):
        _ = (cache / "must-not-write").write_bytes(b"forbidden")


def test_read_only_cache_accepts_a_genuine_read_only_mount_with_writable_mode_bits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    cache = _prewarmed_cache(tmp_path)

    def read_only_statvfs(path: os.PathLike[str]) -> ReadOnlyStatVfs:
        del path
        return ReadOnlyStatVfs(f_flag=os.ST_RDONLY)

    def read_only_open(path: os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        del path, flags, mode
        raise OSError(errno.EROFS, "read-only filesystem")

    monkeypatch.setattr(os, "statvfs", read_only_statvfs)
    monkeypatch.setattr(os, "open", read_only_open)

    # When
    qualification = qualify_read_only_cache(cache)

    # Then
    assert qualification.read_only_mount is True
    assert qualification.write_trap_verified is True


def test_resident_pool_rejects_writable_cache_descendant_even_when_root_is_read_only(
    tmp_path: Path,
) -> None:
    # Given
    cache = _writable_cache(tmp_path)
    pool = _pool(_workers(cache, []))

    # When / Then
    with pytest.raises(ResidentPoolError, match="writable descendant"):
        _ = pool.__enter__()


def _workers(cache: Path, events: list[str]) -> list[FakeResidentWorker]:
    return [
        FakeResidentWorker(device=f"cuda:{index}", cache_dir=cache, events=events)
        for index in range(8)
    ]


def _read_only_cache(tmp_path: Path) -> Path:
    cache = _prewarmed_cache(tmp_path)
    for path in sorted(cache.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    cache.chmod(0o555)
    return cache


def _writable_cache(tmp_path: Path) -> Path:
    cache = _prewarmed_cache(tmp_path)
    writable = cache / "models" / "writable.bin"
    writable.parent.mkdir()
    _ = writable.write_bytes(b"weights")
    for path in (cache / "modules" / "ready.py", cache / "modules", writable.parent, cache):
        path.chmod(0o444 if path.is_file() else 0o555)
    writable.chmod(0o644)
    return cache


def _prewarmed_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "shared-cache"
    modules = cache / "modules"
    modules.mkdir(parents=True)
    _ = (modules / "ready.py").write_text("# prewarmed\n", encoding="utf-8")
    return cache
