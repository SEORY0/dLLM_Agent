"""Eight-GPU resident worker pool contracts for the one-day campaign."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, NamedTuple, Protocol, Self, override

from tools.stable_diffcoder_generation import ReadOnlyCacheError, qualify_read_only_cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

    from tools.runtime.selective_revision_json import ReadyRecord

    from dllm_agent.experiments.openexr_one_day.models import CellSchedule, GpuIndex

GIBIBYTE = 1024**3


@dataclass(frozen=True, slots=True)
class ResidentPoolError(Exception):
    """Raised when worker residency or a resource gate fails closed."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


class ResourceSnapshot(NamedTuple):
    """Host/GPU/process memory observed at one load boundary."""

    host_available_bytes: int
    swap_used_bytes: int
    gpu_free_bytes: int
    process_rss_bytes: int


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Inclusive resident-worker load thresholds."""

    max_process_rss_bytes: int
    min_preload_host_available_bytes: int = 12 * GIBIBYTE
    min_preload_gpu_free_bytes: int = 22 * GIBIBYTE
    min_postload_host_available_bytes: int = 12 * GIBIBYTE
    min_postload_gpu_free_bytes: int = 4 * GIBIBYTE


class ResidentWorker(Protocol):
    """Revision worker surface owned by the pool."""

    device: str
    cache_dir: Path
    read_only_cache: bool
    ready: ReadyRecord | None

    @property
    def process_id_if_started(self) -> int | None: ...  # noqa: D102

    @property
    def last_process_id(self) -> int | None: ...  # noqa: D102

    @property
    def is_alive(self) -> bool: ...  # noqa: D102

    def __enter__(self) -> Self: ...  # noqa: D105

    def close(self) -> None: ...  # noqa: D102


class ResourceTelemetryProbe(Protocol):
    """Memory telemetry boundary kept outside the ready wire schema."""

    def capture(  # noqa: D102
        self, gpu_index: GpuIndex, process_id: int | None
    ) -> ResourceSnapshot: ...


class LoadIntervalMonitor(Protocol):
    """Observe the complete interval of one blocking resident-worker load."""

    def load(  # noqa: D102
        self,
        binding: ResidentWorkerBinding,
        telemetry_probe: ResourceTelemetryProbe,
        record: Callable[[ResourceSnapshot], None],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ResidentWorkerBinding:
    """One physical GPU and its long-lived worker client."""

    gpu_index: GpuIndex
    worker: ResidentWorker


@dataclass(frozen=True, slots=True)
class PollingLoadMonitor:
    """Sample resource state concurrently throughout one blocking load."""

    interval_seconds: float = 0.25

    def load(  # noqa: D102
        self,
        binding: ResidentWorkerBinding,
        telemetry_probe: ResourceTelemetryProbe,
        record: Callable[[ResourceSnapshot], None],
    ) -> None:
        if self.interval_seconds <= 0:
            raise ResidentPoolError(reason="load-monitor interval must be positive")
        finished = Event()

        def start() -> None:
            try:
                _ = binding.worker.__enter__()
            finally:
                finished.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(start)
            while not finished.wait(self.interval_seconds):
                record(
                    telemetry_probe.capture(
                        binding.gpu_index,
                        binding.worker.process_id_if_started,
                    )
                )
            try:
                future.result()
            finally:
                record(
                    telemetry_probe.capture(
                        binding.gpu_index,
                        binding.worker.process_id_if_started,
                    )
                )


@dataclass(frozen=True, slots=True)
class WorkerLoadTelemetry:
    """Immutable load samples and extrema, including failed attempts."""

    gpu_index: GpuIndex
    requested_device: str
    ready_device: str | None
    load_seconds: float
    process_id: int | None
    before: ResourceSnapshot
    after: ResourceSnapshot
    samples: tuple[ResourceSnapshot, ...]
    minimum_host_available_bytes: int
    maximum_swap_used_bytes: int
    minimum_gpu_free_bytes: int
    peak_process_rss_bytes: int


# Pool mutation is the documented resident lifecycle state.
@dataclass(slots=True)  # lifecycle state; # noqa: MUTABLE_OK
class ResidentWorkerPool:
    """Start eight GPU-keyed workers once and reuse them until pool close."""

    bindings: tuple[ResidentWorkerBinding, ...]
    telemetry_probe: ResourceTelemetryProbe
    limits: ResourceLimits
    clock: Callable[[], float] = monotonic
    load_monitor: LoadIntervalMonitor = field(default_factory=PollingLoadMonitor, kw_only=True)
    _load_telemetry: tuple[WorkerLoadTelemetry, ...] = field(default=(), init=False)
    _started: bool = False
    _closer: ExitStack | None = None
    _workers_by_gpu: dict[GpuIndex, ResidentWorker] = field(default_factory=dict)

    @property
    def load_telemetry(self) -> tuple[WorkerLoadTelemetry, ...]:  # noqa: D102
        return self._load_telemetry

    def __enter__(self) -> Self:  # noqa: D105
        if self._started or self._load_telemetry:
            raise ResidentPoolError(reason="resident worker pool may only start once")
        bindings = _validated_bindings(self.bindings)
        telemetry: list[WorkerLoadTelemetry] = []
        with ExitStack() as closer:
            for binding in bindings:
                before = self.telemetry_probe.capture(binding.gpu_index, None)
                _gate_preload(before, self.limits)
                samples = [before]
                _ = closer.callback(binding.worker.close)
                started_at = self.clock()
                try:
                    self.load_monitor.load(binding, self.telemetry_probe, samples.append)
                finally:
                    load_seconds = max(0.0, self.clock() - started_at)
                    telemetry.append(_load_record(binding, tuple(samples), load_seconds))
                    self._load_telemetry = tuple(telemetry)
                ready = binding.worker.ready
                if ready is None:
                    raise ResidentPoolError(reason="resident worker ready record missing")
                if ready.device != binding.worker.device:
                    raise ResidentPoolError(reason="requested and ready device mismatch")
                if binding.worker.process_id_if_started is None:
                    raise ResidentPoolError(reason="resident worker must expose a positive PID")
                if not binding.worker.is_alive:
                    raise ResidentPoolError(reason="resident worker exited during load")
                _gate_load(self._load_telemetry[-1], self.limits)
            self._closer = closer.pop_all()
        self._workers_by_gpu = {binding.gpu_index: binding.worker for binding in bindings}
        self._started = True
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def close(self) -> None:  # noqa: D102
        if (closer := self._closer) is None:
            return
        closer.close()
        self._closer = None
        self._started = False

    def worker_for(self, cell: CellSchedule) -> ResidentWorker:  # noqa: D102
        if not self._started:
            raise ResidentPoolError(reason="resident worker pool is not running")
        if cell.gpu_index is None:
            raise ResidentPoolError(reason="CPU-only cell has no resident model worker")
        worker = self._workers_by_gpu[cell.gpu_index]
        if worker.process_id_if_started is None or not worker.is_alive:
            raise ResidentPoolError(reason="resident worker is no longer alive")
        return worker


def _validated_bindings(
    bindings: tuple[ResidentWorkerBinding, ...],
) -> tuple[ResidentWorkerBinding, ...]:
    ordered = tuple(sorted(bindings, key=lambda binding: int(binding.gpu_index)))
    if tuple(int(binding.gpu_index) for binding in ordered) != tuple(range(8)):
        raise ResidentPoolError(reason="resident pool requires physical GPUs 0 through 7")
    cache_paths = {binding.worker.cache_dir.resolve() for binding in ordered}
    if len(cache_paths) != 1:
        raise ResidentPoolError(reason="resident workers must share one cache path")
    cache_dir = next(iter(cache_paths))
    try:
        _ = qualify_read_only_cache(cache_dir)
    except ReadOnlyCacheError as error:
        raise ResidentPoolError(reason=str(error)) from error
    for binding in ordered:
        if binding.worker.device != f"cuda:{int(binding.gpu_index)}":
            raise ResidentPoolError(reason="worker must be keyed by its physical GPU")
        if not binding.worker.read_only_cache:
            raise ResidentPoolError(reason="campaign worker cache mode must be read-only")
    return ordered


def _gate_preload(snapshot: ResourceSnapshot, limits: ResourceLimits) -> None:
    _validate_snapshot(snapshot)
    if snapshot.host_available_bytes < limits.min_preload_host_available_bytes:
        raise ResidentPoolError(reason="pre-load host MemAvailable below threshold")
    if snapshot.gpu_free_bytes < limits.min_preload_gpu_free_bytes:
        raise ResidentPoolError(reason="pre-load GPU free VRAM below threshold")


def _load_record(
    binding: ResidentWorkerBinding,
    samples: tuple[ResourceSnapshot, ...],
    load_seconds: float,
) -> WorkerLoadTelemetry:
    ready = binding.worker.ready
    return WorkerLoadTelemetry(
        gpu_index=binding.gpu_index,
        requested_device=binding.worker.device,
        ready_device=None if ready is None else ready.device,
        load_seconds=load_seconds,
        process_id=binding.worker.process_id_if_started or binding.worker.last_process_id,
        before=samples[0],
        after=samples[-1],
        samples=samples,
        minimum_host_available_bytes=min(sample.host_available_bytes for sample in samples),
        maximum_swap_used_bytes=max(sample.swap_used_bytes for sample in samples),
        minimum_gpu_free_bytes=min(sample.gpu_free_bytes for sample in samples),
        peak_process_rss_bytes=max(sample.process_rss_bytes for sample in samples),
    )


def _gate_load(telemetry: WorkerLoadTelemetry, limits: ResourceLimits) -> None:
    for sample in telemetry.samples:
        _validate_snapshot(sample)
    if telemetry.minimum_host_available_bytes < limits.min_postload_host_available_bytes:
        raise ResidentPoolError(reason="in-load host MemAvailable below threshold")
    if telemetry.minimum_gpu_free_bytes < limits.min_postload_gpu_free_bytes:
        raise ResidentPoolError(reason="post-load GPU free VRAM or in-load minimum below threshold")
    if telemetry.maximum_swap_used_bytes > telemetry.before.swap_used_bytes:
        raise ResidentPoolError(reason="swap use increased during worker load")
    if telemetry.peak_process_rss_bytes > limits.max_process_rss_bytes:
        raise ResidentPoolError(reason="worker RSS exceeds configured ceiling")


def _validate_snapshot(snapshot: ResourceSnapshot) -> None:
    if min(snapshot) < 0:
        raise ResidentPoolError(reason="resource telemetry values must be nonnegative")
