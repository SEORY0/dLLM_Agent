"""Task 5 wave-runtime concurrency, release, resume, and cancellation regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import pytest

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.openexr_one_day import coordinator
from dllm_agent.experiments.openexr_one_day.cell_runner import (
    DockerSearchLauncher,
    SearchAttemptResult,
    SearchLaunchRequest,
    run_openexr_cell,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CoordinatorError,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    StartGateEvidence,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRecord,
    CorpusIndexRequest,
    load_corpus_index,
    snapshot_corpus,
)
from dllm_agent.experiments.openexr_one_day.failure_taxonomy import (
    ExogenousCampaignError,
)
from dllm_agent.experiments.openexr_one_day.models import (
    MODEL_ARMS,
    CellSchedule,
    MonotonicSeconds,
)
from dllm_agent.experiments.openexr_one_day.recovery import (
    RecoveryError,
    load_campaign_events,
    resume_campaign,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.openexr_one_day_task4_fakes import (
    ManualClock,
    ProcessScript,
    ScriptedSearchLauncher,
)
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakePostSearch,
    FakeResidentPool,
    FakeResidentWorker,
)
from tests.test_openexr_one_day_cell_runner import _request, _seed_corpus
from tests.test_openexr_one_day_coordinator import _config_json

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from dllm_agent.experiments.multitarget.ledger import JsonValue
    from dllm_agent.experiments.openexr_one_day.cell_runner import (
        CellResult,
        CellRunRequest,
    )
    from dllm_agent.revision.controller_models import (
        InitialParticleRequest,
        RevisionRequest,
        RevisionResult,
    )


def test_wave_runtime_overlaps_baselines_banks_and_gpu_preparations_before_atomic_release(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    config = _qualified_config(data_root)
    state = _LifecycleState()
    clock = _ControlledClock(70_000.0)
    model = _BarrierModelExecutor(state)
    runner = _BarrierCellRunner(state, block_baselines=True)
    append_threads: list[int] = []
    original_append = coordinator._append_campaign_event

    def recording_append(
        root: Path,
        phase: Literal["manifest", "launch", "terminal"],
        payload: dict[str, JsonValue],
    ) -> None:
        append_threads.append(threading.get_ident())
        original_append(root, phase, payload)

    monkeypatch.setattr(coordinator, "monotonic", clock.monotonic)
    monkeypatch.setattr(coordinator, "sleep", clock.advance)
    monkeypatch.setattr(coordinator, "_append_campaign_event", recording_append)
    thread, outcomes, errors = _start_campaign(config, state, model, runner)

    # When
    try:
        assert state.wait_for(
            lambda: len(state.baseline_started) == 2 and state.bank_started == {0, 1}
        )
        initial_rows = _assert_wave_zero_is_exact_c0(config)
        state.bank_release.set()
        assert state.wait_for(lambda: len(state.prep_started) >= 2)
        started_gpus = {_gpu_for_cell(cell_id) for cell_id in state.prep_started}
        assert len(started_gpus) >= 2
        clock.set(70_900.0)
        released_cell = "wave-00-rep-00-selective-revision"
        blocked_cell = "wave-00-rep-00-final-only"
        state.release_prep(released_cell)
        assert state.wait_for(lambda: released_cell in state.model_search_started)

        # Then
        assert blocked_cell in state.prep_started
        assert blocked_cell not in state.prep_completed
        release_rows = state.search_index_rows[released_cell]
        assert tuple(row.scheduled_elapsed_seconds for row in release_rows) == (0, 900)
        assert release_rows[0].cumulative_hashes == release_rows[1].cumulative_hashes
        assert release_rows[1].cumulative_set_sha256 == initial_rows[released_cell]
        release_payload = _read_json(
            config.artifact_root / "cells/wave-00/rep-00/selective-revision/corpus-release.json"
        )
        assert release_payload["bank_physical_duration_seconds"] == 0.0
        assert release_payload["bank_ready_elapsed_seconds"] == 0.0
        assert release_payload["model_ready_elapsed_seconds"] == 900.0
        assert release_payload["corpus_release_elapsed_seconds"] == 900.0
        assert release_payload["pre_release_corpus_set_sha256"] == initial_rows[released_cell]
        assert release_payload["producer_gpu_index"] == 0
    finally:
        state.abort_all()
        thread.join(timeout=30.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert outcomes == []
    assert append_threads
    assert set(append_threads) == {thread.ident}


def test_model_ready_after_deadline_launches_no_search_and_keeps_complete_c0_curve(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    config = _qualified_config(data_root)
    state = _LifecycleState()
    state.bank_release.set()
    clock = _ControlledClock(70_000.0)
    model = _BarrierModelExecutor(state)
    runner = _BarrierCellRunner(
        state,
        block_baselines=False,
        interrupt_after_wave_zero=True,
    )
    monkeypatch.setattr(coordinator, "monotonic", clock.monotonic)
    monkeypatch.setattr(coordinator, "sleep", clock.advance)
    thread, outcomes, errors = _start_campaign(config, state, model, runner)
    first_wave_models = {
        f"wave-00-rep-{repetition:02d}-{arm}" for repetition in (0, 1) for arm in MODEL_ARMS
    }

    # When
    try:
        assert state.wait_for(lambda: bool(state.prep_started))
        clock.set(80_800.1)
        state.release_all()
        thread.join(timeout=30.0)
    finally:
        state.release_all()

    # Then
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert outcomes == []
    assert first_wave_models.isdisjoint(state.model_search_started)
    terminal_events = {
        event.cell_id: event
        for event in load_campaign_events(config.artifact_root)
        if event.phase == "terminal"
    }
    launched_ids = {
        event.cell_id
        for event in load_campaign_events(config.artifact_root)
        if event.phase == "launch"
    }
    assert {terminal_events[cell_id].terminal_reason for cell_id in first_wave_models} == {
        "expired-frozen"
    }
    assert first_wave_models.isdisjoint(launched_ids)
    for cell_id in first_wave_models:
        request = _index_request(_cell_root(config, cell_id))
        rows = load_corpus_index(request)
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
        assert len({row.cumulative_set_sha256 for row in rows}) == 1
        assert tuple(path.name for path in request.corpus_root.iterdir()) == (
            f"c0-{config.c0_sha256}.seed",
        )


def test_expired_baselines_freeze_c0_without_submitting_search(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    runner = _ExpiredWaveCellRunner()

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        cell_runner=runner,
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    # Then
    baselines = tuple(cell for cell in outcome.cells if cell.arm == "libfuzzer-only")
    launched_ids = {
        event.cell_id
        for event in load_campaign_events(config.artifact_root)
        if event.phase == "launch"
    }
    assert runner.requests == []
    assert len(baselines) == 10
    assert {cell.terminal_reason for cell in baselines} == {"expired-frozen"}
    assert {cell.cell_id for cell in baselines}.isdisjoint(launched_ids)
    for cell in baselines:
        rows = load_corpus_index(_index_request(cell.artifact_root))
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
        assert tuple(row.observation_kind for row in rows) == (
            "observed",
            *("frozen" for _ in SCHEDULE_SECONDS[1:]),
        )
        assert len({row.cumulative_set_sha256 for row in rows}) == 1
        assert rows[0].cumulative_hashes == (config.c0_sha256,)
        assert not (cell.artifact_root / "attempts").exists()
    reloaded = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        cell_runner=runner,
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )
    assert reloaded == outcome


def test_cell_runner_accepts_validated_preinitialized_corpus_index_prefix(
    data_root: Path,
) -> None:
    # Given
    request = _request(data_root)
    _seed_corpus(request.corpus_root)
    index_request = _index_request(request.artifact_root)
    prefix = (snapshot_corpus(index_request, 0, 0.0),)
    clock = ManualClock(request.start_monotonic_seconds)
    completed = SearchAttemptResult(
        outcome="completed",
        return_code=0,
        log=b"done\n",
        cpu_seconds=1.0,
        artifacts=(),
    )
    launcher = ScriptedSearchLauncher(
        clock,
        (
            ProcessScript(
                request.start_monotonic_seconds + 10.0,
                completed,
                completed,
            ),
        ),
    )

    # When
    result = run_openexr_cell(
        request,
        launcher,
        clock,
        preinitialized_corpus_index=prefix,
    )

    # Then
    rows = load_corpus_index(index_request)
    assert rows[0] == prefix[0]
    assert tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
    assert result.attempt_count == 1


def test_partial_resume_reuses_persisted_bank_prep_and_release_without_regeneration(
    data_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    config = _qualified_config(data_root)
    state = _LifecycleState()
    clock = _ControlledClock(70_000.0)
    monkeypatch.setattr(coordinator, "monotonic", clock.monotonic)
    monkeypatch.setattr(coordinator, "sleep", clock.advance)
    state.bank_release.set()
    state.release_all()
    first_model = _BarrierModelExecutor(state)
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=_BarrierPool(state),
            cell_runner=_InterruptingModelSearchRunner(),
            post_search=FakePostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=first_model,
        )
    bank_repetitions = {
        int(path.parent.name.removeprefix("rep-"))
        for path in config.artifact_root.glob("banks/rep-*/candidate-bank.json")
    }
    prep_cells = {
        _cell_id_from_artifact(path)
        for path in config.artifact_root.glob("cells/**/model-prep-result.json")
    }
    release_cells = {
        _cell_id_from_artifact(path)
        for path in config.artifact_root.glob("cells/**/corpus-release.json")
    }
    resumed_state = _LifecycleState()
    resumed_state.bank_release.set()
    resumed_state.release_all()
    resumed_model = _BarrierModelExecutor(resumed_state)
    resumed_pool = _BarrierPool(resumed_state)

    # When
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=resumed_pool,
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=resumed_model,
    )

    # Then
    regenerated_repetitions = {
        int(request.particle_id[6:8])
        for worker in resumed_pool.workers
        for request in worker.delegate.initial_requests
    }
    assert outcome.launched_cell_count == 50
    assert bank_repetitions.isdisjoint(regenerated_repetitions)
    assert prep_cells.isdisjoint(resumed_state.prep_started)
    assert release_cells.isdisjoint(resumed_state.prep_started)


def test_expired_recovery_freezes_c0_without_requiring_unbuilt_banks(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = coordinator.build_campaign_manifest(config, schedule)
    coordinator._persist_manifest(config.artifact_root, manifest)
    coordinator._append_campaign_event(
        config.artifact_root,
        "manifest",
        coordinator._manifest_payload(manifest),
    )
    pool = FakeResidentPool()

    # When
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=pool,
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        now_monotonic_seconds=200_000.0,
    )

    # Then
    assert pool.enter_count == 0
    assert not (config.artifact_root / "banks").exists()
    assert {cell.terminal_reason for cell in outcome.cells} == {"expired-frozen"}
    events = load_campaign_events(config.artifact_root)
    assert not tuple(event for event in events if event.phase == "launch")
    for cell in outcome.cells:
        rows = load_corpus_index(_index_request(cell.artifact_root))
        assert tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
        assert len({row.cumulative_set_sha256 for row in rows}) == 1
        assert rows[0].cumulative_hashes == (config.c0_sha256,)


@pytest.mark.parametrize("terminal_reason", ["completed", "treatment-failed"])
def test_nonexpired_terminal_without_launch_is_rejected(
    data_root: Path,
    terminal_reason: Literal["completed", "treatment-failed"],
) -> None:
    # Given
    config = _qualified_config(data_root)
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    manifest = coordinator.build_campaign_manifest(config, schedule)
    coordinator._persist_manifest(config.artifact_root, manifest)
    coordinator._append_campaign_event(
        config.artifact_root,
        "manifest",
        coordinator._manifest_payload(manifest),
    )
    cell = next(item for item in schedule.cells if item.arm == "libfuzzer-only")
    request = coordinator._cell_request(
        config,
        manifest,
        cell,
        {int(cell.repetition): config.c0_sha256},
        config.resource_map,
    )
    terminal = coordinator._pending_cell(config, manifest, cell, request).model_copy(
        update={"terminal_reason": terminal_reason}
    )
    coordinator._append_campaign_event(
        config.artifact_root,
        "terminal",
        coordinator._outcome_payload(terminal),
    )

    # When / Then
    with pytest.raises(RecoveryError, match="event after terminal"):
        _ = load_campaign_events(config.artifact_root)


def test_exogenous_failure_uses_one_canonical_wave_artifact_bound_to_all_ten_cells(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)

    # When
    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        cell_runner=FakeCellRunner(),
        post_search=FakePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=_ExogenousModelExecutor(),
    )

    # Then
    wave_cells = tuple(cell for cell in outcome.cells if cell.wave_index == 0)
    paths = tuple(sorted((config.artifact_root / "exogenous").glob("*.json")))
    assert tuple(path.name for path in paths) == tuple(
        f"wave-{wave_index:02d}.json" for wave_index in range(5)
    )
    wave_path = config.artifact_root / "exogenous/wave-00.json"
    payload = _read_json(wave_path)
    affected = payload["affected_cell_ids"]
    assert isinstance(affected, list)
    assert set(affected) == {cell.cell_id for cell in wave_cells}
    assert len(wave_cells) == 10
    assert {cell.exogenous_evidence_path for cell in wave_cells} == {wave_path}
    assert len({cell.exogenous_evidence_sha256 for cell in wave_cells}) == 1


def test_docker_launcher_cancel_active_terms_then_bounded_kills_and_reaps(
    data_root: Path,
) -> None:
    # Given
    child = _TermThenKillChild()
    launcher = DockerSearchLauncher(
        popen_factory=lambda _argv: child,
        monotonic=lambda: 1_000.0,
        cpu_clock=lambda: 0.0,
    )
    launch_request = _docker_launch_request(data_root)
    process = launcher.launch(launch_request)

    # When
    launcher.cancel_active()

    # Then
    assert child.terminate_count == 1
    assert child.kill_count == 1
    assert child.communicate_timeouts == [5.0, 5.0]
    assert process.terminate().outcome == "deadline"


def test_docker_launcher_cancel_does_not_communicate_concurrently_with_waiter(
    data_root: Path,
) -> None:
    # Given
    child = _ConcurrentCancelChild()
    launcher = DockerSearchLauncher(
        popen_factory=lambda _argv: child,
        monotonic=lambda: 1_000.0,
        cpu_clock=lambda: 0.0,
    )
    process = launcher.launch(_docker_launch_request(data_root))
    results: list[SearchAttemptResult | None] = []
    errors: list[BaseException] = []

    def wait_for_search() -> None:
        try:
            results.append(process.wait_until(2_000.0))
        except BaseException as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
            errors.append(error)

    waiter = threading.Thread(target=wait_for_search)
    waiter.start()
    assert child.communicate_started.wait(1.0)

    # When
    launcher.cancel_active()
    waiter.join(1.0)

    # Then
    assert not waiter.is_alive()
    assert errors == []
    assert results
    assert results[0] is not None
    assert child.max_concurrent_communicators == 1
    assert child.terminate_count == 1


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix6-wave-runtime")
    digest = sha256(str(root).encode()).hexdigest()[:12]
    return Path("/data/dllm-agent") / f"task5-fix6-wave-runtime-{digest}"


@dataclass(frozen=True, slots=True)
class _ControlledClock:
    value: list[float]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __init__(self, value: float) -> None:
        object.__setattr__(self, "value", [value])
        object.__setattr__(self, "lock", threading.Lock())

    def monotonic(self) -> float:
        with self.lock:
            return self.value[0]

    def set(self, value: float) -> None:
        with self.lock:
            self.value[0] = value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value[0] += seconds


@dataclass(frozen=True, slots=True)
class _LifecycleState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    bank_release: threading.Event = field(default_factory=threading.Event)
    all_prep_release: threading.Event = field(default_factory=threading.Event)
    baseline_release: threading.Event = field(default_factory=threading.Event)
    abort: threading.Event = field(default_factory=threading.Event)
    bank_started: set[int] = field(default_factory=set)
    baseline_started: set[str] = field(default_factory=set)
    prep_started: set[str] = field(default_factory=set)
    prep_completed: set[str] = field(default_factory=set)
    individually_released: set[str] = field(default_factory=set)
    model_search_started: set[str] = field(default_factory=set)
    search_index_rows: dict[str, tuple[CorpusIndexRecord, ...]] = field(default_factory=dict)

    def note_bank(self, repetition: int) -> None:
        with self.condition:
            self.bank_started.add(repetition)
            self.condition.notify_all()

    def note_cell(self, collection: set[str], cell_id: str) -> None:
        with self.condition:
            collection.add(cell_id)
            self.condition.notify_all()

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
        with self.condition:
            return self.condition.wait_for(predicate, timeout=timeout)

    def wait_for_prep_release(self, cell_id: str) -> bool:
        with self.condition:
            return self.condition.wait_for(
                lambda: self.all_prep_release.is_set() or cell_id in self.individually_released,
                timeout=10.0,
            )

    def release_prep(self, cell_id: str) -> None:
        with self.condition:
            self.individually_released.add(cell_id)
            self.condition.notify_all()

    def release_all(self) -> None:
        self.bank_release.set()
        self.baseline_release.set()
        self.all_prep_release.set()
        with self.condition:
            self.condition.notify_all()

    def abort_all(self) -> None:
        self.abort.set()
        self.release_all()


@dataclass(frozen=True, slots=True)
class _BarrierWorker:
    delegate: FakeResidentWorker
    state: _LifecycleState

    @property
    def process_id_if_started(self) -> int | None:
        return self.delegate.process_id_if_started

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        repetition = int(request.particle_id[6:8])
        if request.particle_id.endswith("-00") and repetition in {0, 1}:
            self.state.note_bank(repetition)
            if not self.state.bank_release.wait(10.0):
                message = "bank release barrier timed out"
                raise AssertionError(message)
        return self.delegate.initial(request)

    def revise(self, request: RevisionRequest) -> RevisionResult:
        return self.delegate.revise(request)


@dataclass(frozen=True, slots=True)
class _BarrierPool:
    state: _LifecycleState
    base: FakeResidentPool = field(default_factory=FakeResidentPool)
    workers: tuple[_BarrierWorker, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workers",
            tuple(_BarrierWorker(self.base.workers[index], self.state) for index in range(8)),
        )

    def __enter__(self) -> Self:
        _ = self.base.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.base.__exit__(exc_type, exc, tb)

    def worker_for(self, cell: CellSchedule) -> _BarrierWorker:
        if cell.gpu_index is None:
            message = "model cell requires a GPU"
            raise AssertionError(message)
        return self.workers[int(cell.gpu_index)]

    def bank_worker_for(self, cell: CellSchedule) -> _BarrierWorker:
        return self.worker_for(cell)


@dataclass(frozen=True, slots=True)
class _BarrierModelExecutor:
    state: _LifecycleState

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        cell_id = request.cell.cell_id
        self.state.note_cell(self.state.prep_started, cell_id)
        if cell_id.startswith("wave-00-") and not self.state.wait_for_prep_release(cell_id):
            message = "model-prep release barrier timed out"
            raise AssertionError(message)
        staging_root = request.staging_root
        if staging_root is None:
            message = "model preparation requires private staging"
            raise AssertionError(message)
        staging_root.mkdir(parents=True, exist_ok=True)
        path = staging_root / f"model-{sha256_bytes(cell_id.encode())}.exr"
        _ = path.write_bytes(cell_id.encode())
        self.state.note_cell(self.state.prep_completed, cell_id)
        return ModelArmExecutionResult(
            written_snapshot_count=1,
            written_paths=(path,),
            nfe=1,
            forward_calls=1,
            gpu_ms=1,
            bank_ready_elapsed_seconds=request.bank_ready_elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class _BarrierCellRunner:
    state: _LifecycleState
    block_baselines: bool
    interrupt_after_wave_zero: bool = False

    def run(self, request: CellRunRequest) -> CellResult:
        if self.interrupt_after_wave_zero and request.cell_id.startswith("wave-01-"):
            raise KeyboardInterrupt
        if request.cell_id.startswith("wave-00-") and request.cell_id.endswith("libfuzzer-only"):
            self.state.note_cell(self.state.baseline_started, request.cell_id)
            if self.block_baselines and not self.state.baseline_release.wait(10.0):
                message = "baseline release barrier timed out"
                raise AssertionError(message)
            if self.state.abort.is_set():
                raise KeyboardInterrupt
        elif request.cell_id.startswith("wave-00-"):
            rows = load_corpus_index(_index_request(request.artifact_root))
            with self.state.condition:
                self.state.model_search_started.add(request.cell_id)
                self.state.search_index_rows[request.cell_id] = rows
                self.state.condition.notify_all()
            if self.state.abort.is_set():
                raise KeyboardInterrupt
        return FakeCellRunner().run(request)

    def cancel_active(self) -> None:
        self.state.release_all()


@dataclass(frozen=True, slots=True)
class _InterruptingModelSearchRunner:
    interrupted: bool = False

    def run(self, request: CellRunRequest) -> CellResult:
        if request.cell_id.endswith("libfuzzer-only"):
            return FakeCellRunner().run(request)
        if not self.interrupted:
            object.__setattr__(self, "interrupted", True)
            raise KeyboardInterrupt
        return FakeCellRunner().run(request)

    def cancel_active(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _ExpiredWaveCellRunner:
    requests: list[CellRunRequest] = field(default_factory=list)

    def wave_monotonic(self, scheduled_start: float) -> float:
        return scheduled_start + 10_800.1

    def run(self, request: CellRunRequest) -> CellResult:
        self.requests.append(request)
        return FakeCellRunner().run(request)


@dataclass(frozen=True, slots=True)
class _ExogenousModelExecutor:
    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        raise ExogenousCampaignError(
            cause="gpu-xid",
            evidence=f"xid:{request.cell.cell_id}",
        )


@dataclass(slots=True)
class _TermThenKillChild:
    returncode: int | None = None
    terminate_count: int = 0
    kill_count: int = 0
    communicate_timeouts: list[float | None] = field(default_factory=list)

    def communicate(
        self,
        input: bytes | None = None,  # noqa: A002
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        del input
        self.communicate_timeouts.append(timeout)
        if self.kill_count == 0:
            raise subprocess.TimeoutExpired(
                cmd="docker", timeout=0.0 if timeout is None else timeout
            )
        self.returncode = -9
        return b"", b""

    def terminate(self) -> None:
        self.terminate_count += 1

    def kill(self) -> None:
        self.kill_count += 1


@dataclass(slots=True)
class _ConcurrentCancelChild:
    returncode: int | None = None
    terminate_count: int = 0
    kill_count: int = 0
    concurrent_communicators: int = 0
    max_concurrent_communicators: int = 0
    communicate_started: threading.Event = field(default_factory=threading.Event)
    finish_communication: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def communicate(
        self,
        input: bytes | None = None,  # noqa: A002
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        del input, timeout
        with self.lock:
            self.concurrent_communicators += 1
            self.max_concurrent_communicators = max(
                self.max_concurrent_communicators, self.concurrent_communicators
            )
        self.communicate_started.set()
        if not self.finish_communication.wait(1.0):
            raise subprocess.TimeoutExpired(cmd="docker", timeout=1.0)
        with self.lock:
            self.concurrent_communicators -= 1
        self.returncode = -15
        return b"", b""

    def terminate(self) -> None:
        self.terminate_count += 1
        _ = threading.Timer(0.05, self.finish_communication.set).start()

    def kill(self) -> None:
        self.kill_count += 1
        self.finish_communication.set()


def _start_campaign(
    config: CampaignConfig,
    state: _LifecycleState,
    model: _BarrierModelExecutor,
    runner: _BarrierCellRunner,
) -> tuple[threading.Thread, list[CampaignOutcome], list[BaseException]]:
    outcomes: list[CampaignOutcome] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            outcomes.append(
                coordinate_campaign(
                    config,
                    resident_pool=_BarrierPool(state),
                    cell_runner=runner,
                    post_search=FakePostSearch(),
                    gate_runner=FakeGateRunner(_passing_gates()),
                    model_executor=model,
                )
            )
        except (CoordinatorError, AssertionError, OSError, ValueError, KeyboardInterrupt) as error:
            errors.append(error)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcomes, errors


def _assert_wave_zero_is_exact_c0(config: CampaignConfig) -> dict[str, str]:
    hashes: dict[str, str] = {}
    schedule = build_campaign_schedule(
        main_start_monotonic_seconds=config.main_start_monotonic_seconds
    )
    for cell in schedule.waves[0].cells:
        cell_id = f"wave-00-rep-{int(cell.repetition):02d}-{cell.arm}"
        root = _cell_root(config, cell_id)
        corpus = root / "corpus"
        assert tuple(path.name for path in corpus.iterdir()) == (f"c0-{config.c0_sha256}.seed",)
        rows = load_corpus_index(_index_request(root))
        assert len(rows) == 1
        assert rows[0].scheduled_elapsed_seconds == 0
        assert rows[0].cumulative_hashes == (config.c0_sha256,)
        hashes[cell_id] = rows[0].cumulative_set_sha256
    assert len(hashes) == 10
    return hashes


def _cell_root(config: CampaignConfig, cell_id: str) -> Path:
    parts = cell_id.split("-", maxsplit=4)
    return config.artifact_root / "cells" / f"wave-{parts[1]}" / f"rep-{parts[3]}" / parts[4]


def _cell_id_from_artifact(path: Path) -> str:
    arm_root = path.parent
    return f"{arm_root.parent.parent.name}-{arm_root.parent.name}-{arm_root.name}"


def _index_request(artifact_root: Path) -> CorpusIndexRequest:
    return CorpusIndexRequest(
        artifact_root=artifact_root,
        corpus_root=artifact_root / "corpus",
        store_root=artifact_root / "corpus-inputs",
        index_path=artifact_root / "corpus-index.jsonl",
    )


def _gpu_for_cell(cell_id: str) -> int:
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(70_000.0))
    return next(
        int(cell.gpu_index)
        for cell in schedule.waves[0].cells
        if cell.gpu_index is not None
        and f"wave-00-rep-{int(cell.repetition):02d}-{cell.arm}" == cell_id
    )


def _docker_launch_request(root: Path) -> SearchLaunchRequest:
    request = _request(root)
    return SearchLaunchRequest(
        cell_id=request.cell_id,
        attempt_index=0,
        attempt_root=request.artifact_root / "attempts/000000",
        artifact_root=request.artifact_root,
        corpus_root=request.corpus_root,
        crash_root=request.crash_root,
        log_root=request.log_root,
        absolute_deadline_monotonic_seconds=request.deadline_monotonic_seconds,
        remaining_seconds=10_800.0,
        fuzzer_seed=request.fuzzer_seed,
        max_input_length=request.max_input_length,
        target_timeout_seconds=request.target_timeout_seconds,
        rss_limit_mb=request.rss_limit_mb,
        cpu_index=request.cpu_index,
        dictionary_path=request.dictionary_path,
        dictionary_sha256=request.dictionary_sha256,
        build=request.build,
    )


def _qualified_config(root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(root)) | {"qualification_state": "qualified"}
    )


def _passing_gates() -> StartGateEvidence:
    return StartGateEvidence(
        hashes_match=True,
        banks_complete=True,
        read_only_cache=True,
        external_coverage_check=True,
        control_arm_equivalence=True,
        resource_map_valid=True,
        disk_space=True,
        vram=True,
        ram=True,
        swap_stable=True,
        reserved_core_load=True,
        smoke_passed=True,
    )


def _read_json(path: Path) -> dict[str, JsonValue]:
    return cast("dict[str, JsonValue]", json.loads(path.read_text(encoding="utf-8")))
