"""Parallel Magma campaign launch and collection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from dllm_agent.experiments.multitarget.campaign_core import arm_candidate_inputs
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerLaunch,
    ContainerRunResult,
    launch_container,
    wait_for_container,
)
from dllm_agent.experiments.multitarget.corpus import StagedCorpus, stage_cell_corpus
from dllm_agent.experiments.multitarget.docker_runtime import ContainerRequest, runtime_profile

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from dllm_agent.experiments.multitarget.budget import Target
    from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
    from dllm_agent.experiments.multitarget.corpus import CandidateInput
    from dllm_agent.experiments.multitarget.schedule import CampaignSchedule, CellSchedule


class LaunchCampaign(Protocol):
    """Container launch seam used by tests and the live runner."""

    def __call__(self, request: ContainerRequest) -> ContainerLaunch:
        """Launch one detached campaign container."""
        ...


class WaitCampaign(Protocol):
    """Container wait seam used by tests and the live runner."""

    def __call__(
        self,
        container_id: str,
        *,
        shared_dir: Path,
        timeout_seconds: int,
    ) -> ContainerRunResult:
        """Wait for one campaign container and collect evidence."""
        ...


@dataclass(slots=True)
class CampaignExecutionError(Exception):
    """Raised when a campaign filesystem boundary is unsafe."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CampaignExecutionPaths:
    """Experiment and pinned Magma source roots."""

    experiment_root: Path
    magma_root: Path


@dataclass(frozen=True, slots=True)
class LaunchedCell:
    """One staged and detached main-campaign cell."""

    cell: CellSchedule
    work_dir: Path
    shared_dir: Path
    staged_corpus: StagedCorpus
    request: ContainerRequest
    launch: ContainerLaunch


@dataclass(frozen=True, slots=True)
class FinishedCell:
    """One completed main campaign with persisted raw logs."""

    cell: CellSchedule
    work_dir: Path
    staged_corpus: StagedCorpus
    request: ContainerRequest
    launch: ContainerLaunch
    result: ContainerRunResult
    log_path: Path


def launch_main_campaigns(  # noqa: PLR0913 - explicit campaign boundaries.
    *,
    schedule: CampaignSchedule,
    durations: CampaignDurations,
    paths: CampaignExecutionPaths,
    run_label: str,
    initial_inputs: Mapping[Target, tuple[CandidateInput, ...]],
    revision_inputs: Mapping[Target, Mapping[str, tuple[CandidateInput, ...]]],
    launch: LaunchCampaign = launch_container,
) -> tuple[LaunchedCell, ...]:
    """Stage and detach every cell before waiting on any one campaign."""
    launched: list[LaunchedCell] = []
    for cell in schedule.cells:
        work_dir = (
            paths.experiment_root / "cells" / cell.target / cell.arm / f"rep-{cell.repetition}"
        )
        if work_dir.exists():
            raise CampaignExecutionError(reason=f"cell work directory already exists: {work_dir}")
        work_dir.mkdir(parents=True)
        shared_dir = work_dir / "shared"
        shared_dir.mkdir()
        staged = stage_cell_corpus(
            _source_corpus(paths.magma_root, cell),
            work_dir / "corpus",
            candidates=arm_candidate_inputs(
                cell.arm,
                initial_inputs[cell.target],
                revision_inputs[cell.target],
            ),
        )
        request = ContainerRequest(
            name=_container_name(run_label, cell),
            profile=runtime_profile(cell.target),
            image_id=cell.image_id,
            shared_dir=shared_dir,
            corpus_dir=staged.corpus_dir,
            duration_seconds=durations.main_seconds(cell.arm),
            fuzzer_seed=cell.seeds.main_fuzzer,
            cpu_index=cell.cpu_index,
            poll_seconds=5,
        )
        launched.append(
            LaunchedCell(
                cell=cell,
                work_dir=work_dir,
                shared_dir=shared_dir,
                staged_corpus=staged,
                request=request,
                launch=launch(request),
            )
        )
    return tuple(launched)


def wait_main_campaigns(
    launched: tuple[LaunchedCell, ...],
    *,
    wait: WaitCampaign = wait_for_container,
    max_workers: int = 15,
) -> tuple[FinishedCell, ...]:
    """Wait concurrently, then return evidence in frozen schedule order."""
    if max_workers < 1:
        raise CampaignExecutionError(reason="max_workers must be positive")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = tuple(executor.submit(_finish_one, item, wait) for item in launched)
        return tuple(future.result() for future in futures)


def _finish_one(item: LaunchedCell, wait: WaitCampaign) -> FinishedCell:
    result = wait(
        item.launch.container_id,
        shared_dir=item.shared_dir,
        timeout_seconds=item.request.duration_seconds + 60,
    )
    log_path = item.work_dir / "main.log"
    if log_path.exists():
        raise CampaignExecutionError(reason=f"campaign log already exists: {log_path}")
    _ = log_path.write_text(result.logs_text, encoding="utf-8")
    return FinishedCell(
        cell=item.cell,
        work_dir=item.work_dir,
        staged_corpus=item.staged_corpus,
        request=item.request,
        launch=item.launch,
        result=result,
        log_path=log_path,
    )


def _source_corpus(magma_root: Path, cell: CellSchedule) -> Path:
    return magma_root / "targets" / cell.target / "corpus" / runtime_profile(cell.target).program


def _container_name(run_label: str, cell: CellSchedule) -> str:
    return f"dllm-{run_label}-{cell.target}-{cell.arm}-r{cell.repetition}"
