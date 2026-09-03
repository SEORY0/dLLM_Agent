"""Main campaign launch/wait integration contracts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from dllm_agent.experiments.multitarget.budget import ExperimentSpec, Target
from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
from dllm_agent.experiments.multitarget.campaign_execution import (
    CampaignExecutionPaths,
    launch_main_campaigns,
    wait_main_campaigns,
)
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerLaunch,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.corpus import CandidateInput
from dllm_agent.experiments.multitarget.schedule import build_campaign_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from dllm_agent.experiments.multitarget.docker_runtime import ContainerRequest

IMAGE_IDS: dict[Target, str] = {
    "libpng": "sha256:" + ("a" * 64),
    "libtiff": "sha256:" + ("b" * 64),
    "libxml2": "sha256:" + ("c" * 64),
}


def test_launches_all_cells_with_exact_arm_population_and_duration(tmp_path: Path) -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    paths = _paths(tmp_path)
    initial: dict[Target, tuple[CandidateInput, ...]] = {
        target: _population(f"{target}-initial") for target in IMAGE_IDS
    }
    revisions: dict[Target, Mapping[str, tuple[CandidateInput, ...]]] = {
        target: {
            arm: _population(f"{target}-{arm}")
            for arm in ("feedback-no-remask", "selective-revision", "pseudo-feedback")
        }
        for target in IMAGE_IDS
    }
    requests: list[ContainerRequest] = []

    def launch(request: ContainerRequest) -> ContainerLaunch:
        requests.append(request)
        return ContainerLaunch(container_id=f"container-{len(requests):02d}")

    launched = launch_main_campaigns(
        schedule=schedule,
        durations=CampaignDurations.smoke(probe_seconds=1, main_seconds=7),
        paths=paths,
        run_label="smoke-01",
        initial_inputs=initial,
        revision_inputs=revisions,
        launch=launch,
    )

    assert len(launched) == 15
    assert len(requests) == 15
    assert tuple(request.cpu_index for request in requests) == tuple(range(15))
    assert all(request.duration_seconds == 7 for request in requests)
    baseline = launched[0]
    one_shot = launched[1]
    selective = next(item for item in launched if item.cell.arm == "selective-revision")
    assert baseline.staged_corpus.candidate_file_count == 0
    assert one_shot.staged_corpus.candidate_file_count == 16
    assert selective.staged_corpus.candidate_file_count == 16
    assert selective.request.fuzzer_seed == selective.cell.seeds.main_fuzzer
    assert selective.request.image_id == selective.cell.image_id
    assert selective.request.duration_seconds == 7
    assert selective.request.corpus_dir == selective.staged_corpus.corpus_dir
    assert selective.request.shared_dir == selective.shared_dir


def test_wait_collects_results_in_schedule_order_and_persists_logs(tmp_path: Path) -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    launched = launch_main_campaigns(
        schedule=schedule,
        durations=CampaignDurations.smoke(probe_seconds=1, main_seconds=1),
        paths=_paths(tmp_path),
        run_label="smoke-02",
        initial_inputs={target: _population(f"{target}-initial") for target in IMAGE_IDS},
        revision_inputs={
            target: {
                arm: _population(f"{target}-{arm}")
                for arm in (
                    "feedback-no-remask",
                    "selective-revision",
                    "pseudo-feedback",
                )
            }
            for target in IMAGE_IDS
        },
        launch=lambda request: ContainerLaunch(container_id=request.name),
    )

    def wait(
        container_id: str,
        *,
        shared_dir: Path,
        timeout_seconds: int,
    ) -> ContainerRunResult:
        assert timeout_seconds == 61
        assert shared_dir.is_dir()
        return _result(container_id)

    finished = wait_main_campaigns(launched, wait=wait, max_workers=15)

    assert tuple(item.cell for item in finished) == tuple(item.cell for item in launched)
    assert all(item.result.exit_code == 0 for item in finished)
    assert all(item.log_path.read_text(encoding="utf-8") for item in finished)
    assert finished[0].log_path.read_text(encoding="utf-8") == launched[0].launch.container_id


def _paths(tmp_path: Path) -> CampaignExecutionPaths:
    magma_root = tmp_path / "magma"
    for target, program in {
        "libpng": "libpng_read_fuzzer",
        "libtiff": "tiff_read_rgba_fuzzer",
        "libxml2": "libxml2_xml_read_memory_fuzzer",
    }.items():
        source = magma_root / "targets" / target / "corpus" / program
        source.mkdir(parents=True)
        _ = (source / "seed").write_bytes(target.encode())
    return CampaignExecutionPaths(experiment_root=tmp_path / "experiment", magma_root=magma_root)


def _population(prefix: str) -> tuple[CandidateInput, ...]:
    values: list[CandidateInput] = []
    for index in range(16):
        content = f"{prefix}-{index}".encode()
        values.append(
            CandidateInput(
                snapshot_sha256=hashlib.sha256(content).hexdigest(),
                content=content,
            )
        )
    return tuple(values)


def _result(container_id: str) -> ContainerRunResult:
    return ContainerRunResult(
        container_id=container_id,
        exit_code=0,
        docker_status="exited",
        oom_killed=False,
        timed_out=False,
        process_error=None,
        logs_text=container_id,
        log_sha256=hashlib.sha256(container_id.encode()).hexdigest(),
        metrics=FuzzerMetrics(1, 2, 3, 4),
        canaries={},
        findings=(),
    )
