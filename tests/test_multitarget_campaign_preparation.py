"""Campaign preparation contracts for selected and revised multitarget particles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from dllm_agent.experiments.multitarget.bank_generation import RevisedCandidate
from dllm_agent.experiments.multitarget.campaign_core import probe_seed
from dllm_agent.experiments.multitarget.campaign_preparation import (
    CampaignPreparationError,
    CandidateProbeBatchRequest,
    candidates_to_inputs,
    revised_candidates_to_inputs,
    run_candidate_micro_probes,
)
from dllm_agent.experiments.multitarget.candidate_bank import Candidate, ProbeObservation
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile
from dllm_agent.experiments.multitarget.probe_runner import ProbeRunRequest, ProbeRunResult
from dllm_agent.experiments.multitarget.slot_sampler import SlotSample
from dllm_agent.targets.structured import MaterializedSnapshot

if TYPE_CHECKING:
    from pathlib import Path


IMAGE_ID = "sha256:" + ("a" * 64)


@dataclass(frozen=True, slots=True)
class ProbeCall:
    """Captured micro-probe request evidence."""

    request: ProbeRunRequest


def test_candidates_to_inputs_accepts_exactly_sixteen_unique_selected_candidates() -> None:
    # Given
    selected = tuple(_candidate(f"initial-{index}") for index in range(16))

    # When
    inputs = candidates_to_inputs(selected)

    # Then
    assert tuple(item.snapshot_sha256 for item in inputs) == tuple(
        candidate.snapshot.sha256 for candidate in selected
    )
    assert tuple(item.content for item in inputs) == tuple(
        candidate.snapshot.content for candidate in selected
    )


def test_revised_candidates_to_inputs_uses_revised_snapshot_order() -> None:
    # Given
    revised = tuple(
        RevisedCandidate(
            parent=_candidate(f"parent-{index}"),
            sample=_sample(index),
            snapshot=_snapshot(f"revised-{index}"),
            used_revision=True,
        )
        for index in range(16)
    )

    # When
    inputs = revised_candidates_to_inputs(revised)

    # Then
    assert tuple(item.snapshot_sha256 for item in inputs) == tuple(
        candidate.snapshot.sha256 for candidate in revised
    )
    assert tuple(item.content for item in inputs) == tuple(
        candidate.snapshot.content for candidate in revised
    )


def test_candidates_to_inputs_rejects_not_sixteen_populations() -> None:
    # Given / When / Then
    with pytest.raises(CampaignPreparationError, match="exactly 16 unique"):
        _ = candidates_to_inputs(tuple(_candidate(f"short-{index}") for index in range(15)))


def test_candidates_to_inputs_rejects_duplicate_populations() -> None:
    # Given
    duplicate = _candidate("duplicate")

    # When / Then
    with pytest.raises(CampaignPreparationError, match="exactly 16 unique"):
        _ = candidates_to_inputs(tuple(duplicate for _ in range(16)))


def test_run_candidate_micro_probes_runs_sixteen_particles_sequentially(
    tmp_path: Path,
) -> None:
    # Given
    selected = tuple(_candidate(f"probe-{index}") for index in range(16))
    calls: list[ProbeCall] = []

    def run_probe(request: ProbeRunRequest) -> ProbeRunResult:
        calls.append(ProbeCall(request=request))
        return _probe_result(request, executions_per_second=len(calls))

    # When
    result = run_candidate_micro_probes(
        CandidateProbeBatchRequest(
            target_label="libpng",
            profile=runtime_profile("libpng"),
            image_id=IMAGE_ID,
            work_root=tmp_path / "preparation",
            candidates=selected,
            duration_seconds=30,
            probe_seed_base=20260821,
            cpu_index=7,
            poll_seconds=5,
        ),
        run_probe=run_probe,
    )

    # Then
    assert tuple(item.request.candidate for item in calls) == selected
    assert tuple(item.request.name for item in calls) == tuple(
        f"dllm-libpng-probe-p{index:02d}" for index in range(16)
    )
    assert tuple(item.request.work_dir for item in calls) == tuple(
        tmp_path / "preparation" / f"particle-{index:02d}" for index in range(16)
    )
    assert tuple(item.request.fuzzer_seed for item in calls) == tuple(
        probe_seed(20260821, index) for index in range(16)
    )
    assert all(item.request.profile == runtime_profile("libpng") for item in calls)
    assert all(item.request.image_id == IMAGE_ID for item in calls)
    assert all(item.request.duration_seconds == 30 for item in calls)
    assert all(item.request.cpu_index == 7 for item in calls)
    assert all(item.request.poll_seconds == 5 for item in calls)
    assert result.observations == tuple(item.observation for item in result.results)
    assert tuple(observation.candidate_sha256 for observation in result.observations) == tuple(
        candidate.snapshot.sha256 for candidate in selected
    )
    assert tuple(observation.executions_per_second for observation in result.observations) == tuple(
        range(1, 17)
    )


def test_run_candidate_micro_probes_rejects_invalid_population_before_running(
    tmp_path: Path,
) -> None:
    # Given
    request = CandidateProbeBatchRequest(
        target_label="libpng",
        profile=runtime_profile("libpng"),
        image_id=IMAGE_ID,
        work_root=tmp_path / "preparation",
        candidates=tuple(_candidate(f"short-{index}") for index in range(15)),
        duration_seconds=30,
        probe_seed_base=20260821,
        cpu_index=7,
        poll_seconds=5,
    )

    # When / Then
    with pytest.raises(CampaignPreparationError, match="exactly 16 unique"):
        _ = run_candidate_micro_probes(request, run_probe=_forbidden_probe)


def _candidate(label: str) -> Candidate:
    snapshot = _snapshot(label)
    return Candidate(
        sample=_sample(int(hashlib.sha256(label.encode()).hexdigest()[:4], 16)),
        snapshot=snapshot,
        source_sample_sha256=hashlib.sha256(f"sample-{label}".encode()).hexdigest(),
    )


def _sample(seed: int) -> SlotSample:
    selector = seed % 251
    return SlotSample(
        token_ids=(selector,),
        selected_slot_token_ids=(selector,),
        selectors=(selector,),
        slot_positions=(0,),
        slot_confidences=(0.5,),
        trajectory_steps=(),
        state_sha256=hashlib.sha256(f"state-{seed}".encode()).hexdigest(),
        parent_state_sha256=None,
        requested_steps=1,
        nfe=1,
        forward_calls=1,
        gpu_ms=1,
    )


def _snapshot(label: str) -> MaterializedSnapshot:
    content = label.encode()
    return MaterializedSnapshot(
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        anchor_sha256="b" * 64,
        field_spans=(),
    )


def _probe_result(request: ProbeRunRequest, *, executions_per_second: int) -> ProbeRunResult:
    observation = ProbeObservation(
        candidate_sha256=request.candidate.snapshot.sha256,
        feedback_source_sha256=request.candidate.snapshot.sha256,
        canary_triggered=False,
        sanitizer_observed=False,
        coverage_edges=10,
        value_profile_features=20,
        executions_per_second=executions_per_second,
        log_sha256=hashlib.sha256(request.name.encode()).hexdigest(),
    )
    return ProbeRunResult(
        observation=observation,
        run_result=ContainerRunResult(
            container_id=request.name,
            exit_code=0,
            docker_status="exited",
            oom_killed=False,
            timed_out=False,
            process_error=None,
            logs_text=request.name,
            log_sha256=observation.log_sha256,
            metrics=FuzzerMetrics(10, 20, executions_per_second, 1000),
            canaries={},
            findings=(),
        ),
        corpus_dir=request.work_dir / "corpus",
        shared_dir=request.work_dir / "shared",
        snapshot_path=request.work_dir / "corpus" / f"{request.candidate.snapshot.sha256}.bin",
        log_path=request.work_dir / "probe.log",
    )


def _forbidden_probe(request: ProbeRunRequest) -> ProbeRunResult:
    _ = request
    pytest.fail("probe seam must not run")
