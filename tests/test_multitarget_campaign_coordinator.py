"""Resident-model multitarget preparation state-machine contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dllm_agent.experiments.multitarget.budget import ExperimentSpec, Target
from dllm_agent.experiments.multitarget.campaign_coordinator import (
    TreatmentPreparationConfig,
    generate_shared_initial_banks,
    prepare_treatments,
)
from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
from dllm_agent.experiments.multitarget.candidate_bank import ProbeObservation
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.probe_runner import ProbeRunRequest, ProbeRunResult
from dllm_agent.experiments.multitarget.schedule import build_campaign_schedule
from dllm_agent.experiments.multitarget.slot_sampler import BlockDenoiseResult

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.experiments.multitarget.campaign_preparation import RunCandidateProbe

IMAGE_IDS: dict[Target, str] = {
    "libpng": "sha256:" + ("a" * 64),
    "libtiff": "sha256:" + ("b" * 64),
    "libxml2": "sha256:" + ("c" * 64),
}


@dataclass(slots=True)
class NumericTokenizer:
    """Tokenizer exposing four numeric selector tokens."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [30 + (sum(text.encode()) % 67)]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return {10: "0", 11: "1", 12: "2", 13: "3"}.get(token_ids[0], "x")

    def __len__(self) -> int:
        return 64


@dataclass(slots=True)
class DiverseModel:
    """Deterministic fake resident model with seed/position diversity."""

    calls: int = 0

    def denoise_block(  # noqa: PLR0913, PLR0917
        self,
        block_index: int,
        token_ids: tuple[int, ...],
        mask_positions: tuple[int, ...],
        allowed_token_ids: tuple[int, ...],
        previous_token_ids: tuple[int | None, ...],
        seed: int,
        step_index: int,
    ) -> BlockDenoiseResult:
        del block_index, token_ids, previous_token_ids
        self.calls += 1
        selected = tuple(
            allowed_token_ids[
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{position}:{step_index}".encode()).digest()[:4],
                    "big",
                )
                % len(allowed_token_ids)
            ]
            for position in mask_positions
        )
        return BlockDenoiseResult(
            token_ids=selected,
            confidences=tuple(0.5 + (position % 4) / 10 for position in mask_positions),
            gpu_ms=1,
        )


def test_preparation_reuses_three_shared_initial_banks_and_probes_once_per_target(
    tmp_path: Path,
) -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    model = DiverseModel()
    tokenizer = NumericTokenizer()
    probe_requests: list[ProbeRunRequest] = []
    prompt_root = tmp_path / "prompts" / "multitarget"
    prompt_root.mkdir(parents=True)
    for target in IMAGE_IDS:
        _ = (prompt_root / f"{target}-slot-particle.md").write_text(
            f"# {target}\nselector experiment\n",
            encoding="utf-8",
        )

    initial = generate_shared_initial_banks(
        schedule=schedule,
        repo_root=tmp_path,
        tokenizer=tokenizer,
        model=model,
    )
    initial_calls = model.calls
    prepared = prepare_treatments(
        initial_banks=initial,
        schedule=schedule,
        config=TreatmentPreparationConfig(
            experiment_root=tmp_path / "experiment",
            durations=CampaignDurations.smoke(probe_seconds=1, main_seconds=1),
            run_label="smoke-test",
        ),
        tokenizer=tokenizer,
        model=model,
        run_probe=_fake_probe(probe_requests),
    )

    assert set(initial) == set(IMAGE_IDS)
    assert all(bank.bank.raw_count == 32 for bank in initial.values())
    assert all(bank.bank.unique_count >= 16 for bank in initial.values())
    assert all(len(bank.selected) == 16 for bank in initial.values())
    assert len(prepared.initial_probe_batches) == 3
    assert sum(len(by_arm) for by_arm in prepared.revision_probe_batches.values()) == 9
    assert len(probe_requests) == (3 * 16) + (9 * 16)
    assert model.calls > initial_calls
    assert all(len(values) == 16 for values in prepared.initial_inputs.values())
    assert all(
        len(values) == 16
        for by_arm in prepared.revision_inputs.values()
        for values in by_arm.values()
    )
    initial_probe_names = [
        request.name for request in probe_requests if "-initial-" in request.name
    ]
    assert len(initial_probe_names) == 48
    assert all(
        name.startswith("dllm-smoke-test-") for name in probe_requests_by_name(probe_requests)
    )
    assert not any("feedback-no-remask-initial" in name for name in initial_probe_names)
    pseudo = prepared.revised_banks["libpng"]["pseudo-feedback"]
    assert all(
        observation.candidate_sha256 != observation.feedback_source_sha256
        for observation in pseudo.feedback_observations
    )


def probe_requests_by_name(requests: list[ProbeRunRequest]) -> tuple[str, ...]:
    """Return captured probe names for concise assertions."""
    return tuple(request.name for request in requests)


def _fake_probe(calls: list[ProbeRunRequest]) -> RunCandidateProbe:
    def run(request: ProbeRunRequest) -> ProbeRunResult:
        calls.append(request)
        coverage = int(request.candidate.snapshot.sha256[:6], 16) % 10_000
        observation = _observation(request, coverage)
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
                metrics=FuzzerMetrics(coverage, coverage + 1, 100, 1_000),
                canaries={},
                findings=(),
            ),
            corpus_dir=request.work_dir / "corpus",
            shared_dir=request.work_dir / "shared",
            snapshot_path=request.work_dir / "corpus" / request.candidate.snapshot.sha256,
            log_path=request.work_dir / "probe.log",
        )

    return run


def _observation(request: ProbeRunRequest, coverage: int) -> ProbeObservation:
    return ProbeObservation(
        candidate_sha256=request.candidate.snapshot.sha256,
        feedback_source_sha256=request.candidate.snapshot.sha256,
        canary_triggered=False,
        sanitizer_observed=False,
        coverage_edges=coverage,
        value_profile_features=coverage + 1,
        executions_per_second=100,
        log_sha256=hashlib.sha256(request.name.encode()).hexdigest(),
    )
