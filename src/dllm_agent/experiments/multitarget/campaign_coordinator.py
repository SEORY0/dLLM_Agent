"""Resident-model preparation state machine for multitarget campaigns."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.bank_generation import (
    BankRevisionRequest,
    GeneratedBank,
    GeneratedInitialBank,
    InitialBankRequest,
    RevisionMode,
    generate_initial_bank,
    revise_bank,
)
from dllm_agent.experiments.multitarget.budget import FEEDBACK_ARMS, TARGETS
from dllm_agent.experiments.multitarget.campaign_preparation import (
    CandidateProbeBatchRequest,
    CandidateProbeBatchResult,
    RunCandidateProbe,
    candidates_to_inputs,
    revised_candidates_to_inputs,
    run_candidate_micro_probes,
)
from dllm_agent.experiments.multitarget.candidate_bank import Candidate
from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile

if TYPE_CHECKING:
    from collections.abc import Mapping
    from concurrent.futures import Future
    from pathlib import Path

    from dllm_agent.experiments.multitarget.budget import FeedbackArm, Target
    from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
    from dllm_agent.experiments.multitarget.corpus import CandidateInput
    from dllm_agent.experiments.multitarget.schedule import CampaignSchedule, CellSchedule
    from dllm_agent.experiments.multitarget.slot_sampler import SlotModel, SlotTokenizer

_INITIAL_STEPS = 8
_REVISION_STEPS = 2
_RAW_PARTICLES = 32


@dataclass(frozen=True, slots=True)
class TreatmentPreparationConfig:
    """Physical paths, phase duration, and dedicated probe CPUs."""

    experiment_root: Path
    durations: CampaignDurations
    run_label: str
    initial_probe_cpu_start: int = 15
    revision_probe_cpu_start: int = 15
    poll_seconds: int = 5


@dataclass(frozen=True, slots=True)
class PreparedTreatments:
    """Initial/revised populations and both executable-feedback rounds."""

    initial_banks: Mapping[Target, GeneratedInitialBank]
    initial_probe_batches: Mapping[Target, CandidateProbeBatchResult]
    revised_banks: Mapping[Target, Mapping[FeedbackArm, GeneratedBank]]
    revision_probe_batches: Mapping[Target, Mapping[FeedbackArm, CandidateProbeBatchResult]]
    initial_inputs: Mapping[Target, tuple[CandidateInput, ...]]
    revision_inputs: Mapping[Target, Mapping[str, tuple[CandidateInput, ...]]]


def generate_shared_initial_banks(
    *,
    schedule: CampaignSchedule,
    repo_root: Path,
    tokenizer: SlotTokenizer,
    model: SlotModel,
) -> dict[Target, GeneratedInitialBank]:
    """Generate one physical 32-trajectory bank for each target."""
    banks: dict[Target, GeneratedInitialBank] = {}
    for target in TARGETS:
        cell = _cell(schedule, target, "structured-one-shot")
        seeds = tuple(cell.seeds.model_particles + offset for offset in range(_RAW_PARTICLES))
        banks[target] = generate_initial_bank(
            InitialBankRequest(
                target_id=target,
                prompt_path=repo_root / "prompts" / "multitarget" / f"{target}-slot-particle.md",
                initial_seeds=seeds,
                initial_steps=_INITIAL_STEPS,
            ),
            tokenizer=tokenizer,
            model=model,
        )
    return banks


def prepare_treatments(  # noqa: PLR0913 - explicit live dependencies.
    *,
    initial_banks: Mapping[Target, GeneratedInitialBank],
    schedule: CampaignSchedule,
    config: TreatmentPreparationConfig,
    tokenizer: SlotTokenizer,
    model: SlotModel,
    run_probe: RunCandidateProbe,
) -> PreparedTreatments:
    """Run shared probes, three revision treatments, and revision probes."""
    initial_batches = _initial_probes(initial_banks, schedule, config, run_probe)
    revised = _revise_all(initial_banks, initial_batches, schedule, tokenizer, model)
    revision_batches = _revision_probes(revised, schedule, config, run_probe)
    return PreparedTreatments(
        initial_banks=initial_banks,
        initial_probe_batches=initial_batches,
        revised_banks=revised,
        revision_probe_batches=revision_batches,
        initial_inputs={
            target: candidates_to_inputs(initial_banks[target].selected) for target in TARGETS
        },
        revision_inputs={
            target: {
                arm: revised_candidates_to_inputs(revised[target][arm].revised)
                for arm in FEEDBACK_ARMS
            }
            for target in TARGETS
        },
    )


def _initial_probes(
    banks: Mapping[Target, GeneratedInitialBank],
    schedule: CampaignSchedule,
    config: TreatmentPreparationConfig,
    run_probe: RunCandidateProbe,
) -> dict[Target, CandidateProbeBatchResult]:
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as executor:
        futures = {
            target: executor.submit(
                run_candidate_micro_probes,
                CandidateProbeBatchRequest(
                    target_label=f"{config.run_label}-{target}-initial",
                    profile=runtime_profile(target),
                    image_id=_cell(schedule, target, "selective-revision").image_id,
                    work_root=config.experiment_root / "probes" / "initial" / target,
                    candidates=banks[target].selected,
                    duration_seconds=config.durations.probe_seconds,
                    probe_seed_base=_cell(
                        schedule, target, "selective-revision"
                    ).seeds.initial_probe,
                    cpu_index=config.initial_probe_cpu_start + index,
                    poll_seconds=config.poll_seconds,
                ),
                run_probe=run_probe,
            )
            for index, target in enumerate(TARGETS)
        }
        return {target: futures[target].result() for target in TARGETS}


def _revise_all(
    banks: Mapping[Target, GeneratedInitialBank],
    initial_batches: Mapping[Target, CandidateProbeBatchResult],
    schedule: CampaignSchedule,
    tokenizer: SlotTokenizer,
    model: SlotModel,
) -> dict[Target, dict[FeedbackArm, GeneratedBank]]:
    revised: dict[Target, dict[FeedbackArm, GeneratedBank]] = {}
    for target in TARGETS:
        by_arm: dict[FeedbackArm, GeneratedBank] = {}
        for arm in FEEDBACK_ARMS:
            cell = _cell(schedule, target, arm)
            by_arm[arm] = revise_bank(
                banks[target],
                BankRevisionRequest(
                    mode=_revision_mode(arm),
                    revision_steps=_REVISION_STEPS,
                    observations=initial_batches[target].observations,
                    pseudo_seed=cell.seeds.pseudo_permutation,
                    revision_seed_base=cell.seeds.model_revision,
                ),
                tokenizer=tokenizer,
                model=model,
            )
        revised[target] = by_arm
    return revised


def _revision_probes(
    revised: Mapping[Target, Mapping[FeedbackArm, GeneratedBank]],
    schedule: CampaignSchedule,
    config: TreatmentPreparationConfig,
    run_probe: RunCandidateProbe,
) -> dict[Target, dict[FeedbackArm, CandidateProbeBatchResult]]:
    with ThreadPoolExecutor(max_workers=len(TARGETS) * len(FEEDBACK_ARMS)) as executor:
        futures: dict[tuple[Target, FeedbackArm], Future[CandidateProbeBatchResult]] = {}
        worker_index = 0
        for target in TARGETS:
            for arm in FEEDBACK_ARMS:
                cell = _cell(schedule, target, arm)
                futures[(target, arm)] = executor.submit(
                    run_candidate_micro_probes,
                    CandidateProbeBatchRequest(
                        target_label=f"{config.run_label}-{target}-{arm}-revision",
                        profile=runtime_profile(target),
                        image_id=cell.image_id,
                        work_root=config.experiment_root / "probes" / "revision" / target / arm,
                        candidates=_revision_probe_candidates(revised[target][arm]),
                        duration_seconds=config.durations.probe_seconds,
                        probe_seed_base=cell.seeds.revision_probe,
                        cpu_index=config.revision_probe_cpu_start + worker_index,
                        poll_seconds=config.poll_seconds,
                    ),
                    run_probe=run_probe,
                )
                worker_index += 1
        return {
            target: {arm: futures[(target, arm)].result() for arm in FEEDBACK_ARMS}
            for target in TARGETS
        }


def _revision_probe_candidates(bank: GeneratedBank) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            sample=item.sample,
            snapshot=item.snapshot,
            source_sample_sha256=hashlib.sha256(canonical_json_bytes(item.sample)).hexdigest(),
        )
        for item in bank.revised
    )


def _revision_mode(arm: FeedbackArm) -> RevisionMode:
    return {
        "feedback-no-remask": RevisionMode.FULL,
        "selective-revision": RevisionMode.SELECTIVE,
        "pseudo-feedback": RevisionMode.PSEUDO,
    }[arm]


def _cell(schedule: CampaignSchedule, target: Target, arm: str) -> CellSchedule:
    for cell in schedule.cells:
        if cell.target == target and cell.arm == arm and cell.repetition == 0:
            return cell
    msg = f"schedule missing cell: {target}/{arm}/0"
    raise ValueError(msg)
