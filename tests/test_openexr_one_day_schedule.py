"""Deterministic schedule contract for the OpenEXR one-day experiment."""

from dataclasses import FrozenInstanceError
from itertools import pairwise

import pytest

from dllm_agent.experiments.openexr_one_day.models import (
    Arm,
    ModelArm,
    MonotonicSeconds,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule

EXPECTED_ARMS: tuple[Arm, ...] = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
    "libfuzzer-only",
)
EXPECTED_MODEL_ARMS: tuple[ModelArm, ...] = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
)


def test_campaign_grid_has_ten_repetitions_in_five_three_hour_waves() -> None:
    # Given
    main_start = MonotonicSeconds(70_000.0)

    # When
    schedule = build_campaign_schedule(main_start_monotonic_seconds=main_start)

    # Then
    assert schedule.preregistration.arms == EXPECTED_ARMS
    assert schedule.preregistration.repetitions == 10
    assert schedule.preregistration.wave_count == 5
    assert schedule.preregistration.wave_duration_seconds == 10_800
    assert tuple(wave.repetitions for wave in schedule.waves) == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    )
    assert len(schedule.cells) == 50
    assert {(cell.repetition, cell.arm) for cell in schedule.cells} == {
        (repetition, arm) for repetition in range(10) for arm in EXPECTED_ARMS
    }


def test_gpu_rotation_is_balanced_when_all_five_waves_are_scheduled() -> None:
    # Given
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(70_000.0))

    # When
    wave_gpu_indices = tuple(
        tuple(sorted(cell.gpu_index for cell in wave.cells if cell.gpu_index is not None))
        for wave in schedule.waves
    )
    per_arm_gpu_counts = {
        arm: tuple(
            sum(cell.arm == arm and cell.gpu_index == gpu_index for cell in schedule.cells)
            for gpu_index in range(8)
        )
        for arm in EXPECTED_MODEL_ARMS
    }

    # Then
    assert wave_gpu_indices == (tuple(range(8)),) * 5
    for counts in per_arm_gpu_counts.values():
        assert sum(counts) == 10
        assert min(counts) == 1
        assert max(counts) == 2
    assert all(cell.gpu_index is None for cell in schedule.cells if cell.arm == "libfuzzer-only")


def test_repetition_seed_namespaces_are_disjoint_and_own_each_arm_seed() -> None:
    # Given
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(70_000.0))

    # When
    namespaces = schedule.seed_namespaces

    # Then
    assert tuple(namespace.repetition for namespace in namespaces) == tuple(range(10))
    assert all(
        current.stop_seed_exclusive <= following.start_seed
        for current, following in pairwise(namespaces)
    )
    for namespace in namespaces:
        cell_seeds = tuple(
            cell.fuzzer_seed for cell in schedule.cells if cell.repetition == namespace.repetition
        )
        assert len(cell_seeds) == 5
        assert len(set(cell_seeds)) == 5
        assert all(
            namespace.start_seed <= seed < namespace.stop_seed_exclusive for seed in cell_seeds
        )


def test_cells_carry_deterministic_absolute_monotonic_deadlines() -> None:
    # Given
    main_start = MonotonicSeconds(70_000.0)

    # When
    schedule = build_campaign_schedule(main_start_monotonic_seconds=main_start)
    rebuilt = build_campaign_schedule(main_start_monotonic_seconds=main_start)

    # Then
    assert tuple(wave.start_monotonic_seconds for wave in schedule.waves) == (
        70_000.0,
        80_800.0,
        91_600.0,
        102_400.0,
        113_200.0,
    )
    assert tuple(wave.deadline_monotonic_seconds for wave in schedule.waves) == (
        80_800.0,
        91_600.0,
        102_400.0,
        113_200.0,
        124_000.0,
    )
    assert all(
        cell.deadline_monotonic_seconds == wave.deadline_monotonic_seconds
        for wave in schedule.waves
        for cell in wave.cells
    )
    assert schedule.absolute_deadline_monotonic_seconds == 124_000.0
    assert schedule == rebuilt


def test_preregistration_and_schedule_are_immutable_machine_contracts() -> None:
    # Given
    schedule = build_campaign_schedule(main_start_monotonic_seconds=MonotonicSeconds(70_000.0))
    manifest = schedule.preregistration
    assign_attribute = setattr

    # When / Then
    assert manifest.confirmatory_treatment == "selective-revision"
    assert manifest.confirmatory_control == "final-only"
    assert manifest.primary_endpoint == "target_branch_coverage_auc"
    assert manifest.target_commit == "777d231a179de1711d2a942810d9559216ab3f4c"
    with pytest.raises(FrozenInstanceError):
        assign_attribute(manifest, "repetitions", 9)
    with pytest.raises(FrozenInstanceError):
        assign_attribute(schedule, "waves", ())
