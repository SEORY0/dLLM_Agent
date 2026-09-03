"""Deterministic schedule and seed provenance for the multitarget campaign."""

from typing import cast

import pytest

from dllm_agent.experiments.multitarget.budget import (
    ARMS,
    FEEDBACK_ARMS,
    TARGETS,
    ExperimentSpec,
    Target,
)
from dllm_agent.experiments.multitarget.schedule import (
    ScheduleError,
    build_campaign_schedule,
)

IMAGE_IDS: dict[Target, str] = {
    "libpng": "sha256:" + ("a" * 64),
    "libtiff": "sha256:" + ("b" * 64),
    "libxml2": "sha256:" + ("c" * 64),
}


def test_default_schedule_expands_exact_fifteen_cpu_pinned_cells() -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)

    assert len(schedule.cells) == 15
    assert tuple(cell.cpu_index for cell in schedule.cells) == tuple(range(15))
    assert {(cell.target, cell.arm, cell.repetition) for cell in schedule.cells} == {
        (target, arm, 0) for target in TARGETS for arm in ARMS
    }
    assert len(schedule.manifest_hash) == 64
    assert (
        schedule.manifest_hash
        == build_campaign_schedule(
            ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS
        ).manifest_hash
    )


def test_schedule_records_exact_worker_budget_by_arm() -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)

    for cell in schedule.cells:
        assert cell.allocation.search_worker_seconds == 86_400
        parts = cell.worker_ledger
        assert sum(item.worker_seconds for item in parts) == 86_400
        if cell.arm in FEEDBACK_ARMS:
            assert [(item.phase, item.worker_seconds) for item in parts] == [
                ("initial-micro-probe", 480),
                ("revision-micro-probe", 480),
                ("main-campaign", 85_440),
            ]
        else:
            assert [(item.phase, item.worker_seconds) for item in parts] == [
                ("main-campaign", 86_400),
            ]


def test_seed_namespaces_share_only_declared_target_level_physical_work() -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    values: dict[int, tuple[str, str, str]] = {}
    shared_components = {"model_particles", "initial_probe"}
    target_shared: dict[tuple[str, str], int] = {}

    for cell in schedule.cells:
        for component, seed in cell.seeds.as_dict().items():
            if component in shared_components:
                key = (cell.target, component)
                _ = target_shared.setdefault(key, seed)
                assert target_shared[key] == seed
                continue
            owner = (cell.target, cell.arm, component)
            assert seed not in values, (seed, values.get(seed), owner)
            values[seed] = owner
    assert len(target_shared) == len(TARGETS) * len(shared_components)


def test_manifest_binds_cells_seeds_and_image_digests() -> None:
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=IMAGE_IDS)
    manifest = schedule.to_manifest()

    assert manifest["raw_particle_count"] == 32
    assert manifest["minimum_unique_count"] == 16
    assert manifest["cell_count"] == 15
    cells = cast("tuple[object, ...]", manifest["cells"])
    assert isinstance(cells, tuple)
    first = cast("dict[str, object]", cells[0])
    assert isinstance(first, dict)
    assert first["target"] == "libpng"
    assert first["arm"] == "libfuzzer-only"
    assert first["cpu_index"] == 0
    image_id = cast("str", first["image_id"])
    assert isinstance(image_id, str)
    assert image_id.startswith("sha256:")
    seeds = cast("dict[str, int]", first["seeds"])
    assert isinstance(seeds, dict)
    assert seeds["model_particles"] != seeds["main_fuzzer"]
    assert seeds["model_revision"] != seeds["revision_probe"]
    assert first["shared_model_bank"] is False
    dllm_cell = next(
        cast("dict[str, object]", cell)
        for cell in cells
        if cast("dict[str, object]", cell)["arm"] == "structured-one-shot"
    )
    assert dllm_cell["shared_model_bank"] is True


def test_schedule_rejects_insufficient_cpus_or_invalid_spec() -> None:
    with pytest.raises(ScheduleError, match="available_cpus"):
        _ = build_campaign_schedule(ExperimentSpec(), available_cpus=14, image_ids=IMAGE_IDS)

    with pytest.raises(ScheduleError, match="raw_particle_count"):
        _ = build_campaign_schedule(
            ExperimentSpec(raw_particle_count=31), available_cpus=40, image_ids=IMAGE_IDS
        )

    with pytest.raises(ScheduleError, match="minimum_unique_count"):
        _ = build_campaign_schedule(
            ExperimentSpec(minimum_unique_count=33), available_cpus=40, image_ids=IMAGE_IDS
        )

    with pytest.raises(ScheduleError, match="image_ids"):
        _ = build_campaign_schedule(
            ExperimentSpec(),
            available_cpus=40,
            image_ids=cast("dict[Target, str]", {}),
        )

    with pytest.raises(ScheduleError, match="sha256"):
        _ = build_campaign_schedule(
            ExperimentSpec(),
            available_cpus=40,
            image_ids={**IMAGE_IDS, "libpng": "libpng:latest"},
        )
