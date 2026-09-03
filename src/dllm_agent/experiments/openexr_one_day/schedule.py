"""Deterministic schedule expansion for the OpenEXR one-day experiment."""

from typing import Final, assert_never

from dllm_agent.experiments.openexr_one_day.models import (
    ARMS,
    GPU_COUNT,
    REPETITION_COUNT,
    REPETITIONS_PER_WAVE,
    WAVE_COUNT,
    WAVE_DURATION_SECONDS,
    Arm,
    CampaignSchedule,
    CellSchedule,
    GpuIndex,
    MonotonicSeconds,
    PreregistrationManifest,
    RepetitionIndex,
    Seed,
    SeedNamespace,
    WaveIndex,
    WaveSchedule,
)

_SEED_NAMESPACE_ROOT: Final = 1_000_000
_SEED_NAMESPACE_WIDTH: Final = 1_000


def build_campaign_schedule(
    *,
    main_start_monotonic_seconds: MonotonicSeconds,
) -> CampaignSchedule:
    """Build the fixed five-wave schedule from one absolute monotonic anchor."""
    seed_namespaces = tuple(
        _seed_namespace(RepetitionIndex(repetition)) for repetition in range(REPETITION_COUNT)
    )
    return CampaignSchedule(
        preregistration=PreregistrationManifest(),
        seed_namespaces=seed_namespaces,
        waves=tuple(
            _build_wave(
                WaveIndex(wave_index),
                main_start_monotonic_seconds,
                seed_namespaces,
            )
            for wave_index in range(WAVE_COUNT)
        ),
    )


def _seed_namespace(repetition: RepetitionIndex) -> SeedNamespace:
    start_seed = Seed(_SEED_NAMESPACE_ROOT + (int(repetition) * _SEED_NAMESPACE_WIDTH))
    return SeedNamespace(
        repetition=repetition,
        start_seed=start_seed,
        stop_seed_exclusive=Seed(int(start_seed) + _SEED_NAMESPACE_WIDTH),
    )


def _build_wave(
    wave_index: WaveIndex,
    main_start: MonotonicSeconds,
    seed_namespaces: tuple[SeedNamespace, ...],
) -> WaveSchedule:
    first_repetition = RepetitionIndex(int(wave_index) * REPETITIONS_PER_WAVE)
    repetitions = (
        first_repetition,
        RepetitionIndex(int(first_repetition) + 1),
    )
    start = MonotonicSeconds(float(main_start) + (int(wave_index) * WAVE_DURATION_SECONDS))
    deadline = MonotonicSeconds(float(start) + WAVE_DURATION_SECONDS)
    return WaveSchedule(
        index=wave_index,
        repetitions=repetitions,
        start_monotonic_seconds=start,
        deadline_monotonic_seconds=deadline,
        cells=tuple(
            CellSchedule(
                repetition=repetition,
                arm=arm,
                wave_index=wave_index,
                seed_namespace=seed_namespaces[int(repetition)],
                fuzzer_seed=Seed(
                    int(seed_namespaces[int(repetition)].start_seed) + ARMS.index(arm) + 1
                ),
                gpu_index=_gpu_index(arm, repetition_offset, wave_index),
                deadline_monotonic_seconds=deadline,
            )
            for repetition_offset, repetition in enumerate(repetitions)
            for arm in ARMS
        ),
    )


def _gpu_index(
    arm: Arm,
    repetition_offset: int,
    wave_index: WaveIndex,
) -> GpuIndex | None:
    match arm:
        case "selective-revision":
            arm_offset = 0
        case "final-only":
            arm_offset = 1
        case "full-regenerate":
            arm_offset = 2
        case "structured-one-shot":
            arm_offset = 3
        case "libfuzzer-only":
            return None
        case _:
            assert_never(arm)
    return GpuIndex(
        (arm_offset + (repetition_offset * (GPU_COUNT // REPETITIONS_PER_WAVE)) + int(wave_index))
        % GPU_COUNT
    )
