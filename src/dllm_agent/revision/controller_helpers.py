# pyright: reportUnnecessaryComparison=false
# noqa: SIZE_OK
"""Private helpers for the deterministic PoC revision controller."""

from typing import assert_never

import anyio
from anyio.to_thread import run_sync
from pydantic import ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    Particle,
    ProbeRecord,
    ProbeRunner,
    PseudoFeedbackParticleCountError,
    RevisionRequest,
    RevisionResult,
    RevisionRunRequest,
    RevisionSampler,
    crash_artifacts_from_results,
    feedback_from_results,
    online_revision_policy,
)
from dllm_agent.revision.models import (
    FieldAttribution,
    ProbeFeedback,
    RemaskDecision,
    RevisionConfig,
)
from dllm_agent.revision.policy import attribute_feedback, decide_remask
from dllm_agent.targets.openexr.particle import (
    FIELD_ORDER,
    MATERIALIZED_MUTABLE_FIELDS,
    ParticleDecodeError,
    SelectorSlots,
    decode_particle,
    materialize_particle,
)
from dllm_agent.targets.openexr.probe import ProbeRequest, ProbeResult

_MIN_PSEUDO_PARTICLES = 2
_EPOCH_SEED_STRIDE = 1_000_000
_REVISION_SEED_OFFSET = 100_000
_RESAMPLE_SEED_OFFSET = 200_000


def initial_result(  # noqa: D103, PLR0913, PLR0917 - sampler boundary needs fixed inputs.
    config: RevisionConfig,
    run_request: RevisionRunRequest,
    sampler: RevisionSampler,
    particle_id: str,
    index: int,
    steps: int | None = None,
) -> RevisionResult | None:
    requested_steps = config.initial_steps if steps is None else steps
    seed = run_request.particle_seed_base + index if steps is None else index
    if run_request.arm == "libfuzzer-only":
        first = index % 4
        second = (index + 1) % 4
        third = (index + 2) % 4
        fourth = (index + 3) % 4
        # fmt: off
        selectors: SelectorSlots = (first, second, third, fourth, first, second, third, fourth, first, second)  # noqa: E501
        # fmt: on
        return RevisionResult(
            selectors=selectors,
            token_ids=selectors,
            slot_positions=tuple(range(len(FIELD_ORDER))),
            selected_slot_token_ids=(),
            token_state_sha256=sha256_bytes(f"libfuzzer:{particle_id}".encode()),
            transfer_evidence_sha256=sha256_bytes(b"libfuzzer-transfer"),
            rng_state_sha256=sha256_bytes(f"libfuzzer-rng:{seed}".encode()),
            nfe=0,
            forward_calls=0,
            gpu_ms=0,
            cache_hit=False,
            trajectory_uncertainty=0.0,
            suggested_field="width",
        )
    try:
        return sampler.initial(
            InitialParticleRequest(
                particle_id=particle_id,
                seed=seed,
                steps=requested_steps,
                arm=run_request.arm,
                mode="initial",
            )
        )
    except (ValidationError, ParticleDecodeError):
        return None


def materialized_particle(  # noqa: D103
    particle_id: str, result: RevisionResult
) -> Particle | None:
    try:
        snapshot = materialize_particle(decode_particle(result.selectors, "validity-preserving"))
    except ParticleDecodeError:
        return None
    return Particle(particle_id=particle_id, result=result, snapshot=snapshot)


async def probe_particles(  # noqa: D103
    config: RevisionConfig, probe: ProbeRunner, particles: list[Particle], dedup_count: int
) -> list[ProbeRecord]:
    limiter = anyio.CapacityLimiter(max(1, min(len(config.fuzzer_seeds), config.particle_count)))
    results: dict[tuple[str, int], ProbeResult] = {}

    async def run_one(particle: Particle, seed: int) -> None:
        request = ProbeRequest(
            snapshot=particle.snapshot.content,
            fuzzer_seed=seed,
            duration_seconds=config.probe_seconds,
            max_len=max(len(particle.snapshot.content), 1),
        )
        results[(particle.particle_id, seed)] = await run_sync(probe.run, request, limiter=limiter)

    async with anyio.create_task_group() as task_group:
        for particle in particles:
            for seed in config.fuzzer_seeds:
                _ = task_group.start_soon(run_one, particle, seed)
    return [
        _probe_record(
            config,
            particle,
            tuple(results[(particle.particle_id, seed)] for seed in config.fuzzer_seeds),
            dedup_count,
        )
        for particle in sorted(particles, key=lambda item: item.particle_id)
    ]


def _probe_record(
    config: RevisionConfig,
    particle: Particle,
    results: tuple[ProbeResult, ...],
    dedup_count: int,
) -> ProbeRecord:
    return ProbeRecord(
        particle=particle,
        feedback=feedback_from_results(config, particle, results, dedup_count),
        cache_hit=any(result.cache_hit for result in results),
        crash_artifacts=crash_artifacts_from_results(results),
    )


def decide_next(  # noqa: D103
    epoch: int, records: list[ProbeRecord], attributions: list[FieldAttribution]
) -> RemaskDecision:
    latest = tuple(record.feedback for record in records)
    if len(attributions) > 0:
        return decide_remask(tuple(attributions), epoch=epoch + 1, latest_feedback=latest)
    suggested = records[0].particle.result.suggested_field
    bootstrap = attribute_feedback(records[0].feedback, records[0].feedback, (suggested,))
    return decide_remask(bootstrap, epoch=epoch + 1, latest_feedback=latest, prefer_untried=False)


def revise_unique(  # noqa: D103, PLR0913, PLR0917
    config: RevisionConfig,
    run_request: RevisionRunRequest,
    sampler: RevisionSampler,
    records: list[ProbeRecord],
    remask: RemaskDecision,
    seen: set[str],
    epoch: int,
) -> tuple[list[Particle], list[Particle], int, int]:
    particles: list[Particle] = []
    duplicates: list[Particle] = []
    dedup = 0
    rejected = 0
    batch_seen = set(seen)
    field = remask.fields[0]
    revision_policy = online_revision_policy(run_request.arm)
    remask_scope = "selected-field" if revision_policy is None else revision_policy.remask_scope
    match remask_scope:
        case "selected-field":
            remask_fields = remask.fields
        case "all-mutable":
            remask_fields = MATERIALIZED_MUTABLE_FIELDS
        case unreachable:
            assert_never(unreachable)
    bitmap = sum(1 << FIELD_ORDER.index(remask_field) for remask_field in remask_fields)
    for record in records:
        particle_index = int(record.particle.particle_id.removeprefix("p"))
        try:
            result = sampler.revise(
                RevisionRequest(
                    particle_id=record.particle.particle_id,
                    parent_token_ids=record.particle.result.token_ids,
                    parent_selectors=record.particle.result.selectors,
                    slot_positions=record.particle.result.slot_positions,
                    remask_fields=remask_fields,
                    changed_field=field,
                    remask_bitmap=bitmap,
                    frozen_bitmap=((1 << len(FIELD_ORDER)) - 1) ^ bitmap,
                    revision_steps=config.revision_steps,
                    seed=_epoch_seed(
                        run_request.particle_seed_base,
                        epoch,
                        particle_index,
                        _REVISION_SEED_OFFSET,
                    ),
                    rng_state_sha256=record.particle.result.rng_state_sha256,
                    feedback_digest=record.feedback.feedback_sha256,
                    mode="selective-revision/v1",
                )
            )
        except (ValidationError, ParticleDecodeError):
            rejected += 1
            continue
        particle = materialized_particle(record.particle.particle_id, result)
        if particle is not None and particle.snapshot.sha256 not in batch_seen:
            particles.append(particle)
            batch_seen.add(particle.snapshot.sha256)
        elif particle is not None:
            duplicates.append(particle)
            dedup += 1
        else:
            rejected += 1
    return particles, duplicates, dedup, rejected


def attribute_records(  # noqa: D103
    run_request: RevisionRunRequest,
    parents: list[ProbeRecord],
    children: list[ProbeRecord],
    remask: RemaskDecision,
) -> tuple[FieldAttribution, ...]:
    parent_feedback = _rotated_feedback(run_request, parents)
    by_child = {record.particle.particle_id: record.feedback for record in children}
    return tuple(
        attribution
        for parent in parents
        if parent.particle.particle_id in by_child
        if parent.particle.particle_id in parent_feedback
        for attribution in attribute_feedback(
            parent_feedback[parent.particle.particle_id],
            by_child[parent.particle.particle_id],
            remask.fields,
        )
    )


def _rotated_feedback(
    run_request: RevisionRunRequest, records: list[ProbeRecord]
) -> dict[str, ProbeFeedback]:
    particle_ids = [record.particle.particle_id for record in records]
    feedback = [record.feedback for record in records]
    if run_request.arm != "pseudo-feedback":
        return dict(zip(particle_ids, feedback, strict=True))
    if len(records) < _MIN_PSEUDO_PARTICLES:
        return {}
    shift = (run_request.pseudo_feedback_rotation_seed % (len(records) - 1)) + 1
    return dict(zip(particle_ids, feedback[shift:] + feedback[:shift], strict=True))


def resample_unique(  # noqa: D103
    config: RevisionConfig,
    run_request: RevisionRunRequest,
    sampler: RevisionSampler,
    epoch: int,
    seen: set[str],
) -> tuple[list[Particle], list[Particle], int, int]:
    particles: list[Particle] = []
    duplicates: list[Particle] = []
    dedup = 0
    rejected = 0
    batch_seen = set(seen)
    for index in range(config.particle_count):
        seed_index = _epoch_seed(
            run_request.particle_seed_base, epoch, index, _RESAMPLE_SEED_OFFSET
        )
        result = initial_result(
            config, run_request, sampler, f"p{index:04d}", seed_index, config.revision_steps
        )
        particle = None if result is None else materialized_particle(f"p{index:04d}", result)
        if particle is not None and particle.snapshot.sha256 not in batch_seen:
            particles.append(particle)
            batch_seen.add(particle.snapshot.sha256)
        elif particle is not None:
            duplicates.append(particle)
            dedup += 1
        else:
            rejected += 1
    return particles, duplicates, dedup, rejected


def reject_impossible_pseudo_feedback(  # noqa: D103
    config: RevisionConfig, run_request: RevisionRunRequest
) -> None:
    if run_request.arm == "pseudo-feedback" and config.particle_count < _MIN_PSEUDO_PARTICLES:
        raise PseudoFeedbackParticleCountError(particle_count=config.particle_count)


def _epoch_seed(base: int, epoch: int, particle_index: int, offset: int) -> int:
    return base + ((epoch + 1) * _EPOCH_SEED_STRIDE) + offset + particle_index
