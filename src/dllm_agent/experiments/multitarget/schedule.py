"""Deterministic multitarget campaign schedule and provenance manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from dllm_agent.experiments.multitarget.budget import (
    FEEDBACK_ARMS,
    Arm,
    BudgetAllocation,
    ExperimentSpec,
    Target,
)
from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile

if TYPE_CHECKING:
    from collections.abc import Mapping

type WorkerPhase = Literal["initial-micro-probe", "revision-micro-probe", "main-campaign"]

_SHA256_IMAGE_LENGTH = 71
_EXPECTED_RAW_PARTICLES = 32
_MINIMUM_UNIQUE = 16


@dataclass(slots=True)
class ScheduleError(Exception):
    """Raised when a campaign schedule would be non-reproducible."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class SeedNamespace:
    """Disjoint deterministic seeds consumed by one campaign cell."""

    model_particles: int
    model_revision: int
    pseudo_permutation: int
    initial_probe: int
    revision_probe: int
    main_fuzzer: int
    proof_replay_a: int
    proof_replay_b: int

    def as_dict(self) -> dict[str, int]:
        """Return canonical seed names for JSON manifests."""
        return {
            "initial_probe": self.initial_probe,
            "main_fuzzer": self.main_fuzzer,
            "model_particles": self.model_particles,
            "model_revision": self.model_revision,
            "proof_replay_a": self.proof_replay_a,
            "proof_replay_b": self.proof_replay_b,
            "pseudo_permutation": self.pseudo_permutation,
            "revision_probe": self.revision_probe,
        }


@dataclass(frozen=True, slots=True)
class WorkerLedgerItem:
    """One charged worker-second phase inside a campaign cell."""

    phase: WorkerPhase
    worker_seconds: int


@dataclass(frozen=True, slots=True)
class CellSchedule:
    """One target/arm/repetition cell with runtime and provenance binding."""

    target: Target
    arm: Arm
    repetition: int
    cpu_index: int
    image_id: str
    allocation: BudgetAllocation
    worker_ledger: tuple[WorkerLedgerItem, ...]
    seeds: SeedNamespace

    def to_manifest(self) -> dict[str, object]:
        """Return a canonical JSON-serializable cell manifest."""
        profile = runtime_profile(self.target)
        return {
            "allocation": {
                "campaign_worker_seconds": self.allocation.campaign_worker_seconds,
                "initial_probe_worker_seconds": self.allocation.initial_probe_worker_seconds,
                "proof_worker_seconds": self.allocation.proof_worker_seconds,
                "revision_probe_worker_seconds": self.allocation.revision_probe_worker_seconds,
                "search_worker_seconds": self.allocation.search_worker_seconds,
            },
            "arm": self.arm,
            "cpu_index": self.cpu_index,
            "image_id": self.image_id,
            "program": profile.program,
            "repetition": self.repetition,
            "seeds": self.seeds.as_dict(),
            "shared_initial_probe": self.arm in FEEDBACK_ARMS,
            "shared_model_bank": self.arm != "libfuzzer-only",
            "target": self.target,
            "upstream_commit": profile.upstream_commit,
            "worker_ledger": tuple(
                {"phase": item.phase, "worker_seconds": item.worker_seconds}
                for item in self.worker_ledger
            ),
        }


@dataclass(frozen=True, slots=True)
class CampaignSchedule:
    """Frozen campaign execution plan with stable manifest hash evidence."""

    spec: ExperimentSpec
    cells: tuple[CellSchedule, ...]
    manifest_hash: str

    def to_manifest(self) -> dict[str, object]:
        """Return the canonical schedule manifest."""
        return {
            "cell_count": len(self.cells),
            "cells": tuple(cell.to_manifest() for cell in self.cells),
            "minimum_unique_count": self.spec.minimum_unique_count,
            "raw_particle_count": self.spec.raw_particle_count,
            "repetitions": self.spec.repetitions,
        }


def build_campaign_schedule(
    spec: ExperimentSpec,
    *,
    available_cpus: int,
    image_ids: Mapping[Target, str],
) -> CampaignSchedule:
    """Expand and validate the exact multitarget campaign schedule."""
    _validate_spec(spec)
    _validate_image_ids(image_ids)
    cells = spec.cells()
    if available_cpus < len(cells):
        raise ScheduleError(reason="available_cpus is lower than campaign cell count")
    scheduled = tuple(
        CellSchedule(
            target=cell.target,
            arm=cell.arm,
            repetition=cell.repetition,
            cpu_index=index,
            image_id=image_ids[cell.target],
            allocation=spec.budget.for_arm(cell.arm),
            worker_ledger=_worker_ledger(cell.arm, spec.budget.for_arm(cell.arm)),
            seeds=_seed_namespace(cell.target, cell.arm, cell.repetition),
        )
        for index, cell in enumerate(cells)
    )
    manifest = {
        "cell_count": len(scheduled),
        "cells": tuple(cell.to_manifest() for cell in scheduled),
        "minimum_unique_count": spec.minimum_unique_count,
        "raw_particle_count": spec.raw_particle_count,
        "repetitions": spec.repetitions,
    }
    return CampaignSchedule(
        spec=spec,
        cells=scheduled,
        manifest_hash=_sha256_json(manifest),
    )


def _validate_spec(spec: ExperimentSpec) -> None:
    if spec.raw_particle_count != _EXPECTED_RAW_PARTICLES:
        raise ScheduleError(reason="raw_particle_count must be exactly 32")
    if (
        spec.minimum_unique_count < _MINIMUM_UNIQUE
        or spec.minimum_unique_count > spec.raw_particle_count
    ):
        raise ScheduleError(reason="minimum_unique_count must be in [16, raw_particle_count]")
    if spec.repetitions < 1:
        raise ScheduleError(reason="repetitions must be positive")


def _validate_image_ids(image_ids: Mapping[Target, str]) -> None:
    expected: tuple[Target, ...] = ("libpng", "libtiff", "libxml2")
    missing = [target for target in expected if target not in image_ids]
    if missing:
        raise ScheduleError(reason=f"image_ids missing targets: {','.join(missing)}")
    for target, image_id in image_ids.items():
        if target not in expected:
            raise ScheduleError(reason=f"image_ids contains unsupported target: {target}")
        if len(image_id) != _SHA256_IMAGE_LENGTH or not image_id.startswith("sha256:"):
            raise ScheduleError(reason="image_ids must be sha256 Docker image IDs")
        if not all(character in "0123456789abcdef" for character in image_id[7:]):
            raise ScheduleError(reason="image_ids must be lowercase sha256 Docker image IDs")


def _worker_ledger(arm: Arm, allocation: BudgetAllocation) -> tuple[WorkerLedgerItem, ...]:
    if arm in FEEDBACK_ARMS:
        return (
            WorkerLedgerItem("initial-micro-probe", allocation.initial_probe_worker_seconds),
            WorkerLedgerItem("revision-micro-probe", allocation.revision_probe_worker_seconds),
            WorkerLedgerItem("main-campaign", allocation.campaign_worker_seconds),
        )
    return (WorkerLedgerItem("main-campaign", allocation.campaign_worker_seconds),)


def _seed_namespace(target: Target, arm: Arm, repetition: int) -> SeedNamespace:
    return SeedNamespace(
        model_particles=_target_seed(target, repetition, "model_particles"),
        model_revision=_seed(target, arm, repetition, "model_revision"),
        pseudo_permutation=_seed(target, arm, repetition, "pseudo_permutation"),
        initial_probe=_target_seed(target, repetition, "initial_probe"),
        revision_probe=_seed(target, arm, repetition, "revision_probe"),
        main_fuzzer=_seed(target, arm, repetition, "main_fuzzer"),
        proof_replay_a=_seed(target, arm, repetition, "proof_replay_a"),
        proof_replay_b=_seed(target, arm, repetition, "proof_replay_b"),
    )


def _seed(target: Target, arm: Arm, repetition: int, component: str) -> int:
    digest = hashlib.sha256(f"{target}:{arm}:{repetition}:{component}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _target_seed(target: Target, repetition: int, component: str) -> int:
    digest = hashlib.sha256(f"{target}:{repetition}:{component}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
