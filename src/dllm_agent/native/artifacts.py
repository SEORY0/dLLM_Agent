"""Probe normalization and deterministic native-controller artifact persistence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import sha256_bytes
from dllm_agent.fuzzing.coverage.runner import CoverageResult
from dllm_agent.fuzzing.runner import FuzzRunResult
from dllm_agent.hunt.artifacts import coverage_evidence
from dllm_agent.hunt.pipeline import fuzz_evidence
from dllm_agent.native.models import (
    ProbeEvent,
    ProbeEvidence,
    SelectionEvent,
    StageEvent,
)

type ControllerEvent = ProbeEvent | StageEvent | SelectionEvent


@dataclass(frozen=True, slots=True)
class ProbeIdentity:
    """Deterministic identity of one candidate seed probe."""

    sequence: int
    stage_index: int
    candidate_id: str
    fuzzer_seed: int


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One event paired with its stable machine-consumed ledger discriminator."""

    kind: Literal["probe", "stage", "selection"]
    event: ControllerEvent


def normalize_probe(
    identity: ProbeIdentity,
    fuzz_result: FuzzRunResult,
    coverage_result: CoverageResult,
) -> ProbeEvent:
    """Normalize bounded runner results while binding every raw log and argv by hash."""
    fuzz = fuzz_evidence(fuzz_result)
    coverage = coverage_evidence(coverage_result)
    return ProbeEvent(
        sequence=identity.sequence,
        stage_index=identity.stage_index,
        candidate_id=identity.candidate_id,
        fuzzer_seed=identity.fuzzer_seed,
        build_status=fuzz_result.build_status,
        fuzz_status=fuzz_result.fuzz_status,
        sanitizer_observed=fuzz.sanitizer is not None,
        coverage=coverage.totals,
        feature_coverage=fuzz.feature_coverage,
        edge_coverage=fuzz.edge_coverage,
        executions_per_second=fuzz.executions_per_second,
        evidence=ProbeEvidence(
            build_log_sha256=fuzz.build_log_sha256,
            fuzz_log_sha256=fuzz.fuzz_log_sha256,
            build_argv_sha256=fuzz.commands.build_argv_sha256,
            fuzz_argv_sha256=fuzz.commands.fuzz_argv_sha256,
            coverage=coverage,
        ),
    )


def controller_events_bytes(entries: tuple[LedgerEntry, ...]) -> bytes:
    """Serialize timestamp-free discriminated events in their monotonic sequence order."""
    return b"".join(_event_line(entry) for entry in entries)


def create_evidence_root(root: Path) -> None:
    """Create a new evidence root and fail closed rather than merge or overwrite."""
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        detail = error.strerror or "could not create native hunt evidence root"
        raise HuntSetupError(detail) from error


def persist_controller_events(root: Path, payload: bytes) -> None:
    """Create the successful controller ledger without replacing existing evidence."""
    try:
        with (root / "controller-events.jsonl").open("xb") as stream:
            _ = stream.write(payload)
    except OSError as error:
        detail = error.strerror or "could not persist controller event ledger"
        raise HuntSetupError(detail) from error


def controller_events_sha256(payload: bytes) -> str:
    """Hash the exact controller JSONL bytes returned by the serializer."""
    return sha256_bytes(payload)


def _event_line(entry: LedgerEntry) -> bytes:
    return (
        b'{"kind":"'
        + entry.kind.encode()
        + b'","event":'
        + entry.event.model_dump_json().encode()
        + b"}\n"
    )
