"""Single-writer hash-chained recording for live campaigns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from dllm_agent.experiments.multitarget.artifacts import write_json_artifact
from dllm_agent.experiments.multitarget.ledger import (
    ArtifactEvidence,
    JsonObject,
    LedgerAppendResult,
    append_event,
    artifact_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


class Clock(Protocol):
    """UTC wall-clock seam."""

    def __call__(self) -> datetime:
        """Return the current instant."""
        ...


@dataclass(frozen=True, slots=True)
class RecordedArtifact:
    """Artifact evidence and the ledger append binding it."""

    artifact: ArtifactEvidence
    ledger: LedgerAppendResult


@dataclass(frozen=True, slots=True)
class ExperimentRecorder:
    """Serialize all campaign provenance through one ledger writer."""

    experiment_root: Path
    now: Clock = lambda: datetime.now(UTC)

    def event(self, event_type: str, payload: JsonObject) -> LedgerAppendResult:
        """Append a hash-chained event without a new artifact."""
        return append_event(
            self.experiment_root / "events.jsonl",
            timestamp_utc=_timestamp(self.now()),
            event_type=event_type,
            payload=payload,
        )

    def record_json(
        self,
        *,
        event_type: str,
        relative_path: Path,
        value: object,
        payload: JsonObject,
    ) -> RecordedArtifact:
        """Atomically persist canonical JSON and append its evidence."""
        evidence = write_json_artifact(
            experiment_root=self.experiment_root,
            relative_path=relative_path,
            value=value,
        )
        return self._record(event_type, evidence, payload)

    def record_existing(
        self,
        *,
        event_type: str,
        artifact_path: Path,
        payload: JsonObject,
    ) -> RecordedArtifact:
        """Hash an already persisted raw artifact and bind it to the ledger."""
        root = self.experiment_root.resolve()
        resolved = artifact_path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            msg = "existing artifact is outside experiment root"
            raise ValueError(msg) from error
        evidence = replace(artifact_evidence(resolved), path=relative.as_posix())
        return self._record(event_type, evidence, payload)

    def _record(
        self,
        event_type: str,
        evidence: ArtifactEvidence,
        payload: JsonObject,
    ) -> RecordedArtifact:
        event_payload = {**dict(payload), "artifact": evidence.to_json()}
        ledger = self.event(event_type, event_payload)
        return RecordedArtifact(artifact=evidence, ledger=ledger)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        msg = "recorder clock must return a timezone-aware datetime"
        raise ValueError(msg)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
