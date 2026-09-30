"""Append-only recovery for the OpenEXR one-day campaign.

# noqa: SIZE_OK
"""

# ruff: noqa: C901, PLR0912, PLR0913

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, override

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.ledger import (
    JsonValue,
    LedgerError,
    append_event,
    validate_ledger,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignOutcome,
    CellRunner,
    CoordinatorError,
    GateRunner,
    ModelExecutor,
    PostSearchRunner,
    ResidentPoolSurface,
    build_campaign_manifest,
    coordinate_campaign,
    expire_recovered_campaign,
    load_complete_campaign_outcome,
)
from dllm_agent.experiments.openexr_one_day.coverage_series import (
    SCHEDULE_SECONDS,
    CorpusIndexRequest,
    CoverageSeriesError,
    load_corpus_index,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dllm_agent.experiments.openexr_one_day.models import ModelArm
    from dllm_agent.revision.controller_models import RevisionSampler

_EVENTS_NAME: Final = "campaign-events.jsonl"
_MANIFEST_NAME: Final = "campaign-manifest.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COMPLETE_CELL_COUNT: Final = 50
_WAVE_DURATION_SECONDS: Final = 10_800


class RecoveryError(Exception):
    """Raised when campaign evidence cannot be trusted for resume."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Store the recovery failure reason."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return self.reason


class CampaignEvent(FrozenModel):
    """Validated campaign checkpoint record."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    sequence: int = Field(ge=0)
    event_type: str
    previous_stream_sha256: Sha256
    stream_sha256: Sha256
    phase: Literal["manifest", "launch", "terminal"]
    cell_id: str = ""
    fuzzer_seed: int = Field(default=0, ge=0)
    deadline_monotonic_seconds: float = 0.0
    artifact_root: Path = Path("/data/dllm-agent/placeholder")
    config_sha256: Sha256 = "0" * 64
    resource_map_sha256: Sha256 = "0" * 64
    manifest_sha256: Sha256 = "0" * 64
    terminal_reason: str = ""
    repetition: int = 0
    wave_index: int = 0
    arm: str = ""
    mutable_root: Path = Path("/data/dllm-agent/placeholder")
    bank_sha256: Sha256 = "0" * 64
    trajectory_set_sha256: Sha256 = "0" * 64
    coverage_frozen_through_seconds: int | None = None
    search_cpu_seconds: float = 0.0


def append_campaign_event(root: Path, phase: str, payload: dict[str, JsonValue]) -> None:
    """Append one hash-chained campaign checkpoint."""
    event_type = "campaign_manifest" if phase == "manifest" else "cell_checkpoint"
    _ = append_event(
        root / _EVENTS_NAME,
        timestamp_utc=phase,
        event_type=event_type,
        payload={"phase": phase, **payload},
    )


def load_campaign_events(root: Path) -> tuple[CampaignEvent, ...]:
    """Validate and load the complete campaign checkpoint chain."""
    path = root / _EVENTS_NAME
    try:
        _ = validate_ledger(path)
    except LedgerError as error:
        raise RecoveryError(reason=str(error)) from error
    manifest = _load_manifest(root)
    events: list[CampaignEvent] = []
    terminal_cells: set[str] = set()
    launched: dict[str, tuple[int, float, Path]] = {}
    try:
        for line in path.read_bytes().splitlines():
            outer = _JSON_OBJECT.validate_json(line)
            payload = _JSON_OBJECT.validate_python(outer["payload"])
            event = CampaignEvent(
                sequence=_int(outer["sequence"]),
                event_type=_str(outer["event_type"]),
                previous_stream_sha256=_str(outer["previous_stream_sha256"]),
                stream_sha256=_str(outer["stream_sha256"]),
                phase=_phase(payload["phase"]),
                cell_id=_str(payload.get("cell_id", "")),
                fuzzer_seed=_int(payload.get("fuzzer_seed", 0)),
                deadline_monotonic_seconds=_float(payload.get("deadline_monotonic_seconds", 0.0)),
                artifact_root=Path(
                    _str(payload.get("artifact_root", "/data/dllm-agent/placeholder"))
                ),
                config_sha256=_str(payload.get("config_sha256", "0" * 64)),
                resource_map_sha256=_str(payload.get("resource_map_sha256", "0" * 64)),
                manifest_sha256=_str(payload.get("manifest_sha256", "0" * 64)),
                terminal_reason=_str(payload.get("terminal_reason", "")),
                repetition=_int(payload.get("repetition", 0)),
                wave_index=_int(payload.get("wave_index", 0)),
                arm=_str(payload.get("arm", "")),
                mutable_root=Path(
                    _str(payload.get("mutable_root", "/data/dllm-agent/placeholder"))
                ),
                bank_sha256=_str(payload.get("bank_sha256", "0" * 64)),
                trajectory_set_sha256=_str(payload.get("trajectory_set_sha256", "0" * 64)),
                coverage_frozen_through_seconds=_optional_int(
                    payload.get("coverage_frozen_through_seconds")
                ),
                search_cpu_seconds=_float(payload.get("search_cpu_seconds", 0.0)),
            )
            _validate_event(root, manifest, event, launched, terminal_cells)
            events.append(event)
    except (KeyError, ValidationError) as error:
        raise RecoveryError(reason="invalid campaign event") from error
    return tuple(events)


def resume_campaign(
    root: Path,
    *,
    resident_pool: ResidentPoolSurface,
    cell_runner: CellRunner,
    post_search: PostSearchRunner,
    gate_runner: GateRunner,
    model_executor: ModelExecutor | None = None,
    bank_sampler: RevisionSampler | None = None,
    revision_delegates: Mapping[ModelArm, RevisionSampler] | None = None,
    now_monotonic_seconds: float | None = None,
) -> CampaignOutcome:
    """Resume unfinished work from canonical evidence."""
    if not root.exists():
        raise RecoveryError(reason="campaign is not resumable")
    events = load_campaign_events(root)
    manifest = _load_manifest(root)

    config = CampaignConfig.model_validate(manifest["config"])
    if (
        len({event.cell_id for event in events if event.phase == "terminal"})
        == _COMPLETE_CELL_COUNT
    ):
        schedule = build_campaign_schedule(
            main_start_monotonic_seconds=config.main_start_monotonic_seconds,
            gpu_count=len(config.resource_map.model_cores),
        )
        campaign_manifest = build_campaign_manifest(config, schedule)
        try:
            return load_complete_campaign_outcome(root, campaign_manifest)
        except CoordinatorError as error:
            raise RecoveryError(reason=f"campaign is not resumable: {error}") from error
    if now_monotonic_seconds is not None:
        schedule = build_campaign_schedule(
            main_start_monotonic_seconds=config.main_start_monotonic_seconds,
            gpu_count=len(config.resource_map.model_cores),
        )
        if now_monotonic_seconds >= float(schedule.absolute_deadline_monotonic_seconds):
            try:
                return expire_recovered_campaign(
                    config=config,
                    post_search=post_search,
                )
            except CoordinatorError as error:
                raise RecoveryError(reason=f"campaign is not resumable: {error}") from error
    try:
        return coordinate_campaign(
            config,
            resident_pool=resident_pool,
            cell_runner=cell_runner,
            post_search=post_search,
            gate_runner=gate_runner,
            model_executor=model_executor,
            bank_sampler=bank_sampler,
            revision_delegates=revision_delegates,
        )
    except CoordinatorError as error:
        raise RecoveryError(reason=f"campaign is not resumable: {error}") from error


def _validate_event(
    root: Path,
    manifest: dict[str, JsonValue],
    event: CampaignEvent,
    launched: dict[str, tuple[int, float, Path]],
    terminal_cells: set[str],
) -> None:
    expected_config = _str(_dict(manifest["config"])["config_sha256"])
    expected_resource = _str(manifest["resource_map_sha256"])
    expected_manifest = _str(manifest["manifest_sha256"])
    if event.phase == "manifest":
        if event.config_sha256 != expected_config or event.resource_map_sha256 != expected_resource:
            raise RecoveryError(reason="config mismatch")
        return
    _require_cell_path(root, event.artifact_root)
    if event.config_sha256 != expected_config:
        raise RecoveryError(reason="config mismatch")
    if event.resource_map_sha256 != expected_resource:
        raise RecoveryError(reason="resource-map mismatch")
    if event.manifest_sha256 != expected_manifest:
        raise RecoveryError(reason="manifest hash mismatch")
    if event.phase == "launch":
        if event.cell_id in terminal_cells:
            raise RecoveryError(reason="event after terminal")
        if event.cell_id in launched:
            seed, deadline, path = launched[event.cell_id]
            if (
                seed != event.fuzzer_seed
                or deadline != event.deadline_monotonic_seconds
                or path != event.artifact_root
            ):
                raise RecoveryError(reason="deadline mutation")
        launched[event.cell_id] = (
            event.fuzzer_seed,
            event.deadline_monotonic_seconds,
            event.artifact_root,
        )
        return
    prior = launched.get(event.cell_id)
    if prior is None:
        if event.cell_id in terminal_cells:
            raise RecoveryError(reason="duplicate terminal")
        expected_c0 = _str(_dict(manifest["config"])["c0_sha256"])
        if _has_never_started_expiration_evidence(event, expected_c0):
            terminal_cells.add(event.cell_id)
            return
        raise RecoveryError(reason="event after terminal")
    seed, deadline, path = prior
    if seed != event.fuzzer_seed:
        raise RecoveryError(reason="seed mutation")
    if deadline != event.deadline_monotonic_seconds:
        raise RecoveryError(reason="deadline mutation")
    if path != event.artifact_root:
        raise RecoveryError(reason="path escape")
    if event.cell_id in terminal_cells:
        raise RecoveryError(reason="duplicate terminal")
    terminal_cells.add(event.cell_id)


def _has_never_started_expiration_evidence(event: CampaignEvent, expected_c0_sha256: str) -> bool:
    if (
        event.terminal_reason != "expired-frozen"
        or not event.cell_id
        or not event.arm
        or event.deadline_monotonic_seconds <= 0.0
        or event.coverage_frozen_through_seconds != _WAVE_DURATION_SECONDS
        or event.search_cpu_seconds != 0.0
    ):
        return False
    search_evidence = (
        event.artifact_root / "run-manifest.json",
        event.artifact_root / "events.jsonl",
        event.artifact_root / "final-result.json",
        event.artifact_root / "attempts",
    )
    if any(path.exists() for path in search_evidence):
        return False
    request = CorpusIndexRequest(
        artifact_root=event.artifact_root,
        corpus_root=event.artifact_root / "corpus",
        store_root=event.artifact_root / "corpus-inputs",
        index_path=event.artifact_root / "corpus-index.jsonl",
    )
    try:
        rows = load_corpus_index(request)
    except (OSError, CoverageSeriesError):
        return False
    return (
        tuple(row.scheduled_elapsed_seconds for row in rows) == SCHEDULE_SECONDS
        and all(row.cumulative_hashes == (expected_c0_sha256,) for row in rows)
        and len({row.cumulative_set_sha256 for row in rows}) == 1
    )


def _load_manifest(root: Path) -> dict[str, JsonValue]:
    path = root / _MANIFEST_NAME
    try:
        raw = path.read_bytes()
        parsed = _JSON_OBJECT.validate_json(raw)
    except (OSError, ValidationError) as error:
        raise RecoveryError(reason="campaign is not resumable") from error
    return parsed


def _require_cell_path(root: Path, path: Path) -> None:
    try:
        _ = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RecoveryError(reason="path escape") from error


def _phase(value: JsonValue) -> Literal["manifest", "launch", "terminal"]:
    text = _str(value)
    if text == "manifest":
        return "manifest"
    if text == "launch":
        return "launch"
    if text == "terminal":
        return "terminal"
    raise RecoveryError(reason="invalid campaign phase")


def _dict(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RecoveryError(reason="invalid campaign manifest")
    return value


def _str(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise RecoveryError(reason="invalid campaign event")
    return value


def _int(value: JsonValue) -> int:
    if not isinstance(value, int):
        raise RecoveryError(reason="invalid campaign event")
    return value


def _optional_int(value: JsonValue | None) -> int | None:
    if value is None:
        return None
    return _int(value)


def _float(value: JsonValue) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise RecoveryError(reason="invalid campaign event")
