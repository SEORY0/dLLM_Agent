"""Hash-bound qualification evidence gates for OpenEXR one-day runs.

# noqa: SIZE_OK
"""

# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

import json
import math
import os
import re
import stat
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Protocol, assert_never

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from tools.stable_diffcoder_payloads import MODEL_ID, MODEL_REVISION

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignManifest,
    CandidateBankEvidence,
    CoordinatorError,
    ResourceMap,
    StartGateEvidence,
    build_campaign_manifest,
    load_persisted_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke import (
    build_qualification_smoke_plan as build_canonical_qualification_smoke_plan,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeArtifact as CanonicalQualificationSmokeArtifact,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeCell as CanonicalQualificationSmokeCell,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeError,
    load_smoke_artifact_file,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokePlan as CanonicalQualificationSmokePlan,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeResult as CanonicalQualificationSmokeResult,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule

if TYPE_CHECKING:
    from subprocess import CompletedProcess

type QualificationGateName = Literal[
    "hashes_match",
    "banks_complete",
    "read_only_cache",
    "external_coverage_check",
    "control_arm_equivalence",
    "resource_map_valid",
    "disk_space",
    "vram",
    "ram",
    "swap_stable",
    "reserved_core_load",
    "smoke_passed",
]

QUALIFICATION_GATE_NAMES: Final[tuple[QualificationGateName, ...]] = (
    "hashes_match",
    "banks_complete",
    "read_only_cache",
    "external_coverage_check",
    "control_arm_equivalence",
    "resource_map_valid",
    "disk_space",
    "vram",
    "ram",
    "swap_stable",
    "reserved_core_load",
    "smoke_passed",
)
_QUALIFICATION_SCHEMA: Final = "dllm-openexr-one-day-qualification/v1"
_QUALIFICATION_ARTIFACT_SCHEMA: Final = "dllm-openexr-one-day-qualification-artifact/v1"
_EXPECTED_REPETITIONS: Final = 10
_EXPECTED_BANK_RAW: Final = 32
_MINIMUM_UNIQUE_BANK: Final = 16
_SUPPLEMENT_RAW_COUNT: Final = 16
_MINIMUM_DISK_FREE_BYTES: Final = 100 * 1024 * 1024 * 1024
_MINIMUM_PRELOAD_VRAM_BYTES: Final = 22 * 1024 * 1024 * 1024
_MINIMUM_POSTLOAD_VRAM_BYTES: Final = 4 * 1024 * 1024 * 1024
_MINIMUM_HOST_AVAILABLE_BYTES: Final = 12 * 1024 * 1024 * 1024
_MAX_RESERVED_CORE_LOAD: Final = 0.05
_SMOKE_PAIR_COUNT: Final = 2
_SMOKE_DEADLINE_SECONDS: Final = 300
_SMOKE_ARM_COUNT: Final = 5
_THREAD_SIBLING_ROOT: Final = Path("/sys/devices/system/cpu")
_LSCPU_TOPOLOGY_COLUMN_COUNT: Final = 4
_NVIDIA_TOPOLOGY_COLUMN_COUNT: Final = 3
_SYSFS_NUMA_PATH_SUFFIX_LEN: Final = 2
_CPU_LIST_RANGE_PART_COUNT: Final = 2
_SHA256_HEX_LEN: Final = 64
_LSCPU_TOPOLOGY_ARGV: Final = ("lscpu", "--parse=CPU,CORE,SOCKET,NODE")
_NVIDIA_GPU_TOPOLOGY_ARGV: Final = (
    "nvidia-smi",
    "--query-gpu=index,uuid,pci.bus_id",
    "--format=csv,noheader,nounits",
)


def nvidia_gpu_topology_argv(physical_gpu_indices: tuple[int, ...]) -> tuple[str, ...]:
    """Return the exact nvidia-smi argv for one topology query.

    Restricted to ``physical_gpu_indices`` via ``-i`` when given, so a shared
    box's other GPUs (in use by unrelated work) never enter this campaign's
    resource_map identity evidence. Empty means "every GPU on the host" --
    the original dedicated-cluster assumption.
    """
    if not physical_gpu_indices:
        return _NVIDIA_GPU_TOPOLOGY_ARGV
    indices = ",".join(str(index) for index in sorted(set(physical_gpu_indices)))
    return (*_NVIDIA_GPU_TOPOLOGY_ARGV, "-i", indices)
_PCI_DEVICE_ROOT: Final = Path("/sys/bus/pci/devices")
_CACHE_WRITE_PROBE_NAME: Final = ".dllm-agent-cache-write-probe"
_NVIDIA_DOMAIN_HEX_LEN: Final = 8
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class QualificationCommandRunner(Protocol):
    """Shell-free command runner for qualification collectors."""

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Run one fixed argv vector and return captured bytes."""
        ...


class QualificationSmokeRunner(Protocol):
    """Runner seam for the isolated two-pair qualification smoke."""

    def run(self, plan: QualificationSmokePlan) -> QualificationSmokeResult:
        """Run the smoke plan and return observed cell/pair outcomes."""
        ...


@dataclass(frozen=True, slots=True)
class LscpuTopology:
    """Raw-derived CPU topology indexed by logical CPU."""

    cpu_numa: dict[int, int]
    physical_groups: dict[tuple[int, int], frozenset[int]]


@dataclass(frozen=True, slots=True)
class NvidiaGpuRecord:
    """Raw NVIDIA GPU identity record."""

    uuid: str
    bus_id: str


@dataclass(frozen=True, slots=True)
class ConfigArtifactMeasurements:
    """Measured artifact files used to freeze a one-day config."""

    prompt_path: Path
    target_harness_path: Path
    remote_code_path: Path
    search_binary_path: Path
    search_image_digest: str
    coverage_binary_path: Path
    coverage_image_digest: str
    coverage_build_manifest_path: Path
    c0_source_path: Path
    dictionary_path: Path
    artifact_root: Path
    config_id: str
    qualification_state: Literal["draft", "qualified"]


class QualificationSmokeCell(CanonicalQualificationSmokeCell):
    """One isolated qualification-smoke cell."""


class QualificationSmokePlan(CanonicalQualificationSmokePlan):
    """Fixed 2-pair x 5-arm x 300-second smoke plan."""


class QualificationSmokeResult(CanonicalQualificationSmokeResult):
    """Canonical smoke lifecycle result from the hash-bound smoke runner."""


class QualificationSmokeArtifact(CanonicalQualificationSmokeArtifact):
    """Hash-bound persisted smoke result produced by the isolated smoke runner."""


class QualificationWriteResult(FrozenModel):
    """Record-backed qualification write result."""

    state: Literal["qualified"]
    evidence: QualificationEvidence
    start_gates: StartGateEvidence


class SystemQualificationCommandRunner:
    """Run production qualification collectors without a shell."""

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Run one fixed argv vector with captured stdout and stderr."""
        return subprocess.run(argv, check=False, capture_output=True)  # noqa: S603


class QualificationEvidence(FrozenModel):
    """Persisted measured facts bound to one manifest and config digest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    schema_version: Literal["dllm-openexr-one-day-qualification/v1"] = _QUALIFICATION_SCHEMA
    manifest_sha256: Sha256
    config_sha256: Sha256
    facts: dict[str, bool] = Field(default_factory=dict)
    records: tuple[QualificationFact, ...] = ()
    evidence_sha256: Sha256

    @classmethod
    def for_manifest(
        cls,
        *,
        manifest_sha256: str,
        config_sha256: str,
        facts: dict[str, bool] | None = None,
        records: tuple[QualificationFact, ...] = (),
    ) -> QualificationEvidence:
        """Build evidence with a detached hash over sorted full records."""
        del facts
        evidence = cls(
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
            records=records,
            evidence_sha256="0" * 64,
        )
        return evidence.model_copy(update={"evidence_sha256": evidence.expected_evidence_sha256()})

    @staticmethod
    def record_adapter() -> TypeAdapter[QualificationFact]:
        """Return the typed record adapter used by tests and artifact builders."""
        return TypeAdapter(QualificationFact)

    def expected_evidence_sha256(self) -> str:
        """Hash canonical evidence with records sorted by gate name."""
        payload = _JSON_OBJECT.validate_python(
            {
                "schema_version": self.schema_version,
                "manifest_sha256": self.manifest_sha256,
                "config_sha256": self.config_sha256,
                "records": [
                    record.model_dump(mode="json")
                    for record in sorted(self.records, key=lambda item: item.name)
                ],
            }
        )
        return sha256_bytes(_canonical_json(payload))


class QualificationFact(FrozenModel):
    """One measured qualification fact backed by a canonical hashed artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=False)

    name: QualificationGateName
    manifest_sha256: Sha256
    config_sha256: Sha256
    passed: bool
    measurement_kind: str = Field(min_length=1)
    measurements: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_path: Path
    artifact_sha256: Sha256
    artifact_size_bytes: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_fact(cls, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        upgraded = dict(value)
        if "measurement" in upgraded and "measurement_kind" not in upgraded:
            measurement = upgraded.pop("measurement")
            upgraded["measurement_kind"] = "legacy"
            upgraded["measurements"] = {"description": measurement}
        path = upgraded.get("artifact_path")
        if "artifact_size_bytes" not in upgraded and isinstance(path, str | Path):
            try:
                upgraded["artifact_size_bytes"] = Path(path).stat().st_size
            except OSError:
                upgraded["artifact_size_bytes"] = 1
        _ = upgraded.setdefault("manifest_sha256", "0" * 64)
        _ = upgraded.setdefault("config_sha256", "0" * 64)
        return upgraded

    @model_validator(mode="after")
    def _artifact_matches_record(self) -> QualificationFact:
        try:
            content = self.artifact_path.read_bytes()
        except OSError as error:
            message = "qualification artifact unavailable"
            raise ValueError(message) from error
        if sha256_bytes(content) != self.artifact_sha256:
            message = "qualification artifact hash mismatch"
            raise ValueError(message)
        if len(content) != self.artifact_size_bytes:
            message = "qualification artifact length mismatch"
            raise ValueError(message)
        return self


class ProductionGateRunner:
    """Evaluate production gates from persisted hash-bound qualification evidence."""

    evidence_path: Path

    def __init__(self, evidence_path: Path) -> None:
        """Bind the persisted qualification evidence path."""
        self.evidence_path = evidence_path

    def evaluate(self, manifest: CampaignManifest) -> StartGateEvidence:
        """Return gates only when persisted evidence matches the manifest."""
        evidence = self._load()
        if (
            evidence.manifest_sha256 != (manifest.manifest_sha256 or "")
            or evidence.config_sha256 != manifest.config.config_sha256
        ):
            raise CoordinatorError(reason="qualification evidence manifest mismatch")
        derived = {
            record.name: _derive_gate_passed(record, manifest)
            for record in sorted(evidence.records, key=lambda item: item.name)
        }
        false_facts = tuple(field for field in QUALIFICATION_GATE_NAMES if not derived[field])
        if false_facts:
            raise CoordinatorError(reason=f"qualification evidence failed: {false_facts[0]}")
        return StartGateEvidence(**derived)

    def _load(self) -> QualificationEvidence:
        try:
            _reject_symlink_path(self.evidence_path)
            raw = self.evidence_path.read_bytes()
            payload = _JSON_OBJECT.validate_json(raw)
            evidence = QualificationEvidence.model_validate(payload)
        except OSError as error:
            raise CoordinatorError(reason="qualification evidence unavailable") from error
        except ValidationError as error:
            detail = str(error)
            if "artifact hash" in detail:
                raise CoordinatorError(reason="qualification artifact hash mismatch") from error
            if "artifact length" in detail:
                raise CoordinatorError(reason="qualification artifact length mismatch") from error
            raise CoordinatorError(reason="qualification evidence unavailable") from error
        if raw != _canonical_json(_JSON_OBJECT.validate_python(evidence.model_dump(mode="json"))):
            raise CoordinatorError(reason="qualification evidence is not canonical")
        _validate_exact_record_set(evidence)
        if evidence.expected_evidence_sha256() != evidence.evidence_sha256:
            raise CoordinatorError(reason="qualification evidence stale")
        _validate_artifacts(evidence, self.evidence_path.parent)
        return evidence


def write_qualification_fact(  # noqa: PLR0913
    *,
    manifest: CampaignManifest,
    evidence_root: Path,
    gate_name: QualificationGateName,
    measurement_kind: str,
    measurements: dict[str, JsonValue],
    passed: bool | None = None,
) -> QualificationFact:
    """Write one canonical qualification artifact from measured gate inputs."""
    evidence_root.mkdir(parents=True, exist_ok=True)
    measured_passed = (
        _derive_gate_passed_from_measurements(gate_name, measurements, manifest)
        if passed is None
        else passed
    )
    payload = _artifact_payload(
        name=gate_name,
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=manifest.config.config_sha256,
        passed=measured_passed,
        measurement_kind=measurement_kind,
        measurements=measurements,
    )
    content = _canonical_json(payload)
    digest = sha256_bytes(content)
    path = evidence_root / f"{gate_name}-{digest}.json"
    _write_once_bytes(path, content, label=f"qualification artifact {gate_name}")
    return QualificationFact(
        name=gate_name,
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=manifest.config.config_sha256,
        passed=measured_passed,
        measurement_kind=measurement_kind,
        measurements=measurements,
        artifact_path=path.resolve(),
        artifact_sha256=digest,
        artifact_size_bytes=len(content),
    )


def build_record_backed_qualification_evidence(
    *,
    manifest: CampaignManifest,
    evidence_root: Path,
    measurements_by_gate: dict[QualificationGateName, dict[str, JsonValue]],
) -> QualificationEvidence:
    """Construct complete record-backed evidence from real measured gate inputs."""
    missing = tuple(gate for gate in QUALIFICATION_GATE_NAMES if gate not in measurements_by_gate)
    if missing:
        raise CoordinatorError(reason=f"qualification measurement missing: {missing[0]}")
    records = tuple(
        write_qualification_fact(
            manifest=manifest,
            evidence_root=evidence_root,
            gate_name=gate,
            measurement_kind=gate,
            measurements=measurements_by_gate[gate],
        )
        for gate in QUALIFICATION_GATE_NAMES
    )
    return QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=manifest.config.config_sha256,
        records=records,
    )


def write_record_backed_qualification_from_measurements(
    *,
    config: CampaignConfig,
    evidence_root: Path,
    measurements_root: Path,
) -> QualificationWriteResult:
    """Write canonical 12-gate qualification evidence from measured JSON files."""
    _reject_placeholder_provenance(config)
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(
            main_start_monotonic_seconds=config.main_start_monotonic_seconds,
            gpu_count=len(config.resource_map.model_cores),
        ),
    )
    measurements: dict[QualificationGateName, dict[str, JsonValue]] = {
        gate: _read_gate_measurements(measurements_root, gate) for gate in QUALIFICATION_GATE_NAMES
    }
    evidence = build_record_backed_qualification_evidence(
        manifest=manifest,
        evidence_root=evidence_root,
        measurements_by_gate=measurements,
    )
    evidence_path = evidence_root / "qualification.json"
    _write_once_bytes(
        evidence_path,
        _canonical_json(_JSON_OBJECT.validate_python(evidence.model_dump(mode="json"))),
        label="qualification.json",
    )
    gates = ProductionGateRunner(evidence_path).evaluate(manifest)
    return QualificationWriteResult(state="qualified", evidence=evidence, start_gates=gates)


def build_qualification_smoke_plan(
    *,
    config: CampaignConfig,
    smoke_root: Path,
) -> QualificationSmokePlan:
    """Build the isolated fixed qualification-smoke plan without launching it."""
    plan = build_canonical_qualification_smoke_plan(
        config_sha256=config.config_sha256,
        smoke_root=smoke_root,
        start_monotonic_seconds=float(config.main_start_monotonic_seconds),
    )
    return QualificationSmokePlan.model_validate(plan.model_dump(mode="python"))


def run_qualification_smoke(
    *,
    config: CampaignConfig,
    smoke_root: Path,
    runner: QualificationSmokeRunner,
) -> dict[str, JsonValue]:
    """Run the fixed qualification smoke through an injected runtime seam."""
    plan = build_qualification_smoke_plan(config=config, smoke_root=smoke_root)
    result = runner.run(plan)
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "dllm-openexr-one-day-smoke-measurements/v1",
            "config_sha256": config.config_sha256,
            "smoke_root": smoke_root.as_posix(),
            "cell_count": plan.cell_count,
            "completed_cell_count": len({cell.cell_id for cell in result.cells}),
            "pairs": _smoke_pair_measurements(plan, result),
        }
    )


def _smoke_pair_measurements(
    plan: QualificationSmokePlan,
    result: CanonicalQualificationSmokeResult,
) -> list[dict[str, JsonValue]]:
    outcomes_by_id = {outcome.cell_id: outcome for outcome in result.cells}
    restart_success = set(result.restart_success_pair_ids)
    pairs: list[dict[str, JsonValue]] = []
    for pair_id in range(_SMOKE_PAIR_COUNT):
        plan_cells = tuple(cell for cell in plan.cells if cell.pair_id == pair_id)
        outcomes = tuple(outcomes_by_id.get(cell.cell_id) for cell in plan_cells)
        schema_valid = len(plan_cells) == _SMOKE_ARM_COUNT and all(
            outcome is not None
            and outcome.pair_id == cell.pair_id
            and outcome.repetition == cell.repetition
            and outcome.arm == cell.arm
            and bool(outcome.cell_outcome_sha256)
            and bool(outcome.cell_result_sha256)
            and bool(outcome.coverage_envelope_sha256)
            and bool(outcome.proof_envelope_sha256)
            and bool(outcome.deep_parse_envelope_sha256)
            for cell, outcome in zip(plan_cells, outcomes, strict=True)
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "deadline_seconds": _SMOKE_DEADLINE_SECONDS,
                "schema_valid": schema_valid,
                "hash_valid": schema_valid,
                "restart_success": pair_id in restart_success,
                "deferred_proof_success": len(plan_cells) == _SMOKE_ARM_COUNT
                and all(
                    outcome is not None
                    and bool(outcome.proof_result_sha256)
                    and bool(outcome.deep_parse_result_sha256)
                    for outcome in outcomes
                ),
            }
        )
    return pairs


def finalize_config_payload_from_artifacts(
    template: dict[str, JsonValue],
    artifacts: ConfigArtifactMeasurements,
) -> dict[str, JsonValue]:
    """Freeze config hash fields from measured artifact bytes and image digests."""
    _reject_placeholder_image_digest(artifacts.search_image_digest)
    _reject_placeholder_image_digest(artifacts.coverage_image_digest)
    payload = dict(template)
    payload.update(
        {
            "config_id": artifacts.config_id,
            "qualification_state": artifacts.qualification_state,
            "artifact_root": artifacts.artifact_root.as_posix(),
            "prompt_path": artifacts.prompt_path.as_posix(),
            "c0_source_path": artifacts.c0_source_path.as_posix(),
            "target_harness_sha256": sha256_bytes(artifacts.target_harness_path.read_bytes()),
            "prompt_sha256": sha256_bytes(artifacts.prompt_path.read_bytes()),
            "remote_code_sha256": sha256_bytes(artifacts.remote_code_path.read_bytes()),
            "search_binary_sha256": sha256_bytes(artifacts.search_binary_path.read_bytes()),
            "search_image_digest": artifacts.search_image_digest,
            "coverage_binary_sha256": sha256_bytes(artifacts.coverage_binary_path.read_bytes()),
            "coverage_image_digest": artifacts.coverage_image_digest,
            "coverage_build_manifest_sha256": sha256_bytes(
                artifacts.coverage_build_manifest_path.read_bytes()
            ),
            "c0_sha256": _decoded_c0_sha256(artifacts.c0_source_path),
            "dictionary_path": artifacts.dictionary_path.as_posix(),
            "dictionary_sha256": sha256_bytes(artifacts.dictionary_path.read_bytes()),
            "config_sha256": "0" * 64,
        }
    )
    checker_sha256 = _manifest_deep_parse_checker_sha256(artifacts.coverage_build_manifest_path)
    if checker_sha256 is not None:
        payload["deep_parse_checker_sha256"] = checker_sha256
    payload["config_sha256"] = config_payload_sha256(payload)
    config = CampaignConfig.model_validate(payload)
    _reject_placeholder_provenance(config)
    return _JSON_OBJECT.validate_python(config.model_dump(mode="json"))


def _manifest_deep_parse_checker_sha256(manifest_path: Path) -> str | None:
    """Return the checker hash the built image's manifest attests, when it declares one.

    The checker binary is compiled inside the image, so its hash is a property of the
    build (a rebuild on another host yields different bytes) and must follow the
    measured manifest rather than a value frozen in the template.
    """
    try:
        manifest = _JSON_OBJECT.validate_json(manifest_path.read_bytes())
    except ValidationError:
        return None
    value = manifest.get("deep_parse_checker_sha256")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    return None


def collect_resource_topology_measurements(
    runner: QualificationCommandRunner,
    *,
    thread_sibling_root: Path = _THREAD_SIBLING_ROOT,
    pci_device_root: Path = _PCI_DEVICE_ROOT,
    physical_gpu_indices: tuple[int, ...] = (),
) -> dict[str, JsonValue]:
    """Collect raw CPU/GPU topology evidence through fixed shell-free commands.

    ``physical_gpu_indices`` restricts the nvidia-smi query to the specific
    GPUs the campaign actually declares (via ``-i``) -- required on a shared
    box where other physical GPUs may be in use by unrelated work and were
    never meant to be part of this resource_map's identity evidence. Empty
    (the historical default) queries every GPU on the host, matching the
    original dedicated-cluster assumption where used GPUs == all GPUs.
    """
    lscpu = runner.run(_LSCPU_TOPOLOGY_ARGV)
    nvidia = runner.run(nvidia_gpu_topology_argv(physical_gpu_indices))
    lscpu_stdout = _decode_stdout(lscpu, "lscpu")
    lscpu_stderr = _decode_stderr(lscpu, "lscpu")
    nvidia_stdout = _decode_stdout(nvidia, "nvidia-smi")
    nvidia_stderr = _decode_stderr(nvidia, "nvidia-smi")
    cpu_ids = tuple(sorted(_parse_lscpu_topology(lscpu_stdout).cpu_numa))
    sibling_records: list[dict[str, JsonValue]] = []
    for cpu_id in cpu_ids:
        path = thread_sibling_root / f"cpu{cpu_id}" / "topology" / "thread_siblings_list"
        sibling_raw = path.read_text(encoding="utf-8").strip()
        sibling_records.append(_raw_path_record(path, {"cpu": cpu_id}, sibling_raw))
    gpu_records: list[dict[str, JsonValue]] = []
    for index, record in sorted(_parse_nvidia_gpu_records(nvidia_stdout).items()):
        path = pci_device_root / _sysfs_pci_device_name(record.bus_id) / "numa_node"
        raw = path.read_text(encoding="utf-8").strip()
        gpu_records.append(
            _raw_path_record(
                path,
                {"index": index, "uuid": record.uuid, "pci_bus_id": record.bus_id},
                raw,
            )
        )
    return _JSON_OBJECT.validate_python(
        {
            "lscpu_argv": list(_LSCPU_TOPOLOGY_ARGV),
            "lscpu_exit_code": lscpu.returncode,
            "lscpu_stdout": lscpu_stdout,
            "lscpu_stderr": lscpu_stderr,
            "lscpu_stdout_sha256": sha256_bytes(lscpu_stdout.encode()),
            "lscpu_stderr_sha256": sha256_bytes(lscpu_stderr.encode()),
            "nvidia_smi_argv": list(nvidia_gpu_topology_argv(physical_gpu_indices)),
            "nvidia_smi_exit_code": nvidia.returncode,
            "nvidia_smi_stdout": nvidia_stdout,
            "nvidia_smi_stderr": nvidia_stderr,
            "nvidia_smi_stdout_sha256": sha256_bytes(nvidia_stdout.encode()),
            "nvidia_smi_stderr_sha256": sha256_bytes(nvidia_stderr.encode()),
            "thread_siblings": sibling_records,
            "gpu_sysfs_numa_nodes": gpu_records,
        }
    )


def collect_read_only_cache_measurements(cache_dir: Path) -> dict[str, JsonValue]:
    """Collect raw evidence for the configured read-only model cache path."""
    resolved = cache_dir.resolve(strict=False)
    probe_path = resolved / _CACHE_WRITE_PROBE_NAME
    probe_succeeded = False
    probe_error = ""
    try:
        fd = os.open(probe_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as error:
        probe_error = error.__class__.__name__
    else:
        os.close(fd)
        probe_succeeded = True
        with suppress(FileNotFoundError):
            probe_path.unlink()
    stat_result = resolved.stat() if resolved.exists() else None
    return _JSON_OBJECT.validate_python(
        {
            "cache_path": cache_dir.as_posix(),
            "resolved_cache_path": resolved.as_posix(),
            "exists": resolved.exists(),
            "is_dir": resolved.is_dir(),
            "mode": None if stat_result is None else stat.S_IMODE(stat_result.st_mode),
            "read_only": False if stat_result is None else not bool(stat_result.st_mode & 0o222),
            "offline": True,
            "write_probe_name": _CACHE_WRITE_PROBE_NAME,
            "write_probe_succeeded": probe_succeeded,
            "write_probe_error": probe_error,
            "write_probe_removed": not probe_path.exists(),
        }
    )


def _validate_exact_record_set(evidence: QualificationEvidence) -> None:
    names = tuple(record.name for record in evidence.records)
    if names == ():
        raise CoordinatorError(reason="qualification evidence must be record-backed")
    if len(names) != len(QUALIFICATION_GATE_NAMES):
        raise CoordinatorError(reason="qualification evidence must contain exactly 12 records")
    if set(names) != set(QUALIFICATION_GATE_NAMES) or len(set(names)) != len(names):
        raise CoordinatorError(reason="qualification evidence must contain exactly 12 records")
    if evidence.facts:
        raise CoordinatorError(reason="qualification evidence facts compatibility is disabled")


def _validate_artifacts(  # noqa: C901
    evidence: QualificationEvidence, evidence_root: Path
) -> None:
    root = evidence_root.resolve(strict=True)
    for record in evidence.records:
        if record.manifest_sha256 != evidence.manifest_sha256:
            raise CoordinatorError(reason="qualification artifact manifest mismatch")
        if record.config_sha256 != evidence.config_sha256:
            raise CoordinatorError(reason="qualification artifact config mismatch")
        path = record.artifact_path
        _reject_symlink_path(path)
        if not path.is_absolute():
            raise CoordinatorError(reason="qualification artifact path must be absolute")
        if path.is_symlink() or not path.is_file():
            raise CoordinatorError(reason="qualification artifact must be an immutable file")
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise CoordinatorError(reason="qualification artifact path escapes evidence root")
            content = path.read_bytes()
        except OSError as error:
            raise CoordinatorError(reason="qualification artifact unavailable") from error
        if sha256_bytes(content) != record.artifact_sha256:
            raise CoordinatorError(reason="qualification artifact hash mismatch")
        if len(content) != record.artifact_size_bytes:
            raise CoordinatorError(reason="qualification artifact length mismatch")
        try:
            artifact_payload = _JSON_OBJECT.validate_json(content)
        except ValidationError as error:
            raise CoordinatorError(reason="qualification artifact invalid") from error
        if _canonical_json(artifact_payload) != content:
            raise CoordinatorError(reason="qualification artifact is not canonical")
        if artifact_payload != _artifact_payload_from_record(record):
            raise CoordinatorError(reason="qualification artifact payload mismatch")


def _derive_gate_passed(record: QualificationFact, manifest: CampaignManifest) -> bool:
    derived = _derive_gate_passed_from_measurements(record.name, record.measurements, manifest)
    if record.passed != derived:
        raise CoordinatorError(reason=f"qualification evidence failed: {record.name}")
    return derived


def _derive_gate_passed_from_measurements(  # noqa: C901, PLR0911, PLR0912
    gate_name: QualificationGateName,
    measurements: dict[str, JsonValue],
    manifest: CampaignManifest,
) -> bool:
    config = manifest.config
    match gate_name:
        case "hashes_match":
            return (
                _str_value(measurements, "target_commit") == config.target_commit
                and _str_value(measurements, "config_sha256") == config.config_sha256
                and _str_value(measurements, "target_harness_sha256")
                == config.target_harness_sha256
                and _str_value(measurements, "prompt_sha256") == config.prompt_sha256
                and _str_value(measurements, "remote_code_sha256") == config.remote_code_sha256
                and _str_value(measurements, "search_image_digest") == config.search_image_digest
                and _str_value(measurements, "search_binary_sha256") == config.search_binary_sha256
                and _str_value(measurements, "coverage_image_digest")
                == config.coverage_image_digest
                and _str_value(measurements, "coverage_binary_sha256")
                == config.coverage_binary_sha256
                and _str_value(measurements, "coverage_build_manifest_sha256")
                == config.coverage_build_manifest_sha256
                and _str_value(measurements, "c0_sha256") == config.c0_sha256
                and _str_value(measurements, "dictionary_sha256") == config.dictionary_sha256
                and _hash_backing_matches(measurements, config)
            )
        case "banks_complete":
            return _banks_complete(measurements, config)
        case "read_only_cache":
            return _read_only_cache_matches(measurements, config)
        case "external_coverage_check":
            return (
                _str_value(measurements, "coverage_image_digest") == config.coverage_image_digest
                and _str_value(measurements, "coverage_binary_sha256")
                == config.coverage_binary_sha256
                and _str_value(measurements, "coverage_build_manifest_sha256")
                == config.coverage_build_manifest_sha256
                and _int_value(measurements, "target_only_branch_count")
                == _int_value(measurements, "repeated_target_only_branch_count")
                and _int_value(measurements, "target_only_branch_count") > 0
                and _coverage_backing_matches(measurements, config)
            )
        case "control_arm_equivalence":
            return _control_configs_equivalent(measurements)
        case "resource_map_valid":
            return _resource_topology_matches(measurements, config.resource_map)
        case "disk_space":
            return _int_value(measurements, "free_bytes") >= _MINIMUM_DISK_FREE_BYTES
        case "vram":
            preload = _int_map_value(measurements, "pre_load_free_bytes")
            postload = _int_map_value(measurements, "post_load_free_bytes")
            gpu_indexes = range(len(config.resource_map.model_cores))
            return (
                set(preload) == {str(index) for index in gpu_indexes}
                and set(postload) == {str(index) for index in gpu_indexes}
                and all(
                    preload[str(index)] >= _MINIMUM_PRELOAD_VRAM_BYTES
                    and postload[str(index)] >= _MINIMUM_POSTLOAD_VRAM_BYTES
                    for index in gpu_indexes
                )
            )
        case "ram":
            return _int_value(measurements, "available_bytes") >= _MINIMUM_HOST_AVAILABLE_BYTES
        case "swap_stable":
            return _int_value(measurements, "swap_used_after_bytes") <= _int_value(
                measurements, "swap_used_before_bytes"
            )
        case "reserved_core_load":
            load = _float_value(measurements, "max_load")
            return (
                0.0 <= load <= _MAX_RESERVED_CORE_LOAD
                and _float_value(measurements, "interval_seconds") > 0.0
                and _int_value(measurements, "minimum_total_jiffy_delta") > 0
            )
        case "smoke_passed":
            return _smoke_passed(measurements, config)
        case unreachable:
            assert_never(unreachable)


def _artifact_payload_from_record(record: QualificationFact) -> dict[str, JsonValue]:
    return _artifact_payload(
        name=record.name,
        manifest_sha256=record.manifest_sha256,
        config_sha256=record.config_sha256,
        passed=record.passed,
        measurement_kind=record.measurement_kind,
        measurements=record.measurements,
    )


def _artifact_payload(  # noqa: PLR0913
    *,
    name: QualificationGateName,
    manifest_sha256: Sha256,
    config_sha256: Sha256,
    passed: bool,
    measurement_kind: str,
    measurements: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": _QUALIFICATION_ARTIFACT_SCHEMA,
            "gate_name": name,
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "passed": passed,
            "measurement_kind": measurement_kind,
            "measurements": measurements,
        }
    )


def _canonical_json(payload: dict[str, JsonValue]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_once_bytes(path: Path, content: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise CoordinatorError(reason=f"{label} mismatch")
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise CoordinatorError(reason=f"{label} mismatch") from None
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _str_value(measurements: dict[str, JsonValue], key: str) -> str:
    value = measurements.get(key)
    if isinstance(value, str):
        return value
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _bool_value(measurements: dict[str, JsonValue], key: str) -> bool:
    value = measurements.get(key)
    if isinstance(value, bool):
        return value
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _int_value(measurements: dict[str, JsonValue], key: str) -> int:
    value = measurements.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _float_value(measurements: dict[str, JsonValue], key: str) -> float:
    value = measurements.get(key)
    if isinstance(value, bool):
        raise CoordinatorError(reason=f"qualification measurement missing: {key}")
    if isinstance(value, int | float):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _sequence_value(measurements: dict[str, JsonValue], key: str) -> tuple[JsonValue, ...]:
    value = measurements.get(key)
    if isinstance(value, list | tuple):
        return tuple(value)
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _dict_value(measurements: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = measurements.get(key)
    if isinstance(value, dict):
        return value
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _read_gate_measurements(root: Path, gate: QualificationGateName) -> dict[str, JsonValue]:
    path = root / f"{gate}.json"
    try:
        content = path.read_bytes()
        payload = _JSON_OBJECT.validate_json(content)
    except OSError as error:
        raise CoordinatorError(reason=f"qualification measurement missing: {gate}") from error
    except ValidationError as error:
        raise CoordinatorError(reason=f"qualification measurement invalid: {gate}") from error
    if _canonical_json(payload) != content:
        raise CoordinatorError(reason=f"qualification measurement is not canonical: {gate}")
    return payload


def _reject_placeholder_provenance(config: CampaignConfig) -> None:
    hash_fields = (
        config.target_harness_sha256,
        config.prompt_sha256,
        config.remote_code_sha256,
        config.search_binary_sha256,
        config.coverage_binary_sha256,
        config.coverage_build_manifest_sha256,
        config.c0_sha256,
        config.dictionary_sha256,
    )
    if any(_is_placeholder_hash(value) for value in hash_fields):
        raise CoordinatorError(reason="qualification placeholder provenance")
    _reject_placeholder_image_digest(config.search_image_digest)
    _reject_placeholder_image_digest(config.coverage_image_digest)


def _reject_placeholder_image_digest(value: str) -> None:
    prefix = "sha256:"
    if not value.startswith(prefix) or _is_placeholder_hash(value.removeprefix(prefix)):
        raise CoordinatorError(reason="qualification placeholder provenance")


def _decoded_c0_sha256(path: Path) -> str:
    try:
        _reject_symlink_path(path)
        return sha256_bytes(bytes.fromhex(path.read_text(encoding="ascii").strip()))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CoordinatorError(reason="invalid C0 source fixture") from error


def _is_placeholder_hash(value: str) -> bool:
    return len(value) == _SHA256_HEX_LEN and len(set(value)) == 1 and value[0] in "0123456789abcdef"


def _strict_int_field(payload: dict[str, JsonValue], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _int_map_value(measurements: dict[str, JsonValue], key: str) -> dict[str, int]:
    value = _dict_value(measurements, key)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value.values()):
        return {name: item for name, item in value.items() if isinstance(item, int)}
    raise CoordinatorError(reason=f"qualification measurement missing: {key}")


def _banks_complete(  # noqa: C901, PLR0911, PLR0912
    measurements: dict[str, JsonValue], config: CampaignConfig
) -> bool:
    repetitions = measurements.get("repetitions")
    if not isinstance(repetitions, list | tuple) or len(repetitions) != _EXPECTED_REPETITIONS:
        return False
    seen: set[int] = set()
    for item in repetitions:
        if not isinstance(item, dict):
            return False
        repetition = item.get("repetition")
        if not isinstance(repetition, int) or isinstance(repetition, bool):
            return False
        try:
            bank_path = Path(_str_value(item, "bank_path"))
            _verify_bound_file(
                root=config.artifact_root / "qualification" / "banks",
                path=bank_path,
                sha256=_str_value(item, "bank_artifact_sha256"),
                size_bytes=_strict_int_field(item, "bank_artifact_size_bytes"),
            )
            bank = _load_bank_evidence(bank_path, repetition)
            canonical_bank = load_persisted_candidate_bank(bank_path, repetition)
            primary_raw_count = _strict_int_field(item, "primary_raw_count")
            supplement_raw_count = _strict_int_field(item, "supplement_raw_count")
            total_raw_count = _strict_int_field(item, "total_raw_count")
            raw_count = _strict_int_field(item, "raw_count")
            primary_unique_count = _strict_int_field(item, "primary_unique_count")
            final_unique_count = _strict_int_field(item, "final_unique_count")
            unique_count = _strict_int_field(item, "unique_count")
            selected_count = _strict_int_field(item, "selected_count")
        except CoordinatorError:
            return False
        raw_parent_count = len(canonical_bank.raw_parents)
        primary_unique = len(
            {
                parent.materialized_sha256
                for parent in canonical_bank.raw_parents[:_EXPECTED_BANK_RAW]
            }
        )
        final_unique = len({parent.materialized_sha256 for parent in canonical_bank.raw_parents})
        supplemented = item.get("supplemented")
        if not isinstance(supplemented, bool):
            return False
        if (
            bank.primary_raw_count != primary_raw_count
            or bank.supplement_raw_count != supplement_raw_count
            or canonical_bank.primary_raw_count != primary_raw_count
            or canonical_bank.supplement_raw_count != supplement_raw_count
            or raw_parent_count != raw_count
            or primary_unique != primary_unique_count
            or final_unique != final_unique_count
            or final_unique != unique_count
            or len(canonical_bank.selected_parents) != selected_count
            or _str_value(item, "raw_token_states_sha256") != canonical_bank.raw_token_states_sha256
            or _str_value(item, "raw_trajectories_sha256") != canonical_bank.raw_trajectories_sha256
            or _str_value(item, "raw_materialized_sha256") != canonical_bank.raw_materialized_sha256
            or _str_value(item, "selected_bank_sha256") != canonical_bank.selected_bank_sha256
        ):
            return False
        if supplement_raw_count not in {0, _SUPPLEMENT_RAW_COUNT}:
            return False
        total = primary_raw_count + supplement_raw_count
        if total_raw_count != total or raw_count != total or unique_count != final_unique_count:
            return False
        if primary_raw_count != _EXPECTED_BANK_RAW or selected_count != _MINIMUM_UNIQUE_BANK:
            return False
        if primary_unique_count > primary_raw_count or final_unique_count > total:
            return False
        if final_unique_count < _MINIMUM_UNIQUE_BANK:
            return False
        if supplemented != (supplement_raw_count == _SUPPLEMENT_RAW_COUNT):
            return False
        if supplement_raw_count == 0 and final_unique_count != primary_unique_count:
            return False
        if (
            supplement_raw_count == _SUPPLEMENT_RAW_COUNT
            and primary_unique_count >= _MINIMUM_UNIQUE_BANK
        ):
            return False
        seen.add(repetition)
    return seen == set(range(_EXPECTED_REPETITIONS))


def _smoke_passed(measurements: dict[str, JsonValue], config: CampaignConfig) -> bool:
    try:
        smoke_path = Path(_str_value(measurements, "smoke_artifact_path"))
        _verify_bound_file(
            root=config.artifact_root,
            path=smoke_path,
            sha256=_str_value(measurements, "smoke_artifact_sha256"),
            size_bytes=_int_value(measurements, "smoke_artifact_size_bytes"),
        )
        plan = build_qualification_smoke_plan(config=config, smoke_root=smoke_path.parent)
        artifact = load_smoke_artifact_file(plan, smoke_path)
        expected_pairs = _smoke_pair_measurements(plan, artifact.result)
    except (CoordinatorError, QualificationSmokeError, OSError, ValidationError):
        return False
    pairs = measurements.get("pairs")
    if not isinstance(pairs, list | tuple) or tuple(pairs) != tuple(expected_pairs):
        return False
    pair_ids: set[int] = set()
    for item in pairs:
        if not isinstance(item, dict):
            return False
        pair_id = item.get("pair_id")
        if (
            not isinstance(pair_id, int)
            or isinstance(pair_id, bool)
            or item.get("deadline_seconds") != _SMOKE_DEADLINE_SECONDS
            or item.get("schema_valid") is not True
            or item.get("hash_valid") is not True
            or item.get("restart_success") is not True
            or item.get("deferred_proof_success") is not True
        ):
            return False
        pair_ids.add(pair_id)
    return pair_ids == set(range(_SMOKE_PAIR_COUNT))


def _resource_topology_matches(  # noqa: PLR0911
    measurements: dict[str, JsonValue], expected: ResourceMap
) -> bool:
    try:
        if tuple(_sequence_value(measurements, "lscpu_argv")) != _LSCPU_TOPOLOGY_ARGV:
            return False
        expected_physical_indices = tuple(record.index for record in expected.gpu_records)
        expected_argv = nvidia_gpu_topology_argv(expected_physical_indices)
        if tuple(_sequence_value(measurements, "nvidia_smi_argv")) != expected_argv:
            return False
        lscpu_stdout = _str_value(measurements, "lscpu_stdout")
        lscpu_stderr = _str_value(measurements, "lscpu_stderr")
        nvidia_stdout = _str_value(measurements, "nvidia_smi_stdout")
        nvidia_stderr = _str_value(measurements, "nvidia_smi_stderr")
        if _int_value(measurements, "lscpu_exit_code") != 0:
            return False
        if _int_value(measurements, "nvidia_smi_exit_code") != 0:
            return False
        if sha256_bytes(lscpu_stdout.encode()) != _str_value(measurements, "lscpu_stdout_sha256"):
            return False
        if sha256_bytes(lscpu_stderr.encode()) != _str_value(measurements, "lscpu_stderr_sha256"):
            return False
        if sha256_bytes(nvidia_stdout.encode()) != _str_value(
            measurements, "nvidia_smi_stdout_sha256"
        ):
            return False
        if sha256_bytes(nvidia_stderr.encode()) != _str_value(
            measurements, "nvidia_smi_stderr_sha256"
        ):
            return False
        topology = _parse_lscpu_topology(lscpu_stdout)
        gpu_uuid_by_index = _parse_nvidia_gpu_records(nvidia_stdout)
        thread_siblings = _parse_thread_sibling_records(measurements)
        gpu_numa = _parse_gpu_sysfs_numa_records(measurements, gpu_uuid_by_index)
    except (CoordinatorError, ValueError):
        return False
    return (
        _core_numa_matches(expected, topology.cpu_numa)
        and _thread_siblings_match(topology, thread_siblings)
        and _gpu_records_match(expected, gpu_uuid_by_index, gpu_numa)
    )


def _decode_stdout(process: CompletedProcess[bytes], label: str) -> str:
    try:
        return process.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CoordinatorError(reason=f"{label} stdout is not utf-8") from error


def _decode_stderr(process: CompletedProcess[bytes], label: str) -> str:
    try:
        return process.stderr.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CoordinatorError(reason=f"{label} stderr is not utf-8") from error


def _parse_lscpu_topology(stdout: str) -> LscpuTopology:
    cpu_numa: dict[int, int] = {}
    physical_groups: dict[tuple[int, int], set[int]] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        parts = tuple(part.strip() for part in stripped.split(","))
        if len(parts) != _LSCPU_TOPOLOGY_COLUMN_COUNT:
            raise CoordinatorError(reason="lscpu topology row must contain four columns")
        cpu_id = int(parts[0])
        core_id = int(parts[1])
        socket_id = int(parts[2])
        numa_node = int(parts[3])
        if cpu_numa.setdefault(cpu_id, numa_node) != numa_node:
            raise CoordinatorError(reason="lscpu duplicate CPU topology row")
        physical_groups.setdefault((socket_id, core_id), set()).add(cpu_id)
    return LscpuTopology(
        cpu_numa=cpu_numa,
        physical_groups={key: frozenset(value) for key, value in physical_groups.items()},
    )


def _parse_nvidia_gpu_records(stdout: str) -> dict[int, NvidiaGpuRecord]:
    records: dict[int, NvidiaGpuRecord] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        parts = tuple(part.strip() for part in stripped.split(","))
        if len(parts) != _NVIDIA_TOPOLOGY_COLUMN_COUNT:
            raise CoordinatorError(reason="nvidia-smi topology row must contain three columns")
        index = int(parts[0])
        uuid = parts[1]
        bus_id = parts[2]
        if not uuid.startswith("GPU-"):
            raise CoordinatorError(reason="nvidia-smi GPU uuid is invalid")
        record = NvidiaGpuRecord(uuid=uuid, bus_id=bus_id)
        if records.setdefault(index, record) != record:
            raise CoordinatorError(reason="nvidia-smi duplicate GPU topology row")
    return records


def _parse_thread_sibling_records(
    measurements: dict[str, JsonValue],
) -> dict[int, frozenset[int]]:
    records = measurements.get("thread_siblings")
    if not isinstance(records, list | tuple):
        raise CoordinatorError(reason="qualification measurement missing: thread_siblings")
    siblings: dict[int, frozenset[int]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise CoordinatorError(reason="thread sibling record must be an object")
        cpu = record.get("cpu")
        raw = record.get("raw")
        digest = record.get("raw_sha256")
        path = record.get("path")
        if (
            not isinstance(cpu, int)
            or isinstance(cpu, bool)
            or not isinstance(raw, str)
            or not isinstance(digest, str)
            or not isinstance(path, str)
        ):
            raise CoordinatorError(reason="thread sibling record is incomplete")
        if Path(path).parts[-3:] != (
            f"cpu{cpu}",
            "topology",
            "thread_siblings_list",
        ):
            raise CoordinatorError(reason="thread sibling record path is invalid")
        if sha256_bytes(raw.encode()) != digest:
            raise CoordinatorError(reason="thread sibling record hash mismatch")
        parsed = frozenset(_parse_cpu_list(raw))
        if siblings.setdefault(cpu, parsed) != parsed:
            raise CoordinatorError(reason="duplicate thread sibling record")
    return siblings


def _parse_cpu_list(raw: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in raw.split(","):
        token = item.strip()
        if token == "":
            continue
        bounds = tuple(part.strip() for part in token.split("-"))
        if len(bounds) == 1:
            values.add(int(bounds[0]))
            continue
        if len(bounds) != _CPU_LIST_RANGE_PART_COUNT:
            raise CoordinatorError(reason="CPU list entry is invalid")
        lower = int(bounds[0])
        upper = int(bounds[1])
        if lower > upper:
            raise CoordinatorError(reason="CPU range is inverted")
        values.update(range(lower, upper + 1))
    return tuple(sorted(values))


def _parse_gpu_sysfs_numa_records(
    measurements: dict[str, JsonValue], gpu_records: dict[int, NvidiaGpuRecord]
) -> dict[int, int]:
    records = measurements.get("gpu_sysfs_numa_nodes")
    if not isinstance(records, list | tuple):
        raise CoordinatorError(reason="qualification measurement missing: gpu_sysfs_numa_nodes")
    numa: dict[int, int] = {}
    seen_keys: set[tuple[int, str, str]] = set()
    for item in records:
        if not isinstance(item, dict):
            raise CoordinatorError(reason="GPU sysfs NUMA record must be an object")
        index = item.get("index")
        uuid = item.get("uuid")
        bus_id = item.get("pci_bus_id")
        path = item.get("path")
        raw = item.get("raw")
        digest = item.get("raw_sha256")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(uuid, str)
            or not isinstance(bus_id, str)
            or not isinstance(path, str)
            or not isinstance(raw, str)
            or not isinstance(digest, str)
        ):
            raise CoordinatorError(reason="GPU sysfs NUMA record is incomplete")
        expected = gpu_records.get(index)
        if expected is None or expected.uuid != uuid or expected.bus_id != bus_id:
            raise CoordinatorError(reason="GPU sysfs NUMA record identity mismatch")
        if sha256_bytes(raw.encode()) != digest:
            raise CoordinatorError(reason="GPU sysfs NUMA record hash mismatch")
        path_parts = Path(path).parts
        if len(path_parts) < _SYSFS_NUMA_PATH_SUFFIX_LEN or path_parts[
            -_SYSFS_NUMA_PATH_SUFFIX_LEN:
        ] != (
            _sysfs_pci_device_name(bus_id),
            "numa_node",
        ):
            raise CoordinatorError(reason="GPU sysfs NUMA path is invalid")
        key = (index, uuid, bus_id)
        if key in seen_keys:
            raise CoordinatorError(reason="duplicate GPU sysfs NUMA record")
        seen_keys.add(key)
        numa[index] = int(raw)
    if set(numa) != set(gpu_records):
        raise CoordinatorError(reason="GPU sysfs NUMA record set mismatch")
    return numa


def _core_numa_matches(expected: ResourceMap, measured: dict[int, int]) -> bool:
    return all(measured.get(core_id) == node for core_id, node in expected.core_numa_nodes.items())


def _thread_siblings_match(topology: LscpuTopology, measured: dict[int, frozenset[int]]) -> bool:
    if set(measured) != set(topology.cpu_numa):
        return False
    expected_by_cpu: dict[int, frozenset[int]] = {}
    for sibling_group in topology.physical_groups.values():
        for cpu in sibling_group:
            expected_by_cpu[cpu] = sibling_group
    if set(expected_by_cpu) != set(topology.cpu_numa):
        return False
    for cpu, expected_group in expected_by_cpu.items():
        if measured.get(cpu) != expected_group:
            return False
    return True


def _gpu_records_match(
    expected: ResourceMap,
    measured_uuid_by_index: dict[int, NvidiaGpuRecord],
    measured_numa: dict[int, int],
) -> bool:
    expected_records = {
        record.index: (record.uuid, record.numa_node) for record in expected.gpu_records
    }
    return (
        set(measured_uuid_by_index) == set(expected_records)
        and set(measured_numa) == set(expected_records)
        and all(
            measured_uuid_by_index[index].uuid == uuid and measured_numa[index] == numa_node
            for index, (uuid, numa_node) in expected_records.items()
        )
    )


def _control_configs_equivalent(measurements: dict[str, JsonValue]) -> bool:
    treatment = measurements.get("treatment_config")
    control = measurements.get("control_config")
    if not isinstance(treatment, dict) or not isinstance(control, dict):
        return False
    treatment_payload = _JSON_OBJECT.validate_python(treatment)
    control_payload = _JSON_OBJECT.validate_python(control)
    treatment_sha = sha256_bytes(_canonical_json(treatment_payload))
    control_sha = sha256_bytes(_canonical_json(control_payload))
    if treatment_sha != _str_value(measurements, "treatment_config_sha256"):
        return False
    if control_sha != _str_value(measurements, "control_config_sha256"):
        return False
    differing = _canonical_diff_paths(treatment_payload, control_payload)
    treatment_policy = measurements.get("treatment_policy")
    control_policy = measurements.get("control_policy")
    return (
        _str_value(measurements, "treatment_arm") == "selective-revision"
        and _str_value(measurements, "control_arm") == "final-only"
        and treatment_sha != control_sha
        and treatment_payload.get("policy") == treatment_policy
        and treatment_payload.get("arm") == "selective-revision"
        and treatment_payload.get("trajectory_visibility") is True
        and control_payload.get("policy") == control_policy
        and control_payload.get("arm") == "final-only"
        and control_payload.get("trajectory_visibility") is False
        and treatment_policy
        == _JSON_OBJECT.validate_python(
            {
                "variant": "selective-revision",
                "trajectory_feature_view": "trajectory-aware",
                "remask_scope": "selected-field",
            }
        )
        and control_policy
        == _JSON_OBJECT.validate_python(
            {
                "variant": "final-only",
                "trajectory_feature_view": "final-only",
                "remask_scope": "selected-field",
            }
        )
        and differing
        == (
            "arm",
            "policy.trajectory_feature_view",
            "policy.variant",
            "trajectory_visibility",
        )
    )


def _canonical_diff_paths(left: JsonValue, right: JsonValue, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = key if prefix == "" else f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(path)
                continue
            paths.extend(_canonical_diff_paths(left[key], right[key], path))
        return tuple(paths)
    if left == right:
        return ()
    return (prefix,)


def _read_only_cache_matches(  # noqa: PLR0911
    measurements: dict[str, JsonValue], config: CampaignConfig
) -> bool:
    try:
        configured = config.model_cache_dir
        configured_resolved = configured.resolve(strict=True)
        _reject_symlink_path(configured)
        if _str_value(measurements, "model_id") != MODEL_ID:
            return False
        if _str_value(measurements, "model_revision") != config.model_revision:
            return False
        if config.model_revision != MODEL_REVISION:
            return False
        if _str_value(measurements, "remote_code_sha256") != config.remote_code_sha256:
            return False
        if _str_value(measurements, "cache_tree_sha256") != _artifact_tree_sha256(configured):
            return False
        cache_path = Path(_str_value(measurements, "cache_path"))
        resolved_cache_path = Path(_str_value(measurements, "resolved_cache_path"))
        return (
            configured.is_absolute()
            and cache_path == configured
            and resolved_cache_path == configured_resolved
            and configured_resolved.is_dir()
            and not bool(configured_resolved.stat().st_mode & 0o222)
            and _bool_value(measurements, "exists")
            and _bool_value(measurements, "is_dir")
            and _bool_value(measurements, "offline")
            and _bool_value(measurements, "read_only")
            and _str_value(measurements, "write_probe_name") == _CACHE_WRITE_PROBE_NAME
            and not _bool_value(measurements, "write_probe_succeeded")
            and _str_value(measurements, "write_probe_error") != ""
            and _bool_value(measurements, "write_probe_removed")
            and not (configured_resolved / _CACHE_WRITE_PROBE_NAME).exists()
        )
    except (CoordinatorError, OSError):
        return False


def _raw_path_record(path: Path, prefix: dict[str, JsonValue], raw: str) -> dict[str, JsonValue]:
    return {
        **prefix,
        "path": path.as_posix(),
        "raw": raw,
        "raw_sha256": sha256_bytes(raw.encode()),
    }


def _hash_backing_matches(measurements: dict[str, JsonValue], config: CampaignConfig) -> bool:
    try:
        return (
            _file_sha256(Path(_str_value(measurements, "target_harness_path")))
            == config.target_harness_sha256
            and Path(_str_value(measurements, "prompt_path")) == config.prompt_path
            and _file_sha256(config.prompt_path) == config.prompt_sha256
            and _file_sha256(Path(_str_value(measurements, "remote_code_path")))
            == config.remote_code_sha256
            and _file_sha256(Path(_str_value(measurements, "search_binary_path")))
            == config.search_binary_sha256
            and _file_sha256(Path(_str_value(measurements, "coverage_binary_path")))
            == config.coverage_binary_sha256
            and _file_sha256(Path(_str_value(measurements, "coverage_build_manifest_path")))
            == config.coverage_build_manifest_sha256
            and Path(_str_value(measurements, "c0_source_path")) == config.c0_source_path
            and _decoded_c0_sha256(Path(_str_value(measurements, "c0_source_path")))
            == config.c0_sha256
            and _file_sha256(Path(_str_value(measurements, "dictionary_path")))
            == config.dictionary_sha256
        )
    except CoordinatorError:
        return False


def _coverage_backing_matches(measurements: dict[str, JsonValue], config: CampaignConfig) -> bool:
    try:
        _verify_bound_file(
            root=config.artifact_root,
            path=Path(_str_value(measurements, "coverage_artifact_path")),
            sha256=_str_value(measurements, "coverage_artifact_sha256"),
            size_bytes=_int_value(measurements, "coverage_artifact_size_bytes"),
        )
        _verify_bound_file(
            root=config.artifact_root,
            path=Path(_str_value(measurements, "repeated_coverage_artifact_path")),
            sha256=_str_value(measurements, "repeated_coverage_artifact_sha256"),
            size_bytes=_int_value(measurements, "repeated_coverage_artifact_size_bytes"),
        )
    except CoordinatorError:
        return False
    return True


def _load_bank_evidence(path: Path, repetition: int) -> CandidateBankEvidence:
    try:
        evidence = CandidateBankEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid candidate-bank evidence") from error
    if evidence.repetition != repetition:
        raise CoordinatorError(reason="candidate-bank repetition mismatch")
    unsigned = _canonical_json(
        _JSON_OBJECT.validate_python(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    )
    if evidence.evidence_sha256 != sha256_bytes(unsigned):
        raise CoordinatorError(reason="candidate-bank evidence hash mismatch")
    return evidence


def _verify_bound_file(*, root: Path, path: Path, sha256: str, size_bytes: int) -> None:
    _reject_symlink_path(root)
    _reject_symlink_path(path)
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise CoordinatorError(reason="qualification backing path escapes root")
    if not resolved_path.is_file():
        raise CoordinatorError(reason="qualification backing is not a file")
    content = resolved_path.read_bytes()
    if len(content) != size_bytes:
        raise CoordinatorError(reason="qualification backing size mismatch")
    if sha256_bytes(content) != sha256:
        raise CoordinatorError(reason="qualification backing hash mismatch")


def _file_sha256(path: Path) -> str:
    try:
        _reject_symlink_path(path)
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise CoordinatorError(reason=f"qualification backing unavailable: {path}") from error


def _artifact_tree_sha256(root: Path, *, exclude: Path | None = None) -> str:
    try:
        _reject_symlink_path(root)
        resolved_root = root.resolve(strict=True)
        resolved_exclude = None if exclude is None else exclude.resolve(strict=False)
        files: list[dict[str, JsonValue]] = []
        for path in sorted(resolved_root.rglob("*")):
            _reject_symlink_path(path)
            if not path.is_file() or path == resolved_exclude:
                continue
            content = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(resolved_root).as_posix(),
                    "sha256": sha256_bytes(content),
                    "size_bytes": len(content),
                }
            )
        return sha256_bytes(_canonical_json(_JSON_OBJECT.validate_python({"files": files})))
    except OSError as error:
        raise CoordinatorError(reason="qualification artifact tree unavailable") from error


def _sysfs_pci_device_name(bus_id: str) -> str:
    domain, rest = bus_id.split(":", 1)
    if len(domain) == _NVIDIA_DOMAIN_HEX_LEN:
        domain = domain[-4:]
    return f"{domain}:{rest}".lower()


def _reject_symlink_path(path: Path) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    for candidate in (*tuple(reversed(absolute.parents)), absolute):
        if candidate.is_symlink():
            raise CoordinatorError(reason="qualification path contains symlink")
