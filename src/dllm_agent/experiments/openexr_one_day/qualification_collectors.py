"""Production collectors for OpenEXR one-day qualification measurements.

# noqa: SIZE_OK
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import Field, TypeAdapter, ValidationError
from tools.stable_diffcoder_payloads import MODEL_ID, MODEL_REVISION

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CandidateBankEvidence,
    CoordinatorError,
    build_campaign_manifest,
    load_persisted_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ProductionGateRunner,
    QualificationGateName,
    build_qualification_smoke_plan,
    collect_read_only_cache_measurements,
    collect_resource_topology_measurements,
    write_record_backed_qualification_from_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    QualificationSmokeError,
    load_smoke_artifact_file,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from dllm_agent.revision.controller_models import online_revision_policy
from dllm_agent.revision.models import RevisionConfig

if TYPE_CHECKING:
    from subprocess import CompletedProcess

    from dllm_agent.experiments.openexr_one_day.qualification import QualificationEvidence

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NVIDIA_VRAM_ARGV: Final = (
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
)
_DOCKER_IMAGE_ID_FORMAT: Final = "{{.Id}}"
_DOCKER_IMAGE_ID_LEN: Final = 71
_MIN_MEMINFO_FIELD_COUNT: Final = 2
_CPU_IDLE_INDEX: Final = 3
_CPU_IOWAIT_INDEX: Final = 4
_SMOKE_ARMS: Final = (
    "selective-revision",
    "final-only",
    "full-regenerate",
    "structured-one-shot",
    "libfuzzer-only",
)


class QualificationCommandRunner(Protocol):
    """Shell-free command runner used by production collectors."""

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        """Run one fixed argv vector and return captured bytes."""
        ...


class ProcReader(Protocol):
    """Read Linux procfs/sysfs text for deterministic tests."""

    def read_text(self, path: Path) -> str:
        """Return the text at a fixed local path."""
        ...

    def monotonic_seconds(self) -> float:
        """Return a monotonic timestamp for interval qualification."""
        ...


class CoverageQualificationRunner(Protocol):
    """Typed external coverage qualification seam."""

    def run(self, request: CoverageQualificationRequest) -> CoverageQualificationResult:
        """Replay target-only coverage for the exact immutable request."""
        ...


@dataclass(frozen=True, slots=True)
class SystemProcReader:
    """Production procfs/sysfs reader."""

    def read_text(self, path: Path) -> str:
        """Read UTF-8 text from a required procfs/sysfs path."""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise CoordinatorError(reason=f"qualification backing unavailable: {path}") from error

    def monotonic_seconds(self) -> float:
        """Return the current monotonic clock."""
        return monotonic()


@dataclass(frozen=True, slots=True)
class HashArtifactInputs:
    """Required backing paths for artifact and image hash collection."""

    target_harness_path: Path
    remote_code_path: Path
    search_binary_path: Path
    coverage_binary_path: Path
    coverage_build_manifest_path: Path
    c0_source_path: Path
    dictionary_path: Path
    search_image_ref: str
    coverage_image_ref: str


@dataclass(frozen=True, slots=True)
class TopologyPaths:
    """Filesystem roots for resource topology evidence."""

    thread_sibling_root: Path = Path("/sys/devices/system/cpu")
    pci_device_root: Path = Path("/sys/bus/pci/devices")


@dataclass(frozen=True, slots=True)
class CollectorInputs:
    """All environment-dependent inputs for the 12 qualification collectors."""

    config: CampaignConfig
    measurements_root: Path
    evidence_root: Path
    hash_artifacts: HashArtifactInputs
    model_id: str
    coverage_runner: CoverageQualificationRunner
    smoke_artifact_path: Path
    disk_path: Path
    command_runner: QualificationCommandRunner
    topology: TopologyPaths = TopologyPaths()
    proc_reader: ProcReader = SystemProcReader()


class CoverageQualificationRequest(FrozenModel):
    """Exact coverage qualification replay request."""

    target_commit: str
    image_digest: str
    binary_sha256: Sha256
    build_manifest_sha256: Sha256
    repeated: bool = False


class CoverageQualificationResult(FrozenModel):
    """Measured branch count from an external coverage runner."""

    target_only_branch_count: int = Field(gt=0)
    artifact_path: Path
    artifact_sha256: Sha256


def collect_all_measurements(inputs: CollectorInputs) -> Path:
    """Collect every gate, write canonical measurement files, then write evidence."""
    _validate_output_root(inputs.config.artifact_root, inputs.measurements_root)
    _validate_output_root(inputs.config.artifact_root, inputs.evidence_root)
    _preflight_publish_root(inputs.measurements_root)
    _preflight_publish_root(inputs.evidence_root)
    evidence_path = inputs.evidence_root / "qualification.json"
    manifest = build_campaign_manifest(
        inputs.config,
        build_campaign_schedule(
            main_start_monotonic_seconds=inputs.config.main_start_monotonic_seconds,
            gpu_count=len(inputs.config.resource_map.model_cores),
        ),
    )
    if inputs.measurements_root.is_dir() and evidence_path.is_file():
        _validate_published_measurements(inputs.measurements_root)
        _ = ProductionGateRunner(evidence_path).evaluate(manifest)
        return evidence_path
    measurements = _collect_measurements(inputs)
    measurement_stage = _stage_measurement_root(inputs.measurements_root)
    evidence_stage = _stage_measurement_root(inputs.evidence_root)
    try:
        for gate in QUALIFICATION_GATE_NAMES:
            _write_canonical_measurement(measurement_stage / f"{gate}.json", measurements[gate])
        result = write_record_backed_qualification_from_measurements(
            config=inputs.config,
            evidence_root=evidence_stage,
            measurements_root=measurement_stage,
        )
        if result.evidence.manifest_sha256 != (manifest.manifest_sha256 or ""):
            raise CoordinatorError(reason="qualification evidence manifest mismatch")
        final_evidence = _finalize_staged_evidence_payload(
            evidence=result.evidence,
            final_evidence_root=inputs.evidence_root,
        )
        _write_canonical_measurement(
            evidence_stage / "qualification.json",
            final_evidence,
        )
        _preflight_publish_content(measurement_stage, inputs.measurements_root)
        _preflight_publish_evidence_content(evidence_stage, inputs.evidence_root)
        _publish_measurement_root(measurement_stage, inputs.measurements_root)
        measurement_stage = Path()
        _publish_evidence_root(evidence_stage, inputs.evidence_root)
        evidence_stage = Path()
    finally:
        _remove_stage_root(measurement_stage)
        _remove_stage_root(evidence_stage)
    return evidence_path


def collect_hash_measurements(
    config: CampaignConfig,
    artifacts: HashArtifactInputs,
    runner: QualificationCommandRunner,
) -> dict[str, JsonValue]:
    """Collect hash and provenance measurements from files and Docker inspect."""
    search_digest = _docker_image_digest(runner, artifacts.search_image_ref)
    coverage_digest = _docker_image_digest(runner, artifacts.coverage_image_ref)
    measurements = {
        "target_commit": config.target_commit,
        "config_sha256": config.config_sha256,
        "target_harness_path": artifacts.target_harness_path.as_posix(),
        "target_harness_sha256": _file_sha256(artifacts.target_harness_path),
        "prompt_path": config.prompt_path.as_posix(),
        "prompt_sha256": _file_sha256(config.prompt_path),
        "remote_code_path": artifacts.remote_code_path.as_posix(),
        "remote_code_sha256": _file_sha256(artifacts.remote_code_path),
        "search_binary_path": artifacts.search_binary_path.as_posix(),
        "search_binary_sha256": _file_sha256(artifacts.search_binary_path),
        "search_image_ref": artifacts.search_image_ref,
        "search_image_digest": search_digest,
        "coverage_binary_path": artifacts.coverage_binary_path.as_posix(),
        "coverage_binary_sha256": _file_sha256(artifacts.coverage_binary_path),
        "coverage_image_ref": artifacts.coverage_image_ref,
        "coverage_image_digest": coverage_digest,
        "coverage_build_manifest_path": artifacts.coverage_build_manifest_path.as_posix(),
        "coverage_build_manifest_sha256": _file_sha256(artifacts.coverage_build_manifest_path),
        "c0_source_path": artifacts.c0_source_path.as_posix(),
        "c0_sha256": _decoded_c0_sha256(artifacts.c0_source_path),
        "dictionary_path": artifacts.dictionary_path.as_posix(),
        "dictionary_sha256": _file_sha256(artifacts.dictionary_path),
    }
    _require_equal("search_image_digest", search_digest, config.search_image_digest)
    _require_equal("coverage_image_digest", coverage_digest, config.coverage_image_digest)
    _require_equal(
        "target_harness_sha256",
        measurements["target_harness_sha256"],
        config.target_harness_sha256,
    )
    _require_equal("prompt_sha256", measurements["prompt_sha256"], config.prompt_sha256)
    _require_equal(
        "remote_code_sha256",
        measurements["remote_code_sha256"],
        config.remote_code_sha256,
    )
    _require_equal(
        "search_binary_sha256",
        measurements["search_binary_sha256"],
        config.search_binary_sha256,
    )
    _require_equal(
        "coverage_binary_sha256",
        measurements["coverage_binary_sha256"],
        config.coverage_binary_sha256,
    )
    _require_equal(
        "coverage_build_manifest_sha256",
        measurements["coverage_build_manifest_sha256"],
        config.coverage_build_manifest_sha256,
    )
    _require_equal("c0_sha256", measurements["c0_sha256"], config.c0_sha256)
    _require_equal("dictionary_sha256", measurements["dictionary_sha256"], config.dictionary_sha256)
    return _JSON_OBJECT.validate_python(measurements)


def collect_banks_measurements(config: CampaignConfig) -> dict[str, JsonValue]:
    """Collect bank measurements by validating all on-disk candidate-bank artifacts."""
    repetitions: list[JsonValue] = []
    for repetition in range(10):
        path = (
            config.artifact_root
            / "qualification"
            / "banks"
            / f"rep-{repetition:02d}"
            / "candidate-bank.json"
        )
        bank = load_persisted_candidate_bank(path, repetition)
        evidence = _load_candidate_bank_evidence(path, repetition)
        primary = bank.raw_parents[:32]
        raw_count = len(bank.raw_parents)
        unique_count = len({parent.materialized_sha256 for parent in bank.raw_parents})
        primary_unique_count = len({parent.materialized_sha256 for parent in primary})
        repetitions.append(
            {
                "repetition": repetition,
                "bank_path": path.as_posix(),
                "bank_artifact_sha256": sha256_bytes(path.read_bytes()),
                "bank_artifact_size_bytes": path.stat().st_size,
                "evidence_sha256": evidence.evidence_sha256 or "",
                "primary_raw_count": evidence.primary_raw_count,
                "supplement_raw_count": evidence.supplement_raw_count,
                "total_raw_count": raw_count,
                "raw_count": raw_count,
                "primary_unique_count": primary_unique_count,
                "final_unique_count": unique_count,
                "unique_count": unique_count,
                "selected_count": len(bank.selected_parents),
                "supplemented": evidence.supplement_raw_count > 0,
                "raw_token_states_sha256": bank.raw_token_states_sha256,
                "raw_trajectories_sha256": bank.raw_trajectories_sha256,
                "raw_materialized_sha256": bank.raw_materialized_sha256,
                "selected_bank_sha256": bank.selected_bank_sha256,
            }
        )
    return _JSON_OBJECT.validate_python({"repetitions": repetitions})


def collect_read_only_model_cache_measurements(
    config: CampaignConfig, model_id: str
) -> dict[str, JsonValue]:
    """Collect cache measurements and bind model provenance fields."""
    if model_id != MODEL_ID or config.model_revision != MODEL_REVISION:
        raise CoordinatorError(reason="qualification model provenance missing")
    measurements = collect_read_only_cache_measurements(config.model_cache_dir)
    measurements.update(
        {
            "model_id": model_id,
            "model_revision": config.model_revision,
            "remote_code_sha256": config.remote_code_sha256,
            "configured_cache_path": config.model_cache_dir.as_posix(),
            "cache_tree_sha256": _artifact_tree_sha256(config.model_cache_dir),
        }
    )
    return _JSON_OBJECT.validate_python(measurements)


def collect_external_coverage_measurements(
    config: CampaignConfig, runner: CoverageQualificationRunner
) -> dict[str, JsonValue]:
    """Run repeated external target-only coverage and record reproducible counts."""
    first = runner.run(_coverage_request(config, repeated=False))
    repeated = runner.run(_coverage_request(config, repeated=True))
    _require_artifact(first.artifact_path, first.artifact_sha256)
    _require_artifact(repeated.artifact_path, repeated.artifact_sha256)
    return _JSON_OBJECT.validate_python(
        {
            "coverage_image_digest": config.coverage_image_digest,
            "coverage_binary_sha256": config.coverage_binary_sha256,
            "coverage_build_manifest_sha256": config.coverage_build_manifest_sha256,
            "target_only_branch_count": first.target_only_branch_count,
            "repeated_target_only_branch_count": repeated.target_only_branch_count,
            "coverage_artifact_path": first.artifact_path.as_posix(),
            "coverage_artifact_sha256": first.artifact_sha256,
            "coverage_artifact_size_bytes": first.artifact_path.stat().st_size,
            "repeated_coverage_artifact_path": repeated.artifact_path.as_posix(),
            "repeated_coverage_artifact_sha256": repeated.artifact_sha256,
            "repeated_coverage_artifact_size_bytes": repeated.artifact_path.stat().st_size,
        }
    )


def collect_control_arm_equivalence_measurements(config: CampaignConfig) -> dict[str, JsonValue]:
    """Construct actual Task2 arm policies and record their only intended difference."""
    seed = (
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds)
        .cells[0]
        .fuzzer_seed
    )
    base = _JSON_OBJECT.validate_python(
        RevisionConfig(
            schema_version="dllm-poc-revision-config/v1",
            particle_count=16,
            epochs=1,
            initial_steps=config.candidate_steps,
            revision_steps=config.candidate_steps,
            probe_seconds=1,
            fuzzer_seeds=(seed, seed + 1),
            proof_seconds=1,
            controller_mode="online-portfolio",
        ).model_dump(mode="json")
    )
    treatment = _JSON_OBJECT.validate_python(
        {
            **base,
            "arm": "selective-revision",
            "policy": _policy_json("selective-revision"),
            "trajectory_visibility": True,
        }
    )
    control = _JSON_OBJECT.validate_python(
        {
            **base,
            "arm": "final-only",
            "policy": _policy_json("final-only"),
            "trajectory_visibility": False,
        }
    )
    return _JSON_OBJECT.validate_python(
        {
            "treatment_arm": "selective-revision",
            "control_arm": "final-only",
            "treatment_config": treatment,
            "control_config": control,
            "treatment_config_sha256": sha256_bytes(canonical_json_bytes(treatment)),
            "control_config_sha256": sha256_bytes(canonical_json_bytes(control)),
            "treatment_policy": _policy_json("selective-revision"),
            "control_policy": _policy_json("final-only"),
        }
    )


def collect_disk_space_measurements(path: Path) -> dict[str, JsonValue]:
    """Collect statvfs free-byte evidence for the filesystem holding path."""
    try:
        stats = os.statvfs(path)
        stat_result = path.stat()
    except OSError as error:
        raise CoordinatorError(reason=f"qualification backing unavailable: {path}") from error
    return _JSON_OBJECT.validate_python(
        {
            "path": path.as_posix(),
            "device": stat_result.st_dev,
            "free_bytes": stats.f_bavail * stats.f_frsize,
            "fragment_size": stats.f_frsize,
            "available_blocks": stats.f_bavail,
        }
    )


def collect_vram_measurements(
    runner: QualificationCommandRunner, *, gpu_count: int
) -> dict[str, JsonValue]:
    """Collect per-GPU free VRAM through fixed nvidia-smi argv."""
    pre = _collect_vram_sample(runner, gpu_count)
    post = _collect_vram_sample(runner, gpu_count)
    return _JSON_OBJECT.validate_python(
        {
            "nvidia_smi_argv": list(_NVIDIA_VRAM_ARGV),
            "pre_load_free_bytes": pre,
            "post_load_free_bytes": post,
        }
    )


def collect_memory_measurements(reader: ProcReader) -> dict[str, dict[str, JsonValue]]:
    """Collect MemAvailable and swap before/after from /proc/meminfo."""
    before = _read_memory(reader)
    after = _read_memory(reader)
    ram = {"available_bytes": after.mem_available_bytes}
    swap = {
        "swap_used_before_bytes": before.swap_used_bytes,
        "swap_used_after_bytes": after.swap_used_bytes,
        "mem_available_before_bytes": before.mem_available_bytes,
        "mem_available_after_bytes": after.mem_available_bytes,
    }
    return {
        "ram": _JSON_OBJECT.validate_python(ram),
        "swap_stable": _JSON_OBJECT.validate_python(swap),
    }


def collect_reserved_core_load_measurements(
    config: CampaignConfig, reader: ProcReader
) -> dict[str, JsonValue]:
    """Collect reserved-core load from two /proc/stat samples."""
    core_ids = tuple(
        assignment.core_id
        for assignment in config.resource_map.proof_cores + config.resource_map.monitor_cores
    )
    before = _read_cpu_stats(reader, core_ids)
    started = reader.monotonic_seconds()
    after = _read_cpu_stats(reader, core_ids)
    finished = reader.monotonic_seconds()
    loads = {
        str(core): _cpu_load(before[core], after[core])
        for core in core_ids
        if before[core].total_jiffies <= after[core].total_jiffies
    }
    if set(loads) != {str(core) for core in core_ids}:
        raise CoordinatorError(reason="reserved-core load interval invalid")
    interval = finished - started
    minimum_delta = min(after[core].total_jiffies - before[core].total_jiffies for core in core_ids)
    if interval <= 0.0 or minimum_delta <= 0:
        raise CoordinatorError(reason="reserved-core load interval invalid")
    return _JSON_OBJECT.validate_python(
        {
            "reserved_core_ids": list(core_ids),
            "per_core_load": loads,
            "max_load": max(loads.values()) if loads else 1.0,
            "interval_seconds": interval,
            "minimum_total_jiffy_delta": minimum_delta,
        }
    )


def collect_smoke_measurements(
    config: CampaignConfig, smoke_artifact_path: Path
) -> dict[str, JsonValue]:
    """Collect smoke measurements from a persisted hash-bound smoke result artifact."""
    try:
        _reject_symlink_path(smoke_artifact_path)
        plan = build_qualification_smoke_plan(
            config=config,
            smoke_root=smoke_artifact_path.parent,
        )
        artifact = load_smoke_artifact_file(plan, smoke_artifact_path)
    except QualificationSmokeError as error:
        raise CoordinatorError(reason=error.reason) from error
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="qualification smoke artifact unavailable") from error
    result = artifact.result
    outcomes_by_id = {outcome.cell_id: outcome for outcome in result.cells}
    restart_success = set(result.restart_success_pair_ids)
    pairs = [
        {
            "pair_id": pair_id,
            "deadline_seconds": 300,
            "schema_valid": True,
            "hash_valid": True,
            "restart_success": pair_id in restart_success,
            "deferred_proof_success": all(
                outcomes_by_id[cell.cell_id].proof_result_sha256
                and outcomes_by_id[cell.cell_id].deep_parse_result_sha256
                for cell in plan.cells
                if cell.pair_id == pair_id
            ),
        }
        for pair_id in range(2)
    ]
    return _JSON_OBJECT.validate_python(
        {
            "smoke_artifact_path": smoke_artifact_path.as_posix(),
            "smoke_artifact_sha256": sha256_bytes(smoke_artifact_path.read_bytes()),
            "smoke_artifact_size_bytes": smoke_artifact_path.stat().st_size,
            "pairs": pairs,
        }
    )


@dataclass(frozen=True, slots=True)
class _MemorySample:
    mem_available_bytes: int
    swap_used_bytes: int


@dataclass(frozen=True, slots=True)
class _CpuSample:
    idle_jiffies: int
    total_jiffies: int


def _collect_measurements(
    inputs: CollectorInputs,
) -> dict[QualificationGateName, dict[str, JsonValue]]:
    memory = collect_memory_measurements(inputs.proc_reader)
    return {
        "hashes_match": collect_hash_measurements(
            inputs.config, inputs.hash_artifacts, inputs.command_runner
        ),
        "banks_complete": collect_banks_measurements(inputs.config),
        "read_only_cache": collect_read_only_model_cache_measurements(
            inputs.config, inputs.model_id
        ),
        "external_coverage_check": collect_external_coverage_measurements(
            inputs.config, inputs.coverage_runner
        ),
        "control_arm_equivalence": collect_control_arm_equivalence_measurements(inputs.config),
        "resource_map_valid": collect_resource_topology_measurements(
            inputs.command_runner,
            thread_sibling_root=inputs.topology.thread_sibling_root,
            pci_device_root=inputs.topology.pci_device_root,
            physical_gpu_indices=tuple(
                record.index for record in inputs.config.resource_map.gpu_records
            ),
        ),
        "disk_space": collect_disk_space_measurements(inputs.disk_path),
        "vram": collect_vram_measurements(
            inputs.command_runner, gpu_count=len(inputs.config.resource_map.model_cores)
        ),
        "ram": memory["ram"],
        "swap_stable": memory["swap_stable"],
        "reserved_core_load": collect_reserved_core_load_measurements(
            inputs.config, inputs.proc_reader
        ),
        "smoke_passed": collect_smoke_measurements(inputs.config, inputs.smoke_artifact_path),
    }


def _docker_image_digest(runner: QualificationCommandRunner, image_ref: str) -> str:
    result = runner.run(
        ("docker", "image", "inspect", "--format", _DOCKER_IMAGE_ID_FORMAT, image_ref)
    )
    if result.returncode != 0:
        raise CoordinatorError(reason=f"Docker image inspect failed: {image_ref}")
    lines = result.stdout.decode("utf-8").splitlines()
    if len(lines) != 1:
        raise CoordinatorError(reason=f"Docker image ID unavailable: {image_ref}")
    image_id = lines[0].strip()
    if (
        len(image_id) != _DOCKER_IMAGE_ID_LEN
        or not image_id.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in image_id.removeprefix("sha256:"))
    ):
        raise CoordinatorError(reason=f"Docker image ID invalid: {image_ref}")
    return image_id


def _file_sha256(path: Path) -> str:
    try:
        _reject_symlink_path(path)
        if not path.is_file():
            raise CoordinatorError(reason=f"qualification backing unavailable: {path}")
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise CoordinatorError(reason=f"qualification backing unavailable: {path}") from error


def _decoded_c0_sha256(path: Path) -> str:
    try:
        _reject_symlink_path(path)
        return sha256_bytes(bytes.fromhex(path.read_text(encoding="ascii").strip()))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CoordinatorError(reason="invalid C0 source fixture") from error


def _require_equal(name: str, observed: JsonValue, expected: str) -> None:
    if observed != expected:
        raise CoordinatorError(reason=f"qualification {name} mismatch")


def _require_artifact(path: Path, digest: str) -> None:
    if _file_sha256(path) != digest:
        raise CoordinatorError(reason=f"qualification artifact hash mismatch: {path}")


def _load_candidate_bank_evidence(path: Path, repetition: int) -> CandidateBankEvidence:
    try:
        evidence = CandidateBankEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoordinatorError(reason="invalid candidate-bank evidence") from error
    if evidence.repetition != repetition:
        raise CoordinatorError(reason="candidate-bank repetition mismatch")
    unsigned = canonical_json_bytes(evidence.model_dump(mode="json", exclude={"evidence_sha256"}))
    if evidence.evidence_sha256 != sha256_bytes(unsigned):
        raise CoordinatorError(reason="candidate-bank evidence hash mismatch")
    return evidence


def _coverage_request(config: CampaignConfig, *, repeated: bool) -> CoverageQualificationRequest:
    return CoverageQualificationRequest(
        target_commit=config.target_commit,
        image_digest=config.coverage_image_digest,
        binary_sha256=config.coverage_binary_sha256,
        build_manifest_sha256=config.coverage_build_manifest_sha256,
        repeated=repeated,
    )


def _policy_json(arm: Literal["selective-revision", "final-only"]) -> dict[str, JsonValue]:
    policy = online_revision_policy(arm)
    if policy is None:
        raise CoordinatorError(reason="Task2 online policy unavailable")
    return _JSON_OBJECT.validate_python(policy.model_dump(mode="json"))


def _collect_vram_sample(runner: QualificationCommandRunner, gpu_count: int) -> dict[str, int]:
    result = runner.run(_NVIDIA_VRAM_ARGV)
    if result.returncode != 0:
        raise CoordinatorError(reason="nvidia-smi VRAM query failed")
    values: dict[str, int] = {}
    for raw_line in result.stdout.decode("utf-8").splitlines():
        line = raw_line.strip()
        if line == "":
            continue
        index, mib = (part.strip() for part in line.split(",", maxsplit=1))
        values[index] = int(mib) * 1024 * 1024
    if set(values) != {str(index) for index in range(gpu_count)}:
        raise CoordinatorError(reason="nvidia-smi VRAM set incomplete")
    return values


def _read_memory(reader: ProcReader) -> _MemorySample:
    fields: dict[str, int] = {}
    for line in reader.read_text(Path("/proc/meminfo")).splitlines():
        parts = line.split()
        if len(parts) < _MIN_MEMINFO_FIELD_COUNT:
            continue
        name = parts[0].rstrip(":")
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            fields[name] = int(parts[1]) * 1024
    try:
        return _MemorySample(
            mem_available_bytes=fields["MemAvailable"],
            swap_used_bytes=fields["SwapTotal"] - fields["SwapFree"],
        )
    except KeyError as error:
        raise CoordinatorError(reason="meminfo qualification fields unavailable") from error


def _read_cpu_stats(reader: ProcReader, core_ids: tuple[int, ...]) -> dict[int, _CpuSample]:
    wanted = {f"cpu{core}" for core in core_ids}
    samples: dict[int, _CpuSample] = {}
    for line in reader.read_text(Path("/proc/stat")).splitlines():
        parts = line.split()
        if not parts or parts[0] not in wanted:
            continue
        values = tuple(int(part) for part in parts[1:])
        core = int(parts[0].removeprefix("cpu"))
        idle = values[_CPU_IDLE_INDEX] + (
            values[_CPU_IOWAIT_INDEX] if len(values) > _CPU_IOWAIT_INDEX else 0
        )
        samples[core] = _CpuSample(idle_jiffies=idle, total_jiffies=sum(values))
    if set(samples) != set(core_ids):
        raise CoordinatorError(reason="reserved-core proc stat unavailable")
    return samples


def _cpu_load(before: _CpuSample, after: _CpuSample) -> float:
    total_delta = after.total_jiffies - before.total_jiffies
    idle_delta = after.idle_jiffies - before.idle_jiffies
    if total_delta < 0 or idle_delta < 0:
        raise CoordinatorError(reason="reserved-core load interval invalid")
    if total_delta <= 0:
        raise CoordinatorError(reason="reserved-core load interval invalid")
    return max(0.0, min(1.0, 1.0 - (idle_delta / total_delta)))


def _write_canonical_measurement(path: Path, payload: dict[str, JsonValue]) -> None:
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            _ = handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _ = temp_path.replace(path)
        _fsync_dir(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage_measurement_root(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )


def _publish_measurement_root(stage: Path, destination: Path) -> None:
    if destination.exists():
        _compare_measurement_root(stage, destination)
        shutil.rmtree(stage)
        return
    _ = stage.replace(destination)
    _fsync_dir(destination.parent)


def _publish_evidence_root(stage: Path, destination: Path) -> None:
    if not destination.exists():
        _ = stage.replace(destination)
        _fsync_dir(destination.parent)
        return
    _preflight_publish_evidence_content(stage, destination)
    for source in sorted(path for path in stage.rglob("*") if path.is_file()):
        relative = source.relative_to(stage)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = source.replace(target)
        _fsync_dir(target.parent)
    shutil.rmtree(stage)


def _compare_measurement_root(stage: Path, destination: Path) -> None:
    expected = sorted(path.relative_to(stage) for path in stage.rglob("*") if path.is_file())
    actual = sorted(
        path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
    )
    if expected != actual:
        raise CoordinatorError(reason="qualification measurements mismatch")
    for relative in expected:
        if (stage / relative).read_bytes() != (destination / relative).read_bytes():
            raise CoordinatorError(reason="qualification measurements mismatch")


def _validate_output_root(artifact_root: Path, path: Path) -> None:
    _reject_symlink_path(path)
    _reject_symlink_path(path.parent)
    resolved_root = artifact_root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise CoordinatorError(reason="qualification output root escapes artifact root")


def _preflight_publish_root(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise CoordinatorError(reason="qualification output root must be a directory")


def _preflight_publish_content(stage: Path, destination: Path) -> None:
    if destination.exists():
        _compare_measurement_root(stage, destination)


def _preflight_publish_evidence_content(stage: Path, destination: Path) -> None:
    if not destination.exists():
        return
    for source in sorted(path for path in stage.rglob("*") if path.is_file()):
        target = destination / source.relative_to(stage)
        if target.exists() and (not target.is_file() or source.read_bytes() != target.read_bytes()):
            raise CoordinatorError(reason="qualification measurements mismatch")


def _validate_published_measurements(root: Path) -> None:
    for gate in QUALIFICATION_GATE_NAMES:
        _ = _read_published_measurement(root, gate)


def _read_published_measurement(root: Path, gate: QualificationGateName) -> dict[str, JsonValue]:
    path = root / f"{gate}.json"
    try:
        content = path.read_bytes()
        payload = _JSON_OBJECT.validate_json(content)
    except OSError as error:
        raise CoordinatorError(reason=f"qualification measurement missing: {gate}") from error
    except ValidationError as error:
        raise CoordinatorError(reason=f"qualification measurement invalid: {gate}") from error
    if canonical_json_bytes(payload) != content:
        raise CoordinatorError(reason=f"qualification measurement is not canonical: {gate}")
    return payload


def _finalize_staged_evidence_payload(
    *,
    evidence: QualificationEvidence,
    final_evidence_root: Path,
) -> dict[str, JsonValue]:
    records = tuple(
        dict(record.model_dump(mode="json"))
        | {"artifact_path": (final_evidence_root / record.artifact_path.name).as_posix()}
        for record in evidence.records
    )
    evidence_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": evidence.schema_version,
                "manifest_sha256": evidence.manifest_sha256,
                "config_sha256": evidence.config_sha256,
                "records": sorted(records, key=lambda item: item["name"]),
            }
        )
    )
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": evidence.schema_version,
            "manifest_sha256": evidence.manifest_sha256,
            "config_sha256": evidence.config_sha256,
            "facts": {},
            "records": records,
            "evidence_sha256": evidence_sha256,
        }
    )


def _remove_stage_root(path: Path) -> None:
    if path != Path() and path.exists():
        shutil.rmtree(path)


def _reject_symlink_path(path: Path) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    for candidate in (*tuple(reversed(absolute.parents)), absolute):
        if candidate.is_symlink():
            raise CoordinatorError(reason="qualification path contains symlink")


def _artifact_tree_sha256(root: Path, *, exclude: Path | None = None) -> str:
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
    return sha256_bytes(canonical_json_bytes({"files": files}))
