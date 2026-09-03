"""Fix5E qualification live-host and forged-evidence regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.config_digest import (
    config_payload_sha256,
    load_verified_campaign_config,
)
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignManifest,
    CoordinatorError,
    ResourceMap,
    build_campaign_manifest,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ProductionGateRunner,
    QualificationEvidence,
    QualificationFact,
    QualificationGateName,
    SystemQualificationCommandRunner,
    collect_read_only_cache_measurements,
    collect_resource_topology_measurements,
    write_qualification_fact,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix4_qualification import _config_payload, _measurements

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.ledger import JsonValue


def test_resource_gate_accepts_normal_smt_lscpu_and_sysfs_gpu_numa(data_root: Path) -> None:
    config = _qualified_config(data_root)
    path = _write_with_replacement(config, data_root, {})

    gates = ProductionGateRunner(path).evaluate(_manifest(config))

    assert gates.resource_map_valid is True


def test_resource_gate_rejects_forged_collector_argv(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("resource_map_valid", manifest)
    forged["lscpu_argv"] = ["lscpu", "-e=CPU,CORE,SOCKET,NODE", "-p"]
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="resource_map_valid",
        measurement_kind="resource_map_valid",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"resource_map_valid": replacement})

    with pytest.raises(CoordinatorError, match="resource_map_valid"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_integer_measurements_reject_bool_values(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("external_coverage_check", manifest)
    forged["target_only_branch_count"] = True
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="external_coverage_check",
        measurement_kind="external_coverage_check",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"external_coverage_check": replacement})

    with pytest.raises(CoordinatorError, match="target_only_branch_count"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_smoke_pair_id_rejects_bool_value(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("smoke_passed", manifest)
    pairs = forged["pairs"]
    if not isinstance(pairs, list):
        raise TypeError
    first = pairs[0]
    if not isinstance(first, dict):
        raise TypeError
    first["pair_id"] = True
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="smoke_passed",
        measurement_kind="smoke_passed",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"smoke_passed": replacement})

    with pytest.raises(CoordinatorError, match="smoke_passed"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_bank_gate_rejects_impossible_unique_counts(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("banks_complete", manifest)
    repetitions = forged["repetitions"]
    if not isinstance(repetitions, list):
        raise TypeError
    first = repetitions[0]
    if not isinstance(first, dict):
        raise TypeError
    first["final_unique_count"] = 49
    first["unique_count"] = 49
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="banks_complete",
        measurement_kind="banks_complete",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"banks_complete": replacement})

    with pytest.raises(CoordinatorError, match="banks_complete"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_control_equivalence_treats_missing_and_null_as_different(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("control_arm_equivalence", manifest)
    treatment = forged["treatment_config"]
    control = forged["control_config"]
    if not isinstance(treatment, dict) or not isinstance(control, dict):
        raise TypeError
    treatment["optional"] = None
    forged["treatment_config_sha256"] = _hash_json(treatment)
    forged["control_config_sha256"] = _hash_json(control)
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="control_arm_equivalence",
        measurement_kind="control_arm_equivalence",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"control_arm_equivalence": replacement})

    with pytest.raises(CoordinatorError, match="control_arm_equivalence"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_qualification_wrapper_and_artifact_symlink_ancestors_reject(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    path = _write_with_replacement(config, data_root, {})
    wrapper_link = data_root / "qualification-link.json"
    wrapper_link.symlink_to(path)

    with pytest.raises(CoordinatorError, match="symlink"):
        _ = ProductionGateRunner(wrapper_link).evaluate(_manifest(config))

    evidence = QualificationEvidence.model_validate_json(path.read_bytes())
    artifact_link_root = data_root / "artifact-link"
    artifact_link_root.symlink_to(data_root / "qualification", target_is_directory=True)
    records: list[QualificationFact] = []
    for record in evidence.records:
        replacement = record
        if record.name == "hashes_match":
            replacement = record.model_copy(
                update={"artifact_path": artifact_link_root / record.artifact_path.name}
            )
        records.append(replacement)
    symlinked = QualificationEvidence.for_manifest(
        manifest_sha256=evidence.manifest_sha256,
        config_sha256=evidence.config_sha256,
        records=tuple(records),
    )
    symlinked_path = data_root / "qualification" / "qualification-symlink-artifact.json"
    _ = symlinked_path.write_bytes(canonical_json_bytes(symlinked.model_dump(mode="json")))

    with pytest.raises(CoordinatorError, match="symlink"):
        _ = ProductionGateRunner(symlinked_path).evaluate(_manifest(config))


def test_cache_gate_rejects_wrong_path_and_successful_probe(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    wrong_cache = data_root / "wrong-cache"
    wrong_cache.mkdir(parents=True)
    wrong = collect_read_only_cache_measurements(wrong_cache)
    wrong["read_only"] = True
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="read_only_cache",
        measurement_kind="read_only_cache",
        measurements=wrong,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"read_only_cache": replacement})

    with pytest.raises(CoordinatorError, match="read_only_cache"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_reserved_core_load_rejects_negative_value(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    forged = _measurements("reserved_core_load", manifest)
    forged["max_load"] = -1
    replacement = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="reserved_core_load",
        measurement_kind="reserved_core_load",
        measurements=forged,
        passed=True,
    )
    path = _write_with_replacement(config, data_root, {"reserved_core_load": replacement})

    with pytest.raises(CoordinatorError, match="reserved_core_load"):
        _ = ProductionGateRunner(path).evaluate(manifest)


def test_resource_map_rejects_bool_integer_fields() -> None:
    payload = _config_json("/data/dllm-agent/fix5e-bool-resource")["resource_map"]
    if not isinstance(payload, dict):
        raise TypeError
    payload["core_numa_nodes"] = {"0": True}

    with pytest.raises(ValidationError):
        _ = ResourceMap.model_validate(payload)


def test_live_host_collector_matches_checked_in_config(tmp_path: Path) -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi unavailable")
    config = load_verified_campaign_config(Path("configs/openexr-v3.4.14-one-day.json"))
    manifest = _manifest(config)
    measurements = collect_resource_topology_measurements(SystemQualificationCommandRunner())
    fact = write_qualification_fact(
        manifest=manifest,
        evidence_root=tmp_path / "qualification",
        gate_name="resource_map_valid",
        measurement_kind="resource_map_valid",
        measurements=measurements,
    )

    assert fact.passed is True
    assert isinstance(measurements["lscpu_stdout_sha256"], str)
    assert isinstance(measurements["nvidia_smi_stdout_sha256"], str)


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix5e")
    return Path("/data/dllm-agent") / f"task5-fix5e-{sha256(str(root).encode()).hexdigest()[:12]}"


def _qualified_config(data_root: Path) -> CampaignConfig:
    payload = _config_payload(data_root, include_gpu_records=True) | {
        "qualification_state": "qualified"
    }
    payload["config_sha256"] = config_payload_sha256(payload)
    return CampaignConfig.model_validate(payload)


def _manifest(config: CampaignConfig) -> CampaignManifest:
    return build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )


def _write_with_replacement(
    config: CampaignConfig,
    data_root: Path,
    replacements: dict[QualificationGateName, QualificationFact],
) -> Path:
    manifest = _manifest(config)
    evidence_root = data_root / "qualification"
    records: list[QualificationFact] = []
    for gate in QUALIFICATION_GATE_NAMES:
        replacement = replacements.get(gate)
        records.append(
            replacement
            if replacement is not None
            else write_qualification_fact(
                manifest=manifest,
                evidence_root=evidence_root,
                gate_name=gate,
                measurement_kind=gate,
                measurements=_measurements(gate, manifest),
            )
        )
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(records),
    )
    path = evidence_root / "qualification.json"
    _ = path.write_bytes(canonical_json_bytes(evidence.model_dump(mode="json")))
    return path


def _hash_json(payload: dict[str, JsonValue]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))
