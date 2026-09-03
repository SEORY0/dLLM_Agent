"""Task 5 fix4B qualification-evidence regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false, reportUnnecessaryComparison=false

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

import pytest
from tools.stable_diffcoder_payloads import MODEL_ID, MODEL_REVISION

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignManifest,
    CoordinatorError,
    build_campaign_manifest,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ProductionGateRunner,
    QualificationEvidence,
    QualificationFact,
    QualificationGateName,
    collect_read_only_cache_measurements,
    write_qualification_fact,
)
from dllm_agent.experiments.openexr_one_day.qualification_collectors import (
    collect_banks_measurements,
    collect_control_arm_equivalence_measurements,
    collect_smoke_measurements,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.test_openexr_one_day_fix1 import _config_json
from tests.test_openexr_one_day_qualification_collectors import (
    _tree_sha,
    _write_banks,
    _write_smoke,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.ledger import JsonValue

_GPU_UUIDS: Final = (
    "GPU-a8821ad6-2a99-42d9-696c-12372fd30262",
    "GPU-86be84f5-18e6-84e6-7873-0d1765821041",
    "GPU-0e44c3d9-ed42-3547-be29-c98766ebb467",
    "GPU-8a084353-dd96-f9af-bd16-5565fbce3dae",
    "GPU-a28e9f6a-9339-62b7-1b07-5b972d12e2e9",
    "GPU-3f84c261-7f83-e7b3-f131-b5696d8725c5",
    "GPU-23d8fb25-fac9-d241-60ce-1d98405ffa08",
    "GPU-4be60505-8935-bdd5-ba9a-116eb034ee56",
)


def test_production_gate_accepts_only_record_backed_canonical_evidence(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    evidence_root = data_root / "qualification"
    records = tuple(
        write_qualification_fact(
            manifest=manifest,
            evidence_root=evidence_root,
            gate_name=gate,
            measurement_kind=gate,
            measurements=_measurements(gate, manifest),
        )
        for gate in QUALIFICATION_GATE_NAMES
    )
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(reversed(records)),
    )
    evidence_path = evidence_root / "qualification.json"
    _ = evidence_path.write_bytes(canonical_json_bytes(evidence.model_dump(mode="json")))

    gates = ProductionGateRunner(evidence_path=evidence_path).evaluate(manifest)
    ordered = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=records,
    )

    assert gates.failing_gate() is None
    assert evidence.evidence_sha256 == ordered.evidence_sha256


def test_production_gate_rejects_facts_only_and_bad_record_cardinality(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    evidence_path = data_root / "qualification" / "qualification.json"
    evidence_path.parent.mkdir(parents=True)
    facts_payload: dict[str, JsonValue] = {
        "schema_version": "dllm-openexr-one-day-qualification/v1",
        "manifest_sha256": manifest.manifest_sha256,
        "config_sha256": config.config_sha256,
        "facts": dict.fromkeys(QUALIFICATION_GATE_NAMES, True),
        "records": [],
        "evidence_sha256": "0" * 64,
    }
    _ = evidence_path.write_bytes(canonical_json_bytes(facts_payload))

    with pytest.raises(CoordinatorError, match="record-backed"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(manifest)

    records = list(_records(data_root, config))
    duplicate_payload = _evidence_payload(config, [*records, records[0]])
    _ = evidence_path.write_bytes(canonical_json_bytes(duplicate_payload))
    with pytest.raises(CoordinatorError, match="exactly 12"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(manifest)


def test_production_gate_rejects_unsafe_artifact_paths_and_tampering(data_root: Path) -> None:
    config = _qualified_config(data_root)
    evidence_path = data_root / "qualification" / "qualification.json"
    records = list(_records(data_root, config))
    payload = _evidence_payload(config, records)
    _ = evidence_path.write_bytes(canonical_json_bytes(payload))
    first_path = QualificationEvidence.model_validate(payload).records[0].artifact_path
    _ = first_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(CoordinatorError, match="artifact hash"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(
            build_campaign_manifest(
                config,
                build_campaign_schedule(
                    main_start_monotonic_seconds=config.main_start_monotonic_seconds
                ),
            )
        )

    outside = data_root.parent / "outside-qualification.json"
    _ = outside.write_text("outside", encoding="utf-8")
    unsafe_records: list[dict[str, JsonValue]] = [dict(record) for record in records]
    unsafe = payload | {"records": unsafe_records}
    unsafe_record = unsafe_records[0]
    unsafe_record["artifact_path"] = str(outside)
    unsafe_record["artifact_sha256"] = sha256_bytes(outside.read_bytes())
    unsafe_record["artifact_size_bytes"] = outside.stat().st_size
    unsafe["evidence_sha256"] = QualificationEvidence.model_validate(
        unsafe
    ).expected_evidence_sha256()
    _ = evidence_path.write_bytes(canonical_json_bytes(unsafe))

    with pytest.raises(CoordinatorError, match="artifact path"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(
            build_campaign_manifest(
                config,
                build_campaign_schedule(
                    main_start_monotonic_seconds=config.main_start_monotonic_seconds
                ),
            )
        )


def test_production_gate_derives_passed_from_measurements_not_record_flag(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    evidence_path = data_root / "qualification" / "qualification.json"
    records = list(_records(data_root, config))
    bad = write_qualification_fact(
        manifest=manifest,
        evidence_root=data_root / "qualification",
        gate_name="smoke_passed",
        measurement_kind="smoke_passed",
        measurements={
            "search_smoke_passed": False,
            "coverage_smoke_passed": True,
            "proof_smoke_passed": True,
        },
        passed=True,
    )
    records[-1] = bad.model_dump(mode="json")
    payload = _evidence_payload(config, records)
    _ = evidence_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(CoordinatorError, match="smoke_passed"):
        _ = ProductionGateRunner(evidence_path=evidence_path).evaluate(manifest)


def test_qualification_hardening_rejects_attacker_supplied_thresholds(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    evidence_root = data_root / "qualification"

    disk = write_qualification_fact(
        manifest=manifest,
        evidence_root=evidence_root,
        gate_name="disk_space",
        measurement_kind="disk_space",
        measurements={"free_bytes": 1, "required_free_bytes": 0},
        passed=True,
    )
    load = write_qualification_fact(
        manifest=manifest,
        evidence_root=evidence_root,
        gate_name="reserved_core_load",
        measurement_kind="reserved_core_load",
        measurements={"max_load": 0.90, "threshold": 1.00},
        passed=True,
    )

    assert disk.passed
    assert load.passed
    with pytest.raises(CoordinatorError, match="disk_space"):
        _ = ProductionGateRunner(
            evidence_path=_write_with_replacements(config, evidence_root, {"disk_space": disk})
        ).evaluate(manifest)
    with pytest.raises(CoordinatorError, match="reserved_core_load"):
        _ = ProductionGateRunner(
            evidence_path=_write_with_replacements(
                config,
                evidence_root,
                {"reserved_core_load": load},
            )
        ).evaluate(manifest)


def test_qualified_resource_map_requires_complete_gpu_identity(data_root: Path) -> None:
    payload = _config_payload(data_root, include_gpu_records=False) | {
        "qualification_state": "qualified"
    }
    payload["config_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="GPU identity"):
        _ = CampaignConfig.model_validate(payload)

    draft_payload = _config_payload(data_root, include_gpu_records=False)
    draft_payload["config_sha256"] = config_payload_sha256(draft_payload)
    draft = CampaignConfig.model_validate(draft_payload)

    assert draft.qualification_state == "draft"


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix4b")
    return Path("/data/dllm-agent") / f"task5-fix4b-{sha256(str(root).encode()).hexdigest()[:12]}"


def _qualified_config(data_root: Path) -> CampaignConfig:
    payload = _config_payload(data_root, include_gpu_records=True) | {
        "qualification_state": "qualified"
    }
    payload["config_sha256"] = config_payload_sha256(payload)
    return CampaignConfig.model_validate(payload)


def _records(data_root: Path, config: CampaignConfig) -> tuple[dict[str, JsonValue], ...]:
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    return tuple(
        write_qualification_fact(
            manifest=manifest,
            evidence_root=data_root / "qualification",
            gate_name=gate,
            measurement_kind=gate,
            measurements=_measurements(gate, manifest),
        ).model_dump(mode="json")
        for gate in QUALIFICATION_GATE_NAMES
    )


def _evidence_payload(
    config: CampaignConfig, records: list[dict[str, JsonValue]] | tuple[dict[str, JsonValue], ...]
) -> dict[str, JsonValue]:
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(
            QualificationEvidence.record_adapter().validate_python(item) for item in records
        ),
    )
    return evidence.model_dump(mode="json")


def _write_with_replacements(
    config: CampaignConfig,
    evidence_root: Path,
    replacements: dict[QualificationGateName, JsonValue | QualificationFact],
) -> Path:
    records = list(_records(evidence_root.parent, config))
    for index, record in enumerate(records):
        name = QualificationEvidence.record_adapter().validate_python(record).name
        replacement = replacements.get(name)
        if replacement is not None:
            records[index] = (
                QualificationEvidence.record_adapter()
                .validate_python(replacement)
                .model_dump(mode="json")
            )
    payload = _evidence_payload(config, records)
    path = evidence_root / f"qualification-{next(iter(replacements))}.json"
    _ = path.write_bytes(canonical_json_bytes(payload))
    return path


def _config_payload(data_root: Path, *, include_gpu_records: bool) -> dict[str, JsonValue]:
    payload = _config_json(str(data_root))
    _bind_backing_artifacts(payload, data_root)
    resource_map = payload["resource_map"]
    if not isinstance(resource_map, dict):
        raise TypeError
    if include_gpu_records:
        resource_map["gpu_records"] = [
            {
                "index": index,
                "uuid": uuid,
                "numa_node": 0 if index < 4 else 1,
            }
            for index, uuid in enumerate(_GPU_UUIDS)
        ]
    else:
        _ = resource_map.pop("gpu_records", None)
    return payload


def _bind_backing_artifacts(payload: dict[str, JsonValue], data_root: Path) -> None:
    artifact_root = data_root / "fixtures"
    artifacts = {
        "target_harness": (artifact_root / "harness.cpp", b"harness"),
        "prompt": (artifact_root / "prompt.md", b"prompt"),
        "remote_code": (artifact_root / "remote.py", b"remote"),
        "search_binary": (artifact_root / "search", b"search"),
        "coverage_binary": (artifact_root / "coverage", b"coverage"),
        "coverage_build_manifest": (artifact_root / "coverage-manifest.json", b'{"ok":true}'),
        "c0_source": (artifact_root / "c0.hex", b"00ff"),
        "dictionary": (artifact_root / "openexr.dict", b'"EXR"'),
    }
    for path, content in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(content)
    c0_path, _ = artifacts["c0_source"]
    payload.update(
        {
            "prompt_path": artifacts["prompt"][0].as_posix(),
            "model_revision": MODEL_REVISION,
            "target_harness_sha256": sha256_bytes(artifacts["target_harness"][0].read_bytes()),
            "prompt_sha256": sha256_bytes(artifacts["prompt"][0].read_bytes()),
            "remote_code_sha256": sha256_bytes(artifacts["remote_code"][0].read_bytes()),
            "search_binary_sha256": sha256_bytes(artifacts["search_binary"][0].read_bytes()),
            "coverage_binary_sha256": sha256_bytes(artifacts["coverage_binary"][0].read_bytes()),
            "coverage_build_manifest_sha256": sha256_bytes(
                artifacts["coverage_build_manifest"][0].read_bytes()
            ),
            "c0_source_path": c0_path.as_posix(),
            "c0_sha256": sha256(bytes.fromhex(c0_path.read_text(encoding="ascii"))).hexdigest(),
            "dictionary_path": artifacts["dictionary"][0].as_posix(),
            "dictionary_sha256": sha256_bytes(artifacts["dictionary"][0].read_bytes()),
        }
    )


def _artifact_paths(artifact_root: Path) -> dict[str, Path]:
    root = artifact_root / "fixtures"
    return {
        "target_harness": root / "harness.cpp",
        "prompt": root / "prompt.md",
        "remote_code": root / "remote.py",
        "search_binary": root / "search",
        "coverage_binary": root / "coverage",
        "coverage_build_manifest": root / "coverage-manifest.json",
        "c0_source": root / "c0.hex",
        "dictionary": root / "openexr.dict",
    }


def _measurements(  # noqa: C901, PLR0911, PLR0912
    gate: QualificationGateName, manifest: CampaignManifest
) -> dict[str, JsonValue]:
    config = manifest.config
    resource_map = config.resource_map.model_dump(mode="json")
    match gate:
        case "hashes_match":
            artifacts = _artifact_paths(config.artifact_root)
            return {
                "target_commit": config.target_commit,
                "config_sha256": config.config_sha256,
                "target_harness_path": artifacts["target_harness"].as_posix(),
                "target_harness_sha256": config.target_harness_sha256,
                "prompt_path": config.prompt_path.as_posix(),
                "prompt_sha256": config.prompt_sha256,
                "remote_code_path": artifacts["remote_code"].as_posix(),
                "remote_code_sha256": config.remote_code_sha256,
                "search_image_digest": config.search_image_digest,
                "search_binary_path": artifacts["search_binary"].as_posix(),
                "search_binary_sha256": config.search_binary_sha256,
                "coverage_image_digest": config.coverage_image_digest,
                "coverage_binary_path": artifacts["coverage_binary"].as_posix(),
                "coverage_binary_sha256": config.coverage_binary_sha256,
                "coverage_build_manifest_path": artifacts["coverage_build_manifest"].as_posix(),
                "coverage_build_manifest_sha256": config.coverage_build_manifest_sha256,
                "c0_source_path": artifacts["c0_source"].as_posix(),
                "c0_sha256": config.c0_sha256,
                "dictionary_path": artifacts["dictionary"].as_posix(),
                "dictionary_sha256": config.dictionary_sha256,
            }
        case "banks_complete":
            _write_banks(config)
            return collect_banks_measurements(config)
        case "read_only_cache":
            config.model_cache_dir.mkdir(parents=True, exist_ok=True)
            config.model_cache_dir.chmod(0o755)
            (config.model_cache_dir / "modules").mkdir(exist_ok=True)
            _ = (config.model_cache_dir / "modules" / "modeling.py").write_text(
                "model",
                encoding="utf-8",
            )
            config.model_cache_dir.chmod(0o555)
            measurements = collect_read_only_cache_measurements(config.model_cache_dir)
            measurements.update(
                {
                    "model_id": MODEL_ID,
                    "model_revision": config.model_revision,
                    "remote_code_sha256": config.remote_code_sha256,
                    "configured_cache_path": config.model_cache_dir.as_posix(),
                    "cache_tree_sha256": _tree_sha(config.model_cache_dir),
                }
            )
            return measurements
        case "external_coverage_check":
            coverage_a = config.artifact_root / "coverage" / "coverage-a.json"
            coverage_b = config.artifact_root / "coverage" / "coverage-b.json"
            coverage_a.parent.mkdir(parents=True, exist_ok=True)
            _ = coverage_a.write_bytes(b'{"branches":123}')
            _ = coverage_b.write_bytes(b'{"branches":123}')
            return {
                "coverage_image_digest": config.coverage_image_digest,
                "coverage_binary_sha256": config.coverage_binary_sha256,
                "coverage_build_manifest_sha256": config.coverage_build_manifest_sha256,
                "target_only_branch_count": 123,
                "repeated_target_only_branch_count": 123,
                "coverage_artifact_path": coverage_a.as_posix(),
                "coverage_artifact_sha256": sha256_bytes(coverage_a.read_bytes()),
                "coverage_artifact_size_bytes": coverage_a.stat().st_size,
                "repeated_coverage_artifact_path": coverage_b.as_posix(),
                "repeated_coverage_artifact_sha256": sha256_bytes(coverage_b.read_bytes()),
                "repeated_coverage_artifact_size_bytes": coverage_b.stat().st_size,
            }
        case "control_arm_equivalence":
            return collect_control_arm_equivalence_measurements(config)
        case "resource_map_valid":
            bus_ids = {
                record.index: f"00000000:{record.index + 1:02x}:00.0"
                for record in config.resource_map.gpu_records
            }
            lscpu_stdout = "# CPU,CORE,SOCKET,NODE\n" + "\n".join(
                f"{cpu_id},{cpu_id % 20},{0 if cpu_id % 20 < 10 else 1},{node}"
                for cpu_id, node in sorted(config.resource_map.core_numa_nodes.items())
            )
            nvidia_stdout = "\n".join(
                f"{record.index}, {record.uuid}, {bus_ids[record.index]}"
                for record in config.resource_map.gpu_records
            )
            thread_siblings: list[JsonValue] = []
            sibling_by_cpu: dict[int, str] = {
                cpu_id: f"{cpu_id % 20},{cpu_id % 20 + 20}"
                for cpu_id in config.resource_map.core_numa_nodes
            }
            seen_cpus: set[int] = set()
            for assignment in (
                config.resource_map.model_cores
                + config.resource_map.search_cores
                + config.resource_map.proof_cores
                + config.resource_map.monitor_cores
            ):
                for cpu_id in (assignment.core_id, assignment.sibling_core_id):
                    if cpu_id in seen_cpus:
                        continue
                    seen_cpus.add(cpu_id)
                    raw = sibling_by_cpu[cpu_id]
                    path = f"/sys/devices/system/cpu/cpu{cpu_id}/topology/thread_siblings_list"
                    record: dict[str, JsonValue] = {
                        "cpu": cpu_id,
                        "path": path,
                        "raw": raw,
                        "raw_sha256": sha256_bytes(raw.encode()),
                    }
                    thread_siblings.append(record)
            gpu_sysfs: list[JsonValue] = []
            for gpu_record in config.resource_map.gpu_records:
                raw = str(gpu_record.numa_node)
                path = f"/sys/bus/pci/devices/0000:{gpu_record.index + 1:02x}:00.0/numa_node"
                gpu_sysfs.append(
                    {
                        "index": gpu_record.index,
                        "uuid": gpu_record.uuid,
                        "pci_bus_id": bus_ids[gpu_record.index],
                        "path": path,
                        "raw": raw,
                        "raw_sha256": sha256_bytes(raw.encode()),
                    }
                )
            return {
                "lscpu_argv": ["lscpu", "--parse=CPU,CORE,SOCKET,NODE"],
                "lscpu_exit_code": 0,
                "lscpu_stdout": lscpu_stdout,
                "lscpu_stderr": "",
                "lscpu_stdout_sha256": sha256_bytes(lscpu_stdout.encode()),
                "lscpu_stderr_sha256": sha256_bytes(b""),
                "nvidia_smi_argv": [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ],
                "nvidia_smi_exit_code": 0,
                "nvidia_smi_stdout": nvidia_stdout,
                "nvidia_smi_stderr": "",
                "nvidia_smi_stdout_sha256": sha256_bytes(nvidia_stdout.encode()),
                "nvidia_smi_stderr_sha256": sha256_bytes(b""),
                "thread_siblings": thread_siblings,
                "gpu_sysfs_numa_nodes": gpu_sysfs,
                "parsed_resource_map": resource_map,
            }
        case "disk_space":
            return {"free_bytes": 101 * 1024 * 1024 * 1024}
        case "vram":
            return {
                "pre_load_free_bytes": {str(index): 23 * 1024 * 1024 * 1024 for index in range(8)},
                "post_load_free_bytes": {str(index): 5 * 1024 * 1024 * 1024 for index in range(8)},
            }
        case "ram":
            return {"available_bytes": 13 * 1024 * 1024 * 1024}
        case "swap_stable":
            return {"swap_used_before_bytes": 1024, "swap_used_after_bytes": 1024}
        case "reserved_core_load":
            return {
                "max_load": 0.05,
                "interval_seconds": 1.0,
                "minimum_total_jiffy_delta": 1,
            }
        case "smoke_passed":
            path = _write_smoke(config.artifact_root / "smoke", config)
            return collect_smoke_measurements(config, path)
        case unreachable:
            assert_never(unreachable)
