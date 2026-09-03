"""Task 7 live qualification collector tests.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from subprocess import CompletedProcess
from typing import cast

import pytest
from pydantic import TypeAdapter
from tools.stable_diffcoder_payloads import MODEL_ID

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.candidate_bank import build_candidate_bank
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    BankGenerationCostEvidence,
    BankParentEvidence,
    CampaignConfig,
    CampaignManifest,
    CandidateBankEvidence,
    CoordinatorError,
    DuplicateParentEvidence,
    build_campaign_manifest,
    load_persisted_candidate_bank,
)
from dllm_agent.experiments.openexr_one_day.models import RepetitionIndex, Seed, SeedNamespace
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ProductionGateRunner,
    QualificationEvidence,
    QualificationGateName,
    QualificationSmokeArtifact,
    QualificationSmokeResult,
    build_qualification_smoke_plan,
    write_qualification_fact,
)
from dllm_agent.experiments.openexr_one_day.qualification_collectors import (
    CollectorInputs,
    CoverageQualificationRequest,
    CoverageQualificationResult,
    HashArtifactInputs,
    TopologyPaths,
    collect_all_measurements,
    collect_banks_measurements,
    collect_control_arm_equivalence_measurements,
    collect_hash_measurements,
    collect_memory_measurements,
    collect_reserved_core_load_measurements,
    collect_smoke_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke import (
    _cell_outcome,
    _result_from_cells,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    WRAPPER_NAME,
    QualificationSmokeBinding,
    SmokeCellResult,
    SmokePostSearchOutcome,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.test_openexr_one_day_candidate_bank import DeterministicInitialSampler


@dataclass(frozen=True, slots=True)
class FakeCommandRunner:
    digests: dict[str, str]
    vram_mib: int = 24 * 1024
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        self.calls.append(argv)
        if argv[:4] == ("docker", "image", "inspect", "--format"):
            ref = argv[5]
            return CompletedProcess(argv, 0, f"{self.digests[ref]}\n".encode(), b"")
        if argv[0] == "nvidia-smi" and "memory.free" in argv[1]:
            lines = "\n".join(f"{index}, {self.vram_mib}" for index in range(8))
            return CompletedProcess(argv, 0, lines.encode(), b"")
        if argv[0] == "lscpu":
            return CompletedProcess(argv, 0, _lscpu_stdout().encode(), b"")
        if argv[0] == "nvidia-smi":
            return CompletedProcess(argv, 0, _nvidia_topology_stdout().encode(), b"")
        raise AssertionError(argv)


@dataclass(slots=True)
class FakeProcReader:
    meminfo: list[str]
    proc_stat: list[str]
    monotonic_values: list[float] = field(default_factory=lambda: [1.0, 2.0])

    def read_text(self, path: Path) -> str:
        if path == Path("/proc/meminfo"):
            return self.meminfo.pop(0)
        if path == Path("/proc/stat"):
            return self.proc_stat.pop(0)
        if path.name == "thread_siblings_list":
            cpu = int(path.parts[-3].removeprefix("cpu"))
            return f"{cpu % 20},{cpu % 20 + 20}\n"
        if path.name == "numa_node":
            index = int(path.parent.name.split(":")[1], 16) - 1
            return f"{0 if index < 4 else 1}\n"
        raise AssertionError(path)

    def monotonic_seconds(self) -> float:
        return self.monotonic_values.pop(0)


@dataclass(frozen=True, slots=True)
class FakeCoverageRunner:
    first_artifact: Path
    repeated_artifact: Path
    count: int = 123
    requests: list[CoverageQualificationRequest] = field(default_factory=list)

    def run(self, request: CoverageQualificationRequest) -> CoverageQualificationResult:
        self.requests.append(request)
        path = self.repeated_artifact if request.repeated else self.first_artifact
        return CoverageQualificationResult(
            target_only_branch_count=self.count,
            artifact_path=path,
            artifact_sha256=sha256_bytes(path.read_bytes()),
        )


def test_hash_collector_measures_files_and_docker_image_ids(
    data_root: Path, tmp_path: Path
) -> None:
    """Given artifact files, when hashes collect, then values are derived from backing bytes."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    runner = FakeCommandRunner(
        {
            "search:qualified": config.search_image_digest,
            "coverage:qualified": config.coverage_image_digest,
        }
    )

    measurements = collect_hash_measurements(config, _hash_inputs(files), runner)

    assert measurements["prompt_path"] == files.prompt.as_posix()
    assert measurements["c0_sha256"] == sha256(b"\x00\xff").hexdigest()
    assert measurements["dictionary_sha256"] == sha256(b'"EXR"').hexdigest()
    assert (
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        "search:qualified",
    ) in runner.calls


def test_hash_collector_rejects_mismatched_backing(data_root: Path, tmp_path: Path) -> None:
    """Given a stale config hash, when hashes collect, then qualification fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    _ = files.prompt.write_bytes(b"tampered")
    runner = FakeCommandRunner(
        {
            "search:qualified": config.search_image_digest,
            "coverage:qualified": config.coverage_image_digest,
        }
    )

    with pytest.raises(CoordinatorError, match="prompt_sha256"):
        _ = collect_hash_measurements(config, _hash_inputs(files), runner)


def test_hash_collector_rejects_symlinked_c0_source(data_root: Path, tmp_path: Path) -> None:
    """Given a symlink C0 source, when hashes collect, then qualification fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    target = files.c0
    link = tmp_path / "c0-link.hex"
    link.symlink_to(target)
    runner = FakeCommandRunner(
        {
            "search:qualified": config.search_image_digest,
            "coverage:qualified": config.coverage_image_digest,
        }
    )
    inputs = replace(_hash_inputs(files), c0_source_path=link)

    with pytest.raises(CoordinatorError, match="symlink"):
        _ = collect_hash_measurements(config, inputs, runner)


def test_bank_collector_recomputes_counts_from_all_ten_artifacts(
    data_root: Path, tmp_path: Path
) -> None:
    """Given persisted candidate banks, when banks collect, then counts are measured."""
    config = _config(data_root, _artifact_files(tmp_path))
    _write_banks(config)

    measurements = collect_banks_measurements(config)
    repetitions = measurements["repetitions"]

    assert isinstance(repetitions, list)
    assert len(repetitions) == 10
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(repetitions)
    assert [_int(row["repetition"]) for row in rows] == list(range(10))
    assert {_int(row["raw_count"]) for row in rows} == {32}
    assert {_int(row["unique_count"]) for row in rows} == {32}
    assert {
        str(row["bank_path"]).startswith(
            (config.artifact_root / "qualification" / "banks").as_posix()
        )
        for row in rows
    } == {True}
    assert not (config.artifact_root / "banks").exists()


def test_bank_collector_rejects_tampered_candidate_bank(data_root: Path, tmp_path: Path) -> None:
    """Given a tampered bank file, when banks collect, then it rejects the artifact."""
    config = _config(data_root, _artifact_files(tmp_path))
    _write_banks(config)
    path = config.artifact_root / "qualification" / "banks" / "rep-03" / "candidate-bank.json"
    _ = path.write_bytes(path.read_bytes().replace(b'"repetition":3', b'"repetition":4'))

    with pytest.raises(CoordinatorError, match="candidate-bank"):
        _ = collect_banks_measurements(config)


def test_candidate_bank_loader_recomputes_persisted_parent_content_hashes(
    data_root: Path, tmp_path: Path
) -> None:
    """Given self-consistent forged parent bytes, when bank loads, then decoded content wins."""
    config = _config(data_root, _artifact_files(tmp_path))
    _write_banks(config)
    path = config.artifact_root / "qualification" / "banks" / "rep-00" / "candidate-bank.json"
    payload = _JSON.validate_python(
        CandidateBankEvidence.model_validate_json(path.read_bytes()).model_dump(mode="json")
    )
    parents = TypeAdapter(list[dict[str, JsonValue]]).validate_python(payload["raw_parents"])
    first = parents[0]
    first["content_hex"] = "00"
    first["materialized_sha256"] = sha256(b"\x00").hexdigest()
    payload["raw_parents"] = cast("JsonValue", parents)
    payload["raw_materialized_sha256"] = _hash_parts(
        tuple(str(parent["materialized_sha256"]) for parent in parents)
    )
    unsigned_payload = _JSON.validate_python(payload)
    removed = unsigned_payload.pop("evidence_sha256", None)
    assert removed is not None
    unsigned = canonical_json_bytes(unsigned_payload)
    payload["evidence_sha256"] = sha256_bytes(unsigned)
    _ = path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(CoordinatorError, match="candidate-bank"):
        _ = load_persisted_candidate_bank(path, 0)


def test_memory_and_reserved_core_collectors_use_fixed_proc_samples(
    data_root: Path, tmp_path: Path
) -> None:
    """Given fixed proc samples, when memory and load collect, then no sleeping is required."""
    config = _config(data_root, _artifact_files(tmp_path))
    reader = FakeProcReader(
        meminfo=[
            "MemAvailable: 14000000 kB\nSwapTotal: 1024 kB\nSwapFree: 512 kB\n",
            "MemAvailable: 15000000 kB\nSwapTotal: 1024 kB\nSwapFree: 512 kB\n",
        ],
        proc_stat=[_proc_stat(idle=1000), _proc_stat(idle=2000)],
    )

    memory = collect_memory_measurements(reader)
    load = collect_reserved_core_load_measurements(config, reader)

    assert memory["ram"]["available_bytes"] == 15_000_000 * 1024
    assert memory["swap_stable"]["swap_used_before_bytes"] == 512 * 1024
    assert load["max_load"] == 0.0


def test_control_arm_equivalence_constructs_task2_policy_payload(
    data_root: Path, tmp_path: Path
) -> None:
    """Given a config, when policy collect runs, then only trajectory visibility differs."""
    config = _config(data_root, _artifact_files(tmp_path))

    measurements = collect_control_arm_equivalence_measurements(config)

    assert measurements["treatment_arm"] == "selective-revision"
    assert measurements["control_arm"] == "final-only"
    assert measurements["treatment_config_sha256"] != measurements["control_config_sha256"]
    treatment = measurements["treatment_config"]
    control = measurements["control_config"]
    assert isinstance(treatment, dict)
    assert isinstance(control, dict)
    assert treatment["arm"] == "selective-revision"
    assert control["arm"] == "final-only"
    assert control["policy"] == measurements["control_policy"]
    assert treatment["policy"] != control["policy"]


def test_control_gate_rejects_selective_revision_control_payload(
    data_root: Path, tmp_path: Path
) -> None:
    """Given a mislabeled control config, when evidence evaluates, then it fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    control = measurements["control_arm_equivalence"]["control_config"]
    treatment = measurements["control_arm_equivalence"]["treatment_config"]
    if not isinstance(control, dict) or not isinstance(treatment, dict):
        raise TypeError
    control["arm"] = "selective-revision"
    control["policy"] = treatment["policy"]
    measurements["control_arm_equivalence"]["control_config_sha256"] = sha256_bytes(
        canonical_json_bytes(_JSON.validate_python(control))
    )
    path = _write_evidence(config, data_root / "qualification-bad-control", measurements)

    with pytest.raises(CoordinatorError, match="control_arm_equivalence"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_bank_gate_revalidates_canonical_bank_content(data_root: Path, tmp_path: Path) -> None:
    """Given forged bank digests, when evidence evaluates, then canonical bank load wins."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    banks = measurements["banks_complete"]
    repetitions = banks["repetitions"]
    if not isinstance(repetitions, list | tuple):
        raise TypeError
    first = repetitions[0]
    if not isinstance(first, dict):
        raise TypeError
    first["raw_token_states_sha256"] = "1" * 64
    path = _write_evidence(config, data_root / "qualification-forged-bank", measurements)

    with pytest.raises(CoordinatorError, match="banks_complete"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_hash_gate_rejects_symlinked_c0_source_path(data_root: Path, tmp_path: Path) -> None:
    """Given a symlink C0 backing path, when evidence evaluates, then it fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    target = files.c0
    link = tmp_path / "gate-c0-link.hex"
    link.symlink_to(target)
    measurements["hashes_match"]["c0_source_path"] = link.as_posix()
    path = _write_evidence(config, data_root / "qualification-symlink-c0", measurements)

    with pytest.raises(CoordinatorError, match="hashes_match"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_hash_gate_requires_measured_c0_source_path_matches_config(
    data_root: Path, tmp_path: Path
) -> None:
    """Given an alternate same-content C0 path, when evidence evaluates, then it fails closed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    alternate = tmp_path / "alternate-c0.hex"
    _ = alternate.write_bytes(files.c0.read_bytes())
    measurements["hashes_match"]["c0_source_path"] = alternate.as_posix()
    path = _write_evidence(config, data_root / "qualification-alt-c0", measurements)

    with pytest.raises(CoordinatorError, match="hashes_match"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_smoke_collector_requires_canonical_plan_hash(data_root: Path, tmp_path: Path) -> None:
    """Given an ad hoc smoke hash, when smoke collects, then the artifact is rejected."""
    config = _config(data_root, _artifact_files(tmp_path))
    path = _write_smoke(config.artifact_root / "smoke", config)
    artifact = QualificationSmokeArtifact.model_validate_json(path.read_bytes())
    legacy = artifact.model_copy(
        update={
            "plan_sha256": sha256_bytes(
                canonical_json_bytes(
                    {"config_sha256": config.config_sha256, "smoke_root": str(path.parent)}
                )
            )
        }
    )
    _ = path.write_bytes(canonical_json_bytes(legacy.model_dump(mode="json")))

    with pytest.raises(CoordinatorError, match="qualification smoke artifact hash mismatch"):
        _ = collect_smoke_measurements(config, path)


def test_smoke_collector_rejects_self_consistent_missing_cell_result(
    data_root: Path, tmp_path: Path
) -> None:
    """Given a wrapper missing one planned cell, when smoke collects, then it fails."""
    config = _config(data_root, _artifact_files(tmp_path))
    path = _write_smoke(config.artifact_root / "smoke-missing-cell", config)
    artifact = QualificationSmokeArtifact.model_validate_json(path.read_bytes())
    result = artifact.result.model_copy(update={"cells": artifact.result.cells[:-1]})
    result_payload = result.model_dump(mode="json", exclude={"result_sha256"})
    result = result.model_copy(
        update={
            "cell_result_set_sha256": sha256_bytes(
                canonical_json_bytes([cell.model_dump(mode="json") for cell in result.cells])
            ),
            "result_sha256": sha256_bytes(canonical_json_bytes(result_payload)),
        }
    )
    forged = artifact.model_copy(
        update={
            "result": result,
            "result_sha256": sha256_bytes(canonical_json_bytes(result.model_dump(mode="json"))),
        }
    )
    _ = path.write_bytes(canonical_json_bytes(forged.model_dump(mode="json")))

    with pytest.raises(CoordinatorError, match="exactly 10 smoke cell outcomes"):
        _ = collect_smoke_measurements(config, path)


def test_production_gate_rejects_shallow_bank_coverage_and_smoke_payloads(
    data_root: Path, tmp_path: Path
) -> None:
    """Given shallow payloads, when evidence evaluates, then backing paths are required."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    measurements["banks_complete"] = {
        "repetitions": [
            {
                "repetition": index,
                "primary_raw_count": 32,
                "supplement_raw_count": 0,
                "total_raw_count": 32,
                "raw_count": 32,
                "primary_unique_count": 32,
                "final_unique_count": 32,
                "unique_count": 32,
                "selected_count": 16,
                "supplemented": False,
            }
            for index in range(10)
        ]
    }
    path = _write_evidence(config, data_root / "qualification-shallow", measurements)

    with pytest.raises(CoordinatorError, match="banks_complete"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_production_gate_rejects_tampered_coverage_and_smoke_backing(
    data_root: Path, tmp_path: Path
) -> None:
    """Given changed backing files, when evidence evaluates, then hashes are reopened."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    path = _write_evidence(config, data_root / "qualification-tamper", measurements)
    coverage_path = Path(_str(measurements["external_coverage_check"], "coverage_artifact_path"))
    _ = coverage_path.write_bytes(b"tampered")

    with pytest.raises(CoordinatorError, match="external_coverage_check"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_production_gate_rejects_wrong_model_and_cache_tree(
    data_root: Path, tmp_path: Path
) -> None:
    """Given forged model cache metadata, when evidence evaluates, then tree hash is recomputed."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    measurements = _passing_measurements(config, files, tmp_path)
    cache = measurements["read_only_cache"]
    cache["model_id"] = "microsoft/CodePori-7B"
    path = _write_evidence(config, data_root / "qualification-cache", measurements)

    with pytest.raises(CoordinatorError, match="read_only_cache"):
        _ = ProductionGateRunner(path).evaluate(_manifest(config))


def test_reserved_core_load_rejects_zero_jiffy_delta(data_root: Path, tmp_path: Path) -> None:
    """Given a zero-delta proc interval, when load collects, then it fails closed."""
    config = _config(data_root, _artifact_files(tmp_path))
    reader = FakeProcReader(
        meminfo=[],
        proc_stat=[_proc_stat(idle=1000), _proc_stat(idle=1000)],
        monotonic_values=[1.0, 2.0],
    )

    with pytest.raises(CoordinatorError, match="reserved-core load interval"):
        _ = collect_reserved_core_load_measurements(config, reader)


def test_collect_all_rejects_symlink_measurement_root(data_root: Path, tmp_path: Path) -> None:
    """Given a symlink output root, when collecting all, then no partial set is published."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    target = data_root / "real-measurements"
    target.mkdir(parents=True)
    symlink = data_root / "measurements-link"
    symlink.symlink_to(target, target_is_directory=True)
    inputs = replace(_collector_inputs(config, files, tmp_path), measurements_root=symlink)

    with pytest.raises(CoordinatorError, match="symlink"):
        _ = collect_all_measurements(inputs)


def test_collect_all_preflights_publish_roots_before_measurement_publish(
    data_root: Path, tmp_path: Path
) -> None:
    """Given an unusable evidence root, when collecting all, then no side is published."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    inputs = _collector_inputs(config, files, tmp_path)
    blocking_path = inputs.evidence_root / "qualification.json"
    blocking_path.mkdir(parents=True)

    with pytest.raises(CoordinatorError, match="qualification measurements mismatch"):
        _ = collect_all_measurements(inputs)

    assert not inputs.measurements_root.exists()
    assert blocking_path.is_dir()


def test_collect_all_writes_canonical_measurements_and_evidence(
    data_root: Path, tmp_path: Path
) -> None:
    """Given all backing artifacts, when collection runs, then production gates accept evidence."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    inputs = _collector_inputs(config, files, tmp_path)

    evidence_path = collect_all_measurements(inputs)
    gates = ProductionGateRunner(evidence_path).evaluate(
        build_campaign_manifest(
            config,
            build_campaign_schedule(
                main_start_monotonic_seconds=config.main_start_monotonic_seconds
            ),
        )
    )

    assert gates.failing_gate() is None
    assert (data_root / "measurements" / "hashes_match.json").read_bytes() == canonical_json_bytes(
        _JSON.validate_json((data_root / "measurements" / "hashes_match.json").read_bytes())
    )


def test_collect_all_rerun_revalidates_published_package_without_volatile_recollect(
    data_root: Path, tmp_path: Path
) -> None:
    """Given published evidence, when volatile inputs change, then rerun accepts the package."""
    files = _artifact_files(tmp_path)
    config = _config(data_root, files)
    inputs = _collector_inputs(config, files, tmp_path)
    evidence_path = collect_all_measurements(inputs)
    volatile_inputs = replace(inputs, disk_path=tmp_path)

    rerun_path = collect_all_measurements(volatile_inputs)

    assert rerun_path == evidence_path


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("qualification-collectors")
    digest = sha256(str(root).encode()).hexdigest()[:12]
    return Path("/data/dllm-agent") / f"task7-collectors-{digest}"


@dataclass(frozen=True, slots=True)
class ArtifactFiles:
    harness: Path
    prompt: Path
    remote: Path
    search: Path
    coverage: Path
    coverage_manifest: Path
    c0: Path
    dictionary: Path


_JSON = TypeAdapter(dict[str, JsonValue])


def _artifact_files(root: Path) -> ArtifactFiles:
    files = ArtifactFiles(
        harness=root / "harness.cpp",
        prompt=root / "prompt.md",
        remote=root / "remote.py",
        search=root / "search",
        coverage=root / "coverage",
        coverage_manifest=root / "coverage-manifest.json",
        c0=root / "c0.hex",
        dictionary=root / "openexr.dict",
    )
    for path, content in (
        (files.harness, b"harness"),
        (files.prompt, b"prompt"),
        (files.remote, b"remote"),
        (files.search, b"search"),
        (files.coverage, b"coverage"),
        (files.coverage_manifest, b'{"coverage":true}'),
        (files.c0, b"00ff"),
        (files.dictionary, b'"EXR"'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(content)
    return files


def _hash_inputs(files: ArtifactFiles) -> HashArtifactInputs:
    return HashArtifactInputs(
        target_harness_path=files.harness,
        remote_code_path=files.remote,
        search_binary_path=files.search,
        coverage_binary_path=files.coverage,
        coverage_build_manifest_path=files.coverage_manifest,
        c0_source_path=files.c0,
        dictionary_path=files.dictionary,
        search_image_ref="search:qualified",
        coverage_image_ref="coverage:qualified",
    )


def _config(data_root: Path, files: ArtifactFiles) -> CampaignConfig:
    payload = _config_json(data_root)
    payload.update(
        {
            "qualification_state": "qualified",
            "model_cache_dir": (data_root / "cache").as_posix(),
            "prompt_path": files.prompt.as_posix(),
            "target_harness_sha256": sha256_bytes(files.harness.read_bytes()),
            "prompt_sha256": sha256_bytes(files.prompt.read_bytes()),
            "remote_code_sha256": sha256_bytes(files.remote.read_bytes()),
            "search_image_digest": "sha256:" + sha256(b"search-image").hexdigest(),
            "search_binary_sha256": sha256_bytes(files.search.read_bytes()),
            "coverage_image_digest": "sha256:" + sha256(b"coverage-image").hexdigest(),
            "coverage_binary_sha256": sha256_bytes(files.coverage.read_bytes()),
            "coverage_build_manifest_sha256": sha256_bytes(files.coverage_manifest.read_bytes()),
            "c0_source_path": files.c0.as_posix(),
            "c0_sha256": sha256(bytes.fromhex(files.c0.read_text(encoding="ascii"))).hexdigest(),
            "dictionary_path": files.dictionary.as_posix(),
            "dictionary_sha256": sha256_bytes(files.dictionary.read_bytes()),
        }
    )
    payload["config_sha256"] = config_payload_sha256(payload)
    return CampaignConfig.model_validate(payload)


def _write_banks(config: CampaignConfig) -> None:
    for repetition in range(10):
        bank = build_candidate_bank(
            DeterministicInitialSampler(primary_unique=32),
            SeedNamespace(
                repetition=RepetitionIndex(repetition),
                start_seed=Seed(1_000_000 + (repetition * 1_000)),
                stop_seed_exclusive=Seed(1_001_000 + (repetition * 1_000)),
            ),
            steps=config.candidate_steps,
        )
        evidence = CandidateBankEvidence(
            repetition=repetition,
            requested_seed_base=bank.requested_seed_base,
            raw_parents=tuple(
                BankParentEvidence(
                    request=parent.request,
                    result=parent.result,
                    content_hex=parent.content.hex(),
                    materialized_sha256=parent.materialized_sha256,
                    raw_parent_sha256=parent.raw_parent_sha256,
                )
                for parent in bank.raw_parents
            ),
            selected_raw_parent_sha256=tuple(
                parent.raw_parent_sha256 for parent in bank.selected_parents
            ),
            duplicates=tuple(
                DuplicateParentEvidence(
                    materialized_sha256=item.materialized_sha256,
                    duplicate_raw_parent_sha256=item.duplicate_raw_parent_sha256,
                    kept_raw_parent_sha256=item.kept_raw_parent_sha256,
                )
                for item in bank.duplicates
            ),
            primary_raw_count=bank.primary_raw_count,
            supplement_raw_count=bank.supplement_raw_count,
            unique_parent_count=bank.unique_parent_count,
            raw_token_states_sha256=bank.raw_token_states_sha256,
            raw_trajectories_sha256=bank.raw_trajectories_sha256,
            raw_materialized_sha256=bank.raw_materialized_sha256,
            duplicate_evidence_sha256=bank.duplicate_evidence_sha256,
            selected_bank_sha256=bank.selected_bank_sha256,
            physical_cost=BankGenerationCostEvidence(
                initial_requests=bank.physical_cost.initial_requests,
                nfe=bank.physical_cost.nfe,
                forward_calls=bank.physical_cost.forward_calls,
                gpu_ms=bank.physical_cost.gpu_ms,
            ),
            measured_elapsed_seconds=1.0,
            physical_duration_seconds=1.0,
            bank_ready_elapsed_seconds=1.0,
            producer_gpu_index=repetition % 8,
        )
        unsigned = canonical_json_bytes(
            evidence.model_dump(mode="json", exclude={"evidence_sha256"})
        )
        evidence = evidence.model_copy(update={"evidence_sha256": sha256_bytes(unsigned)})
        path = (
            config.artifact_root
            / "qualification"
            / "banks"
            / f"rep-{repetition:02d}"
            / "candidate-bank.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(canonical_json_bytes(evidence.model_dump(mode="json")))


def _make_cache(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "modules").mkdir(exist_ok=True)
    _ = (path / "modules" / "modeling.py").write_text("model", encoding="utf-8")
    path.chmod(0o555)


def _write_smoke(root: Path, config: CampaignConfig) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    plan = build_qualification_smoke_plan(config=config, smoke_root=root)
    tree_file = root / "cells" / "rep-00" / "selective-revision" / "cell.json"
    tree_file.parent.mkdir(parents=True, exist_ok=True)
    _ = tree_file.write_text("ok", encoding="utf-8")
    outcomes = tuple(
        _cell_outcome(
            cell,
            QualificationSmokeBinding(
                cell_id=cell.cell_id,
                bank_sha256=sha256_bytes(f"bank:{cell.repetition}".encode()),
                model_prep_sha256=None
                if cell.arm == "libfuzzer-only"
                else sha256_bytes(f"model:{cell.cell_id}".encode()),
            ),
            SmokeCellResult(
                cell_id=cell.cell_id,
                status="completed",
                attempt_count=1,
                deadline_monotonic_seconds=cell.deadline_monotonic_seconds,
                result_sha256=sha256_bytes(f"cell:{cell.cell_id}".encode()),
            ),
            SmokePostSearchOutcome(
                cell_id=cell.cell_id,
                proof_status="available",
                deep_parse_status="available",
                coverage_envelope_sha256=sha256_bytes(f"coverage:{cell.cell_id}".encode()),
                proof_envelope_sha256=sha256_bytes(f"proof:{cell.cell_id}".encode()),
                deep_parse_envelope_sha256=sha256_bytes(f"deep:{cell.cell_id}".encode()),
                proof_result_sha256=sha256_bytes(f"proof-result:{cell.cell_id}".encode()),
                deep_parse_result_sha256=sha256_bytes(f"deep-result:{cell.cell_id}".encode()),
            ),
        )
        for cell in plan.cells
    )
    result = QualificationSmokeResult.model_validate(
        _result_from_cells(plan, outcomes, (0, 1)).model_dump(mode="json")
    )
    artifact = QualificationSmokeArtifact(
        plan_sha256=sha256_bytes(canonical_json_bytes(plan.model_dump(mode="json"))),
        result=result,
        result_sha256=sha256_bytes(canonical_json_bytes(result.model_dump(mode="json"))),
        artifact_tree_sha256=_tree_sha(root, exclude=root / WRAPPER_NAME),
    )
    path = root / WRAPPER_NAME
    _ = path.write_bytes(canonical_json_bytes(artifact.model_dump(mode="json")))
    return path


def _lscpu_stdout() -> str:
    return "# CPU,CORE,SOCKET,NODE\n" + "\n".join(
        f"{cpu},{cpu % 20},{0 if cpu % 20 < 10 else 1},{_numa_node(cpu)}" for cpu in range(40)
    )


def _nvidia_topology_stdout() -> str:
    return "\n".join(
        f"{index}, GPU-00000000-0000-0000-0000-{index:012d}, 00000000:{index + 1:02x}:00.0"
        for index in range(8)
    )


def _proc_stat(*, idle: int) -> str:
    lines = [f"cpu{core} 0 0 0 {idle} 0 0 0 0 0 0" for core in range(40)]
    return "\n".join(lines)


def _numa_node(cpu: int) -> int:
    return 0 if cpu in {*range(10), *range(20, 30)} else 1


def _write_topology_roots(root: Path) -> TopologyPaths:
    cpu_root = root / "sys" / "devices" / "system" / "cpu"
    pci_root = root / "sys" / "bus" / "pci" / "devices"
    for cpu in range(40):
        path = cpu_root / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True, exist_ok=True)
        _ = (path / "thread_siblings_list").write_text(
            f"{cpu % 20},{cpu % 20 + 20}\n",
            encoding="utf-8",
        )
    for index in range(8):
        path = pci_root / f"0000:{index + 1:02x}:00.0"
        path.mkdir(parents=True, exist_ok=True)
        _ = (path / "numa_node").write_text(
            f"{0 if index < 4 else 1}\n",
            encoding="utf-8",
        )
    return TopologyPaths(thread_sibling_root=cpu_root, pci_device_root=pci_root)


def _collector_inputs(
    config: CampaignConfig, files: ArtifactFiles, tmp_path: Path
) -> CollectorInputs:
    _write_banks(config)
    _make_cache(config.model_cache_dir)
    topology = _write_topology_roots(tmp_path)
    smoke = _write_smoke(config.artifact_root / "smoke", config)
    coverage_a = config.artifact_root / "coverage" / "coverage-a.json"
    coverage_b = config.artifact_root / "coverage" / "coverage-b.json"
    coverage_a.parent.mkdir(parents=True, exist_ok=True)
    _ = coverage_a.write_bytes(b'{"branches":123}')
    _ = coverage_b.write_bytes(b'{"branches":123}')
    return CollectorInputs(
        config=config,
        measurements_root=config.artifact_root / "measurements",
        evidence_root=config.artifact_root / "qualification",
        hash_artifacts=_hash_inputs(files),
        model_id=MODEL_ID,
        coverage_runner=FakeCoverageRunner(coverage_a, coverage_b),
        smoke_artifact_path=smoke,
        disk_path=config.artifact_root,
        topology=topology,
        command_runner=FakeCommandRunner(
            {
                "search:qualified": config.search_image_digest,
                "coverage:qualified": config.coverage_image_digest,
            }
        ),
        proc_reader=FakeProcReader(
            meminfo=[
                "MemAvailable: 15000000 kB\nSwapTotal: 1024 kB\nSwapFree: 1024 kB\n",
                "MemAvailable: 15000000 kB\nSwapTotal: 1024 kB\nSwapFree: 1024 kB\n",
            ],
            proc_stat=[_proc_stat(idle=1000), _proc_stat(idle=2000)],
            monotonic_values=[1.0, 2.0],
        ),
    )


def _passing_measurements(
    config: CampaignConfig, files: ArtifactFiles, tmp_path: Path
) -> dict[QualificationGateName, dict[str, JsonValue]]:
    inputs = _collector_inputs(config, files, tmp_path)
    _ = collect_all_measurements(inputs)
    return {
        gate: _JSON.validate_json((inputs.measurements_root / f"{gate}.json").read_bytes())
        for gate in QUALIFICATION_GATE_NAMES
    }


def _hash_parts(parts: tuple[str | int, ...]) -> str:
    return sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _write_evidence(
    config: CampaignConfig,
    evidence_root: Path,
    measurements: dict[QualificationGateName, dict[str, JsonValue]],
) -> Path:
    manifest = _manifest(config)
    records = tuple(
        write_qualification_fact(
            manifest=manifest,
            evidence_root=evidence_root,
            gate_name=gate,
            measurement_kind=gate,
            measurements=measurements[gate],
            passed=True,
        )
        for gate in QUALIFICATION_GATE_NAMES
    )
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=records,
    )
    path = evidence_root / "qualification.json"
    _ = path.write_bytes(canonical_json_bytes(evidence.model_dump(mode="json")))
    return path


def _manifest(config: CampaignConfig) -> CampaignManifest:
    return build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )


def _str(measurements: dict[str, JsonValue], key: str) -> str:
    value = measurements[key]
    if isinstance(value, str):
        return value
    raise AssertionError(value)


def _tree_sha(root: Path, *, exclude: Path | None = None) -> str:
    resolved = root.resolve(strict=True)
    resolved_exclude = None if exclude is None else exclude.resolve(strict=False)
    files: list[dict[str, JsonValue]] = []
    for path in sorted(resolved.rglob("*")):
        if not path.is_file() or path == resolved_exclude:
            continue
        content = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "sha256": sha256_bytes(content),
                "size_bytes": len(content),
            }
        )
    return sha256_bytes(canonical_json_bytes({"files": files}))


def _config_json(root: Path) -> dict[str, JsonValue]:
    return {
        "schema_version": "dllm-openexr-one-day-config/v1",
        "config_id": "task7-collectors-test",
        "qualification_state": "draft",
        "artifact_root": root.as_posix(),
        "model_cache_dir": (root / "cache").as_posix(),
        "main_start_monotonic_seconds": 70_000.0,
        "target_commit": "777d231a179de1711d2a942810d9559216ab3f4c",
        "target_harness_sha256": "1" * 64,
        "prompt_sha256": "2" * 64,
        "model_revision": "10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        "remote_code_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "search_image_digest": "sha256:" + "5" * 64,
        "search_binary_sha256": "6" * 64,
        "sanitizers": ["address", "undefined"],
        "coverage_image_digest": "sha256:" + "7" * 64,
        "coverage_binary_sha256": "8" * 64,
        "coverage_build_manifest_sha256": "9" * 64,
        "c0_source_path": "fixtures/openexr/c0-validity-preserving-zero.hex",
        "c0_sha256": "ae9d361e47cc503dc91b827f4956fdf8ca16ffb3804ecdfcf83dd627f8a64f96",
        "dictionary_sha256": "9419f2003be8f41d286360ea3dec3f8fa967244111eecec0fd292db0ae6fed93",
        "max_input_length": 1_048_576,
        "target_timeout_seconds": 10,
        "rss_limit_mb": 2_048,
        "candidate_steps": 8,
        "proof_duration_seconds": 30,
        "resource_map": _resource_map_json(),
    }


def _resource_map_json() -> dict[str, JsonValue]:
    return {
        "model_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (0, 1, 2, 3, 10, 11, 12, 13)
        ],
        "search_cores": [
            {"core_id": index, "sibling_core_id": index + 20}
            for index in (4, 5, 6, 7, 8, 9, 14, 15, 16, 17)
        ],
        "proof_cores": [{"core_id": 18, "sibling_core_id": 38}],
        "monitor_cores": [{"core_id": 19, "sibling_core_id": 39}],
        "gpu_numa_nodes": [0, 0, 0, 0, 1, 1, 1, 1],
        "gpu_records": [
            {
                "index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
                "numa_node": 0 if index < 4 else 1,
            }
            for index in range(8)
        ],
        "core_numa_nodes": {str(index): _numa_node(index) for index in range(40)},
    }


def _int(value: JsonValue) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise AssertionError(value)
