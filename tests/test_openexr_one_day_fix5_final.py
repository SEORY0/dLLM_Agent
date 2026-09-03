"""Task 5 Fix5C final evidence and runtime regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Self, cast

import pytest
from tools.run_openexr_one_day import qualify_verified

from dllm_agent.core.io import sha256_bytes
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day import runtime
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CampaignManifest,
    CoordinatorError,
    ExogenousFailureEvidence,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    RawArmReportRow,
    StartGateEvidence,
    TreatmentFailureEvidence,
    build_campaign_manifest,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    HashOnlyDeepParseReplayer,
    ProductionPostSearch,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ProductionGateRunner,
    QualificationEvidence,
    collect_resource_topology_measurements,
    write_qualification_fact,
)
from dllm_agent.experiments.openexr_one_day.runtime import (
    ProductionRuntimeFactory,
    RevisionWorkerFactory,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix4_qualification import _measurements
from tests.test_openexr_one_day_fix4_recovery_reporting import (
    CountingCoverageBuilder,
    CountingProofBuilder,
    NoopCoverageReplayer,
    StaticProofClock,
)

if TYPE_CHECKING:
    from types import TracebackType

    from dllm_agent.experiments.multitarget.ledger import JsonValue
    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.models import ModelArm
    from dllm_agent.experiments.openexr_one_day.post_search import PostSearchOutcome
    from dllm_agent.runtime.revision_process import WorkerCpuAffinity
    from dllm_agent.targets.openexr.probe import ProbeConfig


def test_qualification_rejects_noncanonical_wrapper_and_duplicate_smoke_pair(
    data_root: Path,
) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    evidence_root = data_root / "qualification"
    records = [
        write_qualification_fact(
            manifest=manifest,
            evidence_root=evidence_root,
            gate_name=gate,
            measurement_kind=gate,
            measurements=_measurements(gate, manifest),
        )
        for gate in QUALIFICATION_GATE_NAMES
    ]
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(records),
    )
    evidence_path = evidence_root / "qualification.json"
    _ = evidence_path.write_text(json.dumps(evidence.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(CoordinatorError, match="not canonical"):
        _ = ProductionGateRunner(evidence_path).evaluate(manifest)

    smoke = _measurements("smoke_passed", manifest)
    pairs = cast("list[dict[str, JsonValue]]", smoke["pairs"])
    pairs[1]["pair_id"] = 0
    records[-1] = write_qualification_fact(
        manifest=manifest,
        evidence_root=evidence_root,
        gate_name="smoke_passed",
        measurement_kind="smoke_passed",
        measurements=smoke,
        passed=True,
    )
    duplicate = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(records),
    )
    _ = evidence_path.write_bytes(canonical_json_bytes(duplicate.model_dump(mode="json")))
    with pytest.raises(CoordinatorError, match="smoke_passed"):
        _ = ProductionGateRunner(evidence_path).evaluate(manifest)


def test_qualify_verified_write_once_rejects_existing_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "one-day.json"
    payload = _config_json("/data/dllm-agent/fix5c-qualify")
    payload["config_sha256"] = config_payload_sha256(payload)
    _ = config_path.write_bytes(canonical_json_bytes(payload))
    root = tmp_path / "qualification"

    first = qualify_verified(config_path, root)
    _ = (root / "qualification.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CoordinatorError, match=r"qualification\.json"):
        _ = qualify_verified(config_path, root)
    assert first["state"] == "qualified-draft-only"


def test_runtime_graph_owns_probe_context(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _qualified_config(data_root)
    context = TrackingProbeContext()
    monkeypatch.setattr(runtime, "prepare_openexr_probe", context.prepare)

    graph = ProductionRuntimeFactory(
        runtime_python=Path("/usr/bin/python3"),
        source_root=Path("/data/dllm-agent/source/openexr"),
        cache_dir=Path("/data/dllm-agent/cache"),
        prompt_path=Path("prompts/openexr-system.md"),
        worker_factory=cast("RevisionWorkerFactory", cast("object", FakeRuntimeWorkerFactory())),
    ).build(config)

    assert context.entered == 0
    assert context.exited == 0
    assert graph.post_search.deep_parse_replayer is not None
    assert graph.post_search.deep_parse_replayer.__class__.__name__ == "DockerDeepParseReplayer"
    assert graph.post_search.deep_parse_replayer.checker_binary_sha256 == (
        config.deep_parse_checker_sha256
    )
    with graph as entered:
        assert entered is graph
        assert context.entered == 1
        assert context.exited == 0
        assert graph.model_executor.probe.__class__ is FakeProbe

    assert context.exited == 1
    graph.close()
    assert context.exited == 1


def test_resource_topology_derives_from_raw_not_claimed_parsed_map(data_root: Path) -> None:
    config = _qualified_config(data_root)
    manifest = _manifest(config)
    evidence_root = data_root / "qualification"
    records = [
        write_qualification_fact(
            manifest=manifest,
            evidence_root=evidence_root,
            gate_name=gate,
            measurement_kind=gate,
            measurements=_measurements(gate, manifest),
        )
        for gate in QUALIFICATION_GATE_NAMES
    ]
    measurements = _measurements("resource_map_valid", manifest)
    forged_stdout = "0, GPU-forged-raw-identity\n" + "\n".join(
        f"{index}, {record.uuid}"
        for index, record in enumerate(config.resource_map.gpu_records[1:])
    )
    measurements["nvidia_smi_stdout"] = forged_stdout
    measurements["nvidia_smi_stdout_sha256"] = sha256_bytes(forged_stdout.encode())
    measurements["parsed_resource_map"] = config.resource_map.model_dump(mode="json")
    resource_index = QUALIFICATION_GATE_NAMES.index("resource_map_valid")
    records[resource_index] = write_qualification_fact(
        manifest=manifest,
        evidence_root=evidence_root,
        gate_name="resource_map_valid",
        measurement_kind="resource_map_valid",
        measurements=measurements,
        passed=True,
    )
    evidence = QualificationEvidence.for_manifest(
        manifest_sha256=manifest.manifest_sha256 or "",
        config_sha256=config.config_sha256,
        records=tuple(records),
    )
    evidence_path = evidence_root / "qualification.json"
    _ = evidence_path.write_bytes(canonical_json_bytes(evidence.model_dump(mode="json")))

    with pytest.raises(CoordinatorError, match="resource_map_valid"):
        _ = ProductionGateRunner(evidence_path).evaluate(manifest)


def test_collect_resource_topology_measurements_uses_fixed_argv_and_hashes(
    tmp_path: Path,
) -> None:
    sibling_root = tmp_path / "cpu"
    for cpu in range(2):
        path = sibling_root / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        _ = (path / "thread_siblings_list").write_text("0-1\n", encoding="utf-8")
    pci_root = tmp_path / "pci"
    gpu_path = pci_root / "0000:01:00.0"
    gpu_path.mkdir(parents=True)
    _ = (gpu_path / "numa_node").write_text("0\n", encoding="utf-8")
    runner = FakeTopologyRunner(calls=[])

    measurements = collect_resource_topology_measurements(
        runner,
        thread_sibling_root=sibling_root,
        pci_device_root=pci_root,
    )

    assert runner.calls == [
        ("lscpu", "--parse=CPU,CORE,SOCKET,NODE"),
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ),
    ]
    assert measurements["lscpu_exit_code"] == 0
    assert measurements["nvidia_smi_exit_code"] == 0
    assert measurements["lscpu_stdout_sha256"] == sha256_bytes(
        cast("str", measurements["lscpu_stdout"]).encode()
    )
    assert measurements["nvidia_smi_stdout_sha256"] == sha256_bytes(
        cast("str", measurements["nvidia_smi_stdout"]).encode()
    )
    sibling_records = measurements["thread_siblings"]
    assert isinstance(sibling_records, list)
    first_sibling = cast("dict[str, JsonValue]", sibling_records[0])
    assert first_sibling["raw_sha256"] == sha256_bytes(cast("str", first_sibling["raw"]).encode())


def test_model_value_error_becomes_one_typed_treatment_failure(data_root: Path) -> None:
    config = _qualified_config(data_root)
    failing = "wave-00-rep-00-selective-revision"

    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(),
        post_search=EnvelopePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=ValueErrorModelExecutor(failing),
    )

    failures = tuple(cell for cell in outcome.cells if cell.terminal_reason == "treatment-failed")
    assert tuple(cell.cell_id for cell in failures) == (failing,)
    failure = failures[0]
    if failure.treatment_failure_path is None:
        raise AssertionError
    evidence = TreatmentFailureEvidence.model_validate_json(
        failure.treatment_failure_path.read_bytes()
    )
    assert evidence.exception_class == "ValueError"
    rows = _raw_rows(config)
    failed_row = next(row for row in rows if row.cell_id == failing)
    assert failed_row.treatment_failure_sha256 == failure.treatment_failure_sha256


def test_exogenous_failure_binds_pair_and_raw_evidence(data_root: Path) -> None:
    config = _qualified_config(data_root)
    source = "wave-00-rep-00-selective-revision"

    outcome = coordinate_campaign(
        config,
        resident_pool=FakeResidentPool(),
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=FakeCellRunner(exogenous_failures={source: "xid 79"}),
        post_search=EnvelopePostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
    )

    pair = next(item for item in outcome.pairs if item.wave_index == 0)
    assert pair.status == "exogenous-invalidated"
    if pair.exogenous_evidence_path is None:
        raise AssertionError
    evidence = ExogenousFailureEvidence.model_validate_json(
        pair.exogenous_evidence_path.read_bytes()
    )
    assert evidence.cause == "gpu-xid"
    assert evidence.raw_evidence_sha256 == sha256_bytes(b"xid 79")
    rows = _raw_rows(config)
    invalidated = tuple(row for row in rows if row.terminal_reason == "exogenous-invalidated")
    assert invalidated
    assert {row.exogenous_evidence_sha256 for row in invalidated} == {
        pair.exogenous_evidence_sha256
    }


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix5c")
    return Path("/data/dllm-agent") / f"task5-fix5c-{sha256(str(root).encode()).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class ValueErrorModelExecutor:
    failing_cell_id: str

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        if request.cell.cell_id == self.failing_cell_id:
            message = "reviewer smoke failure"
            raise ValueError(message)
        path = request.cell.artifact_root / "corpus" / f"{request.cell.cell_id}.exr"
        payload = request.cell.cell_id.encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(payload)
        return ModelArmExecutionResult(
            written_snapshot_count=1,
            written_paths=(path,),
            nfe=1,
            forward_calls=1,
            gpu_ms=1,
            bank_ready_elapsed_seconds=request.bank_ready_elapsed_seconds,
        )


@dataclass(frozen=True, slots=True)
class EnvelopePostSearch:
    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        return self._post_search().coverage_and_proof(request, result)

    @staticmethod
    def _post_search() -> ProductionPostSearch:
        return ProductionPostSearch(
            coverage_replayer=NoopCoverageReplayer(),
            coverage_builder=CountingCoverageBuilder(),
            proof_builder=CountingProofBuilder(),
            clock=StaticProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        )

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        return ProductionPostSearch(
            coverage_replayer=NoopCoverageReplayer(),
            coverage_builder=CountingCoverageBuilder(),
            proof_builder=CountingProofBuilder(),
            clock=StaticProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        ).treatment_coverage(request)


@dataclass(frozen=True, slots=True)
class TrackingProbeContext:
    entered: int = 0
    exited: int = 0

    def prepare(self, config: ProbeConfig) -> Self:
        del config
        return self

    def __enter__(self) -> FakeProbe:
        object.__setattr__(self, "entered", self.entered + 1)
        return FakeProbe()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        object.__setattr__(self, "exited", self.exited + 1)


@dataclass(frozen=True, slots=True)
class FakeProbe:
    pass


@dataclass(frozen=True, slots=True)
class FakeTopologyRunner:
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[bytes]:
        _ = self.calls.append(argv)
        if argv == ("lscpu", "-e=CPU,CORE,SOCKET,NODE", "-p"):
            return CompletedProcess(
                argv,
                0,
                stdout=b"# CPU,CORE,SOCKET,NODE\n0,0,0,0\n1,1,0,0\n",
                stderr=b"",
            )
        if argv == ("lscpu", "--parse=CPU,CORE,SOCKET,NODE"):
            return CompletedProcess(
                argv,
                0,
                stdout=b"# CPU,Core,Socket,Node\n0,0,0,0\n1,0,0,0\n",
                stderr=b"",
            )
        if argv == (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ):
            return CompletedProcess(
                argv,
                0,
                stdout=b"0, GPU-00000000-0000-0000-0000-000000000000, 00000000:01:00.0\n",
                stderr=b"",
            )
        raise AssertionError(argv)


@dataclass(frozen=True, slots=True)
class FakeRuntimeWorkerFactory:
    affinities: list[WorkerCpuAffinity] = field(default_factory=list)

    def create(self, *, device: str, cpu_affinity: WorkerCpuAffinity) -> FakeRevisionDelegate:
        del device
        self.affinities.append(cpu_affinity)
        return FakeRevisionDelegate()


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified"}
    )


def _manifest(config: CampaignConfig) -> CampaignManifest:
    return build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )


def _passing_gates() -> StartGateEvidence:
    return StartGateEvidence(
        hashes_match=True,
        banks_complete=True,
        read_only_cache=True,
        external_coverage_check=True,
        control_arm_equivalence=True,
        resource_map_valid=True,
        disk_space=True,
        vram=True,
        ram=True,
        swap_stable=True,
        reserved_core_load=True,
        smoke_passed=True,
    )


def _delegates() -> dict[ModelArm, FakeRevisionDelegate]:
    return {
        arm: FakeRevisionDelegate()
        for arm in ("selective-revision", "final-only", "full-regenerate", "structured-one-shot")
    }


def _raw_rows(config: CampaignConfig) -> tuple[RawArmReportRow, ...]:
    return tuple(
        RawArmReportRow.model_validate_json(line)
        for line in (config.artifact_root / "raw-arm-reports.jsonl").read_bytes().splitlines()
    )
