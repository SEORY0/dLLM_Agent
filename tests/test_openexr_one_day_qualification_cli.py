"""Task 7 qualification CLI and smoke lifecycle tests."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Self

import pytest
import tools.run_openexr_one_day as cli
from tools.run_openexr_one_day import _production_factory
from typer.testing import CliRunner

from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.openexr_one_day.config_digest import config_payload_sha256
from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    CoordinatorError,
    build_campaign_manifest,
)
from dllm_agent.experiments.openexr_one_day.qualification import (
    QUALIFICATION_GATE_NAMES,
    ConfigArtifactMeasurements,
    QualificationSmokeResult,
    build_qualification_smoke_plan,
    finalize_config_payload_from_artifacts,
    run_qualification_smoke,
    write_record_backed_qualification_from_measurements,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke import (
    _cell_outcome,
    _result_from_cells,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    WRAPPER_NAME,
    QualificationSmokeArtifact,
    QualificationSmokeBinding,
    SmokeCellResult,
    SmokePostSearchOutcome,
    artifact_tree_sha256,
    write_canonical_once,
)
from dllm_agent.experiments.openexr_one_day.qualification_smoke_models import (
    model_sha256 as smoke_model_sha256,
)
from dllm_agent.experiments.openexr_one_day.schedule import build_campaign_schedule
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix4_qualification import _config_payload, _measurements

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.ledger import JsonValue
    from dllm_agent.experiments.openexr_one_day.qualification import QualificationSmokePlan
    from dllm_agent.experiments.openexr_one_day.runtime import ProductionQualificationSmokeAdapter

RUNNER = CliRunner()


def test_write_record_backed_qualification_rejects_placeholder_provenance(
    data_root: Path,
) -> None:
    """Given placeholder artifact hashes, when qualification writes, then it fails closed."""
    # Given
    config = _draft_config(data_root)
    measurements_root = _write_measurements(data_root / "measurements", config)

    # When / Then
    with pytest.raises(CoordinatorError, match="placeholder provenance"):
        _ = write_record_backed_qualification_from_measurements(
            config=config,
            evidence_root=data_root / "qualification",
            measurements_root=measurements_root,
        )


def test_write_record_backed_qualification_writes_canonical_evidence(
    data_root: Path,
) -> None:
    """Given measured 12-gate JSON, when qualification writes, then evidence starts main gates."""
    # Given
    config = _qualified_config(data_root)
    measurements_root = _write_measurements(data_root / "measurements", config)

    # When
    result = write_record_backed_qualification_from_measurements(
        config=config,
        evidence_root=data_root / "qualification",
        measurements_root=measurements_root,
    )

    # Then
    assert result.state == "qualified"
    assert len(result.evidence.records) == 12
    assert {record.name for record in result.evidence.records} == set(QUALIFICATION_GATE_NAMES)
    evidence_path = data_root / "qualification" / "qualification.json"
    assert evidence_path.read_bytes() == canonical_json_bytes(
        result.evidence.model_dump(mode="json")
    )
    assert result.start_gates.failing_gate() is None


def test_qualification_smoke_plan_is_two_pairs_by_five_arms_for_300_seconds(
    data_root: Path,
) -> None:
    """Given a config, when smoke is planned, then it is isolated from the main campaign."""
    # Given
    config = _qualified_config(data_root)

    # When
    plan = build_qualification_smoke_plan(config=config, smoke_root=data_root / "smoke")

    # Then
    assert plan.pair_count == 2
    assert plan.cell_count == 10
    assert {cell.arm for cell in plan.cells} == {
        "selective-revision",
        "final-only",
        "full-regenerate",
        "structured-one-shot",
        "libfuzzer-only",
    }
    assert {cell.deadline_seconds for cell in plan.cells} == {300}
    assert all(cell.artifact_root.is_relative_to(data_root / "smoke") for cell in plan.cells)
    assert all(not cell.artifact_root.is_relative_to(data_root / "cells") for cell in plan.cells)


def test_run_qualification_smoke_derives_pair_gate_from_runner_result(data_root: Path) -> None:
    """Given a smoke runner, when it omits a cell, then the smoke gate is not forged true."""
    # Given
    config = _qualified_config(data_root)
    plan = build_qualification_smoke_plan(config=config, smoke_root=data_root / "smoke")

    # When
    measurements = run_qualification_smoke(
        config=config,
        smoke_root=data_root / "smoke",
        runner=FakeSmokeRunner(plan, omit_last=True),
    )

    # Then
    pairs = measurements["pairs"]
    assert isinstance(pairs, list)
    assert pairs[1] == {
        "pair_id": 1,
        "deadline_seconds": 300,
        "schema_valid": False,
        "hash_valid": False,
        "restart_success": True,
        "deferred_proof_success": False,
    }


def test_finalize_config_payload_from_artifacts_updates_hashes_and_digest(
    tmp_path: Path,
) -> None:
    """Given measured artifact files, when config is finalized, then hashes bind to bytes."""
    # Given
    template = _config_json("/data/dllm-agent/task7-finalize")
    harness = tmp_path / "harness.cpp"
    prompt = tmp_path / "prompt.md"
    remote = tmp_path / "remote-code.json"
    search = tmp_path / "search-bin"
    coverage = tmp_path / "coverage-bin"
    coverage_manifest = tmp_path / "coverage-manifest.json"
    c0 = tmp_path / "c0.hex"
    dictionary = tmp_path / "openexr.dict"
    for path, content in (
        (harness, b"harness v1"),
        (prompt, b"prompt v1"),
        (remote, b"remote code provenance"),
        (search, b"search binary"),
        (coverage, b"coverage binary"),
        (coverage_manifest, b'{"target_only":true}'),
        (c0, b"00"),
        (dictionary, b'"EXR"'),
    ):
        _ = path.write_bytes(content)

    # When
    finalized = finalize_config_payload_from_artifacts(
        template,
        ConfigArtifactMeasurements(
            prompt_path=prompt,
            target_harness_path=harness,
            remote_code_path=remote,
            search_binary_path=search,
            search_image_digest="sha256:" + sha256(b"search-image").hexdigest(),
            coverage_binary_path=coverage,
            coverage_image_digest="sha256:" + sha256(b"coverage-image").hexdigest(),
            coverage_build_manifest_path=coverage_manifest,
            c0_source_path=c0,
            dictionary_path=dictionary,
            artifact_root=Path("/data/dllm-agent/task7-finalized"),
            config_id="openexr-v3.4.14-one-day-qualified",
            qualification_state="qualified",
        ),
    )

    # Then
    assert finalized["qualification_state"] == "qualified"
    assert finalized["target_harness_sha256"] == sha256(b"harness v1").hexdigest()
    assert finalized["prompt_sha256"] == sha256(b"prompt v1").hexdigest()
    assert finalized["remote_code_sha256"] == sha256(b"remote code provenance").hexdigest()
    assert finalized["search_binary_sha256"] == sha256(b"search binary").hexdigest()
    assert finalized["coverage_binary_sha256"] == sha256(b"coverage binary").hexdigest()
    assert finalized["c0_sha256"] == sha256(b"\x00").hexdigest()
    assert finalized["dictionary_path"] == dictionary.as_posix()
    assert finalized["dictionary_sha256"] == sha256(b'"EXR"').hexdigest()
    assert finalized["config_sha256"] == config_payload_sha256(finalized)


def test_finalize_config_binds_deep_parse_checker_hash_to_build_manifest(tmp_path: Path) -> None:
    """Given a rebuilt image manifest, when config is finalized, then the checker pin follows it."""
    # Given
    template = _config_json("/data/dllm-agent/task7-finalize-checker")
    artifacts = _artifact_files(tmp_path)
    checker = sha256(b"rebuilt checker").hexdigest()
    _ = artifacts.coverage_manifest.write_bytes(
        b'{"deep_parse_checker_sha256":"' + checker.encode() + b'"}'
    )

    # When
    finalized = finalize_config_payload_from_artifacts(
        template,
        ConfigArtifactMeasurements(
            prompt_path=artifacts.prompt,
            target_harness_path=artifacts.harness,
            remote_code_path=artifacts.remote,
            search_binary_path=artifacts.search,
            search_image_digest="sha256:" + sha256(b"search-image").hexdigest(),
            coverage_binary_path=artifacts.coverage,
            coverage_image_digest="sha256:" + sha256(b"coverage-image").hexdigest(),
            coverage_build_manifest_path=artifacts.coverage_manifest,
            c0_source_path=artifacts.c0,
            dictionary_path=artifacts.dictionary,
            artifact_root=Path("/data/dllm-agent/task7-finalized-checker"),
            config_id="openexr-v3.4.14-one-day-qualified",
            qualification_state="qualified",
        ),
    )

    # Then
    assert finalized["deep_parse_checker_sha256"] == checker
    assert finalized["config_sha256"] == config_payload_sha256(finalized)


def test_finalize_config_hashes_decoded_c0_bytes_not_hex_source_file(tmp_path: Path) -> None:
    """Given a hex C0 source, when config is finalized, then C0 binds decoded bytes."""
    # Given
    template = _config_json("/data/dllm-agent/task7-finalize-c0")
    artifacts = _artifact_files(tmp_path, c0_hex=b"00ff")

    # When
    finalized = finalize_config_payload_from_artifacts(
        template,
        ConfigArtifactMeasurements(
            prompt_path=artifacts.prompt,
            target_harness_path=artifacts.harness,
            remote_code_path=artifacts.remote,
            search_binary_path=artifacts.search,
            search_image_digest="sha256:" + sha256(b"search-image").hexdigest(),
            coverage_binary_path=artifacts.coverage,
            coverage_image_digest="sha256:" + sha256(b"coverage-image").hexdigest(),
            coverage_build_manifest_path=artifacts.coverage_manifest,
            c0_source_path=artifacts.c0,
            dictionary_path=artifacts.dictionary,
            artifact_root=Path("/data/dllm-agent/task7-finalized-c0"),
            config_id="openexr-v3.4.14-one-day-qualified",
            qualification_state="qualified",
        ),
    )

    # Then
    assert sha256(b"00ff").hexdigest() != sha256(b"\x00\xff").hexdigest()
    assert finalized["c0_sha256"] == sha256(b"\x00\xff").hexdigest()


def test_production_factory_anchors_relative_qualification_evidence_under_artifact_root(
    data_root: Path,
) -> None:
    """Given a relative evidence path, when factory resolves, then it stays under root."""
    # Given
    config = _qualified_config(data_root)

    # When
    factory = _production_factory(config)

    # Then
    assert factory.qualification_evidence_path == (data_root / "qualification.json").resolve(
        strict=False
    )


def test_production_factory_rejects_qualification_evidence_path_escape(data_root: Path) -> None:
    """Given an escaping evidence path, when factory resolves, then it fails closed."""
    # Given
    payload = _config_json(str(data_root)) | _non_placeholder_artifacts()
    payload["qualification_state"] = "qualified"
    payload["qualification_evidence_path"] = "../escape.json"
    payload["config_sha256"] = config_payload_sha256(payload)
    config = CampaignConfig.model_validate(payload)

    # When / Then
    with pytest.raises(CoordinatorError, match="qualification evidence path escapes"):
        _ = _production_factory(config)


def test_smoke_run_cli_executes_first_pass_then_completed_resume(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given fake production dependencies, when smoke-run is invoked, then it reloads once."""
    # Given
    config = _qualified_config(data_root)
    config_path = tmp_path / "config.json"
    _ = config_path.write_bytes(canonical_json_bytes(config.model_dump(mode="json")))
    smoke_root = data_root / "smoke-cli"
    graph = FakeSmokeGraph()
    calls: list[QualificationSmokePlan] = []

    def fake_factory(loaded: CampaignConfig) -> FakeSmokeFactory:
        assert loaded == config
        return FakeSmokeFactory(graph)

    def fake_execute(
        loaded: CampaignConfig,
        plan: QualificationSmokePlan,
        adapter: ProductionQualificationSmokeAdapter,
    ) -> QualificationSmokeResult:
        assert loaded == config
        assert adapter.smoke_root == smoke_root
        assert adapter.resident_pool is graph.resident_pool
        calls.append(plan)
        result = _canonical_smoke_result(plan)
        wrapper_path = smoke_root / WRAPPER_NAME
        if not wrapper_path.exists():
            wrapper_path.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_once(wrapper_path, _canonical_smoke_artifact(plan, result))
        return result

    monkeypatch.setattr(cli, "_production_factory", fake_factory)
    monkeypatch.setattr(cli, "execute_production_qualification_smoke", fake_execute)

    # When
    result = RUNNER.invoke(
        cli.app,
        ["smoke-run", "--config", str(config_path), "--smoke-root", str(smoke_root)],
    )

    # Then
    assert result.exit_code == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert graph.entered == 1
    assert graph.resident_pool.entered == 1
    payload = canonical_json_bytes(
        _canonical_smoke_artifact(calls[0], _canonical_smoke_result(calls[0])).model_dump(
            mode="json"
        )
    )
    assert result.stdout.strip().encode() == payload


def test_smoke_run_cli_completed_reload_does_not_enter_runtime_graph(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a completed wrapper, when smoke-run is invoked, then runtime is not entered."""
    # Given
    config = _qualified_config(data_root)
    config_path = tmp_path / "config.json"
    _ = config_path.write_bytes(canonical_json_bytes(config.model_dump(mode="json")))
    smoke_root = data_root / "smoke-completed-cli"
    plan = build_qualification_smoke_plan(config=config, smoke_root=smoke_root)
    result = _canonical_smoke_result(plan)
    wrapper = _canonical_smoke_artifact(plan, result)
    write_canonical_once(smoke_root / WRAPPER_NAME, wrapper)

    def forbidden_factory(loaded: CampaignConfig) -> FakeSmokeFactory:
        del loaded
        message = "completed smoke-run must reload before building runtime graph"
        raise AssertionError(message)

    monkeypatch.setattr(cli, "_production_factory", forbidden_factory)

    # When
    completed = RUNNER.invoke(
        cli.app,
        ["smoke-run", "--config", str(config_path), "--smoke-root", str(smoke_root)],
    )

    # Then
    assert completed.exit_code == 0
    assert completed.stdout.strip().encode() == canonical_json_bytes(
        wrapper.model_dump(mode="json")
    )


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    case = tmp_path_factory.mktemp("qualification-cli")
    digest = sha256(str(case).encode()).hexdigest()[:12]
    return Path("/data/dllm-agent") / f"task7-qualification-cli-{digest}"


@dataclass(frozen=True, slots=True)
class FakeSmokeRunner:
    expected_plan: QualificationSmokePlan
    omit_last: bool = False

    def run(self, plan: QualificationSmokePlan) -> QualificationSmokeResult:
        assert plan == self.expected_plan
        cells = tuple(plan.cells[:-1] if self.omit_last else plan.cells)
        outcomes = tuple(
            _cell_outcome(
                cell,
                QualificationSmokeBinding(
                    cell_id=cell.cell_id,
                    bank_sha256=sha256(f"bank:{cell.repetition}".encode()).hexdigest(),
                    model_prep_sha256=None
                    if cell.arm == "libfuzzer-only"
                    else sha256(f"model:{cell.cell_id}".encode()).hexdigest(),
                ),
                SmokeCellResult(
                    cell_id=cell.cell_id,
                    status="completed",
                    attempt_count=1,
                    deadline_monotonic_seconds=cell.deadline_monotonic_seconds,
                    result_sha256=sha256(f"cell:{cell.cell_id}".encode()).hexdigest(),
                ),
                SmokePostSearchOutcome(
                    cell_id=cell.cell_id,
                    proof_status="available",
                    deep_parse_status="available",
                    coverage_envelope_sha256=sha256(
                        f"coverage:{cell.cell_id}".encode()
                    ).hexdigest(),
                    proof_envelope_sha256=sha256(f"proof:{cell.cell_id}".encode()).hexdigest(),
                    deep_parse_envelope_sha256=sha256(f"deep:{cell.cell_id}".encode()).hexdigest(),
                    proof_result_sha256=sha256(f"proof-result:{cell.cell_id}".encode()).hexdigest(),
                    deep_parse_result_sha256=sha256(
                        f"deep-result:{cell.cell_id}".encode()
                    ).hexdigest(),
                ),
            )
            for cell in cells
        )
        if self.omit_last:
            unsigned = QualificationSmokeResult(
                plan_sha256=smoke_model_sha256(plan),
                cells=outcomes,
                cell_result_set_sha256=sha256(
                    canonical_json_bytes([cell.model_dump(mode="json") for cell in outcomes])
                ).hexdigest(),
                restart_success_pair_ids=(0, 1),
                result_sha256="0" * 64,
                passed=True,
            )
            return unsigned.model_copy(
                update={
                    "result_sha256": sha256(
                        canonical_json_bytes(
                            unsigned.model_dump(mode="json", exclude={"result_sha256"})
                        )
                    ).hexdigest()
                }
            )
        return QualificationSmokeResult.model_validate(
            _result_from_cells(plan, outcomes, (0, 1)).model_dump(mode="json")
        )


@dataclass(slots=True)
class FakeSmokePool:
    entered: int = 0

    def __enter__(self) -> Self:
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback


@dataclass(frozen=True, slots=True)
class FakeSmokeComponent:
    name: str


@dataclass(slots=True)
class FakeSmokeGraph:
    resident_pool: FakeSmokePool = field(default_factory=FakeSmokePool)
    cell_runner: FakeSmokeComponent = field(
        default_factory=lambda: FakeSmokeComponent("cell-runner")
    )
    model_executor: FakeSmokeComponent = field(
        default_factory=lambda: FakeSmokeComponent("model-executor")
    )
    post_search: FakeSmokeComponent = field(
        default_factory=lambda: FakeSmokeComponent("post-search")
    )
    entered: int = 0

    def __enter__(self) -> Self:
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback


@dataclass(frozen=True, slots=True)
class FakeSmokeFactory:
    graph: FakeSmokeGraph

    def build(self, config: CampaignConfig) -> FakeSmokeGraph:
        del config
        return self.graph


def _canonical_smoke_result(plan: QualificationSmokePlan) -> QualificationSmokeResult:
    outcomes = tuple(
        _cell_outcome(
            cell,
            QualificationSmokeBinding(
                cell_id=cell.cell_id,
                bank_sha256=sha256(f"bank:{cell.repetition}".encode()).hexdigest(),
                model_prep_sha256=None
                if cell.arm == "libfuzzer-only"
                else sha256(f"model:{cell.cell_id}".encode()).hexdigest(),
            ),
            SmokeCellResult(
                cell_id=cell.cell_id,
                status="completed",
                attempt_count=1,
                deadline_monotonic_seconds=cell.deadline_monotonic_seconds,
                result_sha256=sha256(f"cell:{cell.cell_id}".encode()).hexdigest(),
            ),
            SmokePostSearchOutcome(
                cell_id=cell.cell_id,
                proof_status="available",
                deep_parse_status="available",
                coverage_envelope_sha256=sha256(f"coverage:{cell.cell_id}".encode()).hexdigest(),
                proof_envelope_sha256=sha256(f"proof:{cell.cell_id}".encode()).hexdigest(),
                deep_parse_envelope_sha256=sha256(f"deep:{cell.cell_id}".encode()).hexdigest(),
                proof_result_sha256=sha256(f"proof-result:{cell.cell_id}".encode()).hexdigest(),
                deep_parse_result_sha256=sha256(f"deep-result:{cell.cell_id}".encode()).hexdigest(),
            ),
        )
        for cell in plan.cells
    )
    return QualificationSmokeResult.model_validate(
        _result_from_cells(plan, outcomes, (0, 1)).model_dump(mode="json")
    )


def _canonical_smoke_artifact(
    plan: QualificationSmokePlan, result: QualificationSmokeResult
) -> QualificationSmokeArtifact:
    plan.smoke_root.mkdir(parents=True, exist_ok=True)
    return QualificationSmokeArtifact(
        plan_sha256=smoke_model_sha256(plan),
        result=result,
        result_sha256=smoke_model_sha256(result),
        artifact_tree_sha256=artifact_tree_sha256(
            plan.smoke_root, exclude=plan.smoke_root / WRAPPER_NAME
        ),
    )


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


def _artifact_files(root: Path, *, c0_hex: bytes = b"00") -> ArtifactFiles:
    files = ArtifactFiles(
        harness=root / "harness.cpp",
        prompt=root / "prompt.md",
        remote=root / "remote-code.json",
        search=root / "search-bin",
        coverage=root / "coverage-bin",
        coverage_manifest=root / "coverage-manifest.json",
        c0=root / "c0.hex",
        dictionary=root / "openexr.dict",
    )
    for path, content in (
        (files.harness, b"harness v1"),
        (files.prompt, b"prompt v1"),
        (files.remote, b"remote code provenance"),
        (files.search, b"search binary"),
        (files.coverage, b"coverage binary"),
        (files.coverage_manifest, b'{"target_only":true}'),
        (files.c0, c0_hex),
        (files.dictionary, b'"EXR"'),
    ):
        _ = path.write_bytes(content)
    return files


def _write_measurements(root: Path, config: CampaignConfig) -> Path:
    manifest = build_campaign_manifest(
        config,
        build_campaign_schedule(main_start_monotonic_seconds=config.main_start_monotonic_seconds),
    )
    root.mkdir(parents=True)
    for gate in QUALIFICATION_GATE_NAMES:
        _ = (root / f"{gate}.json").write_bytes(canonical_json_bytes(_measurements(gate, manifest)))
    return root


def _qualified_config(data_root: Path) -> CampaignConfig:
    payload = _config_payload(data_root, include_gpu_records=True)
    payload["qualification_state"] = "qualified"
    payload["search_image_digest"] = "sha256:" + sha256(b"search-image").hexdigest()
    payload["coverage_image_digest"] = "sha256:" + sha256(b"coverage-image").hexdigest()
    payload["config_sha256"] = config_payload_sha256(payload)
    return CampaignConfig.model_validate(payload)


def _draft_config(data_root: Path) -> CampaignConfig:
    payload = _config_json(str(data_root))
    payload["config_sha256"] = config_payload_sha256(payload)
    return CampaignConfig.model_validate(payload)


def _non_placeholder_artifacts() -> dict[str, JsonValue]:
    dictionary = Path("configs/openexr-v3.4.14.dict")
    return {
        "target_harness_sha256": sha256(b"harness").hexdigest(),
        "prompt_sha256": sha256(b"prompt").hexdigest(),
        "remote_code_sha256": sha256(b"remote").hexdigest(),
        "search_image_digest": "sha256:" + sha256(b"search-image").hexdigest(),
        "search_binary_sha256": sha256(b"search-binary").hexdigest(),
        "coverage_image_digest": "sha256:" + sha256(b"coverage-image").hexdigest(),
        "coverage_binary_sha256": sha256(b"coverage-binary").hexdigest(),
        "coverage_build_manifest_sha256": sha256(b"coverage-manifest").hexdigest(),
        "dictionary_path": dictionary.as_posix(),
        "dictionary_sha256": sha256(dictionary.read_bytes()).hexdigest(),
    }
