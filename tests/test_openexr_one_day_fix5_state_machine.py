"""Task 5 fix5A state-machine and recovery regressions.

# noqa: SIZE_OK
"""

# pyright: reportPrivateUsage=false, reportUnnecessaryComparison=false

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

from dllm_agent.experiments.openexr_one_day.coordinator import (
    CampaignConfig,
    ModelArmExecutionRequest,
    ModelArmExecutionResult,
    StartGateEvidence,
    coordinate_campaign,
)
from dllm_agent.experiments.openexr_one_day.post_search import (
    HashOnlyDeepParseReplayer,
    PostSearchOutcome,
    ProductionPostSearch,
)
from dllm_agent.experiments.openexr_one_day.recovery import load_campaign_events, resume_campaign
from tests.openexr_one_day_task5_fakes import (
    FakeCellRunner,
    FakeGateRunner,
    FakeResidentPool,
    FakeRevisionDelegate,
    fake_bank_sampler,
)
from tests.test_openexr_one_day_coordinator import _config_json
from tests.test_openexr_one_day_fix4_recovery_reporting import (
    CountingCoverageBuilder,
    CountingProofBuilder,
    NoopCoverageReplayer,
    StaticProofClock,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dllm_agent.experiments.openexr_one_day.cell_runner import CellResult, CellRunRequest
    from dllm_agent.experiments.openexr_one_day.coordinator import CampaignManifest
    from dllm_agent.experiments.openexr_one_day.models import ModelArm
    from dllm_agent.revision.controller_models import InitialParticleRequest, RevisionRequest


def test_interrupt_does_not_serialize_pending_model_placeholders(
    data_root: Path,
) -> None:
    # Given
    config = _qualified_config(data_root)

    # When / Then
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(interrupt_after=1),
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )
    terminals = tuple(
        event for event in load_campaign_events(config.artifact_root) if event.phase == "terminal"
    )
    launches = tuple(
        event for event in load_campaign_events(config.artifact_root) if event.phase == "launch"
    )
    terminal_ids = {event.cell_id for event in terminals}
    launch_ids = {event.cell_id for event in launches}
    assert terminal_ids <= launch_ids
    assert all(cell_id.endswith("-libfuzzer-only") for cell_id in terminal_ids)
    assert launch_ids == {
        "wave-00-rep-00-libfuzzer-only",
        "wave-00-rep-01-libfuzzer-only",
    }


def test_complete_reentry_uses_outcome_envelope_without_side_effects(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    first_pool = FakeResidentPool()
    first_runner = FakeCellRunner()
    first_post = PathBackedPostSearch()
    first_model = CountingModelExecutor()
    first = coordinate_campaign(
        config,
        resident_pool=first_pool,
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=first_runner,
        post_search=first_post,
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=first_model,
    )
    raw_before = (config.artifact_root / "raw-arm-reports.jsonl").read_bytes()
    outcome_before = (config.artifact_root / "campaign-outcome.json").read_bytes()
    second_pool = FakeResidentPool()
    second_runner = FakeCellRunner()
    second_post = PathBackedPostSearch()
    second_model = CountingModelExecutor()

    # When
    second = coordinate_campaign(
        config,
        resident_pool=second_pool,
        bank_sampler=fake_bank_sampler(),
        revision_delegates=_delegates(),
        cell_runner=second_runner,
        post_search=second_post,
        gate_runner=ExplodingGateRunner(),
        model_executor=second_model,
    )

    # Then
    assert second == first
    assert (config.artifact_root / "raw-arm-reports.jsonl").read_bytes() == raw_before
    assert (config.artifact_root / "campaign-outcome.json").read_bytes() == outcome_before
    assert second_pool.enter_count == 0
    assert second_runner.requests == []
    assert second_post.calls == []
    assert second_model.calls == []
    assert first_model.calls
    assert first.accounting.search_cpu_seconds == second.accounting.search_cpu_seconds


def test_partial_resume_reuses_banks_launch_requests_and_model_prep(data_root: Path) -> None:
    # Given
    config = _qualified_config(data_root)
    with pytest.raises(KeyboardInterrupt):
        _ = coordinate_campaign(
            config,
            resident_pool=FakeResidentPool(),
            bank_sampler=fake_bank_sampler(),
            revision_delegates=_delegates(),
            cell_runner=FakeCellRunner(interrupt_after=2),
            post_search=PathBackedPostSearch(),
            gate_runner=FakeGateRunner(_passing_gates()),
            model_executor=CountingModelExecutor(),
        )
    bank_bytes = _artifact_bytes(config.artifact_root.glob("banks/rep-*/candidate-bank.json"))
    prep_bytes = _artifact_bytes(config.artifact_root.glob("cells/**/model-prep-result.json"))
    corpus_bytes = _artifact_bytes(config.artifact_root.glob("cells/**/corpus/*"))
    launch_bytes = (config.artifact_root / "campaign-events.jsonl").read_bytes()
    runner = FakeCellRunner()
    model = CountingModelExecutor()

    # When
    outcome = resume_campaign(
        config.artifact_root,
        resident_pool=FakeResidentPool(),
        bank_sampler=ExplodingBankSampler(),
        revision_delegates=_delegates(),
        cell_runner=runner,
        post_search=PathBackedPostSearch(),
        gate_runner=FakeGateRunner(_passing_gates()),
        model_executor=model,
    )

    # Then
    assert outcome.launched_cell_count == 50
    persisted_prep_cells = {Path(path).parent.name for path, _content in prep_bytes}
    assert not persisted_prep_cells & set(model.calls)
    after_banks = dict(
        _artifact_bytes(config.artifact_root.glob("banks/rep-*/candidate-bank.json"))
    )
    for path, content in bank_bytes:
        assert after_banks[path] == content
    after_prep = dict(_artifact_bytes(config.artifact_root.glob("cells/**/model-prep-result.json")))
    for path, content in prep_bytes:
        assert after_prep[path] == content
    assert (config.artifact_root / "campaign-events.jsonl").read_bytes().startswith(launch_bytes)
    after_corpus = dict(_artifact_bytes(config.artifact_root.glob("cells/**/corpus/*")))
    for path, content in corpus_bytes:
        assert after_corpus[path] == content


@pytest.fixture
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fix5")
    return Path("/data/dllm-agent") / f"task5-fix5-{sha256(str(root).encode()).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class CountingModelExecutor:
    calls: list[str] = field(default_factory=list)

    def prepare_corpus(self, request: ModelArmExecutionRequest) -> ModelArmExecutionResult:
        self.calls.append(request.cell.cell_id)
        paths = tuple(
            request.cell.artifact_root / "corpus" / f"snapshot-{index:02d}.exr"
            for index in range(2)
        )
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_bytes(f"{request.cell.cell_id}:{index}".encode())
        return ModelArmExecutionResult(
            written_snapshot_count=len(paths),
            written_paths=paths,
            nfe=7,
            forward_calls=8,
            gpu_ms=9,
            bank_ready_elapsed_seconds=0.25,
        )


@dataclass(frozen=True, slots=True)
class PathBackedPostSearch:
    calls: list[str] = field(default_factory=list)

    def coverage_and_proof(self, request: CellRunRequest, result: CellResult) -> PostSearchOutcome:
        self.calls.append(request.cell_id)
        return ProductionPostSearch(
            coverage_replayer=NoopCoverageReplayer(),
            coverage_builder=CountingCoverageBuilder(),
            proof_builder=CountingProofBuilder(),
            clock=StaticProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        ).coverage_and_proof(request, result)

    def treatment_coverage(self, request: CellRunRequest) -> PostSearchOutcome:
        self.calls.append(request.cell_id)
        return ProductionPostSearch(
            coverage_replayer=NoopCoverageReplayer(),
            coverage_builder=CountingCoverageBuilder(),
            proof_builder=CountingProofBuilder(),
            clock=StaticProofClock(),
            deep_parse_replayer=HashOnlyDeepParseReplayer(),
        ).treatment_coverage(request)


@dataclass(frozen=True, slots=True)
class ExplodingGateRunner:
    def evaluate(self, manifest: CampaignManifest) -> StartGateEvidence:
        del manifest
        raise AssertionError


@dataclass(frozen=True, slots=True)
class ExplodingBankSampler:
    def initial(self, request: InitialParticleRequest) -> Never:
        del request
        raise AssertionError

    def revise(self, request: RevisionRequest) -> Never:
        del request
        raise AssertionError


def _artifact_bytes(paths: Iterable[Path]) -> tuple[tuple[str, bytes], ...]:
    return tuple(sorted((path.as_posix(), path.read_bytes()) for path in paths))


def _qualified_config(data_root: Path) -> CampaignConfig:
    return CampaignConfig.model_validate(
        _config_json(str(data_root)) | {"qualification_state": "qualified"}
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
