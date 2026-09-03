"""Versioned schemas for live dLLM harness generation and fuzzing."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from dllm_agent.core.models import FrozenModel, Probability, Sha256
from dllm_agent.targets.models import DockerImageId as _DockerImageId
from dllm_agent.targets.models import TargetEvidence as _TargetEvidence

TransferSchemaVersion = Literal["stable-diffcoder-transfer-evidence/v1"]


class TokenTransferEvidence(FrozenModel):
    """One committed Stable-DiffCoder token with generated-slice position evidence."""

    generated_position: int = Field(ge=0)
    token_id: int = Field(ge=0)
    text: str
    confidence: Probability


class DenoisingStep(FrozenModel):
    """One observed Stable-DiffCoder token-transfer step."""

    step_index: int = Field(ge=1)
    masked_tokens_before: int = Field(ge=1)
    committed_tokens: int = Field(ge=1)
    mean_confidence: Probability
    state_sha256: Sha256
    transfer_schema_version: TransferSchemaVersion | None = None
    transfers: tuple[TokenTransferEvidence, ...] = ()

    @model_validator(mode="after")
    def _validate_transfer_evidence(self) -> Self:
        if not self.transfers:
            if self.transfer_schema_version is not None:
                msg = "transfer_schema_version requires transfer records"
                raise ValueError(msg)
            return self
        if self.transfer_schema_version != "stable-diffcoder-transfer-evidence/v1":
            msg = "transfer records require stable-diffcoder-transfer-evidence/v1"
            raise ValueError(msg)
        if len(self.transfers) != self.committed_tokens:
            msg = "transfer record count must match committed_tokens"
            raise ValueError(msg)
        positions = tuple(transfer.generated_position for transfer in self.transfers)
        if positions != tuple(sorted(positions)):
            msg = "transfer generated positions must be monotonic"
            raise ValueError(msg)
        return self


class GenerationEvidence(FrozenModel):
    """Generation controls that determine the denoising budget."""

    steps: int = Field(ge=1, le=4096)
    gen_length: int = Field(ge=4, le=4096)
    block_length: int = Field(ge=1, le=128)
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]


class LiveModelProvenance(FrozenModel):
    """Pinned live-model and runtime evidence."""

    model_id: Literal["ByteDance-Seed/Stable-DiffCoder-8B-Instruct"]
    revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"]
    remote_code_sha256: Sha256
    python_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    device: str = Field(min_length=1)
    generation: GenerationEvidence


class HarnessCandidate(FrozenModel):
    """One finalized C++ harness with its observed denoising trajectory."""

    candidate_id: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=32_768)
    source_sha256: Sha256
    seed: int = Field(ge=0)
    nfe: int = Field(ge=1)
    steps: tuple[DenoisingStep, ...] = Field(min_length=1)


class CandidateBank(FrozenModel):
    """Strict JSON boundary emitted by the separate model runtime."""

    schema_version: Literal["dllm-candidate-bank/v1"]
    prompt_sha256: Sha256
    model: LiveModelProvenance
    candidates: tuple[HarnessCandidate, ...] = Field(min_length=2, max_length=8)


class HarnessCandidateScore(FrozenModel):
    """Transparent live-harness ranking components."""

    candidate_id: str
    score: float
    trajectory_confidence: Probability
    api_coverage: Probability
    nfe_efficiency: Probability
    nfe: int


class CandidateRejection(FrozenModel):
    """Generated candidate excluded by the static harness boundary."""

    candidate_id: str
    reason: str


class SandboxEvidence(FrozenModel):
    """Security properties encoded in every Docker argv."""

    image_id: _DockerImageId
    network: Literal["none"]
    read_only_root: Literal[True]
    capabilities: Literal["none"]
    no_new_privileges: Literal[True]
    user: Literal["65534:65534"]


class ArtifactEvidence(FrozenModel):
    """Hash and size for one corpus or crash artifact."""

    name: str
    sha256: Sha256
    size: int = Field(ge=0)


class CommandEvidence(FrozenModel):
    """Hashes of the exact fixed argument vectors used by the runner."""

    compiler_version_argv_sha256: Sha256
    fuzzer_version_argv_sha256: Sha256
    build_argv_sha256: Sha256
    fuzz_argv_sha256: Sha256


class FuzzEvidence(FrozenModel):
    """Normalized dynamic validation result and hashed raw evidence."""

    requested_sanitizers: Literal["address,undefined", "address"] = "address,undefined"
    build_status: Literal["succeeded", "failed", "timed_out"]
    fuzz_status: Literal["completed", "crashed", "timed_out", "skipped"]
    oracle_status: Literal["proved", "crash_without_sanitizer", "not_observed"]
    compiler_version_sha256: Sha256
    fuzzer_version_sha256: Sha256
    build_log_sha256: Sha256
    fuzz_log_sha256: Sha256
    executed_units: int | None = Field(default=None, ge=0)
    edge_coverage: int | None = Field(default=None, ge=0)
    feature_coverage: int | None = Field(default=None, ge=0)
    corpus_count: int | None = Field(default=None, ge=0)
    executions_per_second: int | None = Field(default=None, ge=0)
    sanitizer: str | None
    corpus_artifacts: tuple[ArtifactEvidence, ...]
    crash_artifacts: tuple[ArtifactEvidence, ...]
    commands: CommandEvidence


class CoverageTotalsEvidence(FrozenModel):
    """LLVM source coverage percentages over the selected target tree."""

    regions: Annotated[float, Field(ge=0.0, le=100.0)]
    functions: Annotated[float, Field(ge=0.0, le=100.0)]
    lines: Annotated[float, Field(ge=0.0, le=100.0)]
    branches: Annotated[float, Field(ge=0.0, le=100.0)]


class CoverageCommandEvidence(FrozenModel):
    """Hashes of fixed argv used for frozen-corpus coverage evaluation."""

    build_argv_sha256: Sha256
    run_argv_sha256: Sha256
    merge_argv_sha256: Sha256
    report_argv_sha256: Sha256


class CoverageEvidence(FrozenModel):
    """Post-campaign LLVM coverage result and raw-evidence hashes."""

    build_status: str
    run_status: str
    merge_status: str
    report_status: str
    build_log_sha256: Sha256
    run_log_sha256: Sha256
    merge_log_sha256: Sha256
    report_log_sha256: Sha256
    totals: CoverageTotalsEvidence | None
    commands: CoverageCommandEvidence


class HuntReport(FrozenModel):
    """Hash-carrying report for one real dLLM-guided fuzz campaign."""

    schema_version: Literal["dllm-hunt-report/v1"]
    hunt_id: str
    target: _TargetEvidence
    model: LiveModelProvenance
    prompt_sha256: Sha256
    candidate_bank_sha256: Sha256
    candidate_scores: tuple[HarnessCandidateScore, ...]
    rejected_candidates: tuple[CandidateRejection, ...]
    selected_candidate_id: str
    selected_harness_sha256: Sha256
    sandbox: SandboxEvidence
    fuzz: FuzzEvidence
    coverage: CoverageEvidence
