"""CLI command boundary for live PoC revision runs."""

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from dllm_agent.core.errors import BoundaryError, HuntSetupError
from dllm_agent.core.io import parse_json, read_bytes, sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.revision.controller_models import RevisionRunRequest
from dllm_agent.revision.models import PocRevisionReport, RevisionConfig
from dllm_agent.targets.models import DockerImageId, GitCommit, HuntManifest, TargetEvidence
from dllm_agent.targets.openexr.probe import ProbeConfig
from dllm_agent.targets.staging import verify_target_archive

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
_IMAGE_ID: Final = "sha256:7591279f51256ef5e80341c788d31145ecf1dd0f7a64db7f0cdeaba527ff5add"
_TARGET_COMMIT: Final = "777d231a179de1711d2a942810d9559216ab3f4c"
_HARNESS_SHA256: Final = "4cf61b34216777285bf0ec2deba85611d04208d42151cf07bab4962dc4400f8f"
_MODEL_ID: Final = "ByteDance-Seed/Stable-DiffCoder-8B-Instruct"
_MODEL_REVISION: Final = "10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
POC_REVISION_CLI_ERRORS = (BoundaryError, HuntSetupError, ValueError)


class WorkerReadyProvenance(FrozenModel):
    """Resident worker ready record recorded after one model load."""

    model_id: Literal["ByteDance-Seed/Stable-DiffCoder-8B-Instruct"]
    revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"]
    remote_code_sha256: Sha256
    device: str = Field(min_length=1)
    load_count: int = Field(ge=0)


class RunManifest(FrozenModel):
    """Compact hash-bound manifest written beside bulky artifacts."""

    schema_version: Literal["dllm-poc-revision-run-manifest/v1"]
    config_sha256: Sha256
    prompt_path: str = Field(min_length=1)
    prompt_sha256: Sha256
    target: TargetEvidence
    model_id: Literal["ByteDance-Seed/Stable-DiffCoder-8B-Instruct"]
    model_revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"]
    model_device: str = Field(min_length=1)
    model_timeout_seconds: float = Field(ge=0)
    model_worker_started: bool
    worker_ready: WorkerReadyProvenance | None
    worker_argv: tuple[str, ...]
    worker_argv_sha256: Sha256
    image_id: DockerImageId
    harness_sha256: Sha256
    arm: str = Field(min_length=1)
    particle_count: int = Field(ge=0)
    epochs: int = Field(ge=0)
    initial_steps: int = Field(ge=0)
    revision_steps: int = Field(ge=0)
    probe_seconds: int = Field(ge=0)
    proof_seconds: int = Field(ge=0)
    fuzzer_seeds: tuple[int, ...]
    particle_seed_base: int = Field(ge=0)
    particle_seeds: tuple[int, ...]
    proof_seed: int = Field(ge=0)
    report_sha256: Sha256


class TargetPin(FrozenModel):
    """Pinned target source and disclosure boundary."""

    repository_url: str = Field(min_length=1)
    commit_sha: GitCommit
    security_policy_url: str = Field(min_length=1)
    security_policy_sha256: Sha256


class ModelConfig(FrozenModel):
    """Resident model worker launch controls."""

    model_id: Literal["ByteDance-Seed/Stable-DiffCoder-8B-Instruct"] = _MODEL_ID
    revision: Literal["10cdaf9b486f1cf0273ad968459dbe2d21a1482a"] = _MODEL_REVISION
    device: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)


class PromptConfig(FrozenModel):
    """Repository-local prompt with pinned bytes."""

    path: str = Field(min_length=1)
    sha256: Sha256


class PocRevisionCommandConfig(FrozenModel):
    """Strict installed command configuration."""

    schema_version: Literal["dllm-poc-revision-command/v1"]
    revision: RevisionConfig
    run: RevisionRunRequest
    target: TargetPin
    probe: ProbeConfig
    model: ModelConfig
    prompt: PromptConfig


class VerifiedPocRevisionCommand(FrozenModel):
    """Command config after fail-closed pins and local paths are checked."""

    config: PocRevisionCommandConfig
    config_sha256: Sha256
    prompt_path: Path
    prompt_sha256: Sha256
    target: TargetEvidence


def load_poc_revision_command(
    config_path: Path,
    source: Path,
    runtime_python: Path,
) -> VerifiedPocRevisionCommand:
    """Parse and verify command inputs before any artifact directory is created."""
    config_bytes = read_bytes(config_path, "poc revision config")
    config = parse_json(config_path, PocRevisionCommandConfig, "poc revision config")
    prompt = _prompt_path(config.prompt)
    _check_prompt(prompt, config.prompt.sha256)
    _check_static_pins(config)
    _check_runtime(runtime_python)
    target = _check_source(source, config)
    return VerifiedPocRevisionCommand(
        config=config,
        config_sha256=sha256_bytes(config_bytes),
        prompt_path=prompt,
        prompt_sha256=config.prompt.sha256,
        target=target,
    )


def _prompt_path(prompt: PromptConfig) -> Path:
    path = (_REPOSITORY_ROOT / prompt.path).resolve()
    if not path.is_relative_to(_REPOSITORY_ROOT):
        detail = "prompt path escapes repository root"
        raise HuntSetupError(detail)
    return path


def _check_prompt(path: Path, expected_sha256: str) -> None:
    observed = sha256_bytes(read_bytes(path, "poc revision prompt"))
    if observed != expected_sha256:
        detail = "prompt sha256 mismatch"
        raise HuntSetupError(detail)


def _check_static_pins(config: PocRevisionCommandConfig) -> None:
    if config.probe.image_id != _IMAGE_ID:
        detail = "image digest does not match pinned OpenEXR probe image"
        raise HuntSetupError(detail)
    if config.probe.harness_sha256 != _HARNESS_SHA256:
        detail = "harness sha256 does not match pinned OpenEXR probe harness"
        raise HuntSetupError(detail)
    if config.target.commit_sha != _TARGET_COMMIT:
        detail = "target commit does not match pinned OpenEXR v3.4.14 source"
        raise HuntSetupError(detail)


def _check_runtime(runtime_python: Path) -> None:
    if not runtime_python.is_file():
        detail = "runtime python does not exist"
        raise HuntSetupError(detail)


def _check_source(source: Path, config: PocRevisionCommandConfig) -> TargetEvidence:
    verified = verify_target_archive(source, _hunt_manifest(config))
    return verified.evidence


def _hunt_manifest(config: PocRevisionCommandConfig) -> HuntManifest:
    return HuntManifest(
        schema_version="dllm-hunt/v1",
        hunt_id="openexr-poc-revision",
        target_id="openexr-v3.4.14",
        repository_url=config.target.repository_url,
        commit_sha=config.target.commit_sha,
        security_policy_url=config.target.security_policy_url,
        security_policy_sha256=config.target.security_policy_sha256,
        prompt=config.prompt.path,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        image_id=config.probe.image_id,
        target_profile="openexr-v3.4",
        sanitizers="address",
        allowed_includes=("OpenEXR/ImfInputFile.h",),
        required_symbols=("LLVMFuzzerTestOneInput",),
        seed_hex=("00",),
        fuzz_seconds=config.revision.probe_seconds,
        max_len=1_048_576,
        rss_limit_mb=config.probe.rss_limit_mb,
        timeout_seconds=config.probe.timeout_seconds,
    )


def build_run_manifest(
    command: VerifiedPocRevisionCommand,
    *,
    report_sha256: str,
    worker_ready: WorkerReadyProvenance | None,
    worker_argv: tuple[str, ...],
) -> RunManifest:
    """Build compact run evidence after the report exists."""
    config = command.config
    worker_started = len(worker_argv) > 0
    return RunManifest(
        schema_version="dllm-poc-revision-run-manifest/v1",
        config_sha256=command.config_sha256,
        prompt_path=command.config.prompt.path,
        prompt_sha256=command.prompt_sha256,
        target=command.target,
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        model_device=config.model.device,
        model_timeout_seconds=config.model.timeout_seconds if worker_started else 0.0,
        model_worker_started=worker_started,
        worker_ready=worker_ready,
        worker_argv=worker_argv,
        worker_argv_sha256=_argv_sha256(worker_argv),
        image_id=config.probe.image_id,
        harness_sha256=config.probe.harness_sha256,
        arm=config.run.arm,
        particle_count=config.revision.particle_count,
        epochs=config.revision.epochs,
        initial_steps=config.revision.initial_steps if worker_started else 0,
        revision_steps=config.revision.revision_steps if worker_started else 0,
        probe_seconds=config.revision.probe_seconds,
        proof_seconds=config.revision.proof_seconds,
        fuzzer_seeds=config.revision.fuzzer_seeds,
        particle_seed_base=config.run.particle_seed_base,
        particle_seeds=_particle_seeds(
            config.run.particle_seed_base, config.revision.particle_count
        ),
        proof_seed=config.run.proof_seed,
        report_sha256=report_sha256,
    )


def report_json_bytes(report: PocRevisionReport) -> bytes:
    """Serialize a PoC revision report exactly as the CLI writes it."""
    return (report.model_dump_json(indent=2) + "\n").encode()


def write_poc_revision_report(path: Path, report: PocRevisionReport) -> str:
    """Atomically write the report and return the exact file-byte SHA-256."""
    payload = report_json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _ = temporary.write_bytes(payload)
    _ = temporary.replace(path)
    return sha256_bytes(payload)


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically persist a compact PoC revision run manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _ = temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    _ = temporary.replace(path)


def _argv_sha256(argv: tuple[str, ...]) -> str:
    return sha256_bytes(json.dumps(argv, separators=(",", ":")).encode())


def _particle_seeds(base: int, count: int) -> tuple[int, ...]:
    return tuple(base + index for index in range(count))
