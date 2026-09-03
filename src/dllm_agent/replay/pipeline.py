"""End-to-end replay scan and plan orchestration."""

from pathlib import Path

from dllm_agent.core.errors import BoundaryError
from dllm_agent.core.io import parse_json, parse_json_bytes, read_bytes, sha256_bytes
from dllm_agent.core.models import (
    Accounting,
    ExpandedPlan,
    ExperimentConfig,
    Finding,
    Provenance,
    ReplayFixture,
    ScanReport,
)
from dllm_agent.replay.sampler import ReplaySampler
from dllm_agent.replay.scoring import rank_trajectories, rotate_trace_for_qa
from dllm_agent.replay.validator import ValidatorRequest, validate

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRUSTED_TARGET = Path("fixtures/cwe121/vuln.c")
_TRUSTED_FIXTURE = Path("fixtures/replay/stable-diffcoder-cwe121.json")
_TRUSTED_TARGET_SHA256 = "42e55e7371d23312996c73b374e47d2459469e9fc4ef2f0dac592c3b15985144"
_TRUSTED_FIXTURE_SHA256 = "bbd074c8f0110fc17ff3669de9025c72df0b05a8392c209413fa91d4c9de7571"


def scan(target: Path, fixture_path: Path) -> ScanReport:
    """Run the deterministic offline scan pipeline."""
    fixture_boundary = "fixture"
    target, target_bytes = _read_trusted_file(
        target, _TRUSTED_TARGET, _TRUSTED_TARGET_SHA256, "target"
    )
    fixture_path, fixture_bytes = _read_trusted_file(
        fixture_path, _TRUSTED_FIXTURE, _TRUSTED_FIXTURE_SHA256, fixture_boundary
    )
    fixture = parse_json_bytes(fixture_bytes, fixture_path, ReplayFixture, fixture_boundary)
    target_hash = sha256_bytes(target_bytes)
    proof = verify_replay_fixture(fixture, fixture_path, target_hash)

    trajectories = ReplaySampler(fixture).sample()
    ranked = rank_trajectories(trajectories)
    selected_id = ranked[0].candidate_id
    selected = next(item for item in trajectories if item.candidate_id == selected_id)
    finding = Finding(
        candidate_id=selected.candidate_id,
        cwe=selected.cwe,
        span=selected.span,
        explanation=selected.explanation,
    )
    verdict = validate(ValidatorRequest(target.name, target_bytes, finding, proof, 2.0))
    evidence = (
        f"{target.name}:{finding.span.start_line}",
        verdict.sanitizer or "sanitizer evidence absent",
    )
    return ScanReport(
        schema_version="dllm-scan-report/v1",
        finding=finding,
        trajectory_summary=ranked,
        synthetic_trace_rotation=rotate_trace_for_qa(selected),
        validator=verdict,
        evidence=evidence,
        accounting=Accounting(
            candidate_count=len(trajectories),
            total_nfe=sum(item.nfe for item in trajectories),
            validated_candidates=1,
        ),
        provenance=Provenance(
            target_sha256=target_hash,
            fixture_sha256=sha256_bytes(fixture_bytes),
            proof_input_sha256=fixture.proof_input_sha256,
            model_revision=fixture.compatible_revision,
            replay_kind=fixture.provenance_kind,
        ),
    )


def verify_replay_fixture(fixture: ReplayFixture, fixture_path: Path, target_hash: str) -> bytes:
    """Verify the hashes claimed inside a trusted replay fixture."""
    fixture_boundary = "fixture"
    if target_hash != fixture.target_sha256:
        detail = "target SHA-256 does not match"
        raise BoundaryError(fixture_boundary, fixture_path, detail)
    proof = fixture.proof_input.encode("ascii")
    if sha256_bytes(proof) != fixture.proof_input_sha256:
        detail = "proof input SHA-256 does not match"
        raise BoundaryError(fixture_boundary, fixture_path, detail)
    return proof


def _read_trusted_file(
    path: Path, relative_path: Path, expected_sha256: str, boundary: str
) -> tuple[Path, bytes]:
    """Read one immutable, repository-owned allowlisted scan input."""
    try:
        repository_root = _REPOSITORY_ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
        allowed = (repository_root / relative_path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        detail = getattr(error, "strerror", None) or "file does not exist"
        raise BoundaryError(boundary, path, detail) from error
    if not resolved.is_relative_to(repository_root) or resolved != allowed:
        detail = "path is not an allowlisted trusted repository fixture"
        raise BoundaryError(boundary, path, detail)
    data = read_bytes(resolved, boundary)
    if sha256_bytes(data) != expected_sha256:
        detail = "allowlisted file SHA-256 does not match immutable manifest"
        raise BoundaryError(boundary, resolved, detail)
    return resolved, data


def expand_plan(config_path: Path) -> ExpandedPlan:
    """Validate and deterministically expand a checked experiment config."""
    config = parse_json(config_path, ExperimentConfig, "config")
    base = config_path.parent.parent
    target = (base / config.target).resolve()
    fixture = (base / config.fixture).resolve()
    target_bytes = read_bytes(target, "target")
    fixture_bytes = read_bytes(fixture, "fixture")
    _ = parse_json(fixture, ReplayFixture, "fixture")
    return ExpandedPlan(
        schema_version="dllm-plan/v1",
        experiment_id=config.experiment_id,
        candidate_bank_size=config.candidate_bank_size,
        validator_timeout_seconds=config.validator_timeout_seconds,
        expected_cwe=config.expected_cwe,
        target=str(target),
        fixture=str(fixture),
        target_sha256=sha256_bytes(target_bytes),
        fixture_sha256=sha256_bytes(fixture_bytes),
    )
