"""End-to-end replay scan and plan orchestration."""

from pathlib import Path

from dllm_agent.errors import BoundaryError
from dllm_agent.io import parse_json, read_bytes, sha256_bytes
from dllm_agent.models import (
    Accounting,
    ExpandedPlan,
    ExperimentConfig,
    Finding,
    Provenance,
    ReplayFixture,
    ScanReport,
)
from dllm_agent.sampler import ReplaySampler
from dllm_agent.scoring import no_fixed_point_permutation, rank_trajectories
from dllm_agent.validator import ValidatorRequest, validate


def scan(target: Path, fixture_path: Path) -> ScanReport:
    """Run the deterministic offline scan pipeline."""
    fixture_boundary = "fixture"
    target_bytes = read_bytes(target, "target")
    fixture_bytes = read_bytes(fixture_path, "fixture")
    fixture = parse_json(fixture_path, ReplayFixture, fixture_boundary)
    target_hash = sha256_bytes(target_bytes)
    if target_hash != fixture.target_sha256:
        detail = "target SHA-256 does not match"
        raise BoundaryError(fixture_boundary, fixture_path, detail)
    proof = fixture.proof_input.encode("ascii")
    if sha256_bytes(proof) != fixture.proof_input_sha256:
        detail = "proof input SHA-256 does not match"
        raise BoundaryError(fixture_boundary, fixture_path, detail)

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
    verdict = validate(ValidatorRequest(target, finding, proof, 2.0))
    evidence = (
        f"{target.name}:{finding.span.start_line}",
        verdict.sanitizer or "sanitizer evidence absent",
    )
    return ScanReport(
        schema_version="dllm-scan-report/v1",
        finding=finding,
        trajectory_summary=ranked,
        permutation_control=no_fixed_point_permutation(selected),
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
