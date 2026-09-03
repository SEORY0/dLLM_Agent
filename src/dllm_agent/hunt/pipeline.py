"""End-to-end live candidate selection and sandboxed fuzz orchestration."""

import re
from collections.abc import Callable
from pathlib import Path

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import parse_json_bytes, read_bytes, sha256_bytes
from dllm_agent.fuzzing.coverage.runner import CoverageRequest, CoverageResult, run_coverage
from dllm_agent.fuzzing.runner import FuzzArtifact, FuzzRunRequest, FuzzRunResult, run_fuzz
from dllm_agent.hunt.artifacts import argv_sha256, coverage_evidence, persist_artifacts
from dllm_agent.hunt.models import (
    ArtifactEvidence,
    CandidateBank,
    CandidateRejection,
    CommandEvidence,
    FuzzEvidence,
    HarnessCandidate,
    HuntReport,
    SandboxEvidence,
)
from dllm_agent.hunt.scoring import rank_harness_candidates
from dllm_agent.hunt.validation import validate_candidate_bank, validate_harness_source
from dllm_agent.targets.models import HuntManifest, TargetEvidence
from dllm_agent.targets.staging import staged_target, verify_target_archive

_FUZZ_STATS_PREFIX = rb"#(?P<units>\d+).*?\bcov:\s*(?P<cov>\d+).*?\bft:\s*(?P<ft>\d+)"
_FUZZ_STATS_SUFFIX = rb".*?\bcorp:\s*(?P<corpus>\d+)/.*?\bexec/s:\s*(?P<execs>\d+)"
_FUZZ_STATS = re.compile(_FUZZ_STATS_PREFIX + _FUZZ_STATS_SUFFIX)
_FINAL_UNITS = re.compile(rb"stat::number_of_executed_units:\s*(?P<value>\d+)")
_FINAL_EXECS = re.compile(rb"stat::average_exec_per_sec:\s*(?P<value>\d+)")


def run_hunt(  # noqa: PLR0913
    manifest_path: Path,
    candidate_bank_path: Path,
    source_root: Path,
    *,
    fuzz_runner: Callable[[FuzzRunRequest], FuzzRunResult] = run_fuzz,
    coverage_runner: Callable[[CoverageRequest], CoverageResult] = run_coverage,
    artifact_dir: Path | None = None,
    selected_candidate_id: str | None = None,
    fuzz_seconds: int | None = None,
) -> HuntReport:
    """Select one live dLLM harness and validate it in the Docker boundary."""
    manifest_bytes = read_bytes(manifest_path, "hunt manifest")
    manifest = parse_json_bytes(manifest_bytes, manifest_path, HuntManifest, "hunt manifest")
    prompt_path = resolve_prompt(manifest_path, manifest.prompt)
    prompt = read_bytes(prompt_path, "model prompt")
    bank_bytes = read_bytes(candidate_bank_path, "candidate bank")
    bank = parse_json_bytes(bank_bytes, candidate_bank_path, CandidateBank, "candidate bank")
    validate_candidate_bank(bank, prompt, candidate_bank_path)
    if bank.model.model_id != manifest.model_id or bank.model.revision != manifest.model_revision:
        detail = "candidate bank model does not match hunt manifest"
        raise HuntSetupError(detail)
    verified_target = verify_target_archive(source_root, manifest)
    target = verified_target.evidence
    valid, rejected = filter_candidates(bank, manifest)
    if not valid:
        detail = "all generated harness candidates were rejected"
        raise HuntSetupError(detail)
    ranked = rank_harness_candidates(valid, manifest.required_symbols)
    selected_id = ranked[0].candidate_id if selected_candidate_id is None else selected_candidate_id
    selected = next(
        (candidate for candidate in valid if candidate.candidate_id == selected_id), None
    )
    if selected is None:
        detail = f"selected candidate is not statically valid: {selected_id}"
        raise HuntSetupError(detail)
    duration_seconds = manifest.fuzz_seconds if fuzz_seconds is None else fuzz_seconds
    if duration_seconds <= 0:
        detail = "fuzz duration override must be positive"
        raise HuntSetupError(detail)
    with staged_target(verified_target) as staged_source:
        request = FuzzRunRequest(
            image_id=manifest.image_id,
            source_root=staged_source,
            harness_bytes=selected.source.encode(),
            corpus=tuple(bytes.fromhex(seed) for seed in manifest.seed_hex),
            duration_seconds=duration_seconds,
            max_len=manifest.max_len,
            rss_limit_mb=manifest.rss_limit_mb,
            timeout_seconds=float(manifest.timeout_seconds),
            sanitizers=manifest.sanitizers,
            target_profile=manifest.target_profile,
        )
        result = fuzz_runner(request)
        coverage_corpus = tuple(item.content for item in result.corpus_artifacts) or request.corpus
        coverage_result = coverage_runner(
            CoverageRequest(
                image_id=manifest.image_id,
                source_root=request.source_root,
                harness_bytes=request.harness_bytes,
                corpus=coverage_corpus,
                timeout_seconds=max(float(manifest.timeout_seconds), 60.0),
                target_profile=manifest.target_profile,
            )
        )
    if artifact_dir is not None:
        persist_artifacts(artifact_dir, selected, result, coverage_result)
    return HuntReport(
        schema_version="dllm-hunt-report/v1",
        hunt_id=manifest.hunt_id,
        target=target,
        model=bank.model,
        prompt_sha256=sha256_bytes(prompt),
        candidate_bank_sha256=sha256_bytes(bank_bytes),
        candidate_scores=ranked,
        rejected_candidates=rejected,
        selected_candidate_id=selected.candidate_id,
        selected_harness_sha256=selected.source_sha256,
        sandbox=SandboxEvidence(
            image_id=manifest.image_id,
            network="none",
            read_only_root=True,
            capabilities="none",
            no_new_privileges=True,
            user="65534:65534",
        ),
        fuzz=fuzz_evidence(result),
        coverage=coverage_evidence(coverage_result),
    )


def verify_target(source_root: Path, manifest: HuntManifest) -> TargetEvidence:
    """Verify exact clean Git bytes and disclosure policy before Docker mounting."""
    return verify_target_archive(source_root, manifest).evidence


def filter_candidates(
    bank: CandidateBank, manifest: HuntManifest
) -> tuple[tuple[HarnessCandidate, ...], tuple[CandidateRejection, ...]]:
    """Partition generated candidates through the static harness boundary."""
    valid: list[HarnessCandidate] = []
    rejected: list[CandidateRejection] = []
    for candidate in bank.candidates:
        try:
            validate_harness_source(
                candidate,
                allowed_includes=manifest.allowed_includes,
                required_symbols=manifest.required_symbols,
            )
        except HuntSetupError as error:
            rejected.append(
                CandidateRejection(candidate_id=candidate.candidate_id, reason=error.detail)
            )
        else:
            valid.append(candidate)
    return tuple(valid), tuple(rejected)


def fuzz_evidence(result: FuzzRunResult) -> FuzzEvidence:
    """Normalize one fuzz result into bounded hash-carrying report evidence."""
    sanitizer = _sanitizer(result.fuzz_log)
    oracle = "not_observed"
    if result.crash_artifacts:
        oracle = "proved" if sanitizer is not None else "crash_without_sanitizer"
    stats = tuple(_FUZZ_STATS.finditer(result.fuzz_log))
    last = stats[-1] if stats else None
    return FuzzEvidence(
        requested_sanitizers=result.sanitizers,
        build_status=result.build_status,
        fuzz_status=result.fuzz_status,
        oracle_status=oracle,
        compiler_version_sha256=sha256_bytes(result.compiler_version_summary),
        fuzzer_version_sha256=sha256_bytes(result.fuzzer_version_summary),
        build_log_sha256=sha256_bytes(result.build_log),
        fuzz_log_sha256=sha256_bytes(result.fuzz_log),
        executed_units=_final_stat(result.fuzz_log, _FINAL_UNITS, _stat(last, "units")),
        edge_coverage=_stat(last, "cov"),
        feature_coverage=_stat(last, "ft"),
        corpus_count=_stat(last, "corpus"),
        executions_per_second=_final_stat(result.fuzz_log, _FINAL_EXECS, _stat(last, "execs")),
        sanitizer=sanitizer,
        corpus_artifacts=_artifact_evidence(result.corpus_artifacts),
        crash_artifacts=_artifact_evidence(result.crash_artifacts),
        commands=CommandEvidence(
            compiler_version_argv_sha256=argv_sha256(result.compiler_version_argv),
            fuzzer_version_argv_sha256=argv_sha256(result.fuzzer_version_argv),
            build_argv_sha256=argv_sha256(result.build_argv),
            fuzz_argv_sha256=argv_sha256(result.fuzz_argv),
        ),
    )


def resolve_prompt(manifest_path: Path, relative: str) -> Path:
    """Resolve a manifest prompt without permitting repository escape."""
    base = manifest_path.resolve().parent.parent
    prompt = (base / relative).resolve()
    if not prompt.is_relative_to(base):
        detail = "model prompt escapes the manifest repository"
        raise HuntSetupError(detail)
    return prompt


def _artifact_evidence(artifacts: tuple[FuzzArtifact, ...]) -> tuple[ArtifactEvidence, ...]:
    return tuple(
        ArtifactEvidence(name=item.name, sha256=sha256_bytes(item.content), size=len(item.content))
        for item in artifacts
    )


def _sanitizer(log: bytes) -> str | None:
    if b"ERROR: AddressSanitizer" in log:
        return "AddressSanitizer"
    if b"UndefinedBehaviorSanitizer" in log or b"runtime error:" in log:
        return "UndefinedBehaviorSanitizer"
    return None


def _stat(match: re.Match[bytes] | None, name: str) -> int | None:
    return int(match.group(name)) if match is not None else None


def _final_stat(log: bytes, pattern: re.Pattern[bytes], fallback: int | None) -> int | None:
    matches = tuple(pattern.finditer(log))
    return int(matches[-1].group("value")) if matches else fallback
