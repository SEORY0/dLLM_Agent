"""Persistence and coverage evidence helpers for live hunts."""

import hashlib
from pathlib import Path

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import sha256_bytes
from dllm_agent.fuzzing.coverage.runner import CoverageResult
from dllm_agent.fuzzing.runner import FuzzArtifact, FuzzRunResult
from dllm_agent.hunt.models import (
    CoverageCommandEvidence,
    CoverageEvidence,
    CoverageTotalsEvidence,
    HarnessCandidate,
)


def argv_sha256(argv: tuple[str, ...]) -> str:
    """Hash a fixed argument vector with unambiguous separators."""
    return hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()


def persist_artifacts(
    root: Path,
    selected: HarnessCandidate,
    result: FuzzRunResult,
    coverage: CoverageResult,
) -> None:
    """Persist only the fixed allowlist of campaign artifacts without overwriting."""
    try:
        root.mkdir(parents=True, exist_ok=False)
        corpus = root / "corpus"
        crashes = root / "crashes"
        corpus.mkdir()
        crashes.mkdir()
        _ = (root / "selected-harness.cpp").write_text(selected.source, encoding="utf-8")
        _ = (root / "compiler-version.txt").write_bytes(result.compiler_version_summary)
        _ = (root / "fuzzer-version.txt").write_bytes(result.fuzzer_version_summary)
        _ = (root / "build.log").write_bytes(result.build_log)
        _ = (root / "fuzz.log").write_bytes(result.fuzz_log)
        _ = (root / "coverage-build.log").write_bytes(coverage.build_log)
        _ = (root / "coverage-run.log").write_bytes(coverage.run_log)
        _ = (root / "coverage-merge.log").write_bytes(coverage.merge_log)
        _ = (root / "coverage-report.txt").write_bytes(coverage.report_log)
        _write_artifact_group(corpus, result.corpus_artifacts)
        _write_artifact_group(crashes, result.crash_artifacts)
    except OSError as error:
        detail = error.strerror or "could not persist hunt artifacts"
        raise HuntSetupError(detail) from error


def coverage_evidence(result: CoverageResult) -> CoverageEvidence:
    """Normalize coverage output and hash every raw input and argv."""
    totals = None
    if result.coverage is not None:
        totals = CoverageTotalsEvidence(
            regions=result.coverage.regions,
            functions=result.coverage.functions,
            lines=result.coverage.lines,
            branches=result.coverage.branches,
        )
    return CoverageEvidence(
        build_status=result.build_status,
        run_status=result.run_status,
        merge_status=result.merge_status,
        report_status=result.report_status,
        build_log_sha256=sha256_bytes(result.build_log),
        run_log_sha256=sha256_bytes(result.run_log),
        merge_log_sha256=sha256_bytes(result.merge_log),
        report_log_sha256=sha256_bytes(result.report_log),
        totals=totals,
        commands=CoverageCommandEvidence(
            build_argv_sha256=argv_sha256(result.build_argv),
            run_argv_sha256=argv_sha256(result.run_argv),
            merge_argv_sha256=argv_sha256(result.merge_argv),
            report_argv_sha256=argv_sha256(result.report_argv),
        ),
    )


def _write_artifact_group(root: Path, artifacts: tuple[FuzzArtifact, ...]) -> None:
    for artifact in artifacts:
        if Path(artifact.name).name != artifact.name:
            detail = f"unsafe artifact name: {artifact.name}"
            raise HuntSetupError(detail)
        _ = (root / artifact.name).write_bytes(artifact.content)
