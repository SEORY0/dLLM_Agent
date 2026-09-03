"""Safe Docker-backed LLVM source coverage evaluation."""

import tempfile
from pathlib import Path

from dllm_agent.fuzzing.coverage.contracts import (
    CommandRunner,
    CoverageRequest,
    CoverageResult,
    CoverageTotals,
)
from dllm_agent.fuzzing.coverage.execution import (
    COMPILER_FLAGS,
    MERGE_TAIL,
    PROFILE_FILE,
    REPORT_TAIL,
    coverage_if_reported,
    docker_prefix,
    log_bytes,
    require_regular_output,
    run_command,
    skip,
    stage_inputs,
    status,
    validate_request,
)

__all__ = [
    "CommandRunner",
    "CoverageRequest",
    "CoverageResult",
    "CoverageTotals",
    "run_coverage",
]


def run_coverage(request: CoverageRequest, runner: CommandRunner = run_command) -> CoverageResult:
    """Compile a harness, replay a frozen corpus, and parse LLVM coverage totals."""
    validate_request(request)
    with tempfile.TemporaryDirectory(prefix="dllm-coverage-") as directory:
        work_root = Path(directory)
        corpus_root = work_root / "corpus"
        corpus_root.mkdir()
        work_root.chmod(0o777)
        corpus_root.chmod(0o777)
        stage_inputs(work_root, corpus_root, request)

        environment = {"LLVM_PROFILE_FILE": PROFILE_FILE}
        prefix = docker_prefix(request, work_root)
        match request.target_profile:
            case "header-only":
                build_argv = (*prefix, "clang++", *COMPILER_FLAGS)
                report_argv = (*prefix, *REPORT_TAIL)
            case "openexr-v3.4":
                build_argv = (*prefix, "/opt/dllm/bin/build-coverage")
                report_argv = (
                    *prefix,
                    "llvm-cov",
                    "report",
                    "/work/coverage_target",
                    "-instr-profile=/work/coverage.profdata",
                    "/opt/openexr-src/src/lib",
                )
        run_argv = (*prefix, "/work/coverage_target", "/work/corpus", "-runs=0")
        merge_argv = (*prefix, *MERGE_TAIL)

        process_timeout = max(request.timeout_seconds, 60.0)
        build = runner(build_argv, environment, process_timeout)
        run = skip(run_argv, "build failed; run skipped")
        merge = skip(merge_argv, "coverage merge skipped")
        report = skip(report_argv, "coverage report skipped")
        if build.returncode == 0:
            run = runner(run_argv, environment, process_timeout)
        if run.returncode == 0:
            require_regular_output(work_root / "coverage.profraw")
            merge = runner(merge_argv, environment, process_timeout)
        if merge.returncode == 0:
            require_regular_output(work_root / "coverage.profdata")
            report = runner(report_argv, environment, process_timeout)

        return CoverageResult(
            build_status=status((), build, "succeeded"),
            run_status=status((build,), run, "completed"),
            merge_status=status((build, run), merge, "succeeded"),
            report_status=status((build, run, merge), report, "succeeded"),
            build_log=log_bytes(build),
            run_log=log_bytes(run),
            merge_log=log_bytes(merge),
            report_log=log_bytes(report),
            build_argv=build_argv,
            run_argv=run_argv,
            merge_argv=merge_argv,
            report_argv=report_argv,
            coverage=coverage_if_reported(report),
        )
