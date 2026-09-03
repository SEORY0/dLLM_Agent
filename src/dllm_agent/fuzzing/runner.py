"""Safe Docker-backed libFuzzer execution."""

import tempfile
from pathlib import Path

from dllm_agent.fuzzing.contracts import (
    BuildStatus,
    CommandRunner,
    FuzzArtifact,
    FuzzRunRequest,
    FuzzRunResult,
    FuzzStatus,
)
from dllm_agent.fuzzing.execution import (
    build_status,
    collect_corpus_artifacts,
    collect_crash_artifacts,
    compiler_flags,
    docker_prefix,
    fuzz_environment,
    fuzz_status,
    run_command,
    run_fuzzer_if_built,
    stage_inputs,
    validate_request,
)

__all__ = [
    "BuildStatus",
    "CommandRunner",
    "FuzzArtifact",
    "FuzzRunRequest",
    "FuzzRunResult",
    "FuzzStatus",
    "run_fuzz",
]


def run_fuzz(request: FuzzRunRequest, runner: CommandRunner = run_command) -> FuzzRunResult:
    """Compile a generated harness and run bounded libFuzzer inside Docker."""
    validate_request(request)
    with tempfile.TemporaryDirectory(prefix="dllm-fuzz-") as directory:
        work_root = Path(directory)
        corpus_root = work_root / "corpus"
        artifact_root = work_root / "artifacts"
        corpus_root.mkdir()
        artifact_root.mkdir()
        work_root.chmod(0o777)
        corpus_root.chmod(0o777)
        artifact_root.chmod(0o777)
        stage_inputs(work_root, corpus_root, request)

        environment = fuzz_environment()
        prefix = docker_prefix(request, work_root)
        compiler_version_argv = (*prefix, "clang++", "--version")
        fuzzer_version_argv = (
            *prefix,
            "clang++",
            "-fsanitize=fuzzer",
            "-###",
        )
        match request.target_profile:
            case "header-only":
                build_argv = (*prefix, "clang++", *compiler_flags(request.sanitizers))
            case "openexr-v3.4":
                build_argv = (*prefix, "/opt/dllm/bin/build-fuzz")
        fuzz_argv = (
            *prefix,
            "/work/fuzz_target",
            "/work/corpus",
            "-artifact_prefix=/work/artifacts/",
            f"-max_total_time={request.duration_seconds}",
            f"-timeout={int(request.timeout_seconds)}",
            f"-rss_limit_mb={request.rss_limit_mb}",
            f"-max_len={request.max_len}",
            f"-seed={request.fuzzer_seed}",
            "-print_final_stats=1",
        )

        setup_timeout = max(request.timeout_seconds, 30.0)
        compiler_version = runner(compiler_version_argv, environment, setup_timeout)
        fuzzer_version = runner(fuzzer_version_argv, environment, setup_timeout)
        build = runner(build_argv, environment, setup_timeout)
        fuzz = run_fuzzer_if_built(build, fuzz_argv, environment, request, runner)
        return FuzzRunResult(
            compiler_version_summary=compiler_version.stdout + compiler_version.stderr,
            fuzzer_version_summary=fuzzer_version.stdout + fuzzer_version.stderr,
            build_status=build_status(build),
            build_log=build.stdout + build.stderr,
            fuzz_status=fuzz_status(build, fuzz),
            fuzz_log=fuzz.stdout + fuzz.stderr,
            corpus_artifacts=collect_corpus_artifacts(corpus_root),
            crash_artifacts=collect_crash_artifacts(artifact_root),
            compiler_version_argv=compiler_version_argv,
            fuzzer_version_argv=fuzzer_version_argv,
            build_argv=build_argv,
            fuzz_argv=fuzz_argv,
            sanitizers=request.sanitizers,
        )
