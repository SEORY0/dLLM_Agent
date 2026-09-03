"""Installed command-line interface."""

from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Protocol

import typer
import uvicorn
from rich.console import Console

from dllm_agent.core.errors import BoundaryError, HuntSetupError
from dllm_agent.core.io import parse_json, write_model
from dllm_agent.fuzzing.coverage.runner import run_coverage
from dllm_agent.fuzzing.runner import run_fuzz
from dllm_agent.hunt.pipeline import run_hunt
from dllm_agent.native.pipeline import run_native_hunt
from dllm_agent.replay.pipeline import expand_plan
from dllm_agent.replay.pipeline import scan as run_scan
from dllm_agent.revision.command import (
    POC_REVISION_CLI_ERRORS,
    VerifiedPocRevisionCommand,
    WorkerReadyProvenance,
    build_run_manifest,
    load_poc_revision_command,
    write_poc_revision_report,
    write_run_manifest,
)
from dllm_agent.revision.controller_models import (
    InitialParticleRequest,
    RevisionRequest,
    RevisionResult,
)
from dllm_agent.revision.models import PocRevisionReport
from dllm_agent.runtime.live_sampler import StableDiffCoderProcessSampler
from dllm_agent.targets.models import HuntManifest
from dllm_agent.web.app import WebSettings, create_app

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
_ERROR_CONSOLE = Console(stderr=True)
_DEFAULT_RUNTIME_PYTHON = Path("/home/seory0/.cache/dllm-agent/model-venv/bin/python")
_DEFAULT_CACHE_DIR = Path("/home/seory0/.cache/dllm-agent/huggingface")
_DEFAULT_SOURCE_ROOT = Path("targets/jsoncons")
_DEFAULT_JOBS_ROOT = Path("artifacts/web/jobs")


class _ReadyRecordView(Protocol):
    """Structural view of Task5 worker ready provenance."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def remote_code_sha256(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def load_count(self) -> int: ...


@app.command()
def scan(
    target: Annotated[Path, typer.Argument()],
    fixture: Annotated[Path, typer.Option("--fixture")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Scan a target with a versioned replay fixture."""
    try:
        report = run_scan(target, fixture)
        write_model(output, report)
    except BoundaryError as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command()
def plan(
    config: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Validate and expand an experiment configuration."""
    try:
        expanded = expand_plan(config)
        write_model(output, expanded)
    except BoundaryError as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command("sample")
def sample_live_harnesses(  # noqa: PLR0913, PLR0917
    manifest: Annotated[Path, typer.Argument()],
    runtime_python: Annotated[Path, typer.Option("--runtime-python")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")],
    output: Annotated[Path, typer.Option("--output")],
    count: Annotated[int, typer.Option(min=2, max=8)] = 2,
    seed: Annotated[int, typer.Option(min=0)] = 20_260_819,
    steps: Annotated[int, typer.Option(min=1)] = 256,
    gen_length: Annotated[int, typer.Option("--gen-length", min=4)] = 256,
    block_length: Annotated[int, typer.Option("--block-length", min=1)] = 4,
    temperature: Annotated[float, typer.Option(min=0.0, max=2.0)] = 0.2,
    device: Annotated[str, typer.Option()] = "cuda:0",
) -> None:
    """Generate real Stable-DiffCoder harness candidates and trajectories."""
    try:
        config = parse_json(manifest, HuntManifest, "hunt manifest")
        prompt = _manifest_prompt(manifest, config)
        sampler = StableDiffCoderProcessSampler(
            runtime_python=runtime_python,
            prompt_path=prompt,
            cache_dir=cache_dir,
            count=count,
            seed=seed,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            device=device,
            timeout_seconds=3600.0,
        )
        write_model(output, sampler.sample())
    except (BoundaryError, HuntSetupError) as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command("hunt")
def hunt_live_target(
    manifest: Annotated[Path, typer.Argument()],
    candidate_bank: Annotated[Path, typer.Option("--candidate-bank")],
    source: Annotated[Path, typer.Option("--source")],
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Rank one live candidate and run isolated ASan/libFuzzer validation."""
    try:
        report = run_hunt(
            manifest,
            candidate_bank,
            source,
            artifact_dir=artifact_dir,
        )
        write_model(output, report)
    except (BoundaryError, HuntSetupError, ValueError) as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command("native-hunt")
def native_hunt_live_target(  # noqa: PLR0913, PLR0917
    manifest: Annotated[Path, typer.Argument()],
    candidate_bank: Annotated[Path, typer.Option("--candidate-bank")],
    controller: Annotated[Path, typer.Option("--controller")],
    source: Annotated[Path, typer.Option("--source")],
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Probe live candidates, select one, and run independent proof fuzzing."""
    try:
        report = run_native_hunt(
            manifest,
            candidate_bank,
            controller,
            source,
            fuzz_runner=run_fuzz,
            coverage_runner=run_coverage,
            artifact_dir=artifact_dir,
        )
        write_model(output, report)
    except (BoundaryError, HuntSetupError, ValueError) as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command("poc-revision")
def poc_revision_live_target(  # noqa: PLR0913, PLR0917
    config: Annotated[Path, typer.Option("--config")],
    source: Annotated[Path, typer.Option("--source")],
    runtime_python: Annotated[Path, typer.Option("--runtime-python")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir")],
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run live feedback-conditioned OpenEXR PoC revision."""
    try:
        report, command, worker_ready, model_argv = run_poc_revision_command(
            config,
            source,
            runtime_python,
            cache_dir=cache_dir,
            artifact_dir=artifact_dir,
        )
        report_sha256 = write_poc_revision_report(output, report)
        run_manifest = build_run_manifest(
            command,
            report_sha256=report_sha256,
            worker_ready=worker_ready,
            worker_argv=model_argv,
        )
        write_run_manifest(artifact_dir / "run-manifest.json", run_manifest)
    except POC_REVISION_CLI_ERRORS as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command("web")
def serve_web_console(  # noqa: PLR0913, PLR0917
    host: Annotated[str, typer.Option(help="Loopback address only.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65_535)] = 8080,
    runtime_python: Annotated[
        Path,
        typer.Option("--runtime-python", help="Python executable in the Stable-DiffCoder runtime."),
    ] = _DEFAULT_RUNTIME_PYTHON,
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Pinned Hugging Face cache directory."),
    ] = _DEFAULT_CACHE_DIR,
    source_root: Annotated[
        Path,
        typer.Option("--source-root", help="Pinned target checkout used by hunt jobs."),
    ] = _DEFAULT_SOURCE_ROOT,
    jobs_root: Annotated[
        Path,
        typer.Option("--jobs-root", help="Server-owned job output directory."),
    ] = _DEFAULT_JOBS_ROOT,
) -> None:
    """Serve the repository-bound dLLM Hunt Console on loopback."""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        _ERROR_CONSOLE.print("[red]web console must bind to a loopback address[/red]")
        raise typer.Exit(code=2)

    web_app = create_app(
        WebSettings(
            workspace=Path.cwd(),
            runtime_python=runtime_python,
            cache_dir=cache_dir,
            source_root=source_root,
            jobs_root=jobs_root,
        )
    )
    uvicorn.run(web_app, host=host, port=port, access_log=False)


def _manifest_prompt(manifest: Path, config: HuntManifest) -> Path:
    base = manifest.resolve().parent.parent
    prompt = (base / config.prompt).resolve()
    if not prompt.is_relative_to(base):
        detail = "model prompt escapes the manifest repository"
        raise HuntSetupError(detail)
    return prompt


def run_poc_revision_command(
    config_path: Path,
    source: Path,
    runtime_python: Path,
    *,
    cache_dir: Path,
    artifact_dir: Path,
) -> tuple[
    PocRevisionReport,
    VerifiedPocRevisionCommand,
    WorkerReadyProvenance | None,
    tuple[str, ...],
]:
    """Run the live PoC revision command with production adapters."""
    command = load_poc_revision_command(config_path, source, runtime_python)

    from dllm_agent.revision.pipeline import run_poc_revision  # noqa: PLC0415
    from dllm_agent.runtime.revision_client import RevisionWorkerClient  # noqa: PLC0415
    from dllm_agent.runtime.revision_process import revision_worker_command  # noqa: PLC0415
    from dllm_agent.targets.openexr.probe import prepare_openexr_probe  # noqa: PLC0415
    from dllm_agent.targets.openexr.proof import (  # noqa: PLC0415
        OpenExrProofRunner,
        ProofBoundaryError,
    )

    try:
        with ExitStack() as stack:
            probe = stack.enter_context(prepare_openexr_probe(command.config.probe))
            if command.config.run.arm == "libfuzzer-only":
                sampler = _UnreachableRevisionSampler()
                ready = None
                model_argv: tuple[str, ...] = ()
            else:
                model_argv = revision_worker_command(
                    runtime_python,
                    cache_dir,
                    command.config.model.device,
                    command.prompt_path,
                    command.prompt_sha256,
                )
                worker = stack.enter_context(
                    RevisionWorkerClient(
                        runtime_python=runtime_python,
                        cache_dir=cache_dir,
                        prompt_path=command.prompt_path,
                        prompt_sha256=command.prompt_sha256,
                        device=command.config.model.device,
                        timeout_seconds=command.config.model.timeout_seconds,
                    )
                )
                ready = _worker_ready(worker.ready)
                sampler = worker
            report = run_poc_revision(
                command.config.revision,
                command.config.run,
                sampler,
                probe,
                OpenExrProofRunner(),
                artifact_dir,
            )
    except ProofBoundaryError as error:
        raise HuntSetupError(str(error)) from error
    return report, command, ready, model_argv


class _UnreachableRevisionSampler:
    """Sampler adapter that must never be called by the libfuzzer-only arm."""

    def initial(self, request: InitialParticleRequest) -> RevisionResult:
        del request
        detail = "libfuzzer-only arm must not request model initialization"
        raise HuntSetupError(detail)

    def revise(self, request: RevisionRequest) -> RevisionResult:
        del request
        detail = "libfuzzer-only arm must not request model revision"
        raise HuntSetupError(detail)


def _worker_ready(ready: _ReadyRecordView | None) -> WorkerReadyProvenance:
    if ready is None:
        detail = "revision worker ready provenance missing"
        raise HuntSetupError(detail)
    if ready.model_id != "ByteDance-Seed/Stable-DiffCoder-8B-Instruct":
        detail = "revision worker model id does not match command pin"
        raise HuntSetupError(detail)
    if ready.revision != "10cdaf9b486f1cf0273ad968459dbe2d21a1482a":
        detail = "revision worker model revision does not match command pin"
        raise HuntSetupError(detail)
    return WorkerReadyProvenance(
        model_id="ByteDance-Seed/Stable-DiffCoder-8B-Instruct",
        revision="10cdaf9b486f1cf0273ad968459dbe2d21a1482a",
        remote_code_sha256=ready.remote_code_sha256,
        device=ready.device,
        load_count=ready.load_count,
    )
