"""Run the frozen 3-target x 5-arm Stable-DiffCoder comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

if __package__ in {None, ""}:
    _REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_REPO / "src"))

from dllm_agent.experiments.multitarget.budget import TARGETS, ExperimentSpec, Target
from dllm_agent.experiments.multitarget.campaign_coordinator import (
    PreparedTreatments,
    TreatmentPreparationConfig,
    generate_shared_initial_banks,
    prepare_treatments,
)
from dllm_agent.experiments.multitarget.campaign_core import CampaignDurations
from dllm_agent.experiments.multitarget.campaign_evidence import (
    finished_cell_manifest,
    launched_cell_manifest,
    probe_batch_manifest,
)
from dllm_agent.experiments.multitarget.campaign_execution import (
    CampaignExecutionPaths,
    FinishedCell,
    LaunchedCell,
    launch_main_campaigns,
    wait_main_campaigns,
)
from dllm_agent.experiments.multitarget.campaign_recording import ExperimentRecorder
from dllm_agent.experiments.multitarget.container_monitor import supervisor_exited_cleanly
from dllm_agent.experiments.multitarget.docker_runtime import runtime_profile
from dllm_agent.experiments.multitarget.ledger import artifact_evidence, validate_ledger
from dllm_agent.experiments.multitarget.model_adapter import StableDiffCoderSlotModel
from dllm_agent.experiments.multitarget.probe_runner import (
    ProbeRunRequest,
    ProbeRunResult,
    run_candidate_probe,
)
from dllm_agent.experiments.multitarget.proof import (
    ProofReplayRequest,
    ProofReplayResult,
    prove_input,
)
from dllm_agent.experiments.multitarget.schedule import CampaignSchedule, build_campaign_schedule
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.stable_diffcoder_generation import load_model
from tools.stable_diffcoder_payloads import (
    MODEL_ID,
    MODEL_REVISION,
    REMOTE_CODE_SHA256,
    RunnerConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dllm_agent.experiments.multitarget.bank_generation import GeneratedInitialBank
    from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
    from dllm_agent.experiments.multitarget.slot_sampler import SlotTokenizer

SEARCH_IMAGES: Final[dict[Target, str]] = {
    "libpng": "sha256:e304abee1c0d1a8366e10110b6267701eff4b761cc023ee882ebc7d46f74cf41",
    "libtiff": "sha256:6c98e35375e446c9ea08e2936f1470c1e3abf8dfcc2431d4a9c9fe47257e4894",
    "libxml2": "sha256:76cd8aab328b4cd7cd4bf143448b6c1719ba821bd94a46567dedad8db7eb1046",
}
PROOF_IMAGES: Final[dict[Target, str]] = {
    "libpng": "sha256:32dc1104f9397670c9bc87ebfc72a75a2234ced77eea75435352f6d7c087c965",
    "libtiff": "sha256:79c3496b95953fcb6004d6e2bbcdd1d9c02a32772cbd544a803c3a44f4fc9ee4",
    "libxml2": "sha256:edaba3bfc994869e79a5cac5bfca4d766fdd0a80797a08ce3c6d4993b58ae720",
}


@dataclass(frozen=True, slots=True)
class RunArguments:
    """Typed command-line inputs for one immutable live run."""

    experiment_root: Path
    smoke_root: Path
    magma_root: Path
    model_cache: Path
    device: str
    smoke_probe_seconds: int
    smoke_main_seconds: int


class CudaRuntime(Protocol):
    """Minimal dynamic torch.cuda surface used after model preparation."""

    def is_available(self) -> bool:
        """Return whether CUDA is available."""
        ...

    def empty_cache(self) -> None:
        """Release unused allocator cache blocks."""
        ...


class TorchRuntime(Protocol):
    """Minimal dynamically imported torch module surface."""

    cuda: CudaRuntime


def main(arguments: Sequence[str] | None = None) -> int:  # noqa: PLR0915
    """Run real smoke, full search, independent proof, and Markdown reporting."""
    args = parse_arguments(arguments)
    repo_root = Path(__file__).resolve().parents[1]
    full_root = args.experiment_root.resolve()
    smoke_root = args.smoke_root.resolve()
    magma_root = args.magma_root.resolve()
    model_cache = args.model_cache.resolve()
    validate_roots(full_root, smoke_root, magma_root, model_cache)
    smoke_root.mkdir(parents=True)
    full_root.mkdir(parents=True, exist_ok=True)
    full_recorder = ExperimentRecorder(full_root)
    smoke_recorder = ExperimentRecorder(smoke_root)
    schedule = build_campaign_schedule(ExperimentSpec(), available_cpus=40, image_ids=SEARCH_IMAGES)
    _record_preflight(full_recorder, repo_root, schedule, args)
    _record_schedule(
        smoke_recorder,
        schedule,
        CampaignDurations.smoke(
            probe_seconds=args.smoke_probe_seconds,
            main_seconds=args.smoke_main_seconds,
        ),
        "qualification-smoke",
    )

    _ = full_recorder.event("model-load-started", {"device": args.device, "model_id": MODEL_ID})
    load_started = time.perf_counter()
    tokenizer, raw_model, device = load_model(
        RunnerConfig(
            prompt=repo_root / "prompts/multitarget/libpng-slot-particle.md",
            output=full_root / "unused-model-output.json",
            count=2,
            seed=0,
            steps=8,
            gen_length=8,
            block_length=4,
            temperature=0.0,
            cache_dir=model_cache,
            device=args.device,
        )
    )
    load_seconds = time.perf_counter() - load_started
    _record_model(full_recorder, raw_model, device, load_seconds)
    slot_model = StableDiffCoderSlotModel(model=raw_model, device=device, block_size=64)

    _ = full_recorder.event("initial-bank-generation-started", {"target_count": 3})
    generation_started = time.perf_counter()
    initial_banks = generate_shared_initial_banks(
        schedule=schedule,
        repo_root=repo_root,
        tokenizer=tokenizer,
        model=slot_model,
    )
    generation_seconds = time.perf_counter() - generation_started
    bank_evidence = _record_initial_banks(full_recorder, initial_banks, generation_seconds)
    _ = smoke_recorder.event(
        "shared-initial-banks-reused",
        {
            "physical_owner": str(full_root),
            "artifacts": {target: evidence.to_json() for target, evidence in bank_evidence.items()},
        },
    )

    smoke_durations = CampaignDurations.smoke(
        probe_seconds=args.smoke_probe_seconds,
        main_seconds=args.smoke_main_seconds,
    )
    smoke_launched = _prepare_and_launch(
        root=smoke_root,
        recorder=smoke_recorder,
        schedule=schedule,
        durations=smoke_durations,
        run_label=run_label("smoke", smoke_root),
        initial_banks=initial_banks,
        tokenizer=tokenizer,
        model=slot_model,
        magma_root=magma_root,
    )
    smoke_finished = _finish_campaign(smoke_root, smoke_recorder, smoke_launched)
    _require_clean(smoke_finished, phase="qualification smoke")
    _ = smoke_recorder.event("qualification-smoke-passed", {"cell_count": 15})

    full_durations = CampaignDurations.full()
    _record_schedule(full_recorder, schedule, full_durations, "full-24h")
    full_launched = _prepare_and_launch(
        root=full_root,
        recorder=full_recorder,
        schedule=schedule,
        durations=full_durations,
        run_label=run_label("full", full_root),
        initial_banks=initial_banks,
        tokenizer=tokenizer,
        model=slot_model,
        magma_root=magma_root,
    )
    del slot_model, raw_model
    _release_cuda_cache()
    full_finished = _finish_campaign(full_root, full_recorder, full_launched)
    _require_clean(full_finished, phase="full campaign")

    proofs = _prove_findings(full_root, full_recorder, full_finished)
    report = _render_report(schedule, full_finished, proofs)
    report_path = full_root / "results" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(report, encoding="utf-8")
    _ = full_recorder.record_existing(
        event_type="markdown-report-written",
        artifact_path=report_path,
        payload={"format": "markdown"},
    )
    docs_path = repo_root / "docs" / "multitarget-24h-results-20260822.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    _ = docs_path.write_text(report, encoding="utf-8")
    _ = full_recorder.record_existing(
        event_type="repository-report-copied",
        artifact_path=report_path,
        payload={"repository_path": str(docs_path.relative_to(repo_root))},
    )
    validation = validate_ledger(full_root / "events.jsonl")
    _sidecar_path, tree_hash = write_evidence_tree_sidecar(full_root)
    _ = sys.stdout.write(f"completed cells={len(full_finished)} proofs={len(proofs)}\n")
    _ = sys.stdout.write(f"ledger_stream_sha256={validation.final_stream_sha256}\n")
    _ = sys.stdout.write(f"evidence_tree_sha256={tree_hash}\n")
    return 0


def _prepare_and_launch(  # noqa: PLR0913 - explicit live campaign boundaries.
    *,
    root: Path,
    recorder: ExperimentRecorder,
    schedule: CampaignSchedule,
    durations: CampaignDurations,
    run_label: str,
    initial_banks: Mapping[Target, GeneratedInitialBank],
    tokenizer: SlotTokenizer,
    model: StableDiffCoderSlotModel,
    magma_root: Path,
) -> tuple[LaunchedCell, ...]:
    _ = recorder.event(
        "treatment-preparation-started",
        {"duration_label": durations.label, "probe_seconds": durations.probe_seconds},
    )
    prepared = prepare_treatments(
        initial_banks=initial_banks,
        schedule=schedule,
        config=TreatmentPreparationConfig(
            experiment_root=root,
            durations=durations,
            run_label=run_label,
        ),
        tokenizer=tokenizer,
        model=model,
        run_probe=_run_probe,
    )
    _record_treatments(recorder, prepared)
    _ = recorder.event("main-launch-started", {"cell_count": 15, "run_label": run_label})
    launched = launch_main_campaigns(
        schedule=schedule,
        durations=durations,
        paths=CampaignExecutionPaths(experiment_root=root, magma_root=magma_root),
        run_label=run_label,
        initial_inputs=prepared.initial_inputs,
        revision_inputs=prepared.revision_inputs,
    )
    _ = recorder.record_json(
        event_type="main-campaigns-launched",
        relative_path=Path("manifests/main-launches.json"),
        value=tuple(launched_cell_manifest(root, item) for item in launched),
        payload={"cell_count": len(launched)},
    )
    return launched


def _finish_campaign(
    root: Path,
    recorder: ExperimentRecorder,
    launched: tuple[LaunchedCell, ...],
) -> tuple[FinishedCell, ...]:
    finished = wait_main_campaigns(launched, max_workers=15)
    for item in finished:
        _ = recorder.record_json(
            event_type="main-cell-finished",
            relative_path=Path(
                f"results/cells/{item.cell.target}/{item.cell.arm}/rep-{item.cell.repetition}.json"
            ),
            value=finished_cell_manifest(root, item),
            payload={
                "arm": item.cell.arm,
                "exit_code": item.result.exit_code,
                "target": item.cell.target,
            },
        )
    return finished


def _run_probe(request: ProbeRunRequest) -> ProbeRunResult:
    return run_candidate_probe(request)


def _record_preflight(
    recorder: ExperimentRecorder,
    repo_root: Path,
    schedule: CampaignSchedule,
    args: RunArguments,
) -> None:
    _record_schedule(recorder, schedule, CampaignDurations.full(), "frozen-full-24h")
    _ = recorder.record_json(
        event_type="source-manifest-frozen",
        relative_path=Path("provenance/source-manifest.json"),
        value=source_manifest(repo_root),
        payload={"git_head": _git_output(repo_root, ("rev-parse", "HEAD")).strip()},
    )
    _ = recorder.record_json(
        event_type="runner-arguments-frozen",
        relative_path=Path("provenance/runner-arguments.json"),
        value=asdict(args),
        payload={"runner": "tools/run_multitarget_24h.py"},
    )


def _record_schedule(
    recorder: ExperimentRecorder,
    schedule: CampaignSchedule,
    durations: CampaignDurations,
    run_kind: str,
) -> None:
    _ = recorder.record_json(
        event_type="campaign-schedule-frozen",
        relative_path=Path(f"manifests/schedule-{run_kind}.json"),
        value={"durations": durations, "schedule": schedule.to_manifest()},
        payload={
            "is_full": durations.is_full,
            "manifest_sha256": schedule.manifest_hash,
            "run_kind": run_kind,
        },
    )


def _record_model(
    recorder: ExperimentRecorder,
    model: object,
    device: str,
    load_seconds: float,
) -> None:
    source_path = inspect.getsourcefile(type(model))
    if source_path is None:
        msg = "could not locate loaded Stable-DiffCoder remote code"
        raise RuntimeError(msg)
    loaded_hash = artifact_evidence(Path(source_path)).sha256
    if loaded_hash != REMOTE_CODE_SHA256:
        msg = "loaded Stable-DiffCoder remote code hash mismatch"
        raise RuntimeError(msg)
    _ = recorder.record_json(
        event_type="model-loaded-resident",
        relative_path=Path("provenance/model-loaded.json"),
        value={
            "device": device,
            "load_wall_seconds": load_seconds,
            "model_id": MODEL_ID,
            "remote_code_sha256": loaded_hash,
            "revision": MODEL_REVISION,
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
        },
        payload={"device": device, "remote_code_sha256": loaded_hash},
    )


def _record_initial_banks(
    recorder: ExperimentRecorder,
    banks: Mapping[Target, GeneratedInitialBank],
    generation_seconds: float,
) -> dict[Target, ArtifactEvidence]:
    recorded: dict[Target, ArtifactEvidence] = {}
    for target in TARGETS:
        result = recorder.record_json(
            event_type="initial-bank-generated",
            relative_path=Path(f"banks/{target}/initial-bank.json"),
            value=banks[target],
            payload={
                "generation_wall_seconds_total": generation_seconds,
                "raw_count": banks[target].bank.raw_count,
                "target": target,
                "unique_count": banks[target].bank.unique_count,
            },
        )
        recorded[target] = result.artifact
    return recorded


def _record_treatments(recorder: ExperimentRecorder, prepared: PreparedTreatments) -> None:
    root = recorder.experiment_root
    for target in TARGETS:
        _ = recorder.record_json(
            event_type="initial-probe-batch-finished",
            relative_path=Path(f"probes/initial/{target}/batch.json"),
            value=probe_batch_manifest(root, prepared.initial_probe_batches[target]),
            payload={"physical_runs": 16, "target": target},
        )
        for arm in ("feedback-no-remask", "selective-revision", "pseudo-feedback"):
            _ = recorder.record_json(
                event_type="revision-bank-generated",
                relative_path=Path(f"banks/{target}/{arm}/revision-bank.json"),
                value=prepared.revised_banks[target][arm],
                payload={"arm": arm, "target": target},
            )
            _ = recorder.record_json(
                event_type="revision-probe-batch-finished",
                relative_path=Path(f"probes/revision/{target}/{arm}/batch.json"),
                value=probe_batch_manifest(root, prepared.revision_probe_batches[target][arm]),
                payload={"arm": arm, "physical_runs": 16, "target": target},
            )
    _ = recorder.record_json(
        event_type="treatment-populations-frozen",
        relative_path=Path("manifests/treatment-populations.json"),
        value={
            "initial": prepared.initial_inputs,
            "revisions": prepared.revision_inputs,
        },
        payload={"target_count": 3},
    )


def _prove_findings(
    root: Path,
    recorder: ExperimentRecorder,
    finished: tuple[FinishedCell, ...],
) -> dict[tuple[Target, str], ProofReplayResult]:
    owners: dict[tuple[Target, str], tuple[Path, FinishedCell]] = {}
    for item in finished:
        for finding in item.result.findings:
            _ = owners.setdefault(
                (item.cell.target, finding.sha256),
                (Path(finding.path), item),
            )
    proofs: dict[tuple[Target, str], ProofReplayResult] = {}
    for index, ((target, digest), (source, owner)) in enumerate(sorted(owners.items())):
        input_path = root / "proof" / "inputs" / target / digest
        input_path.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.exists():
            _ = shutil.copy2(source, input_path)
        request = ProofReplayRequest(
            profile=runtime_profile(target),
            image_id=PROOF_IMAGES[target],
            input_path=input_path,
            input_sha256=digest,
            first_seed=owner.cell.seeds.proof_replay_a + index,
            second_seed=owner.cell.seeds.proof_replay_b + index,
            first_container_name=f"dllm-proof-{target}-{digest[:10]}-a",
            second_container_name=f"dllm-proof-{target}-{digest[:10]}-b",
            first_cpu=24,
            second_cpu=25,
            timeout_seconds=120,
        )
        result = prove_input(request)
        proofs[(target, digest)] = result
        _ = recorder.record_json(
            event_type="independent-asan-proof-finished",
            relative_path=Path(f"proof/results/{target}/{digest}.json"),
            value={
                "request": {
                    **asdict(request),
                    "input_path": input_path.relative_to(root).as_posix(),
                },
                "result": result,
            },
            payload={"input_sha256": digest, "status": result.status, "target": target},
        )
    _ = recorder.event(
        "independent-asan-proof-phase-finished",
        {
            "input_count": len(proofs),
            "proved_count": sum(result.status == "proved" for result in proofs.values()),
        },
    )
    return proofs


def _render_report(
    schedule: CampaignSchedule,
    finished: tuple[FinishedCell, ...],
    proofs: Mapping[tuple[Target, str], ProofReplayResult],
) -> str:
    rows: list[str] = []
    for item in finished:
        triggered = sum(
            bool(series.triggered and series.triggered[-1][1] > 0)
            for series in item.result.canaries.values()
        )
        proved = sum(
            result.status == "proved"
            for (target, _digest), result in proofs.items()
            if target == item.cell.target
        )
        rows.append(
            "".join(
                (
                    f"| {item.cell.target} | {item.cell.arm} | ",
                    f"{item.result.metrics.coverage_edges} | ",
                    f"{item.result.metrics.value_profile_features} | ",
                    f"{item.result.metrics.executed_units} | ",
                    f"{len(item.result.findings)} | {triggered} | {proved} |",
                )
            )
        )
    proved_total = sum(result.status == "proved" for result in proofs.values())
    return "\n".join(
        (
            "# dLLM-native multi-target 24-hour comparison results",
            "",
            "Status: completed 15/15 search cells. One repetition is descriptive only.",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[32 raw trajectories per target] --> B[minimum 16 unique]",
            "    B --> C[initial 30 s probes]",
            "    C --> D[full/selective/pseudo revision]",
            "    D --> E[revision 30 s probes]",
            "    E --> F[15 parallel main campaigns]",
            "    F --> G[two fresh non-fatal ASan replays]",
            "```",
            "",
            (
                "| Target | Arm | Final coverage | Features | Executed units | "
                "Search findings | Triggered canaries | ASan-proved inputs |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"Independent ASan-proved inputs: **{proved_total}**.",
            (
                "Magma fatal-canary findings are benchmark reachability only and are not "
                "vulnerability claims."
            ),
            (
                "Each full target-arm cell consumed exactly 86,400 logical search worker-seconds; "
                "proof cost is separate."
            ),
            f"Frozen schedule hash: `{schedule.manifest_hash}`.",
            "",
        )
    )


def source_manifest(repo_root: Path) -> dict[str, object]:
    """Hash every execution-relevant source, prompt, and lock file."""
    paths = sorted(
        (
            *repo_root.glob("src/dllm_agent/experiments/multitarget/*.py"),
            *repo_root.glob("src/dllm_agent/targets/**/*.py"),
            *repo_root.glob("tools/*.py"),
            *repo_root.glob("prompts/multitarget/*.md"),
            repo_root / "pyproject.toml",
            repo_root / "uv.lock",
        ),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    files: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        evidence = artifact_evidence(path)
        files.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": evidence.sha256,
                "size_bytes": evidence.size_bytes,
            }
        )
    return {
        "files": files,
        "git_head": _git_output(repo_root, ("rev-parse", "HEAD")).strip(),
        "git_status_short": _git_output(repo_root, ("status", "--short")),
    }


def _git_output(repo_root: Path, arguments: tuple[str, ...]) -> str:
    process = subprocess.run(  # noqa: S603 - fixed Git executable and internal argv only.
        ("/usr/bin/git", *arguments),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout


def _require_clean(finished: tuple[FinishedCell, ...], *, phase: str) -> None:
    bad = [item for item in finished if not supervisor_exited_cleanly(item.result)]
    if bad:
        cells = ", ".join(f"{item.cell.target}/{item.cell.arm}" for item in bad)
        msg = f"{phase} has failed cells: {cells}"
        raise RuntimeError(msg)


def run_label(prefix: str, root: Path) -> str:
    """Return a deterministic Docker-safe label scoped to one experiment root."""
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:8]
    return f"{prefix}-{suffix}"


def _release_cuda_cache() -> None:
    """Release the resident model before the CPU-only 24-hour wait."""
    _ = gc.collect()
    torch_runtime = cast(
        "TorchRuntime",
        cast("object", importlib.import_module("torch")),
    )
    if torch_runtime.cuda.is_available():
        torch_runtime.cuda.empty_cache()


def validate_roots(
    full_root: Path,
    smoke_root: Path,
    magma_root: Path,
    model_cache: Path,
) -> None:
    """Fail closed when immutable runtime roots are missing or ambiguous."""
    if full_root == smoke_root:
        msg = "smoke and full experiment roots must differ"
        raise ValueError(msg)
    if full_root.exists() and any(full_root.iterdir()):
        msg = "full experiment root must be empty"
        raise ValueError(msg)
    if smoke_root.exists() and any(smoke_root.iterdir()):
        msg = "smoke experiment root must be empty"
        raise ValueError(msg)
    if not (magma_root / "targets").is_dir():
        msg = "Magma root is missing targets"
        raise ValueError(msg)
    if not model_cache.is_dir():
        msg = "model cache does not exist"
        raise ValueError(msg)


def write_evidence_tree_sidecar(experiment_root: Path) -> tuple[Path, str]:
    """Write a reproducible tree hash beside, never inside, the hashed root."""
    digest = evidence_tree_sha256(experiment_root)
    sidecar = experiment_root.with_name(f"{experiment_root.name}.evidence-tree-sha256.txt")
    if sidecar.exists():
        msg = f"evidence tree sidecar already exists: {sidecar}"
        raise ValueError(msg)
    _ = sidecar.write_text(f"{digest}  {experiment_root.name}\n", encoding="utf-8")
    return sidecar, digest


def parse_arguments(arguments: Sequence[str] | None) -> RunArguments:
    """Parse CLI inputs into a fully typed immutable boundary object."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--experiment-root", type=Path, required=True)
    _ = parser.add_argument("--smoke-root", type=Path, required=True)
    _ = parser.add_argument("--magma-root", type=Path, required=True)
    _ = parser.add_argument("--model-cache", type=Path, required=True)
    _ = parser.add_argument("--device", default="cuda:0")
    _ = parser.add_argument("--smoke-probe-seconds", type=int, default=1)
    _ = parser.add_argument("--smoke-main-seconds", type=int, default=5)
    parsed = parser.parse_args(arguments)
    return RunArguments(
        experiment_root=cast("Path", parsed.experiment_root),
        smoke_root=cast("Path", parsed.smoke_root),
        magma_root=cast("Path", parsed.magma_root),
        model_cache=cast("Path", parsed.model_cache),
        device=cast("str", parsed.device),
        smoke_probe_seconds=cast("int", parsed.smoke_probe_seconds),
        smoke_main_seconds=cast("int", parsed.smoke_main_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
