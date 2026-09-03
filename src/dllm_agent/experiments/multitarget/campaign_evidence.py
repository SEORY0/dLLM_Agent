"""Relative-path manifests for every live campaign surface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from dllm_agent.experiments.multitarget.campaign_execution import LaunchedCell
from dllm_agent.experiments.multitarget.docker_runtime import container_argv
from dllm_agent.experiments.multitarget.ledger import (
    ArtifactEvidence,
    artifact_evidence,
    command_evidence,
)

if TYPE_CHECKING:
    from dllm_agent.experiments.multitarget.campaign_execution import (
        FinishedCell,
    )
    from dllm_agent.experiments.multitarget.campaign_preparation import (
        CandidateProbeBatchResult,
    )
    from dllm_agent.experiments.multitarget.container_monitor import ContainerRunResult
    from dllm_agent.experiments.multitarget.probe_runner import ProbeRunResult


def probe_batch_manifest(
    experiment_root: Path,
    batch: CandidateProbeBatchResult,
) -> dict[str, object]:
    """Describe one 16-particle probe batch without absolute paths or duplicate logs."""
    return {
        "observations": batch.observations,
        "runs": tuple(_probe_run_manifest(experiment_root, item) for item in batch.results),
    }


def launched_cell_manifest(
    experiment_root: Path,
    item: LaunchedCell,
) -> dict[str, object]:
    """Describe one staged and launched main cell, including its full argv."""
    argv = container_argv(item.request)
    return {
        "cell": item.cell.to_manifest(),
        "container_id": item.launch.container_id,
        "corpus": {
            "base_file_count": item.staged_corpus.base_file_count,
            "candidate_file_count": item.staged_corpus.candidate_file_count,
            "corpus_dir": _relative(experiment_root, item.staged_corpus.corpus_dir),
            "files": item.staged_corpus.files,
            "tree_sha256": item.staged_corpus.tree_sha256,
        },
        "request": {
            "argv": argv,
            "command_evidence": command_evidence(argv, {}).to_json(),
            "corpus_dir": _relative(experiment_root, item.request.corpus_dir),
            "duration_seconds": item.request.duration_seconds,
            "fuzzer_seed": item.request.fuzzer_seed,
            "name": item.request.name,
            "poll_seconds": item.request.poll_seconds,
            "shared_dir": _relative(experiment_root, item.request.shared_dir),
        },
    }


def finished_cell_manifest(
    experiment_root: Path,
    item: FinishedCell,
) -> dict[str, object]:
    """Describe one completed cell with exact raw-log and finding hashes."""
    return {
        "launch": launched_cell_manifest(
            experiment_root,
            _as_launched(item),
        ),
        "log": _relative_evidence(experiment_root, artifact_evidence(item.log_path)).to_json(),
        "result": _run_result_manifest(experiment_root, item.result),
    }


def _probe_run_manifest(experiment_root: Path, item: ProbeRunResult) -> dict[str, object]:
    return {
        "corpus_dir": _relative(experiment_root, item.corpus_dir),
        "log": _relative_evidence(experiment_root, artifact_evidence(item.log_path)).to_json(),
        "observation": item.observation,
        "result": _run_result_manifest(experiment_root, item.run_result),
        "shared_dir": _relative(experiment_root, item.shared_dir),
        "snapshot_path": _relative(experiment_root, item.snapshot_path),
    }


def _run_result_manifest(
    experiment_root: Path,
    result: ContainerRunResult,
) -> dict[str, object]:
    return {
        "canaries": result.canaries,
        "container_id": result.container_id,
        "docker_status": result.docker_status,
        "exit_code": result.exit_code,
        "findings": tuple(
            _relative_evidence(experiment_root, evidence).to_json() for evidence in result.findings
        ),
        "log_sha256": result.log_sha256,
        "metrics": result.metrics,
        "oom_killed": result.oom_killed,
        "process_error": result.process_error,
        "timed_out": result.timed_out,
    }


def _relative_evidence(root: Path, evidence: ArtifactEvidence) -> ArtifactEvidence:
    path = Path(evidence.path)
    relative = _relative(root, path) if path.is_absolute() else path.as_posix()
    return replace(evidence, path=relative)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        msg = f"evidence path is outside experiment root: {path}"
        raise ValueError(msg) from error


def _as_launched(item: FinishedCell) -> LaunchedCell:
    return LaunchedCell(
        cell=item.cell,
        work_dir=item.work_dir,
        shared_dir=item.request.shared_dir,
        staged_corpus=item.staged_corpus,
        request=item.request,
        launch=item.launch,
    )
