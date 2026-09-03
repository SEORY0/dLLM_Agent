"""Relative-path live campaign evidence manifest contracts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from dllm_agent.experiments.multitarget.campaign_evidence import probe_batch_manifest
from dllm_agent.experiments.multitarget.campaign_preparation import CandidateProbeBatchResult
from dllm_agent.experiments.multitarget.candidate_bank import ProbeObservation
from dllm_agent.experiments.multitarget.container_monitor import (
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
from dllm_agent.experiments.multitarget.probe_runner import ProbeRunResult

if TYPE_CHECKING:
    from pathlib import Path


def test_probe_manifest_keeps_exact_hashes_and_only_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    work = root / "probes" / "initial" / "libpng" / "particle-00"
    corpus = work / "corpus"
    shared = work / "shared"
    corpus.mkdir(parents=True)
    shared.mkdir()
    snapshot = corpus / "seed.bin"
    log = work / "probe.log"
    finding = shared / "findings" / "crash"
    finding.parent.mkdir()
    _ = snapshot.write_bytes(b"seed")
    _ = log.write_text("exact log\n", encoding="utf-8")
    _ = finding.write_bytes(b"finding")
    observation = ProbeObservation(
        candidate_sha256=hashlib.sha256(b"seed").hexdigest(),
        feedback_source_sha256=hashlib.sha256(b"seed").hexdigest(),
        canary_triggered=True,
        sanitizer_observed=False,
        coverage_edges=10,
        value_profile_features=20,
        executions_per_second=30,
        log_sha256=hashlib.sha256(b"exact log\n").hexdigest(),
    )
    result = ProbeRunResult(
        observation=observation,
        run_result=ContainerRunResult(
            container_id="container",
            exit_code=0,
            docker_status="exited",
            oom_killed=False,
            timed_out=False,
            process_error=None,
            logs_text="exact log\n",
            log_sha256=observation.log_sha256,
            metrics=FuzzerMetrics(10, 20, 30, 40),
            canaries={},
            findings=(
                ArtifactEvidence(
                    path=str(finding),
                    sha256=hashlib.sha256(b"finding").hexdigest(),
                    size_bytes=7,
                ),
            ),
        ),
        corpus_dir=corpus,
        shared_dir=shared,
        snapshot_path=snapshot,
        log_path=log,
    )

    manifest = probe_batch_manifest(
        root,
        CandidateProbeBatchResult(observations=(observation,), results=(result,)),
    )

    rendered = repr(manifest)
    assert str(root) not in rendered
    runs = cast("tuple[dict[str, object], ...]", manifest["runs"])
    run = runs[0]
    log_evidence = cast("dict[str, object]", run["log"])
    assert log_evidence["path"] == "probes/initial/libpng/particle-00/probe.log"
    assert run["snapshot_path"] == "probes/initial/libpng/particle-00/corpus/seed.bin"
    result_manifest = cast("dict[str, object]", run["result"])
    findings = cast("tuple[dict[str, object], ...]", result_manifest["findings"])
    finding_path = cast("str", findings[0]["path"])
    assert finding_path.endswith("shared/findings/crash")
    assert result_manifest["log_sha256"] == observation.log_sha256
