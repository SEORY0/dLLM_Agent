"""Deterministic trajectory-aware native probe controller and proof handoff."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dllm_agent.core.errors import HuntSetupError
from dllm_agent.core.io import parse_json_bytes, read_bytes
from dllm_agent.fuzzing.coverage.runner import CoverageRequest, CoverageResult
from dllm_agent.fuzzing.runner import FuzzRunRequest, FuzzRunResult
from dllm_agent.hunt.artifacts import persist_artifacts
from dllm_agent.hunt.models import CandidateBank, FuzzEvidence, HarnessCandidate
from dllm_agent.hunt.pipeline import filter_candidates, resolve_prompt, run_hunt
from dllm_agent.hunt.validation import validate_candidate_bank
from dllm_agent.native.artifacts import (
    LedgerEntry,
    ProbeIdentity,
    controller_events_bytes,
    controller_events_sha256,
    create_evidence_root,
    normalize_probe,
    persist_controller_events,
)
from dllm_agent.native.models import (
    ControllerEvidence,
    NativeHuntConfig,
    NativeHuntReport,
    ProbeEvent,
    SelectionEvent,
    StageEvent,
)
from dllm_agent.native.scoring import extract_trajectory_features, rank_stage
from dllm_agent.targets.models import HuntManifest
from dllm_agent.targets.staging import staged_target, verify_target_archive

_CANDIDATE_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True, slots=True)
class NativeInputs:
    """Parsed and validated controller boundaries used throughout one native hunt."""

    manifest: HuntManifest
    config: NativeHuntConfig
    candidates: tuple[HarnessCandidate, ...]


def run_native_hunt(  # noqa: PLR0913
    manifest_path: Path,
    candidate_bank_path: Path,
    controller_path: Path,
    source_root: Path,
    *,
    fuzz_runner: Callable[[FuzzRunRequest], FuzzRunResult],
    coverage_runner: Callable[[CoverageRequest], CoverageResult],
    artifact_dir: Path,
) -> NativeHuntReport:
    """Probe valid candidates equally, rank stages, and run an independent proof."""
    inputs = _parse_inputs(manifest_path, candidate_bank_path, controller_path)
    verified_target = verify_target_archive(source_root, inputs.manifest)
    features = tuple(extract_trajectory_features(item) for item in inputs.candidates)
    features_by_id = {item.candidate_id: item for item in features}
    candidates = {item.candidate_id: item for item in inputs.candidates}
    active = tuple(
        sorted(
            inputs.candidates,
            key=lambda item: (
                -features_by_id[item.candidate_id].trajectory_quality,
                item.candidate_id,
            ),
        )
    )
    corpus = tuple(bytes.fromhex(seed) for seed in inputs.manifest.seed_hex)
    _ensure_survivor_counts(inputs.config, len(active))
    create_evidence_root(artifact_dir)
    sequence = 1
    ledger: list[LedgerEntry] = []
    probe_events: list[ProbeEvent] = []
    stage_events: list[StageEvent] = []
    failed_stage: int | None = None
    with staged_target(verified_target) as staged_source:
        for stage_index, stage in enumerate(inputs.config.stages):
            current_events: list[ProbeEvent] = []
            for candidate in active:
                for seed in stage.fuzzer_seeds:
                    request = FuzzRunRequest(
                        image_id=inputs.manifest.image_id,
                        source_root=staged_source,
                        harness_bytes=candidate.source.encode(),
                        corpus=corpus,
                        duration_seconds=stage.duration_seconds,
                        max_len=inputs.manifest.max_len,
                        rss_limit_mb=inputs.manifest.rss_limit_mb,
                        timeout_seconds=float(inputs.manifest.timeout_seconds),
                        sanitizers=inputs.manifest.sanitizers,
                        fuzzer_seed=seed,
                        target_profile=inputs.manifest.target_profile,
                    )
                    fuzz_result = fuzz_runner(request)
                    replay_corpus = (
                        tuple(item.content for item in fuzz_result.corpus_artifacts) or corpus
                    )
                    coverage_result = coverage_runner(
                        CoverageRequest(
                            image_id=inputs.manifest.image_id,
                            source_root=staged_source,
                            harness_bytes=request.harness_bytes,
                            corpus=replay_corpus,
                            timeout_seconds=max(float(inputs.manifest.timeout_seconds), 60.0),
                            target_profile=inputs.manifest.target_profile,
                        )
                    )
                    probe_root = (
                        artifact_dir
                        / "probes"
                        / f"stage-{stage_index:02d}"
                        / candidate.candidate_id
                        / f"seed-{seed:010d}"
                    )
                    persist_artifacts(probe_root, candidate, fuzz_result, coverage_result)
                    event = normalize_probe(
                        ProbeIdentity(sequence, stage_index, candidate.candidate_id, seed),
                        fuzz_result,
                        coverage_result,
                    )
                    current_events.append(event)
                    probe_events.append(event)
                    ledger.append(LedgerEntry("probe", event))
                    sequence += 1
            if not any(event.build_status == "succeeded" for event in current_events):
                failed_stage = stage_index
                break
            active_features = tuple(features_by_id[candidate.candidate_id] for candidate in active)
            ranked_ids = rank_stage(tuple(current_events), active_features)
            promoted_ids = ranked_ids[: stage.survivors]
            stage_event = StageEvent(
                sequence=sequence,
                stage_index=stage_index,
                ranked_candidate_ids=ranked_ids,
                promoted_candidate_ids=promoted_ids,
            )
            stage_events.append(stage_event)
            ledger.append(LedgerEntry("stage", stage_event))
            sequence += 1
            active = tuple(candidates[candidate_id] for candidate_id in promoted_ids)
    if failed_stage is not None:
        detail = f"probe stage {failed_stage} had zero successful builds"
        raise HuntSetupError(detail)
    selected_id = active[0].candidate_id
    selection = SelectionEvent(sequence=sequence, selected_candidate_id=selected_id)
    ledger.append(LedgerEntry("selection", selection))
    ledger_bytes = controller_events_bytes(tuple(ledger))
    proof = run_hunt(
        manifest_path,
        candidate_bank_path,
        source_root,
        fuzz_runner=fuzz_runner,
        coverage_runner=coverage_runner,
        artifact_dir=artifact_dir / "proof",
        selected_candidate_id=selected_id,
        fuzz_seconds=inputs.config.proof_duration_seconds,
    )
    _require_successful_proof(proof.fuzz)
    persist_controller_events(artifact_dir, ledger_bytes)
    return NativeHuntReport(
        schema_version="dllm-native-hunt-report/v1",
        config=inputs.config,
        trajectory_features=features,
        controller=ControllerEvidence(
            controller_events_sha256=controller_events_sha256(ledger_bytes),
            probe_events=tuple(probe_events),
            stage_events=tuple(stage_events),
            selection=selection,
        ),
        proof=proof,
    )


def _parse_inputs(
    manifest_path: Path, candidate_bank_path: Path, controller_path: Path
) -> NativeInputs:
    manifest_bytes = read_bytes(manifest_path, "hunt manifest")
    manifest = parse_json_bytes(manifest_bytes, manifest_path, HuntManifest, "hunt manifest")
    prompt = read_bytes(resolve_prompt(manifest_path, manifest.prompt), "model prompt")
    bank_bytes = read_bytes(candidate_bank_path, "candidate bank")
    bank = parse_json_bytes(bank_bytes, candidate_bank_path, CandidateBank, "candidate bank")
    validate_candidate_bank(bank, prompt, candidate_bank_path)
    if any(_CANDIDATE_ID.fullmatch(item.candidate_id) is None for item in bank.candidates):
        detail = "candidate id is not a safe evidence path segment"
        raise HuntSetupError(detail)
    if bank.model.model_id != manifest.model_id or bank.model.revision != manifest.model_revision:
        detail = "candidate bank model does not match hunt manifest"
        raise HuntSetupError(detail)
    config_bytes = read_bytes(controller_path, "native hunt config")
    config = parse_json_bytes(config_bytes, controller_path, NativeHuntConfig, "native hunt config")
    valid, _rejected = filter_candidates(bank, manifest)
    if not valid:
        detail = "all generated harness candidates were rejected"
        raise HuntSetupError(detail)
    return NativeInputs(manifest, config, valid)


def _ensure_survivor_counts(config: NativeHuntConfig, candidate_count: int) -> None:
    active_count = candidate_count
    for stage in config.stages:
        if stage.survivors > active_count:
            detail = "probe stage survivors exceed active candidate count"
            raise HuntSetupError(detail)
        active_count = stage.survivors


def _require_successful_proof(evidence: FuzzEvidence) -> None:
    if evidence.build_status != "succeeded" or evidence.fuzz_status not in {
        "completed",
        "crashed",
    }:
        detail = (
            "final proof execution did not complete: "
            f"build={evidence.build_status}, fuzz={evidence.fuzz_status}"
        )
        raise HuntSetupError(detail)
