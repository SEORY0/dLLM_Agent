"""Canonical multitarget experiment artifact persistence tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from dllm_agent.experiments.multitarget.artifacts import (
    ArtifactPersistenceError,
    JsonValue,
    canonical_json_bytes,
    write_json_artifact,
)
from dllm_agent.experiments.multitarget.candidate_bank import (
    Candidate,
    CandidateBank,
    ProbeObservation,
)
from dllm_agent.experiments.multitarget.container_monitor import (
    CanarySeries,
    ContainerRunResult,
    FuzzerMetrics,
)
from dllm_agent.experiments.multitarget.docker_runtime import RuntimeProfile, runtime_profile
from dllm_agent.experiments.multitarget.ledger import ArtifactEvidence
from dllm_agent.experiments.multitarget.proof import (
    AsanSignature,
    ProofReplayResult,
    ProofRunEvidence,
)
from dllm_agent.experiments.multitarget.slot_sampler import SlotSample, TrajectoryStep
from dllm_agent.targets.structured import FieldSpan, MaterializedSnapshot

_SHA_ZERO = "0" * 64
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _ArtifactKind(StrEnum):
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class _Manifest:
    kind: _ArtifactKind
    root: Path


def test_canonical_json_includes_full_slot_trajectory_and_is_byte_stable() -> None:
    """Given equivalent slot samples, when encoded, then trajectory evidence is exact."""
    sample = _sample()

    first = canonical_json_bytes(sample)
    second = canonical_json_bytes(sample)

    assert first == second
    payload = _decode_object(first)
    assert _array(payload["trajectory_steps"]) == [
        {
            "block_index": 0,
            "committed_positions": [4, 8],
            "confidences": [0.875, 0.75],
            "masked_before": 3,
            "remaining_masked": 1,
            "state_sha256": _SHA_ZERO,
            "step_index": 1,
            "token_ids": [11, 13],
        }
    ]
    assert b" " not in first
    assert first.endswith(b"}")


def test_nested_multitarget_dataclasses_paths_enums_and_bytes_are_canonical() -> None:
    """Given campaign objects, when encoded, then all nested evidence is JSON-safe."""
    value = {
        "bank": _bank(),
        "selection": (_bank().candidates[0],),
        "observations": (_observation(),),
        "container": _container_result(),
        "proof": _proof_result(),
        "manifest": _Manifest(kind=_ArtifactKind.SNAPSHOT, root=Path("runs/exp-001")),
    }

    payload = _decode_object(canonical_json_bytes(value))
    bank = _object(payload["bank"])
    first_candidate = _object(_array(bank["candidates"])[0])
    sample = _object(first_candidate["sample"])
    first_step = _object(_array(sample["trajectory_steps"])[0])
    selection = _object(_array(payload["selection"])[0])
    snapshot = _object(selection["snapshot"])
    content = _object(snapshot["content"])
    container = _object(payload["container"])
    canaries = _object(container["canaries"])
    bug = _object(canaries["BUG-1"])
    proof = _object(payload["proof"])
    run = _object(_array(proof["runs"])[0])
    log = _object(run["log"])

    assert first_step["token_ids"] == [11, 13]
    assert content["sha256"] == hashlib.sha256(b"snapshot").hexdigest()
    assert bug["triggered"] == [[2, 1]]
    assert log["data_base64"] == "YXNhbg=="
    assert payload["manifest"] == {"kind": "snapshot", "root": "runs/exp-001"}


def test_write_json_artifact_is_atomic_and_returns_hash_size_evidence(tmp_path: Path) -> None:
    """Given an experiment root, when writing JSON, then exact evidence is returned."""
    root = tmp_path / "experiment"
    root.mkdir()

    evidence = write_json_artifact(
        experiment_root=root,
        relative_path=Path("artifacts/sample.json"),
        value=_sample(),
    )

    written = root / "artifacts/sample.json"
    assert written.read_bytes() == canonical_json_bytes(_sample())
    assert evidence == ArtifactEvidence(
        path="artifacts/sample.json",
        sha256=hashlib.sha256(written.read_bytes()).hexdigest(),
        size_bytes=written.stat().st_size,
    )


def test_write_json_artifact_accepts_identical_existing_bytes(tmp_path: Path) -> None:
    """Given identical existing bytes, when writing, then the artifact is reused."""
    root = tmp_path / "experiment"
    path = root / "sample.json"
    path.parent.mkdir()
    payload = canonical_json_bytes(_sample())
    _ = path.write_bytes(payload)

    evidence = write_json_artifact(
        experiment_root=root,
        relative_path=Path("sample.json"),
        value=_sample(),
    )

    assert evidence.sha256 == hashlib.sha256(payload).hexdigest()
    assert evidence.size_bytes == len(payload)


def test_write_json_artifact_rejects_escape_and_conflicting_existing_bytes(tmp_path: Path) -> None:
    """Given unsafe destinations, when writing, then persistence fails closed."""
    root = tmp_path / "experiment"
    root.mkdir()
    existing = root / "sample.json"
    _ = existing.write_bytes(b"{}")

    with pytest.raises(ArtifactPersistenceError, match="outside experiment root"):
        _ = write_json_artifact(
            experiment_root=root,
            relative_path=Path("../escape.json"),
            value=_sample(),
        )
    with pytest.raises(ArtifactPersistenceError, match="already exists"):
        _ = write_json_artifact(
            experiment_root=root,
            relative_path=Path("sample.json"),
            value=_sample(),
        )


def _sample() -> SlotSample:
    return SlotSample(
        token_ids=(101, 102, 103),
        selectors=(1, 2, 3),
        slot_positions=(4, 8, 12),
        slot_confidences=(0.875, 0.75, 0.5),
        selected_slot_token_ids=(11, 13, 17),
        requested_steps=8,
        nfe=2,
        forward_calls=2,
        gpu_ms=19,
        trajectory_steps=(
            TrajectoryStep(
                step_index=1,
                block_index=0,
                masked_before=3,
                committed_positions=(4, 8),
                token_ids=(11, 13),
                confidences=(0.875, 0.75),
                remaining_masked=1,
                state_sha256=_SHA_ZERO,
            ),
        ),
        state_sha256="1" * 64,
        parent_state_sha256=None,
    )


def _decode_object(payload: bytes) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(payload)


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _bank() -> CandidateBank:
    snapshot = MaterializedSnapshot(
        content=b"snapshot",
        sha256=hashlib.sha256(b"snapshot").hexdigest(),
        anchor_sha256=hashlib.sha256(b"anchor").hexdigest(),
        field_spans=(FieldSpan(field="width", start=0, end=4),),
    )
    return CandidateBank(
        candidates=(Candidate(sample=_sample(), snapshot=snapshot, source_sample_sha256="2" * 64),),
        duplicates=(),
        raw_count=32,
        unique_count=1,
        minimum_unique_count=16,
        raw_trajectory_sha256="3" * 64,
        unique_snapshot_sha256="4" * 64,
    )


def _observation() -> ProbeObservation:
    return ProbeObservation(
        candidate_sha256="5" * 64,
        feedback_source_sha256="6" * 64,
        canary_triggered=True,
        sanitizer_observed=False,
        coverage_edges=10,
        value_profile_features=20,
        executions_per_second=30,
        log_sha256="7" * 64,
    )


def _container_result() -> ContainerRunResult:
    return ContainerRunResult(
        container_id="container-1",
        exit_code=0,
        docker_status="exited",
        oom_killed=False,
        timed_out=False,
        process_error=None,
        logs_text="clean",
        log_sha256=hashlib.sha256(b"clean").hexdigest(),
        metrics=FuzzerMetrics(
            coverage_edges=10,
            value_profile_features=20,
            executions_per_second=30,
            executed_units=40,
        ),
        canaries={"BUG-1": CanarySeries(reached=((1, 1),), triggered=((2, 1),))},
        findings=(ArtifactEvidence(path="findings/id", sha256="8" * 64, size_bytes=9),),
    )


def _proof_result() -> ProofReplayResult:
    return ProofReplayResult(
        status="proved",
        profile=_proof_profile(),
        image_id="sha256:" + ("9" * 64),
        input_sha256="a" * 64,
        runs=(
            ProofRunEvidence(
                container_name="proof-1",
                proof_seed=1,
                return_code=1,
                log=b"asan",
                log_sha256=hashlib.sha256(b"asan").hexdigest(),
                argv_sha256="b" * 64,
                signature=AsanSignature(
                    sanitizer_class="heap-buffer-overflow",
                    first_target_frame="png_read_info",
                ),
                error=None,
            ),
        ),
        signature=AsanSignature(
            sanitizer_class="heap-buffer-overflow",
            first_target_frame="png_read_info",
        ),
        error=None,
    )


def _proof_profile() -> RuntimeProfile:
    return runtime_profile("libpng")
