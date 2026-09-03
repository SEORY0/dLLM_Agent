"""Pure orchestration contracts for the real multi-target campaign."""

import hashlib

import pytest

from dllm_agent.experiments.multitarget.campaign_core import (
    CampaignCoreError,
    CampaignDurations,
    arm_candidate_inputs,
    probe_seed,
)
from dllm_agent.experiments.multitarget.corpus import CandidateInput


def test_full_durations_preserve_exact_worker_budget() -> None:
    durations = CampaignDurations.full()

    assert durations.probe_seconds == 30
    assert durations.main_seconds("libfuzzer-only") == 86_400
    assert durations.main_seconds("structured-one-shot") == 86_400
    assert durations.main_seconds("selective-revision") == 85_440
    assert durations.logical_worker_seconds("selective-revision") == 86_400
    assert durations.label == "full-24h"


def test_smoke_durations_are_explicitly_not_a_full_experiment() -> None:
    durations = CampaignDurations.smoke(probe_seconds=1, main_seconds=2)

    assert durations.main_seconds("pseudo-feedback") == 2
    assert durations.logical_worker_seconds("pseudo-feedback") == 34
    assert durations.label == "smoke"
    assert durations.is_full is False


def test_arm_candidate_inputs_bind_shared_initial_and_revision_banks() -> None:
    initial = tuple(_candidate(f"initial-{index}") for index in range(16))
    revised = {
        "feedback-no-remask": tuple(_candidate(f"full-{index}") for index in range(16)),
        "selective-revision": tuple(_candidate(f"selective-{index}") for index in range(16)),
        "pseudo-feedback": tuple(_candidate(f"pseudo-{index}") for index in range(16)),
    }

    assert arm_candidate_inputs("libfuzzer-only", initial, revised) == ()
    assert arm_candidate_inputs("structured-one-shot", initial, revised) == initial
    assert (
        arm_candidate_inputs("selective-revision", initial, revised)
        == revised["selective-revision"]
    )


def test_arm_candidate_inputs_reject_missing_or_duplicate_population() -> None:
    unique = tuple(_candidate(f"seed-{index}") for index in range(16))

    with pytest.raises(CampaignCoreError, match="missing revision"):
        _ = arm_candidate_inputs("pseudo-feedback", unique, {})
    with pytest.raises(CampaignCoreError, match="16 unique"):
        _ = arm_candidate_inputs("structured-one-shot", (*unique[:-1], unique[0]), {})


def test_probe_seed_is_stable_and_particle_specific() -> None:
    values = tuple(probe_seed(1234, index) for index in range(16))

    assert values == tuple(probe_seed(1234, index) for index in range(16))
    assert len(set(values)) == 16
    assert all(value >= 0 for value in values)


def _candidate(label: str) -> CandidateInput:
    content = label.encode()
    return CandidateInput(snapshot_sha256=hashlib.sha256(content).hexdigest(), content=content)
