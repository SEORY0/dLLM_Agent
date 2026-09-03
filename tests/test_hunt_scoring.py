"""Live dLLM harness candidate scoring tests."""

import hashlib

from dllm_agent.hunt.models import DenoisingStep, HarnessCandidate
from dllm_agent.hunt.scoring import rank_harness_candidates, score_harness_candidate


def _candidate(candidate_id: str, confidence: float, source: str, nfe: int) -> HarnessCandidate:
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return HarnessCandidate(
        candidate_id=candidate_id,
        source=source,
        source_sha256=source_sha256,
        seed=7,
        nfe=nfe,
        steps=(
            DenoisingStep(
                step_index=1,
                masked_tokens_before=8,
                committed_tokens=4,
                mean_confidence=confidence,
                state_sha256="1" * 64,
            ),
            DenoisingStep(
                step_index=2,
                masked_tokens_before=4,
                committed_tokens=4,
                mean_confidence=confidence,
                state_sha256="2" * 64,
            ),
        ),
    )


def test_live_score_combines_trajectory_confidence_api_coverage_and_nfe() -> None:
    # Given
    source = "LLVMFuzzerTestOneInput try_decode_ubjson ubjson_bytes_cursor"
    candidate = _candidate("candidate-01", 0.9, source, 64)

    # When
    score = score_harness_candidate(
        candidate,
        ("LLVMFuzzerTestOneInput", "try_decode_ubjson", "ubjson_bytes_cursor"),
    )

    # Then
    assert score.score == 0.9275
    assert score.trajectory_confidence == 0.9
    assert score.api_coverage == 1.0


def test_live_ranking_is_deterministic_for_equal_scores() -> None:
    # Given
    source = "LLVMFuzzerTestOneInput try_decode_ubjson"
    later = _candidate("candidate-02", 0.8, source, 64)
    earlier = _candidate("candidate-01", 0.8, source, 64)

    # When
    ranked = rank_harness_candidates(
        (later, earlier), ("LLVMFuzzerTestOneInput", "try_decode_ubjson")
    )

    # Then
    assert tuple(item.candidate_id for item in ranked) == ("candidate-01", "candidate-02")
