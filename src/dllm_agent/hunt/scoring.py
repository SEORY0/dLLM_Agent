"""Transparent ranking for live dLLM-generated fuzz harnesses."""

from dllm_agent.hunt.models import HarnessCandidate, HarnessCandidateScore


def score_harness_candidate(
    candidate: HarnessCandidate, required_symbols: tuple[str, ...]
) -> HarnessCandidateScore:
    """Score observed confidence, target-API coverage, and NFE efficiency."""
    committed = sum(step.committed_tokens for step in candidate.steps)
    weighted_confidence = sum(
        step.mean_confidence * step.committed_tokens for step in candidate.steps
    )
    confidence = weighted_confidence / committed
    present = sum(symbol in candidate.source for symbol in required_symbols)
    api_coverage = present / len(required_symbols)
    nfe_efficiency = 1.0 - min(candidate.nfe / 512.0, 1.0)
    score = 0.6 * confidence + 0.3 * api_coverage + 0.1 * nfe_efficiency
    return HarnessCandidateScore(
        candidate_id=candidate.candidate_id,
        score=round(score, 6),
        trajectory_confidence=round(confidence, 6),
        api_coverage=round(api_coverage, 6),
        nfe_efficiency=round(nfe_efficiency, 6),
        nfe=candidate.nfe,
    )


def rank_harness_candidates(
    candidates: tuple[HarnessCandidate, ...], required_symbols: tuple[str, ...]
) -> tuple[HarnessCandidateScore, ...]:
    """Rank live candidates deterministically by score then identifier."""
    scores = (score_harness_candidate(candidate, required_symbols) for candidate in candidates)
    return tuple(sorted(scores, key=lambda item: (-item.score, item.candidate_id)))
