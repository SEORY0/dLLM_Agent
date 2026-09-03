# pyright: reportUnnecessaryComparison=false
"""Online exact-proof cause classification and family caps."""

import json
import re
from typing import Literal, Protocol, assert_never

from pydantic import Field

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256

type OnlineProofStatus = Literal[
    "not_reproduced",
    "crash_without_sanitizer",
    "proved_duplicate_cause",
    "proved_new_cause",
]
type CompletedProofStatus = Literal["proved", "not_reproduced", "crash_without_sanitizer"]

_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

__all__ = [
    "CauseArchive",
    "CompletedProofStatus",
    "OnlineProofStatus",
    "OnlineProofVerdict",
    "ProofFamilyAttempt",
    "classify_online_proof",
]


class CompletedProofResult(Protocol):
    """Completed exact proof fields consumed by the online archive."""

    @property
    def oracle_status(self) -> CompletedProofStatus: ...

    @property
    def input_sha256(self) -> Sha256: ...

    @property
    def proof_log_sha256(self) -> Sha256: ...

    @property
    def target_commit(self) -> str: ...

    @property
    def sanitizer_class(self) -> str | None: ...

    @property
    def root_signature(self) -> str | None: ...


class ProofFamilyAttempt(FrozenModel):
    """Attempt count for one proof family."""

    family_key: Sha256
    attempts: int = Field(ge=0)


class OnlineProofVerdict(FrozenModel):
    """Cause-aware online proof verdict with hash-bound evidence."""

    oracle_status: OnlineProofStatus
    input_sha256: Sha256
    proof_log_sha256: Sha256
    family_key: Sha256
    cause_id: Sha256 | None = None


class CauseArchive(FrozenModel):
    """Immutable set of proved causes and provisional proof-family attempts."""

    cause_ids: tuple[Sha256, ...] = ()
    family_attempts: tuple[ProofFamilyAttempt, ...] = ()

    def record(self, verdict: OnlineProofVerdict) -> "CauseArchive":
        """Return an archive updated with one completed proof verdict."""
        cause_ids = self.cause_ids
        if verdict.cause_id is not None and verdict.cause_id not in self.cause_ids:
            cause_ids = tuple(sorted((*self.cause_ids, verdict.cause_id)))
        return CauseArchive(
            cause_ids=cause_ids,
            family_attempts=_increment_family(self.family_attempts, verdict.family_key),
        )

    def record_attempt(self, family_key: Sha256) -> "CauseArchive":
        """Return an archive updated for one actual proof runner invocation."""
        return CauseArchive(
            cause_ids=self.cause_ids,
            family_attempts=_increment_family(self.family_attempts, family_key),
        )

    def record_cause(self, cause_id: Sha256) -> "CauseArchive":
        """Return an archive updated with a proved rich cause without spending budget."""
        if cause_id in self.cause_ids:
            return self
        return CauseArchive(
            cause_ids=tuple(sorted((*self.cause_ids, cause_id))),
            family_attempts=self.family_attempts,
        )

    def family_attempt_count(self, family_key: Sha256) -> int:
        """Return attempts already spent for a proof family."""
        for item in self.family_attempts:
            if item.family_key == family_key:
                return item.attempts
        return 0

    def can_attempt(self, *, proof_log_sha256: Sha256, attempt_cap: int) -> bool:
        """Return whether another summary-only proof attempt is still allowed."""
        return self.can_attempt_family(
            family_key=provisional_family_key(proof_log_sha256), attempt_cap=attempt_cap
        )

    def can_attempt_family(self, *, family_key: Sha256, attempt_cap: int) -> bool:
        """Return whether another proof attempt is still allowed for a known family."""
        return self.family_attempt_count(family_key) < attempt_cap


def classify_online_proof(proof: CompletedProofResult, archive: CauseArchive) -> OnlineProofVerdict:
    """Classify a completed exact proof without using probe-side sanitizer signals."""
    match proof.oracle_status:
        case "proved":
            return _proved_verdict(proof, archive)
        case "not_reproduced":
            return _failed_verdict("not_reproduced", proof)
        case "crash_without_sanitizer":
            return _failed_verdict("crash_without_sanitizer", proof)
        case unreachable:
            assert_never(unreachable)


def provisional_family_key(proof_log_sha256: Sha256) -> Sha256:
    """Return the provisional family key for proof results without a rich cause."""
    return sha256_bytes(f"provisional-proof-log:{proof_log_sha256}".encode())


def _proved_verdict(proof: CompletedProofResult, archive: CauseArchive) -> OnlineProofVerdict:
    cause_id = _rich_cause_id(proof)
    if cause_id is None:
        return OnlineProofVerdict(
            oracle_status="proved_duplicate_cause",
            input_sha256=proof.input_sha256,
            proof_log_sha256=proof.proof_log_sha256,
            family_key=provisional_family_key(proof.proof_log_sha256),
            cause_id=None,
        )
    return OnlineProofVerdict(
        oracle_status="proved_duplicate_cause"
        if cause_id in archive.cause_ids
        else "proved_new_cause",
        input_sha256=proof.input_sha256,
        proof_log_sha256=proof.proof_log_sha256,
        family_key=cause_id,
        cause_id=cause_id,
    )


def _failed_verdict(status: OnlineProofStatus, proof: CompletedProofResult) -> OnlineProofVerdict:
    return OnlineProofVerdict(
        oracle_status=status,
        input_sha256=proof.input_sha256,
        proof_log_sha256=proof.proof_log_sha256,
        family_key=provisional_family_key(proof.proof_log_sha256),
        cause_id=None,
    )


def _rich_cause_id(proof: CompletedProofResult) -> Sha256 | None:
    if proof.sanitizer_class is None or proof.root_signature is None:
        return None
    sanitizer_class = proof.sanitizer_class.strip()
    root_signature = _normalized_root_signature(proof.root_signature)
    if sanitizer_class == "" or root_signature == "":
        return None
    canonical = {
        "root_signature": root_signature,
        "sanitizer_class": sanitizer_class,
        "target_commit": proof.target_commit.strip().lower(),
    }
    return sha256_bytes(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode())


def _normalized_root_signature(root_signature: str) -> str:
    without_addresses = _HEX_RE.sub("0xADDR", root_signature)
    return " ".join(without_addresses.split())


def _increment_family(
    attempts: tuple[ProofFamilyAttempt, ...], family_key: Sha256
) -> tuple[ProofFamilyAttempt, ...]:
    records = {
        item.family_key: item.attempts + 1 if item.family_key == family_key else item.attempts
        for item in attempts
    }
    if family_key not in records:
        records[family_key] = 1
    return tuple(
        ProofFamilyAttempt(family_key=key, attempts=records[key]) for key in sorted(records)
    )
