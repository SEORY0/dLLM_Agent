"""Deterministic corpus staging for Magma campaign cells."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, override

if TYPE_CHECKING:
    from pathlib import Path

_SHA256_HEX_LENGTH: Final = 64
_MAX_BASE_FILES: Final = 16

type CorpusSourceKind = Literal["base", "particle"]


@dataclass(slots=True)
class CorpusError(Exception):
    """Raised when a corpus cannot be staged reproducibly."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class CandidateInput:
    """One dLLM materialized input to inject into a campaign corpus."""

    snapshot_sha256: str
    content: bytes


@dataclass(frozen=True, slots=True)
class CorpusFileEvidence:
    """Hash evidence for one file inside the staged corpus."""

    relative_path: str
    size_bytes: int
    sha256: str
    source_kind: CorpusSourceKind


@dataclass(frozen=True, slots=True)
class StagedCorpus:
    """Staged corpus directory and deterministic tree hash evidence."""

    corpus_dir: Path
    base_file_count: int
    candidate_file_count: int
    files: tuple[CorpusFileEvidence, ...]
    tree_sha256: str


def stage_cell_corpus(
    source_corpus: Path,
    output_dir: Path,
    *,
    candidates: tuple[CandidateInput, ...],
) -> StagedCorpus:
    """Copy base seeds and deterministic dLLM candidates into one corpus dir."""
    if not source_corpus.is_dir():
        raise CorpusError(reason="source corpus must exist")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CorpusError(reason="destination corpus must be empty")
    _validate_candidates(candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[CorpusFileEvidence] = []
    base_count = _stage_base_corpus(source_corpus, output_dir, files)
    for candidate in candidates:
        evidence = _stage_candidate(output_dir, candidate)
        if evidence is not None:
            files.append(evidence)
    return StagedCorpus(
        corpus_dir=output_dir,
        base_file_count=base_count,
        candidate_file_count=sum(1 for item in files if item.source_kind == "particle"),
        files=tuple(files),
        tree_sha256=_tree_hash(files),
    )


def _validate_candidates(candidates: tuple[CandidateInput, ...]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        if len(candidate.snapshot_sha256) != _SHA256_HEX_LENGTH or not all(
            character in "0123456789abcdef" for character in candidate.snapshot_sha256
        ):
            raise CorpusError(reason="candidate snapshot_sha256 must be a 64-char hash")
        actual = hashlib.sha256(candidate.content).hexdigest()
        if actual != candidate.snapshot_sha256:
            raise CorpusError(reason="candidate sha256 mismatch")
        if candidate.snapshot_sha256 in seen:
            raise CorpusError(reason="duplicate candidate snapshot hash")
        seen.add(candidate.snapshot_sha256)


def _stage_base_corpus(
    source: Path,
    output: Path,
    files: list[CorpusFileEvidence],
) -> int:
    selected = _select_base_files(source)
    if not selected:
        raise CorpusError(reason="no valid base corpus files")
    for sha256, path, size in selected:
        destination = output / sha256
        _link_or_copy(path, destination)
        files.append(
            CorpusFileEvidence(
                relative_path=sha256,
                size_bytes=size,
                sha256=sha256,
                source_kind="base",
            )
        )
    return len(selected)


def _select_base_files(source: Path) -> tuple[tuple[str, Path, int], ...]:
    by_hash: dict[str, tuple[Path, int]] = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        if not content:
            continue
        digest = hashlib.sha256(content).hexdigest()
        _ = by_hash.setdefault(digest, (path, len(content)))
    return tuple((digest, path, size) for digest, (path, size) in sorted(by_hash.items()))[
        :_MAX_BASE_FILES
    ]


def _stage_candidate(output: Path, candidate: CandidateInput) -> CorpusFileEvidence | None:
    destination = output / candidate.snapshot_sha256
    if destination.exists():
        if destination.read_bytes() == candidate.content:
            return None
        raise CorpusError(reason="candidate output hash collision")
    _ = destination.write_bytes(candidate.content)
    return CorpusFileEvidence(
        relative_path=candidate.snapshot_sha256,
        size_bytes=len(candidate.content),
        sha256=candidate.snapshot_sha256,
        source_kind="particle",
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        _ = shutil.copy2(source, destination, follow_symlinks=False)


def _tree_hash(files: tuple[CorpusFileEvidence, ...] | list[CorpusFileEvidence]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_path.encode())
        digest.update(b"\0")
        digest.update(item.source_kind.encode())
        digest.update(b"\0")
        digest.update(str(item.size_bytes).encode())
        digest.update(b"\0")
        digest.update(item.sha256.encode())
        digest.update(b"\0")
    return digest.hexdigest()
