"""Corpus staging contracts for Magma multitarget cells."""

import hashlib
from pathlib import Path

import pytest

from dllm_agent.experiments.multitarget.corpus import (
    CandidateInput,
    CorpusError,
    stage_cell_corpus,
)


def test_stage_cell_corpus_copies_base_and_writes_candidate_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    base_a = _sha256(b"base-a")
    base_b = _sha256(b"base-b")
    candidate_a = _sha256(b"candidate-a")
    candidate_b = _sha256(b"candidate-b")
    _ = (source / "z-seed-a").write_bytes(b"base-a")
    _ = (source / "a-seed-b").write_bytes(b"base-b")
    _ = (source / "duplicate").write_bytes(b"base-a")
    _ = (source / "empty").write_bytes(b"")
    _ = (source / "link").symlink_to(source / "a-seed-b")
    out = tmp_path / "out"

    staged = stage_cell_corpus(
        source,
        out,
        candidates=(
            CandidateInput(snapshot_sha256=candidate_a, content=b"candidate-a"),
            CandidateInput(snapshot_sha256=candidate_b, content=b"candidate-b"),
        ),
    )

    assert staged.corpus_dir == out
    assert staged.base_file_count == 2
    assert staged.candidate_file_count == 2
    assert tuple(file.relative_path for file in staged.files) == (
        base_a,
        base_b,
        candidate_a,
        candidate_b,
    )
    assert tuple(file.source_kind for file in staged.files) == (
        "base",
        "base",
        "particle",
        "particle",
    )
    assert (out / base_a).read_bytes() == b"base-a"
    assert (out / base_b).read_bytes() == b"base-b"
    assert (out / candidate_a).read_bytes() == b"candidate-a"
    assert (out / candidate_b).read_bytes() == b"candidate-b"
    assert all(not path.is_symlink() for path in out.iterdir())
    assert len(staged.tree_sha256) == 64


def test_stage_cell_corpus_is_stable_under_source_filename_order_changes(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _ = (left / "b").write_bytes(b"beta")
    _ = (left / "a").write_bytes(b"alpha")
    _ = (right / "different-name").write_bytes(b"alpha")
    _ = (right / "other-name").write_bytes(b"beta")

    first = stage_cell_corpus(
        left,
        tmp_path / "out-left",
        candidates=(CandidateInput(snapshot_sha256=_sha256(b"gamma"), content=b"gamma"),),
    )
    second = stage_cell_corpus(
        right,
        tmp_path / "out-right",
        candidates=(CandidateInput(snapshot_sha256=_sha256(b"gamma"), content=b"gamma"),),
    )

    assert first.tree_sha256 == second.tree_sha256
    assert tuple(file.relative_path for file in first.files) == tuple(
        file.relative_path for file in second.files
    )


def test_stage_cell_corpus_limits_base_corpus_to_sixteen_sha_ordered_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(20):
        _ = (source / f"seed-{index}").write_bytes(f"seed-{index}".encode())

    staged = stage_cell_corpus(source, tmp_path / "out", candidates=())

    assert staged.base_file_count == 16
    assert tuple(file.relative_path for file in staged.files) == tuple(
        sorted(_sha256(f"seed-{index}".encode()) for index in range(20))[:16]
    )


def test_stage_cell_corpus_fails_closed_without_deleting_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _ = (source / "seed").write_bytes(b"base")
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "stale"
    _ = stale.write_bytes(b"stale")

    with pytest.raises(CorpusError, match="destination"):
        _ = stage_cell_corpus(source, out, candidates=())
    assert stale.read_bytes() == b"stale"


def test_stage_cell_corpus_rejects_missing_empty_or_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="source corpus"):
        _ = stage_cell_corpus(tmp_path / "missing", tmp_path / "out", candidates=())

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(CorpusError, match="no valid base"):
        _ = stage_cell_corpus(source, tmp_path / "empty-out", candidates=())

    _ = (source / "seed").write_bytes(b"base")
    with pytest.raises(CorpusError, match="duplicate candidate"):
        _ = stage_cell_corpus(
            source,
            tmp_path / "out",
            candidates=(
                CandidateInput(snapshot_sha256=_sha256(b"a"), content=b"a"),
                CandidateInput(snapshot_sha256=_sha256(b"a"), content=b"a"),
            ),
        )

    with pytest.raises(CorpusError, match="candidate sha256 mismatch"):
        _ = stage_cell_corpus(
            source,
            tmp_path / "other-out",
            candidates=(CandidateInput(snapshot_sha256="e" * 64, content=b"not-e"),),
        )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
