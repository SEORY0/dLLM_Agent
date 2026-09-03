"""Live multitarget runner boundary tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from tools.evidence_tree_sha256 import evidence_tree_sha256
from tools.run_multitarget_24h import (
    RunArguments,
    parse_arguments,
    run_label,
    source_manifest,
    validate_roots,
    write_evidence_tree_sidecar,
)


def test_cli_arguments_are_typed_and_run_labels_are_deterministic(tmp_path: Path) -> None:
    parsed = parse_arguments(
        (
            "--experiment-root",
            str(tmp_path / "full"),
            "--smoke-root",
            str(tmp_path / "smoke"),
            "--magma-root",
            str(tmp_path / "magma"),
            "--model-cache",
            str(tmp_path / "model"),
            "--device",
            "cuda:3",
        )
    )

    assert isinstance(parsed, RunArguments)
    assert parsed.device == "cuda:3"
    assert parsed.smoke_probe_seconds == 1
    assert parsed.smoke_main_seconds == 5
    assert run_label("full", parsed.experiment_root) == run_label("full", parsed.experiment_root)


def test_root_validation_rejects_nonempty_smoke_or_missing_pins(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    _ = (smoke / "stale").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke experiment root must be empty"):
        validate_roots(
            tmp_path / "full",
            smoke,
            tmp_path / "magma",
            tmp_path / "model",
        )

    full = tmp_path / "full"
    full.mkdir()
    _ = (full / "stale").write_text("stale", encoding="utf-8")
    empty_smoke = tmp_path / "empty-smoke"
    empty_smoke.mkdir()
    with pytest.raises(ValueError, match="full experiment root must be empty"):
        validate_roots(full, empty_smoke, tmp_path / "magma", tmp_path / "model")


def test_source_manifest_hashes_runner_and_never_records_absolute_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    manifest = source_manifest(repo_root)

    files = cast("list[dict[str, object]]", manifest["files"])
    runner = next(item for item in files if item["path"] == "tools/run_multitarget_24h.py")
    assert len(cast("str", runner["sha256"])) == 64
    assert str(repo_root) not in repr(files)


def test_tree_hash_sidecar_does_not_change_hashed_experiment_root(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    _ = (root / "evidence.json").write_text('{"ok":true}\n', encoding="utf-8")

    sidecar, digest = write_evidence_tree_sidecar(root)

    assert sidecar.parent == root.parent
    assert sidecar.parent != root
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  {root.name}\n"
    assert evidence_tree_sha256(root) == digest
