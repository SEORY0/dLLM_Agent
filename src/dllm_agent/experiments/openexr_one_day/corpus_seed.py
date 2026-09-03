"""Private corpus seeding for OpenEXR one-day cells."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from pydantic import Field, TypeAdapter, ValidationError

from dllm_agent.core.io import sha256_bytes
from dllm_agent.core.models import FrozenModel, Sha256
from dllm_agent.experiments.multitarget.artifacts import canonical_json_bytes
from dllm_agent.experiments.multitarget.ledger import JsonValue
from dllm_agent.experiments.openexr_one_day.models import MODEL_ARMS

if TYPE_CHECKING:
    from pathlib import Path

    from dllm_agent.experiments.openexr_one_day.candidate_bank import CandidateBank


class SeedConfig(Protocol):
    """Config fields required to seed a private corpus."""

    @property
    def c0_sha256(self) -> str:
        """Configured C0 corpus digest."""
        ...

    @property
    def c0_source_path(self) -> Path:
        """Configured exact C0 fixture path."""
        ...


class SeedCell(Protocol):
    """Cell fields required to seed a private corpus."""

    @property
    def artifact_root(self) -> Path:
        """Private cell artifact root."""
        ...

    @property
    def arm(self) -> str:
        """Scheduled campaign arm."""
        ...


_MANIFEST_NAME: Final = "seed-manifest.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class CorpusSeedSummary(FrozenModel):
    """Summary of C0 and selected-bank bytes materialized into one private corpus."""

    c0_sha256: Sha256
    bank_parent_count: int = Field(ge=0)
    bank_parent_sha256: tuple[Sha256, ...]


def seed_cell_corpus(
    config: SeedConfig,
    cell: SeedCell,
    bank: CandidateBank | None = None,
) -> CorpusSeedSummary:
    """Materialize C0 for every cell and selected-parent bytes for model arms."""
    cell.artifact_root.mkdir(parents=True, exist_ok=True)
    cell.artifact_root.joinpath("corpus").mkdir(parents=True, exist_ok=True)
    from dllm_agent.experiments.openexr_one_day.c0_source import (  # noqa: PLC0415
        ExactC0Source,
        load_exact_c0_source,
    )

    c0 = load_exact_c0_source(ExactC0Source(path=config.c0_source_path, sha256=config.c0_sha256))
    c0_path = cell.artifact_root / "corpus" / f"c0-{c0.sha256}.seed"
    if not c0_path.exists():
        _ = c0_path.write_bytes(c0.content)
    parent_hashes = _materialize_parents(cell.artifact_root / "corpus", bank) if bank else ()
    if not parent_hashes:
        parent_hashes = _load_manifest_parent_hashes(cell.artifact_root)
    if cell.arm not in MODEL_ARMS:
        parent_hashes = ()
    summary = CorpusSeedSummary(
        c0_sha256=config.c0_sha256,
        bank_parent_count=len(parent_hashes),
        bank_parent_sha256=parent_hashes,
    )
    _persist_seed_manifest(cell.artifact_root, summary)
    return summary


def _materialize_parents(corpus_root: Path, bank: CandidateBank) -> tuple[str, ...]:
    hashes: list[str] = []
    for index, parent in enumerate(bank.selected_parents):
        actual = sha256_bytes(parent.content)
        if actual != parent.materialized_sha256:
            message = "candidate bank selected parent materialization hash mismatch"
            raise ValueError(message)
        path = corpus_root / f"bank-parent-{index:02d}-{actual}.exr"
        if not path.exists():
            _ = path.write_bytes(parent.content)
        hashes.append(actual)
    return tuple(hashes)


def _load_manifest_parent_hashes(cell_root: Path) -> tuple[str, ...]:
    path = cell_root / _MANIFEST_NAME
    if not path.exists():
        return ()
    try:
        payload = _JSON_OBJECT.validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return ()
    values = payload.get("bank_parent_sha256", ())
    if not isinstance(values, list | tuple):
        return ()
    return tuple(value for value in values if isinstance(value, str))


def _persist_seed_manifest(cell_root: Path, summary: CorpusSeedSummary) -> None:
    path = cell_root / _MANIFEST_NAME
    payload = canonical_json_bytes(summary.model_dump(mode="json"))
    if path.exists() and path.read_bytes() == payload:
        return
    _ = path.write_bytes(payload)
