#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run tools/evidence_tree_sha256.py EVIDENCE_ROOT [EVIDENCE_ROOT ...]
# 3. Or make executable and run:
#      chmod +x tools/evidence_tree_sha256.py && ./tools/evidence_tree_sha256.py EVIDENCE_ROOT
# ──────────────────

"""Compute a reproducible SHA-256 over a regular-file evidence tree."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

_FIRST_ROOT_ARGUMENT: Final = 1
_USAGE_ERROR: Final = 2


class EvidenceTreeError(Exception):
    """A requested evidence root cannot be hashed unambiguously."""

    def __init__(self, path: Path, detail: str) -> None:
        """Store the rejected path and stable failure detail."""
        self.path: Path = path
        self.detail: str = detail
        super().__init__(f"{path}: {detail}")


def evidence_tree_sha256(root: Path) -> str:
    """Hash sorted UTF-8 relative paths, NUL delimiters, and raw file digests."""
    if root.is_symlink() or not root.is_dir():
        raise EvidenceTreeError(root, "not a non-symlink directory")
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in entries:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise EvidenceTreeError(path, "not a regular file or directory")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main(arguments: Sequence[str]) -> int:
    """Print one reproducible digest for every requested evidence root."""
    if len(arguments) <= _FIRST_ROOT_ARGUMENT:
        _ = sys.stderr.write("usage: evidence_tree_sha256.py EVIDENCE_ROOT [EVIDENCE_ROOT ...]\n")
        return _USAGE_ERROR
    try:
        results = tuple(
            (Path(raw), evidence_tree_sha256(Path(raw))) for raw in arguments[_FIRST_ROOT_ARGUMENT:]
        )
    except (EvidenceTreeError, OSError, UnicodeError) as error:
        _ = sys.stderr.write(f"{error}\n")
        return _USAGE_ERROR
    for root, digest in results:
        _ = sys.stdout.write(f"{digest}  {root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
