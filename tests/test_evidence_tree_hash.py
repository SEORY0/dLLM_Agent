"""Evidence-tree hash command tests."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools/evidence_tree_sha256.py"


def test_evidence_tree_command_hashes_sorted_paths_and_file_digests(tmp_path: Path) -> None:
    # Given
    evidence = tmp_path / "evidence"
    nested = evidence / "nested"
    nested.mkdir(parents=True)
    _ = (nested / "b.bin").write_bytes(b"\x00\xff")
    _ = (evidence / "a.txt").write_bytes(b"A")

    # When
    result = subprocess.run(  # noqa: S603 - checked-in read-only tool
        (sys.executable, str(SCRIPT), str(evidence)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then
    expected = "4f3ac87e95b85ca9a32906c7b9ecaaa4a34ca912b45750b5fb40fa7c996ba8ae"
    assert result.returncode == 0
    assert result.stdout == f"{expected}  {evidence}\n"
    assert result.stderr == ""
    assert (evidence / "a.txt").read_bytes() == b"A"
    assert (nested / "b.bin").read_bytes() == b"\x00\xff"
