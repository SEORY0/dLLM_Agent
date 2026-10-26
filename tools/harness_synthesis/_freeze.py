"""RQ4b P1 freeze: which files define the protocol, their recorded hashes, reserved seeds.

``rq4b_freeze.py`` writes the manifest once, before execution. ``rq4b_functional_pilot.py``
calls ``verify_freeze`` before any GPU run (P1 or smoke) and refuses to start if a
frozen file differs from its recorded hash. Changing the protocol after the freeze
therefore requires a new, separately recorded manifest; it cannot happen silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "docs/assets/harness-rq4b-p1-freeze-20261026.json"

# Everything whose content can change what P1 generates, evaluates or concludes.
FROZEN_FILES: tuple[str, ...] = (
    "docs/superpowers/specs/2026-10-26-rq4b-functional-repair-benchmark.md",
    "docs/assets/harness-rq4b-benchmark-validation-20261026.json",
    "src/dllm_agent/harness_synthesis/body_analysis.py",
    "src/dllm_agent/harness_synthesis/functional_eval.py",
    "src/dllm_agent/harness_synthesis/localize.py",
    "src/dllm_agent/harness_synthesis/remask.py",
    "src/dllm_agent/harness_synthesis/repair_arms.py",
    "src/dllm_agent/harness_synthesis/repair_benchmark.py",
    "src/dllm_agent/harness_synthesis/repair_stats.py",
    "tools/harness_synthesis/_compile.py",
    "tools/harness_synthesis/_dream_generate.py",
    "tools/harness_synthesis/_dream_tokenize.py",
    "tools/harness_synthesis/_freeze.py",
    "tools/harness_synthesis/_functional_eval.py",
    "tools/harness_synthesis/_replay.py",
    "tools/harness_synthesis/kb_diversity_experiment.py",
    "tools/harness_synthesis/kb_v3_fixed_scaffold.py",
    "tools/harness_synthesis/rq4b_analyze.py",
    "tools/harness_synthesis/rq4b_functional_pilot.py",
    "tools/openexr_runtime/CMakeLists.txt",
    "tools/openexr_runtime/harness-compile-only/build-harness",
    "fixtures/openexr/c0-validity-preserving-zero.hex",
)

# Inclusive seed ranges used or reserved by earlier and planned stages; a smoke test
# must stay outside all of them. 9M-12.999M: RQ1/RQ3/RQ4 (12M reserved by RQ4).
# 13.0M-13.499M: RQ4b pilots. 13.5M-13.999M: RQ4b confirmatory.
RESERVED_SEED_BLOCKS: tuple[tuple[int, int], ...] = (
    (9_000_000, 12_999_999),
    (13_000_000, 13_499_999),
    (13_500_000, 13_999_999),
)


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict[str, str]:
    """Hashes of every frozen file as it is on disk now."""
    return {name: sha256_file(REPO / name) for name in FROZEN_FILES}


def verify_freeze() -> list[str]:
    """Problems found comparing disk to the manifest; empty means frozen and unchanged."""
    if not MANIFEST.exists():
        return [f"no freeze manifest at {MANIFEST.relative_to(REPO)}"]
    recorded: dict[str, str] = json.loads(MANIFEST.read_text())["file_sha256"]
    problems = [f"not in manifest: {n}" for n in FROZEN_FILES if n not in recorded]
    problems += [
        f"changed since freeze: {name}"
        for name, digest in current_hashes().items()
        if name in recorded and recorded[name] != digest
    ]
    return problems
