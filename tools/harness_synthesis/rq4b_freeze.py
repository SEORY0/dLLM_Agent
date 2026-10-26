# ruff: noqa: T201 -- deliberate console report, not a library module
"""Freeze RQ4b P1: record file hashes, the full seed/mask allocation, and the environment.

CPU only. It tokenizes with Dream's tokenizer files, compiles the mutants in the pinned
image for their diagnostics, and reads model metadata files (``config.json``, the
safetensors index, and file sizes). It never loads weights or touches a GPU.

It refuses to freeze from a dirty tree: every frozen file must be committed, so that
``code_commit`` identifies the frozen code exactly.

The allocation table lists, for every one of P1's 60 units, every generation call:
arm, frozen order index, seed, seed slot, masked positions, steps. It also checks that
the infill length/slot assignment is balanced (each length meets each slot once per
sample).

    uv run python tools/harness_synthesis/rq4b_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from _compile import IMAGE
from _freeze import FROZEN_FILES, MANIFEST, REPO, RESERVED_SEED_BLOCKS, current_hashes, sha256_file
from _replay import DREAM_VENV_PYTHON, MODEL_PATH
from rq4b_functional_pilot import prepare

from dllm_agent.harness_synthesis.repair_arms import (
    INFILL_MASK_OFFSETS,
    PILOT_REPLICATES,
    PILOT_SEED_BASE,
    RESAMPLE_NEW_TOKENS,
    SUFFIX_SLACK,
    K,
    plan,
    seed_slot,
    seed_slot_balance,
)
from dllm_agent.harness_synthesis.repair_benchmark import samples


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        ["git", "-C", str(REPO), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _environment() -> dict[str, object]:
    model = Path(MODEL_PATH)
    image = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    transformers = subprocess.run(  # noqa: S603 - fixed argv, shell=False.
        [
            str(DREAM_VENV_PYTHON),
            "-c",
            "import transformers, torch; print(transformers.__version__, torch.__version__)",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"CUDA_VISIBLE_DEVICES": ""},
    ).stdout.split()
    return {
        "compile_image": IMAGE,
        "compile_image_id": image,
        "model_path": MODEL_PATH,
        "model_config_sha256": sha256_file(model / "config.json"),
        "model_safetensors_index_sha256": sha256_file(model / "model.safetensors.index.json")
        if (model / "model.safetensors.index.json").exists()
        else None,
        "model_weight_file_sizes": {
            p.name: p.stat().st_size for p in sorted(model.glob("*.safetensors"))
        },
        "dream_venv": {"transformers": transformers[0], "torch": transformers[1]},
        "host_python": platform.python_version(),
    }


def main() -> None:
    """Write the manifest; refuse if the tree is dirty or the allocation is unbalanced."""
    dirty = _git("status", "--porcelain", "--", *FROZEN_FILES)
    if dirty:
        raise SystemExit("commit the frozen files first:\n" + dirty)
    bench = list(samples())
    prepared = prepare(bench)
    allocation = []
    balance_ok = True
    for sample_index, prep in enumerate(prepared):
        sample_plans = []
        for replicate in range(PILOT_REPLICATES):
            plans = plan(
                prep.window,
                len(prep.ids),
                sample_index=sample_index,
                replicate=replicate,
                base=PILOT_SEED_BASE,
                oracle_length=prep.reference_window.length,
            )
            sample_plans.extend(plans)
            allocation.extend(
                {
                    "sample_id": prep.sample.sample_id,
                    "replicate": replicate,
                    **asdict(p),
                    "seed_slot": seed_slot(p.seed, base=PILOT_SEED_BASE),
                }
                for p in plans
            )
        balance = seed_slot_balance(sample_plans, base=PILOT_SEED_BASE)
        balance_ok &= all(slots == set(range(K)) for slots in balance.values())
    if not balance_ok:
        reason = "infill length/seed-slot assignment is not balanced"
        raise SystemExit(reason)
    seeds = [a["seed"] for a in allocation]
    if len(set(seeds)) != len(
        {(a["sample_id"], a["replicate"], a["seed_slot"]) for a in allocation}
    ):
        reason = "seed collision between distinct slots"
        raise SystemExit(reason)
    table = json.dumps(allocation, sort_keys=True).encode()
    manifest = {
        "status": "RQ4b P1 frozen before execution; no GPU run has used this protocol yet",
        "code_commit": _git("rev-parse", "HEAD"),
        "file_sha256": current_hashes(),
        "protocol": {
            "samples": [s.sample_id for s in bench],
            "replicates": PILOT_REPLICATES,
            "K": K,
            "infill_mask_offsets_in_selection_order": list(INFILL_MASK_OFFSETS),
            "infill_seed_slot_rule": "length position k in replicate r uses slot (k + r) % K",
            "suffix_slack": SUFFIX_SLACK,
            "resample_new_tokens": RESAMPLE_NEW_TOKENS,
            "seed_formula": "13_000_000 + 1000*sample_index + 10*replicate + slot",
            "diagnostic_seed_slot": 9,
            "feedback": "none",
            "generations": len(allocation),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
        },
        "reserved_seed_blocks": RESERVED_SEED_BLOCKS,
        "allocation_sha256": hashlib.sha256(table).hexdigest(),
        "allocation": allocation,
        "environment": _environment(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)}  sha256={sha256_file(MANIFEST)}")
    print(f"allocation_sha256={manifest['allocation_sha256']}  generations={len(allocation)}")
    print(f"seeds {min(seeds)}-{max(seeds)}  code_commit={manifest['code_commit']}")


if __name__ == "__main__":
    main()
