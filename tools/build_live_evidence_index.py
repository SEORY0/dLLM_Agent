"""Build the deterministic index for the 2026-08-19 jsoncons live hunt."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_FILES = (
    "configs/jsoncons-ubjson-hunt.json",
    "configs/jsoncons-ubjson-hunt-nonempty.json",
    "prompts/jsoncons-ubjson-harness.md",
    "prompts/jsoncons-ubjson-harness-nonempty.md",
    "artifacts/live/jsoncons-ubjson-prompt-attempt-01.md",
    "artifacts/live/jsoncons-ubjson-nonempty-prompt-attempt-01.md",
    "artifacts/live/jsoncons-ubjson-smoke-candidate-bank.json",
    "artifacts/live/jsoncons-ubjson-candidate-bank-attempt-01.json",
    "artifacts/live/jsoncons-ubjson-candidate-bank.json",
    "artifacts/live/jsoncons-ubjson-nonempty-candidate-bank-attempt-01.json",
    "artifacts/live/jsoncons-ubjson-nonempty-candidate-bank.json",
    "artifacts/live/jsoncons-ubjson-report-attempt-01.json",
    "artifacts/live/jsoncons-ubjson-run-attempt-01/build.log",
    "artifacts/live/jsoncons-ubjson-report.json",
    "artifacts/live/jsoncons-ubjson-run-final-20260819/fuzz.log",
    "artifacts/live/jsoncons-ubjson-report-repro-01.json",
    "artifacts/live/jsoncons-ubjson-run-repro-01/fuzz.log",
    "artifacts/live/jsoncons-ubjson-minimal-reproducer.cpp",
    "artifacts/live/jsoncons-ubjson-minimal-triage/triage.json",
    "artifacts/live/jsoncons-ubjson-minimal-triage/sanitizer-matrix.json",
    "artifacts/live/jsoncons-ubjson-nonempty-report.json",
    "artifacts/live/jsoncons-ubjson-nonempty-run-20260819/fuzz.log",
    (
        "artifacts/live/jsoncons-ubjson-nonempty-run-20260819/crashes/"
        "crash-cdd48c7dd3cafdf219bb389866dfe1c8ce3bc46d"
    ),
    "artifacts/live/jsoncons-ubjson-zero-string-reproducer.cpp",
    "artifacts/live/jsoncons-ubjson-zero-string-triage/sanitizer-matrix.json",
    "artifacts/live/jsoncons-ubjson-nonempty-asan-report.json",
    "artifacts/live/jsoncons-ubjson-nonempty-asan-run-20260819/fuzz.log",
    "artifacts/live/jsoncons-ubjson-nonempty-asan-run-20260819/coverage-report.txt",
    "artifacts/live/jsoncons-ubjson-final-code-smoke-report.json",
    "artifacts/live/jsoncons-ubjson-final-code-smoke/fuzz.log",
    "artifacts/live/jsoncons-ubjson-final-code-smoke/coverage-report.txt",
    "docs/live-hunt-results-20260819.md",
)

MODEL_FILES = (
    {
        "name": "model-00001-of-00004.safetensors",
        "sha256": "77a8bc0635b404f9c5ba4d718df63937aee18fab86c4260854b54805c86a876f",
        "size": 4_924_209_128,
    },
    {
        "name": "model-00002-of-00004.safetensors",
        "sha256": "00aea7f9d293b05ee5aeaf5b849c802cfd9c762d1d493b0a0b56e65ba3ce5a0a",
        "size": 4_915_916_128,
    },
    {
        "name": "model-00003-of-00004.safetensors",
        "sha256": "420649c9036b55fb5dd32fb56095946d5fd2dd8a8bc831081d6c8e8ebb6f7911",
        "size": 4_999_819_336,
    },
    {
        "name": "model-00004-of-00004.safetensors",
        "sha256": "dc291c1d4b67456be5c5e22a51c9885fc4890065c7fc74649a6fb4b1ba56d740",
        "size": 1_661_013_648,
    },
    {
        "name": "tokenizer.json",
        "sha256": "2c9765e3f23ba5809c82adcaf0851d83f6b294f2503f6956fcaa31f62c4dac6d",
        "size": 11_891_620,
    },
    {
        "name": "modeling_stable_diffcoder.py",
        "sha256": "298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104",
        "size": 10_030,
    },
)


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _file_evidence(relative_path: str) -> dict[str, str | int]:
    path = ROOT / relative_path
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _matrix_result(matrix: dict[str, Any], profile: str) -> dict[str, Any]:
    result = matrix["results"][profile]
    return {
        "returncode": result["run_returncode"],
        "asan_observed": result["asan_observed"],
        "ubsan_observed": result["ubsan_observed"],
        "run_log_sha256": result["run_log_sha256"],
    }


def _build_index() -> dict[str, Any]:
    primary = _read_json("artifacts/live/jsoncons-ubjson-report.json")
    nonempty = _read_json("artifacts/live/jsoncons-ubjson-nonempty-report.json")
    asan = _read_json("artifacts/live/jsoncons-ubjson-nonempty-asan-report.json")
    smoke = _read_json("artifacts/live/jsoncons-ubjson-final-code-smoke-report.json")
    finding_a = _read_json("artifacts/live/jsoncons-ubjson-minimal-triage/triage.json")
    matrix_a = _read_json("artifacts/live/jsoncons-ubjson-minimal-triage/sanitizer-matrix.json")
    matrix_b = _read_json("artifacts/live/jsoncons-ubjson-zero-string-triage/sanitizer-matrix.json")

    target = primary["target"]
    expected_target = target["commit_sha"]
    for report in (nonempty, asan, smoke):
        if report["target"]["commit_sha"] != expected_target:
            detail = "hunt reports do not reference one target commit"
            raise ValueError(detail)
    if smoke["fuzz"]["oracle_status"] != "proved":
        detail = "final-code manual QA did not reproduce the sanitizer oracle"
        raise ValueError(detail)
    if asan["fuzz"]["requested_sanitizers"] != "address":
        detail = "ASan continuation report has the wrong sanitizer profile"
        raise ValueError(detail)

    selected = primary["candidate_scores"][0]
    return {
        "schema_version": "dllm-live-evidence-index/v1",
        "execution_date": "2026-08-19",
        "status": "reproduced_ubsan_only_findings_unconfirmed",
        "claim_boundary": {
            "engineering_run_complete": True,
            "d_llm_superiority_established": False,
            "zero_day_confirmed": False,
            "exploitability_confirmed": False,
            "duplicate_status": "unconfirmed",
            "external_submission": False,
        },
        "model": {
            **primary["model"],
            "available_gpus": {"model": "NVIDIA GeForce RTX 3090", "count": 8},
            "load_strategy": "single_gpu_for_two_candidate_baseline",
            "measured_gpu_bytes": {
                "allocated": 16_500_925_440,
                "reserved": 16_636_706_816,
            },
            "verified_files": MODEL_FILES,
        },
        "target": target,
        "sandbox": primary["sandbox"],
        "candidate_selection": {
            "candidate_bank_sha256": primary["candidate_bank_sha256"],
            "selected_candidate_id": primary["selected_candidate_id"],
            "selected_harness_sha256": primary["selected_harness_sha256"],
            "score": selected["score"],
            "trajectory_confidence": selected["trajectory_confidence"],
            "nfe": selected["nfe"],
        },
        "findings": (
            {
                "finding_id": "A",
                "classification": "ubsan_only_undefined_behavior",
                "observed_location": finding_a["observed_location"],
                "input_size": finding_a["input_size"],
                "input_sha256": finding_a["input_sha256"],
                "source_sha256": finding_a["source_sha256"],
                "asan_only": _matrix_result(matrix_a, "asan-only"),
                "ubsan_only": _matrix_result(matrix_a, "ubsan-only"),
            },
            {
                "finding_id": "B",
                "classification": "ubsan_only_undefined_behavior",
                "observed_location": matrix_b["observed_location"],
                "input_size": matrix_b["input_size"],
                "input_sha256": matrix_b["input_sha256"],
                "source_sha256": matrix_b["source_sha256"],
                "asan_only": _matrix_result(matrix_b, "asan-only"),
                "ubsan_only": _matrix_result(matrix_b, "ubsan-only"),
            },
        ),
        "asan_campaign": {
            "report_sha256": _file_evidence(
                "artifacts/live/jsoncons-ubjson-nonempty-asan-report.json"
            )["sha256"],
            "requested_sanitizers": asan["fuzz"]["requested_sanitizers"],
            "fuzz_status": asan["fuzz"]["fuzz_status"],
            "oracle_status": asan["fuzz"]["oracle_status"],
            "executed_units": asan["fuzz"]["executed_units"],
            "executions_per_second": asan["fuzz"]["executions_per_second"],
            "edge_coverage": asan["fuzz"]["edge_coverage"],
            "feature_coverage": asan["fuzz"]["feature_coverage"],
            "final_corpus_count": asan["fuzz"]["corpus_count"],
            "preserved_corpus_count": len(asan["fuzz"]["corpus_artifacts"]),
            "crash_count": len(asan["fuzz"]["crash_artifacts"]),
            "llvm_coverage_percent": asan["coverage"]["totals"],
        },
        "final_code_manual_qa": {
            "surface": "dllm-agent hunt",
            "report_sha256": _file_evidence(
                "artifacts/live/jsoncons-ubjson-final-code-smoke-report.json"
            )["sha256"],
            "selected_candidate_id": smoke["selected_candidate_id"],
            "build_status": smoke["fuzz"]["build_status"],
            "fuzz_status": smoke["fuzz"]["fuzz_status"],
            "oracle_status": smoke["fuzz"]["oracle_status"],
            "sanitizer": smoke["fuzz"]["sanitizer"],
            "executed_units": smoke["fuzz"]["executed_units"],
            "crash_input_sha256": smoke["fuzz"]["crash_artifacts"][0]["sha256"],
        },
        "files": {path: _file_evidence(path) for path in EVIDENCE_FILES},
    }


def main() -> None:
    """Write the index after checking cross-report invariants."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/live/jsoncons-ubjson-evidence-index.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_build_index(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
